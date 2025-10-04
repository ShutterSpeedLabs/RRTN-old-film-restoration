import sys
import os
import cv2
import importlib
import argparse
import yaml
from tqdm import tqdm
import gc
import time
import random
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.dirname(sys.path[0]))

from VP_code.data.dataset import Film_dataset_1
from VP_code.utils.util import frame_to_video
from VP_code.utils.data_util import tensor2img
from VP_code.metrics.psnr_ssim import calculate_psnr


def main_worker(local_rank, config_dict, opts):
    if opts.distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f'using GPU {local_rank} for inference')

        torch.distributed.init_process_group(
            backend='nccl',
            init_method=opts.dist_url,
            world_size=opts.world_size,
            rank=local_rank,
            group_name='mtorch'
        )
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)
        print('using single GPU for inference')

    opts.local_rank = local_rank
    opts.global_rank = local_rank  # For single node, global_rank = local_rank

    # Run main inference
    main_inference(config_dict, opts)


def main_inference(config_dict, opts):
    """Main inference function that processes video folders"""
    # Get list of video folders
    video_folders = [d for d in os.listdir(opts.input_video_url)
                    if os.path.isdir(os.path.join(opts.input_video_url, d))]
    video_folders.sort()  # Process in sorted order

    print(f"\nFound {len(video_folders)} folders to process:")
    for i, folder in enumerate(video_folders, 1):
        print(f"{i}. {folder}")

    # Process each folder
    for i, folder in enumerate(video_folders, 1):
        print(f"\n{'='*50}")
        print(f"Processing folder {i}/{len(video_folders)}: {folder}")
        print(f"{'='*50}")

        folder_path = os.path.join(opts.input_video_url, folder)

        # Set input paths for current folder
        config_dict['datasets']['val']['dataroot_gt'] = folder_path
        config_dict['datasets']['val']['dataroot_lq'] = folder_path
        config_dict['val']['val_frame_num'] = opts.temporal_length

        # Process single folder
        process_single_folder(opts, config_dict, folder_path)

        # Force cleanup between folders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"Completed {i}/{len(video_folders)} folders")


def load_model(opts, which_model='first'):
    """Load model with distributed support"""
    assert which_model in ['first', 'second']

    net = importlib.import_module('VP_code.models.' + opts.model_name)
    netG = net.Video_Backbone()

    if which_model == 'first':
        model_path = opts.model_path_first
    elif which_model == 'second':
        model_path = opts.model_path_second
    else:
        raise ValueError('`which_model` should be "first" or "second"')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    netG.load_state_dict(checkpoint['netG'])

    # Move to GPU and wrap with DDP if distributed
    if opts.distributed and torch.cuda.is_available():
        netG.cuda(opts.local_rank)
        netG = DDP(netG, device_ids=[opts.local_rank], find_unused_parameters=True)
    elif torch.cuda.is_available():
        netG.cuda()
    else:
        print("Warning: CUDA not available, using CPU")
        netG.cpu()

    print("Finish loading model ...")
    return netG


def load_dataset(config_dict, opts):
    """Load dataset with distributed sampler support"""
    val_dataset = Film_dataset_1(config_dict['datasets']['val'])

    # Use distributed sampler if in distributed mode
    if opts.distributed:
        val_sampler = DistributedSampler(val_dataset, num_replicas=opts.world_size, rank=opts.global_rank)
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=val_sampler
    )

    print("Finish loading dataset ...")
    print("Test set statistics:")
    print(f'\n\tNumber of test videos: {len(val_dataset)}')
    return val_loader


def validation(opts, config_dict, loaded_model, val_loader, recursion_step=1):
    """Validation function with distributed support"""
    psnr = 0.0

    # Get the actual model (unwrap DDP if needed)
    if opts.distributed:
        model = loaded_model.module
    else:
        model = loaded_model

    model.eval()

    video_pbar = tqdm(val_loader, desc=f"Processing videos (Recursion {recursion_step})", leave=True)

    for val_data in video_pbar:
        val_frame_num = config_dict['val']['val_frame_num']
        all_len = val_data['lq'].shape[1]
        all_output = []

        clip_name, _ = val_data['key'][0].split('/')
        test_clip_par_folder = val_data['video_name'][0]
        frame_name_list = val_data['name_list']

        frame_pbar = tqdm(range(0, all_len, opts.temporal_stride),
                         desc=f"Processing frames for {clip_name}",
                         leave=False)

        for i in frame_pbar:
            current_part = {}
            current_part['lq'] = val_data['lq'][:, i:min(i + val_frame_num, all_len), :, :, :]
            current_part['gt'] = val_data['gt'][:, i:min(i + val_frame_num, all_len), :, :, :]
            current_part['key'] = val_data['key']
            current_part['frame_list'] = val_data['frame_list'][i:min(i + val_frame_num, all_len)]

            part_lq = current_part['lq']
            if torch.cuda.is_available():
                part_lq = part_lq.cuda()

            with torch.no_grad():
                try:
                    # Get the actual model for inference
                    if opts.distributed:
                        model = loaded_model.module
                    else:
                        model = loaded_model
                    part_output = model(part_lq)
                    if torch.cuda.is_available():
                        part_output = part_output.cpu()
                except RuntimeError as e:
                    print("Warning: runtime error", e)
                    part_output = part_lq.clone()

            if i == 0:
                all_output.append(part_output.detach().squeeze(0))
            else:
                restored_temporal_length = min(i + val_frame_num, all_len) - i - (
                    val_frame_num - opts.temporal_stride)
                all_output.append(part_output[:, 0 - restored_temporal_length:, :, :, :]
                                .detach().squeeze(0))

            del part_lq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_output = torch.cat(all_output, dim=0)
        gt = val_data['gt'].squeeze(0)
        lq = val_data['lq'].squeeze(0)
        if config_dict['datasets']['val']['normalizing']:
            val_output = (val_output + 1) / 2
            gt = (gt + 1) / 2
            lq = (lq + 1) / 2

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gt_imgs = []
        sr_imgs = []

        img_pbar = tqdm(range(len(val_output)), desc="Converting tensors to images", leave=False)
        for j in img_pbar:
            gt_imgs.append(tensor2img(gt[j]))
            sr_imgs.append(tensor2img(val_output[j]))

        save_pbar = tqdm(enumerate(sr_imgs), desc="Saving processed images", leave=False)
        for id, sr_img in save_pbar:
            save_place = os.path.join(opts.save_place, opts.name,
                                    'test_results_' + str(opts.temporal_length) + "_rec" + str(recursion_step),
                                    test_clip_par_folder, clip_name, frame_name_list[id][0])
            dir_name = os.path.abspath(os.path.dirname(save_place))
            os.makedirs(dir_name, exist_ok=True)
            cv2.imwrite(save_place, sr_img)

        if test_clip_par_folder == os.path.basename(opts.input_video_url):
            input_clip_url = os.path.join(opts.input_video_url, clip_name)
        else:
            input_clip_url = os.path.join(opts.input_video_url, test_clip_par_folder, clip_name)

        restored_clip_url = os.path.join(opts.save_place, opts.name,
                                       'test_results_' + str(opts.temporal_length) + "_rec" + str(recursion_step),
                                       test_clip_par_folder, clip_name)
        video_save_url = os.path.join(opts.save_place, opts.name,
                                     'test_results_' + str(opts.temporal_length) + "_rec" + str(recursion_step),
                                     test_clip_par_folder, clip_name + '.avi')

        print(f"Converting frames to video for {clip_name}")
        frame_to_video(input_clip_url, restored_clip_url, video_save_url)

        psnr_pbar = tqdm(zip(sr_imgs, gt_imgs), total=len(sr_imgs), desc="Calculating PSNR", leave=False)
        psnr_this_video = [calculate_psnr(sr, gt) for sr, gt in psnr_pbar]
        psnr += sum(psnr_this_video) / len(psnr_this_video)

        video_pbar.set_postfix({'Current PSNR': f"{psnr_this_video[-1]:.2f}"})

    psnr /= len(val_loader)
    print(f'# Average PSNR: {psnr:.2f}')
    return psnr


def process_single_folder(opts, config_dict, folder_path):
    """Process a single video folder with recursion support"""
    print(f"Processing {os.path.basename(folder_path)}")

    # First recursion
    print("Loading first model...")
    loaded_model = load_model(opts, which_model='first')
    val_loader = load_dataset(config_dict, opts)

    print('=======')
    print('Recursion: 1')
    psnr = validation(opts, config_dict, loaded_model, val_loader, recursion_step=1)

    # Clear first model
    del loaded_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Further recursions
    print("Loading second model...")
    loaded_model = load_model(opts, which_model='second')
    for i in range(opts.max_recursion - 1):
        if psnr >= opts.recursion_threshold:
            break

        recursion_step = i + 2
        print('=======')
        print(f'Recursion: {recursion_step}')

        video_url = os.path.join(
            opts.save_place,
            opts.name,
            'test_results_' + str(opts.temporal_length) + f"_rec{recursion_step-1}",
            os.path.basename(folder_path)
        )
        config_dict['datasets']['val']['dataroot_gt'] = video_url
        config_dict['datasets']['val']['dataroot_lq'] = video_url

        val_loader = load_dataset(config_dict, opts)
        psnr = validation(opts, config_dict, loaded_model, val_loader, recursion_step=recursion_step)

    # Clear second model
    del loaded_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='The name of this experiment')
    parser.add_argument('--model_name', type=str, default='', help='The name of adopted model')
    parser.add_argument('--model_path_first', type=str, default='pretrained_models/rrtn_128_first.pth', help='Path to the first pretrained model')
    parser.add_argument('--model_path_second', type=str, default='pretrained_models/rrtn_128_second.pth', help='Path to the second pretrained model')
    parser.add_argument('--input_video_url', type=str, default='', help='degraded video input')
    parser.add_argument('--temporal_length', type=int, default=15, help='How many frames should be processed in one forward')
    parser.add_argument('--temporal_stride', type=int, default=3, help='Stride value while sliding window')
    parser.add_argument('--recursion_threshold', type=float, default=43, help='Threshold for further recursion. If PSNR between the input video and output video is smaller than the threshold, the recursion will be continued. Default is 43.')
    parser.add_argument('--max_recursion', type=int, default=4, help='Max recursion steps. Default is 4.')
    parser.add_argument('--save_place', type=str, default='OUTPUT', help='save place')

    # DDP arguments
    parser.add_argument('--gpus', type=int, default=None, help='how many GPUs in one node (default: auto-detect)')
    parser.add_argument('--node_rank', type=int, default=0, help='the id of this machine (default: only one machine with id 0)')
    parser.add_argument('--dist_url', type=str, default="", help='Port Address')

    opts = parser.parse_args()
    opts.isTrain = False

    # Auto-detect available GPUs if not specified
    if opts.gpus is None:
        if torch.cuda.is_available():
            opts.gpus = torch.cuda.device_count()
            print(f"Auto-detected {opts.gpus} GPU(s)")
        else:
            opts.gpus = 1
            print("No GPUs detected, using CPU")

    # Validate GPU count
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if opts.gpus > available_gpus:
            print(f"Warning: Requested {opts.gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
            opts.gpus = available_gpus

    opts.world_size = opts.gpus

    with open(os.path.join('./configs', opts.name + '.yaml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # Used for the communication of multi-host
    if opts.dist_url == "":
        port_num = str(random.randint(20000, 30000))
        opts.dist_url = 'tcp://127.0.0.1:' + port_num

    # Validate GPU setup
    if opts.gpus > 1 and not torch.cuda.is_available():
        print("Error: Multiple GPUs requested but CUDA is not available. Using single GPU mode.")
        opts.gpus = 1
        opts.distributed = False
    elif opts.gpus == 1:
        opts.distributed = False
    else:
        opts.distributed = True

    if opts.distributed:
        mp.spawn(main_worker, nprocs=opts.gpus, args=(config_dict, opts,))
    else:
        main_inference(config_dict, opts)
