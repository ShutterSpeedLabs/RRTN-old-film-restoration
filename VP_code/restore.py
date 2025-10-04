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
    """Load dataset with distributed sampler support and minimal memory footprint"""
    val_dataset = Film_dataset_1(config_dict['datasets']['val'])

    # Use distributed sampler if in distributed mode
    if opts.distributed:
        val_sampler = DistributedSampler(val_dataset, num_replicas=opts.world_size, rank=opts.global_rank, shuffle=False)
    else:
        val_sampler = None

    # Minimize memory usage: no workers, no pin_memory, batch_size=1
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing overhead
        pin_memory=False,  # Reduce memory usage
        sampler=val_sampler,
        prefetch_factor=None,  # No prefetching
        persistent_workers=False
    )

    print("Finish loading dataset ...")
    print("Test set statistics:")
    print(f'\n\tNumber of test videos: {len(val_dataset)}')
    if opts.distributed:
        print(f'\tGPU {opts.global_rank} will process {len(val_sampler)} videos')
    return val_loader


def validation(opts, config_dict, loaded_model, val_loader, recursion_step=1):
    """Memory-efficient validation with frame-by-frame processing"""
    total_psnr = 0.0
    video_count = 0

    # Get the actual model (unwrap DDP if needed)
    if opts.distributed:
        model = loaded_model.module
    else:
        model = loaded_model

    model.eval()

    video_pbar = tqdm(val_loader, desc=f"Processing videos (Recursion {recursion_step})", leave=True)

    for val_data in video_pbar:
        clip_name, _ = val_data['key'][0].split('/')
        test_clip_par_folder = val_data['video_name'][0]
        frame_name_list = val_data['name_list']
        all_len = val_data['lq'].shape[1]

        # Create output directory
        output_dir = os.path.join(opts.save_place, opts.name,
                                'test_results_' + str(opts.temporal_length) + "_rec" + str(recursion_step),
                                test_clip_par_folder, clip_name)
        os.makedirs(output_dir, exist_ok=True)

        # Process frames one by one to minimize RAM usage
        frame_psnr_values = []
        
        frame_pbar = tqdm(range(all_len), desc=f"Processing frames for {clip_name}", leave=False)
        
        for frame_idx in frame_pbar:
            # Get single frame data
            frame_lq = val_data['lq'][:, frame_idx:frame_idx+1, :, :, :]
            frame_gt = val_data['gt'][:, frame_idx:frame_idx+1, :, :, :]
            frame_name = frame_name_list[frame_idx][0]

            # Move to device
            if torch.cuda.is_available():
                frame_lq = frame_lq.cuda()

            # Process frame
            with torch.no_grad():
                try:
                    frame_output = model(frame_lq)
                    # Move to CPU immediately
                    frame_output = frame_output.cpu()
                    frame_lq_cpu = frame_lq.cpu()
                except RuntimeError as e:
                    print(f"Warning: runtime error for frame {frame_name}: {e}")
                    frame_output = frame_lq.cpu().clone()
                    frame_lq_cpu = frame_lq.cpu()

            # Denormalize if needed
            if config_dict['datasets']['val']['normalizing']:
                frame_output = (frame_output + 1) / 2
                frame_gt = (frame_gt + 1) / 2

            # Convert to images
            gt_img = tensor2img(frame_gt.squeeze(0).squeeze(0))
            sr_img = tensor2img(frame_output.squeeze(0).squeeze(0))

            # Save image immediately
            save_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(save_path, sr_img)

            # Calculate PSNR for this frame
            frame_psnr = calculate_psnr(sr_img, gt_img)
            frame_psnr_values.append(frame_psnr)

            # Aggressive memory cleanup
            del frame_lq, frame_gt, frame_output, frame_lq_cpu, gt_img, sr_img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Periodic garbage collection
            if frame_idx % 10 == 0:
                gc.collect()

        # Calculate average PSNR for this video
        video_psnr = sum(frame_psnr_values) / len(frame_psnr_values) if frame_psnr_values else 0.0
        total_psnr += video_psnr
        video_count += 1

        video_pbar.set_postfix({'Current PSNR': f"{video_psnr:.2f}"})

        # Create video from saved frames
        if test_clip_par_folder == os.path.basename(opts.input_video_url):
            input_clip_url = os.path.join(opts.input_video_url, clip_name)
        else:
            input_clip_url = os.path.join(opts.input_video_url, test_clip_par_folder, clip_name)

        restored_clip_url = output_dir
        video_save_url = os.path.join(opts.save_place, opts.name,
                                     'test_results_' + str(opts.temporal_length) + "_rec" + str(recursion_step),
                                     test_clip_par_folder, clip_name + '.avi')

        print(f"Converting frames to video for {clip_name}")
        frame_to_video(input_clip_url, restored_clip_url, video_save_url)

        # Clean up video data
        del val_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    average_psnr = total_psnr / video_count if video_count > 0 else 0.0
    print(f'# Average PSNR: {average_psnr:.2f}')
    return average_psnr


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

    # Auto-detect and validate GPU count
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        
        # If not specified, use all available GPUs (up to 2)
        if opts.gpus is None:
            opts.gpus = min(available_gpus, 2)
            print(f"Auto-detected {available_gpus} GPU(s), using {opts.gpus} for inference")
        elif opts.gpus > available_gpus:
            print(f"Warning: Requested {opts.gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
            opts.gpus = available_gpus
        else:
            print(f"Using {opts.gpus} GPU(s) for inference")
    else:
        print("Warning: CUDA not available, falling back to CPU")
        opts.gpus = 1

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
