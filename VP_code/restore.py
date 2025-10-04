import sys
import os
import cv2
import importlib
import argparse
import yaml
from tqdm import tqdm
import torch.nn as nn
import gc
import time

sys.path.append(os.path.dirname(sys.path[0]))

import torch

from VP_code.data.dataset import Film_dataset_1
from VP_code.utils.util import frame_to_video
from VP_code.utils.data_util import tensor2img
from VP_code.metrics.psnr_ssim import calculate_psnr

from torch.utils.data import DataLoader


def get_device():
    """Get the best available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPU(s)")
        if n_gpus >= 2:
            print("Using multiple GPUs")
            return 'cuda', True
        else:
            print("Using single GPU")
            return 'cuda', False
    else:
        print("No GPU found, using CPU")
        return 'cpu', False


def get_gpu_memory():
    """Get available memory for each GPU in GB"""
    available_memory = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        available = total_memory - (reserved + allocated)
        available_memory.append(available)
    return available_memory


def load_model(opts, which_model='first'):
    assert which_model in ['first', 'second']

    net = importlib.import_module('VP_code.models.' + opts.model_name)
    netG = net.Video_Backbone()

    if which_model == 'first':
        model_path = opts.model_path_first
    elif which_model == 'second':
        model_path = opts.model_path_second
    else:
        raise ValueError('`which_model` should be "first" or "second"')

    device, use_multi_gpu = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    netG.load_state_dict(checkpoint['netG'])
    
    if use_multi_gpu:
        netG = nn.DataParallel(netG)
    netG.to(device)
    print("Finish loading model ...")

    return netG


def load_dataset(config_dict):
    val_dataset = Film_dataset_1(config_dict['datasets']['val'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, sampler=None)
    print("Finish loading dataset ...")
    print("Test set statistics:")
    print(f'\n\tNumber of test videos: {len(val_dataset)}')
    return val_loader


def validation(opts, config_dict, loaded_model, val_loader, recursion_step=1):
    psnr = 0.0
    loaded_model.eval()
    device = next(loaded_model.parameters()).device

    video_pbar = tqdm(val_loader, desc=f"Processing videos (Recursion {recursion_step})", leave=True)
    
    for val_data in video_pbar:
        val_frame_num = config_dict['val']['val_frame_num']
        all_len = val_data['lq'].shape[1]
        all_output = []

        clip_name, _ = val_data['key'][0].split('/')
        test_clip_par_folder = val_data['video_name'][0]
        frame_name_list = val_data['name_list']

        part_output = None
        # Calculate optimal batch size based on available GPU memory
        available_memory = get_gpu_memory()
        min_memory = min(available_memory)
        batch_size = min(opts.temporal_length, max(1, int(min_memory * 0.4)))  # Use 40% of available memory
        print(f"Processing with batch size: {batch_size}")

        frame_pbar = tqdm(range(0, all_len, batch_size), 
                         desc=f"Processing frames for {clip_name}", 
                         leave=False)
        
        for i in frame_pbar:
            # Clear memory before processing each batch
            torch.cuda.empty_cache()
            gc.collect()

            current_part = {}
            end_idx = min(i + batch_size, all_len)
            current_part['lq'] = val_data['lq'][:, i:end_idx, :, :, :]
            current_part['gt'] = val_data['gt'][:, i:end_idx, :, :, :]
            current_part['key'] = val_data['key']
            current_part['frame_list'] = val_data['frame_list'][i:end_idx]

            # Process on GPUs with memory monitoring
            with torch.no_grad():
                try:
                    part_lq = current_part['lq'].to(device)
                    part_output = loaded_model(part_lq)
                    part_output = part_output.cpu()
                    del part_lq
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    print(f"Warning: runtime error - {e}")
                    print("Reducing batch size and retrying...")
                    batch_size = max(1, batch_size // 2)
                    part_output = current_part['lq'].clone()

            # Process output
            if i == 0:
                all_output.append(part_output.squeeze(0))
            else:
                all_output.append(part_output.squeeze(0))

            del current_part
            gc.collect()

        val_output = torch.cat(all_output, dim=0)
        gt = val_data['gt'].squeeze(0)
        lq = val_data['lq'].squeeze(0)
        if config_dict['datasets']['val']['normalizing']:
            val_output = (val_output + 1) / 2
            gt = (gt + 1) / 2
            lq = (lq + 1) / 2
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
    """Process a single video folder"""
    print(f"\n{'='*50}")
    print(f"Processing {os.path.basename(folder_path)}")
    print(f"{'='*50}")
    
    # Set input paths for current folder
    config_dict['datasets']['val']['dataroot_gt'] = folder_path
    config_dict['datasets']['val']['dataroot_lq'] = folder_path
    config_dict['val']['val_frame_num'] = opts.temporal_length

    # First recursion
    print("Loading first model...")
    loaded_model = load_model(opts, which_model='first')
    val_loader = load_dataset(config_dict)
    
    print('=======')
    print('Recursion: 1')
    psnr = validation(opts, config_dict, loaded_model, val_loader, recursion_step=1)
    
    # Clear first model
    del loaded_model
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

        val_loader = load_dataset(config_dict)
        psnr = validation(opts, config_dict, loaded_model, val_loader, recursion_step=recursion_step)

    # Clear second model
    del loaded_model
    torch.cuda.empty_cache()
    gc.collect()


def wait_for_user():
    """Wait for user confirmation before proceeding"""
    print("\nFolder processing completed. Press Enter when ready to process next folder...")
    input()
    # Additional delay to ensure GPU memory is cleared
    time.sleep(5)


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
    parser.add_argument('--batch_size', type=int, default=5, help='Initial batch size for processing')
    
    opts = parser.parse_args()

    with open(os.path.join('./configs', opts.name + '.yaml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # Get list of video folders
    video_folders = [d for d in os.listdir(opts.input_video_url) 
                    if os.path.isdir(os.path.join(opts.input_video_url, d))]
    video_folders.sort()  # Process in sorted order

    print(f"\nFound {len(video_folders)} folders to process:")
    for i, folder in enumerate(video_folders, 1):
        print(f"{i}. {folder}")

    # Process each folder with user confirmation
    for i, folder in enumerate(video_folders, 1):
        print(f"\nPreparing to process folder {i}/{len(video_folders)}: {folder}")
        folder_path = os.path.join(opts.input_video_url, folder)
        
        process_single_folder(opts, config_dict, folder_path)
        
        # Force cleanup between folders
        torch.cuda.empty_cache()
        gc.collect()
        
        if i < len(video_folders):  # Don't wait after last folder
            wait_for_user()
        
        print(f"Completed {i}/{len(video_folders)} folders")
