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

    # Optimize GPU computation (Priority 1 optimization)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # Enable auto-tuning for faster kernels
        torch.backends.cudnn.deterministic = False  # Allow faster but non-deterministic ops
        print("GPU optimizations enabled: cuDNN benchmark=True")

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
    """Load model with distributed support and optimizations"""
    assert which_model in ['first', 'second']

    net = importlib.import_module('VP_code.models.' + opts.model_name)
    netG = net.Video_Backbone()

    if which_model == 'first':
        model_path = opts.model_path_first
    elif which_model == 'second':
        model_path = opts.model_path_second
    else:
        raise ValueError('`which_model` should be "first" or "second"')

    # Priority 1 Optimization: Load checkpoint directly to target GPU device
    if torch.cuda.is_available():
        if opts.distributed:
            device = torch.device(f'cuda:{opts.local_rank}')
        else:
            device = torch.device('cuda:0')
        checkpoint = torch.load(model_path, map_location=device)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    
    netG.load_state_dict(checkpoint['netG'])
    del checkpoint  # Free memory immediately
    print(f"Model loaded directly to {device if torch.cuda.is_available() else 'CPU'}")

    # Move to GPU and wrap with DDP if distributed
    if opts.distributed and torch.cuda.is_available():
        netG.cuda(opts.local_rank)
        netG = DDP(netG, device_ids=[opts.local_rank], find_unused_parameters=True)
    elif torch.cuda.is_available():
        netG.cuda()
    else:
        print("Warning: CUDA not available, using CPU")
        netG.cpu()

    # Priority 1 Optimization: Disable gradient computation for inference
    netG.eval()
    for param in netG.parameters():
        param.requires_grad = False

    print("Finish loading model with inference optimizations ...")
    return netG


def load_dataset(config_dict, opts):
    """Load dataset with distributed sampler support and optimal memory footprint"""
    val_dataset = Film_dataset_1(config_dict['datasets']['val'])

    # Use distributed sampler if in distributed mode
    if opts.distributed:
        val_sampler = DistributedSampler(val_dataset, num_replicas=opts.world_size, rank=opts.global_rank, shuffle=False)
    else:
        val_sampler = None

    # Optimize num_workers: scale with GPU count to prevent CPU/GPU imbalance
    if torch.cuda.is_available():
        cpu_count = mp.cpu_count()
        # Reserve 2 cores per GPU for worker threads, rest for OS
        num_workers = min(6, max(2, (cpu_count - opts.gpus - 2) // opts.gpus))
        print(f"Auto-scaling num_workers to {num_workers} based on {opts.gpus} GPU(s) and {cpu_count} CPU cores")
    else:
        num_workers = 0
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,  # Enable workers for GPU performance
        pin_memory=torch.cuda.is_available(),  # Pin memory for GPU transfer
        sampler=val_sampler,
        prefetch_factor=max(2, 4 if num_workers > 0 else 0),  # Prefetch 4 batches for better overlap
        persistent_workers=(num_workers > 0)  # Keep workers alive
    )

    print("Finish loading dataset ...")
    print("Test set statistics:")
    print(f'\n\tNumber of test videos: {len(val_dataset)}')
    print(f'\tDataLoader: num_workers={num_workers}, prefetch_factor={max(2, 4 if num_workers > 0 else 0)}, pin_memory={torch.cuda.is_available()}')
    if opts.distributed:
        print(f'\tGPU {opts.global_rank} will process {len(val_sampler)} videos')
    return val_loader


def validation(opts, config_dict, loaded_model, val_loader, recursion_step=1):
    """Memory-efficient validation with sliding window processing"""
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

        # Process frames in chunks (sliding window) to use temporal information
        frame_psnr_values = []
        
        # Initialize frame progress bar
        frame_pbar = tqdm(total=all_len, desc=f"Processing frames for {clip_name}", leave=False)
        
        for start_idx in range(0, all_len, opts.temporal_length):
            end_idx = min(start_idx + opts.temporal_length, all_len)
            current_batch_size = end_idx - start_idx

            # Get chunk data: [1, T, C, H, W]
            chunk_lq = val_data['lq'][:, start_idx:end_idx, :, :, :]
            chunk_gt = val_data['gt'][:, start_idx:end_idx, :, :, :]

            # Move to device asynchronously
            device = None
            if torch.cuda.is_available():
                if opts.distributed:
                    device = torch.device(f'cuda:{opts.local_rank}')
                else:
                    device = torch.device('cuda:0')
                chunk_lq = chunk_lq.to(device, non_blocking=True)
                chunk_gt = chunk_gt.to(device, non_blocking=True)

            # Priority 2 Optimization: Async GPU computation and transfer
            # Process chunk and move output to CPU asynchronously
            with torch.no_grad():
                try:
                    chunk_output = model(chunk_lq)
                    
                    # Move to CPU asynchronously (start transfer, don't wait for it yet)
                    chunk_output_cpu = chunk_output.cpu()
                    chunk_gt_cpu = chunk_gt.cpu()
                except RuntimeError as e:
                    print(f"Warning: runtime error for frames {start_idx}-{end_idx}: {e}")
                    # Fallback: just copy input if model fails
                    chunk_output_cpu = chunk_lq.cpu().clone()
                    chunk_gt_cpu = chunk_gt.cpu()
                
                # Synchronize GPU only when we need the data (overlaps with I/O)
                if device is not None:
                    torch.cuda.synchronize()
                
                chunk_output = chunk_output_cpu
                chunk_gt = chunk_gt_cpu

            # Denormalize if needed (vectorized)
            if config_dict['datasets']['val']['normalizing']:
                chunk_output = (chunk_output + 1) / 2
                chunk_gt = (chunk_gt + 1) / 2

            # Save individual frames and calculate PSNR
            for i in range(current_batch_size):
                frame_idx = start_idx + i
                frame_name = frame_name_list[frame_idx][0]
                
                # Extract single frame tensors [C, H, W]
                # chunk_output is [1, T, C, H, W], so get [0, i, ...]
                sr_tensor = chunk_output[0, i, :, :, :]
                gt_tensor = chunk_gt[0, i, :, :, :]
                
                # Convert to images
                gt_img = tensor2img(gt_tensor)
                sr_img = tensor2img(sr_tensor)

                # Save image immediately
                save_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(save_path, sr_img)

                # Calculate PSNR for this frame
                frame_psnr = calculate_psnr(sr_img, gt_img)
                frame_psnr_values.append(frame_psnr)
                
                # Clean up extracted numpy arrays
                del gt_img, sr_img

            # Update progress bar
            frame_pbar.update(current_batch_size)

            # Priority 2 Optimization: Selective memory cleanup (less frequent)
            # Only empty cache every 10 chunks to reduce overhead
            del chunk_lq, chunk_gt, chunk_output, chunk_output_cpu, chunk_gt_cpu
            
            if torch.cuda.is_available() and start_idx % (opts.temporal_length * 10) == 0:
                torch.cuda.empty_cache()
            
            # Less frequent garbage collection (every 20 frames)
            if start_idx % (opts.temporal_length * 20) == 0:
                gc.collect()

        frame_pbar.close()

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

        # Priority 1 Optimization: Only master process creates videos (avoid redundant work)
        # This prevents each GPU from doing the same CPU-intensive task
        if not opts.distributed or opts.global_rank == 0:
            print(f"Converting frames to video for {clip_name}")
            frame_to_video(input_clip_url, restored_clip_url, video_save_url)

        # Clean up video data
        del val_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for torch.distributed.launch (ignored, used by launcher only)')

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
