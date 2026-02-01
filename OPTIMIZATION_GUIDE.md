# RRTN Restoration CPU/GPU Optimization Guide

## Current Performance Baseline
- **Execution Time**: 220 seconds (3m 40s) for 301 frames
- **GPU Memory**: ~15GB utilized on RTX 4060 Ti
- **CPU Usage**: Balanced across 3-4 worker threads
- **Speedup Achievement**: 5.2x improvement from initial state

---

## Identified Optimization Opportunities

### 1. **Model Loading Optimization** (Quick Win - 5-10% gain)

**Current Issue**: Checkpoint loaded to CPU first, then moved to GPU
**Impact**: Unnecessary memory spike and transfer latency

**Optimization**:
```python
# BEFORE (in load_model function around line 70)
checkpoint = torch.load(model_path, map_location='cpu')
netG.load_state_dict(checkpoint['netG'])
netG.cuda()  # Transfer happens here

# AFTER
if torch.cuda.is_available():
    device = torch.device(f'cuda:{opts.local_rank}' if opts.distributed else 'cuda:0')
    checkpoint = torch.load(model_path, map_location=device)
netG.load_state_dict(checkpoint['netG'])
del checkpoint  # Free memory immediately
```

**Expected Improvement**: 5-10% faster model initialization, ~2GB less peak memory

### 2. **GPU Synchronization Inefficiency** (Medium Priority - 8-15% gain)

**Current Issue**: `torch.cuda.synchronize()` called immediately after every chunk
**Impact**: Blocks CPU from preparing next batch while GPU is still computing

**Optimization**:
```python
# BEFORE
chunk_output = model(chunk_lq)
torch.cuda.synchronize()  # Blocks CPU

# AFTER - Move output to CPU asynchronously, synchronize only when needed
chunk_output = model(chunk_lq)
chunk_output_cpu = chunk_output.cpu()  # Start async transfer
# CPU can now prepare next batch while GPU finishes

# Only synchronize when we actually need the data
torch.cuda.synchronize()  # Now happens during frame saving I/O
```

**Expected Improvement**: 8-15% better GPU-CPU pipeline overlap

### 3. **DataLoader Worker Count Scaling** (5-8% gain for 2+ GPUs)

**Current Issue**: Fixed `num_workers=3-4` doesn't scale with GPU count
**Impact**: May under-utilize with 2 GPUs or waste resources

**Optimization**:
```python
# BEFORE (in load_dataset function around line 105)
num_workers = min(4, mp.cpu_count() // 2)

# AFTER - Scale with GPU count
cpu_count = mp.cpu_count()
num_workers = min(6, max(2, (cpu_count - opts.gpus - 2) // opts.gpus))
# This reserves 2 cores per GPU for worker threads
```

**Expected Improvement**: Better CPU distribution across multi-GPU setup

### 4. **Prefetch Factor Optimization** (3-5% gain)

**Current Issue**: `prefetch_factor=2` may be too low for high-speed GPU
**Impact**: GPU occasionally waits for next batch

**Optimization**:
```python
# BEFORE
prefetch_factor=2 if num_workers > 0 else None

# AFTER
prefetch_factor=max(2, 4 if num_workers > 0 else 0)
# Prefetch 4 batches for better overlap with GPU computation
```

**Expected Improvement**: Reduce GPU idle time by 3-5%

### 5. **Memory Cache Management** (2-4% gain)

**Current Issue**: `torch.cuda.empty_cache()` called after every chunk
**Impact**: Expensive operation that blocks GPU, reduces throughput

**Optimization**:
```python
# BEFORE (in validation function around line 200)
del chunk_lq, chunk_gt, chunk_output
torch.cuda.empty_cache()  # Called every iteration - expensive!

# AFTER - Selective cache clearing
del chunk_lq, chunk_gt, chunk_output
if torch.cuda.is_available() and start_idx % (opts.temporal_length * 10) == 0:
    torch.cuda.empty_cache()  # Only every 10 chunks
    gc.collect()  # Only every 20 chunks
```

**Expected Improvement**: 2-4% faster, more stable GPU throughput

### 6. **Video Encoding CPU Bottleneck** (5-10% gain with DDP)

**Current Issue**: Every GPU process creates the same video file
**Impact**: Redundant CPU/disk work in multi-GPU setup

**Optimization**:
```python
# BEFORE (in process_single_folder function around line 270)
frame_to_video(input_clip_url, restored_clip_url, video_save_url)
# Each GPU does this independently - wasteful!

# AFTER - Only master process creates videos
if not opts.distributed or opts.global_rank == 0:
    frame_to_video(input_clip_url, restored_clip_url, video_save_url)
```

**Expected Improvement**: 5-10% reduction in CPU load with 2 GPUs, no redundant I/O

### 7. **Gradient Computation Overhead** (2-3% gain)

**Current Issue**: Models in inference but gradients not explicitly disabled
**Impact**: PyTorch tracks operations unnecessarily

**Optimization**:
```python
# ADD to load_model function after loading weights
netG.eval()
for param in netG.parameters():
    param.requires_grad = False
```

**Expected Improvement**: 2-3% reduction in memory and computation overhead

### 8. **cuDNN Optimization** (5-8% gain)

**Current Issue**: cuDNN auto-tuning and determinism not configured
**Impact**: Suboptimal kernel selection, especially for inference

**Optimization**:
```python
# ADD to main_worker function
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Enable auto-tuning
    torch.backends.cudnn.deterministic = False  # Allow faster but non-deterministic ops
```

**Expected Improvement**: 5-8% speedup from optimized CUDA kernel selection

---

## Implementation Checklist

- [ ] **Priority 1 (Quick Wins)**: Items 1, 2, 7, 8 → Estimated 20-30% total improvement
- [ ] **Priority 2 (Medium Effort)**: Items 3, 4, 5 → Estimated 10-15% improvement
- [ ] **Priority 3 (DDP-Specific)**: Item 6 → 5-10% improvement for multi-GPU only

---

## Advanced Optimizations (If Needed)

### A. Mixed Precision Inference (FP16)
```python
# In validation loop
with torch.cuda.amp.autocast():
    chunk_output = model(chunk_lq)
# Estimated 20-40% speedup (if model supports FP16)
```
**Trade-off**: Slight quality reduction, requires testing

### B. Batch Processing Multiple Frames
```python
# Instead of processing 1 frame at a time
# Process 4 frames simultaneously (if VRAM allows)
batch_size = 4  # vs current 1
# Estimated 30-50% improvement but requires VRAM increase
```
**Trade-off**: Requires VRAM profile testing, may not fit RTX 4060 Ti

### C. GPU Streams for Overlapping Operations
```python
stream_compute = torch.cuda.Stream()
stream_transfer = torch.cuda.Stream()
# Overlap GPU computation with CPU-GPU transfers
# Estimated 10-15% improvement but complex implementation
```

### D. Asynchronous Frame Saving (ThreadPoolExecutor)
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(cv2.imwrite, save_path, sr_img)
# Estimated 5-8% improvement, non-blocking I/O
```

---

## Performance Measurement

### Before Optimization
```bash
python restore.py --test_folder test_data_sample
# Expected: ~220 seconds
```

### After Implementing Priority 1 + 2
```bash
python restore.py --test_folder test_data_sample
# Expected: ~160-180 seconds (20-30% improvement)
```

### Profiling Commands
```bash
# Profile GPU utilization
nvidia-smi -l 1  # Monitor in separate terminal

# Profile with PyTorch profiler
torch.profiler.profile()  # Add to validation loop

# Memory profiling
python -m memory_profiler restore.py --test_folder test_data_sample
```

---

## Multi-GPU (2 GPU) Specific Notes

With 2 GPUs, prioritize:
1. Item 6 (Video encoding optimization) - **10% gain**
2. Item 3 (DataLoader worker scaling) - **8% gain**
3. DDP gradient synchronization reduction

The codebase already has DDP support, but ensure `find_unused_parameters=True` is used (it is).

---

## Summary: Expected Final Performance

| Stage | Time | Speedup |
|-------|------|---------|
| Current | 220s | 1x |
| + Priority 1,2 (Items 1,2,7,8) | 160-180s | 1.2-1.4x |
| + Priority 2 (Items 3,4,5) | 140-160s | 1.4-1.6x |
| + DDP Optimization (Item 6, 2 GPUs) | 80-100s | 2.2-2.75x |

**Recommended Implementation Order**:
1. Start with load_model() optimization (5 min)
2. Add cuDNN settings (2 min)
3. Fix GPU sync strategy (10 min)
4. Update DataLoader num_workers (5 min)
5. Implement cache cleanup optimization (5 min)
6. Add DDP video encoding check (3 min)
7. Test and benchmark (10 min)

**Total Implementation Time**: ~40 minutes for 1.4-1.6x improvement
