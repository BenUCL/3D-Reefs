# Gaussian Splatting Pipeline for Large-Scale Reconstructions

This pipeline processes large COLMAP reconstructions into high-quality 3D gaussian splats through spatial patching, parallel training, and quality-based cleanup.

## Outstanding TODO list
- Had to switch optimisation strategy to MCMC as was getting this [CUDA error](https://github.com/MrNeRF/LichtFeld-Studio/issues/549) for patches seemingly at random. The downside of this is without MCMC more floaters seem to be added, and supposedly splat quality should be better with MCMC. Also...
- As a consequence of switching to MCMC, it seems the `max-cap` feature no longer works, it will happily blow through the cap. This can be mitigated by limiting training iters, but this is not optimal. It seems `max-cap` is not actually supported for the default optimiser, but copilot said it could be added. However, unsure if this would create new issues (e.g get stuck with bad splats)
- Buffer is supposedly in metres but is clearly not when you view the visualiser. Not sure how it would have any reference to metres as now scale was provided. Could make buffer setting clearer (e.g., portion of total area rather than metres).
- Currently not 100% confident in whether the point cloud is being downsampled or not during patching.
- The pipeline is probably fragile, not tested on new datasets or edge cases, expect issues with these.

## Overview

**Input**: COLMAP reconstruction (cameras.bin, images.bin, points3D.bin, source images). Note, it may be that images from different cameras must be the same dimension or very close to it, though I am unsure.
**Output**: Cleaned gaussian splat models ready for rendering or merging

**Pipeline**: Patch → Train → Clean

## Prerequisites

- COLMAP reconstruction with camera poses and 3D points
- LichtFeld-Studio for gaussian splatting training
- wildflow Python package for patching and cleanup
- Multi-camera images organized by camera (e.g., `images/left/`, `images/right/`)

## Configuration

All settings are centralized in `splat_config.yml`.

## Step-by-Step Workflow

### Step 0: Preprocess Raw Images

Before running COLMAP, downsample and convert raw images using the preprocessing script:

```bash
./downsample_imgs.sh
```

**What it does:**
- Reads settings from `splat_config.yml` (image_preprocessing section)
- Downsamples images from `raw_images_dir` to `max_dimension` 
- Converts to specified format (PNG/JPG)
- Applies resampling filter (Lanczos by default for sharp details)
- Outputs to `processed_images_dir`

### Step 1: Determine Patch Size

Point config to data and run this script to open the patch visualiser to determine optimal `max_cameras` and `buffer` values. Too many images in a patch will OOM GPU. However, his visualiser also helps pick `buffer` mainly, it is met to be set in metres but I find the scale is way off so important to check with this visualiser.

```bash
3D-Reefs/process_data/patch_visualiser.py
```

Update `patching.max_cameras` in `splat_config.yml` with chosen values.

### Step 2: Patch the COLMAP Reconstruction

Split large reconstruction into manageable spatial patches:

```bash
python patch_colmap_data.py --config splat_config.yml
```

**What it does:**
- Analyzes camera positions to create patches seen in patch visualiser
- Splits cameras.bin and images.bin per patch
- Optionally splits points3D.bin (COLMAP points) or external dense PLY
- Creates patch directories: `patches_dir/p0/`, `p1/`, `p2/`, etc.

**Output structure per patch:**
```
p0/
  sparse/0/
    cameras.bin/txt    # Camera intrinsics for this patch
    images.bin/txt     # Image poses for this patch
    points3D.bin/txt   # 3D points for initialization (optional)
```

**Key parameters:**
- `max_cameras`: Controls patch size (from Step 1)
- `buffer_meters`: Overlap between patches
- `use_colmap_points`: Use sparse COLMAP points (points3d.bin file) vs dense PLY

### Step 3: Train Gaussian Splats

Train using LichtFeld-Studio:

```bash
./batch_train_splat.sh

```

**What it does:**
- Creates temporary directory structure compatible with LichtFeld-Studio
- Symlinks sparse reconstruction and images (no data duplication)
- Runs LichtFeld with configured parameters (iterations, max_cap, pose_opt)
- Captures training progress and metrics
- Saves splat models as `splat_iter.ply` (iter = training iteration count)

**Output per patch:**
```
p0/sparse/splat/
  p0_splat_20000.ply     # Trained gaussian splat (prefixed with patch name)
  run_report.txt         # Training metrics and progress
  run.log                # Full training log
```

**Training details:**
- Uses headless mode for batch processing (without it will open the LFS visualiser)
- Handles multi-camera datasets (left/right cameras), set these in config
- Logs training loss and splat count progression
- Continues to next patch if one fails (logs error, moves to next patch)

**Log location:** `patches_dir/splat_training_log.txt`

**TODO**
- Currently setting `max_cap` in the config doesn't seem to do anything. It will just keeping adding gaussians.
- Had to disable MCMC as was getting "CUDA error: invalid configuration argument" in MulBackward1 during adaptive Gaussian growth. However, MCMC is meant to produce more accurate splats so I would like this to work.

### Step 4: Clean Gaussian Splats

Remove floaters and giant splats using spatial and neighbor-based filtering. 

NOTE: May want to view some splats straight after training, before this clean up step, so as to check for any funny business or failures.

Set params in the confgi and then run clean up with:
```bash
./batch_clean_splat.sh
```

**What it does:**
- Finds highest iteration splat file (e.g., `splat_20000.ply`)
- Filters splats using wildflow.splat.cleanup_splats:
  - If `filter_boundaries=true`, removes splats outside patch core (excludes buffer zone added during patching)
  - `max_area`: Filters out large, low-quality splats (default: 0.004)
  - `min_neighbors`: Requires this many neighbors within radius (default: 20)
  - `radius`: Neighbor search radius in meters (default: 0.2m)
  - Higher `min_neighbors` = more aggressive filtering
- Creates cleaned version: `splat_20000_clean.ply`

**Output:**
```
p0/sparse/splat/
  p0_splat_20000.ply        # Original splat
  p0_splat_20000_clean.ply  # Cleaned splat (use this for rendering)
```

**Cleanup parameters:**


**Log location:** `patches_dir/splat_cleanup_log.txt`

## Script Reference

### Core Scripts
- **`splat_config.yml`**: set configuration for all pipeline steps
- **`patch_colmap_data.py`**: Spatial patching of COLMAP reconstructions
- **`train_splat.py`**: Single-patch gaussian splatting training
- **`batch_train_splat.sh`**: Batch training wrapper with logging
- **`clean_splats.py`**: Single-patch splat cleanup
- **`batch_clean_splat.sh`**: Batch cleanup wrapper with logging

