# Gaussian Splatting Pipeline for Large-Scale Reconstructions

This pipeline processes large COLMAP reconstructions into high-quality 3D gaussian splats through spatial patching, parallel training, and quality-based cleanup.

## Overview

**Input**: COLMAP reconstruction (cameras.bin, images.bin, points3D.bin, source images)  
**Output**: Cleaned gaussian splat models ready for rendering or merging

**Pipeline**: Patch → Train → Clean

## Prerequisites

- COLMAP reconstruction with camera poses and 3D points
- LichtFeld-Studio for gaussian splatting training
- wildflow Python package for patching and cleanup
- Multi-camera images organized by camera (e.g., `images/left/`, `images/right/`)

## Configuration

All settings are centralized in `splat_config.yml`:

```yaml
paths:
  lichtfeld_bin: /path/to/LichtFeld-Studio
  patches_dir: /path/to/output/patches
  images_dir: /path/to/images

patching:
  sparse_dir: /path/to/colmap/sparse/0
  max_cameras: 400 # images per patch
  buffer_meters: 0.1 # buffer size, supposedly in metres but not sure. Use patch visualiser to set this!
  use_colmap_points: true
  pointcloud_path:          # Optional: path to dense PLY
  sample_percentage: 5.0    # Percentage of PLY points to sample (reduces point count, only used with pointcloud_path)

camera_mapping:
  left: 1                  # Map folder names to camera IDs
  right: 2

training:
  iterations: 20000
  max_cap: 1000000
  run_batch: true          # Train all patches or single patch

cleanup:
  max_area: 0.004 # Remove oversized splats
  min_neighbors: 20 # Remove isolated splats
  radius: 0.2 # Neighbour search area, remove those that don't meet this.
  filter_boundaries: false  # Remove splats outside patch core (excludes buffer zone)
  boundary_buffer: 0.1      # Buffer to exclude if filter_boundaries=true
  run_batch: true
```

## Step-by-Step Workflow

### Step 1: Determine Patch Size

Opens the patch visualiser to determine optimal `max_cameras` and `buffer` values. Too many images in a patch will OOM GPU, this visualiser helps pick `buffer` mainly.

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
- `buffer_meters`: Overlap between patches (0.5-2.0m typical)
- `use_colmap_points`: Use sparse COLMAP points vs external dense PLY

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
- Saves splat models as `splat_N.ply` (N = iteration count)

**Output per patch:**
```
p0/sparse/splat/
  splat_20000.ply        # Trained gaussian splat
  run_report.txt         # Training metrics and progress
  run.log                # Full training log
```

**Training details:**
- Uses headless mode for batch processing (without it will open the LFS visualiser)
- Handles multi-camera datasets (left/right cameras)
- Logs training loss and splat count progression
- Continues on failure (logs error, moves to next patch)

**Log location:** `patches_dir/splat_training_log.txt`

### Step 4: Clean Gaussian Splats

Remove low-quality splats using spatial and neighbor-based filtering:

```bash
./batch_clean_splat.sh
```

**What it does:**
- Finds highest iteration splat file (e.g., `splat_20000.ply`)
- Filters splats using wildflow.splat.cleanup_splats:
  - Removes oversized splats (area > `max_area`)
  - Removes isolated splats (< `min_neighbors` within `radius`)
  - If `filter_boundaries=true`, removes splats outside patch core (excludes buffer zone added during patching)
- Creates cleaned version: `splat_20000_clean.ply`

**Output:**
```
p0/sparse/splat/
  splat_20000.ply        # Original splat
  splat_20000_clean.ply  # Cleaned splat (use this for rendering)
```

**Cleanup parameters:**
- `max_area`: Filters out large, low-quality splats (default: 0.004)
- `min_neighbors`: Requires this many neighbors within radius (default: 20)
- `radius`: Neighbor search radius in meters (default: 0.2m)
- Higher `min_neighbors` = more aggressive filtering

**Log location:** `patches_dir/splat_cleanup_log.txt`

## Script Reference

### Core Scripts

- **`patch_colmap_data.py`**: Spatial patching of COLMAP reconstructions
- **`train_splat.py`**: Single-patch gaussian splatting training
- **`batch_train_splat.sh`**: Batch training wrapper with logging
- **`clean_splats.py`**: Single-patch splat cleanup
- **`batch_clean_splat.sh`**: Batch cleanup wrapper with logging

### Configuration

- **`splat_config.yml`**: Centralized configuration for all pipeline steps

