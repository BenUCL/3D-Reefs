# Automated MASt3R-SLAM → Gaussian Splatting Pipeline

This directory contains an automated pipeline that orchestrates the complete workflow from raw images to trained Gaussian splats.

## Environment Setup

**CRITICAL:** This pipeline requires:

1. **System COLMAP** (installed via apt, NOT conda)
   ```bash
   sudo apt-get update && sudo apt-get install colmap
   ```

2. **One conda environment** with MASt3R-SLAM + Python packages
   - Use `mast3r-slam-blackwell` (or your MASt3R-SLAM environment name)
   - See `M-SLAM_BLACKWELL_SETUP.md` for complete setup guide
   - Also needs: `pycolmap`, `open3d`, `numpy`, `pyyaml`, `pillow`

**Why system COLMAP?**
- Conda COLMAP 3.13.0 has bundle adjustment bugs
- System COLMAP 3.9.1 (via apt) works reliably
- Pipeline automatically filters conda library paths to prevent conflicts

## Quick Start

```bash
# 1. Ensure system COLMAP is installed (see above)

# 2. Create your config from template
cp pipeline_config_template.yaml my_config.yaml

# 3. Edit my_config.yaml (set run_name, images_path, etc.)

# 4. Run the full pipeline
cd /home/ben/encode/code/3D-Reefs/m-splam
conda activate mast3r-slam-blackwell
python run_pipeline.py --config my_config.yaml
```

### Steps to perform before running this pipeline
- Downsample all images (raw gopro images are huge 4mb files). use `downsample_img.sh`
- If using stereo cameras, they may have a tiny difference in pixel count (e.g., I previously found: 1600x1399 vs 1600x1397). Use `crop_images_uniform.py` to crop to the smaller of the sizes. This will shave the excess pixels off the larger images.
- Make sure all images are PNG's, use `jpeg2png.py`. TODO: A [PR](https://github.com/rmurai0610/MASt3R-SLAM/pull/19) on the m-slam repo supports pngs or jpgs. although consider if this could cause memory issues.

## Splatting Modes: Keyframe vs High-Resolution

The pipeline supports two modes for Gaussian splatting, controlled by the `use_highres_for_splatting` flag in your config:

### Mode A: Splat with M-SLAM Keyframes (Default, `use_highres_for_splatting: false`)
- **Images Used**: MASt3R-SLAM's undistorted keyframes (~512px width, resized to maintain aspect ratio)
- **Camera Model**: PINHOLE (no distortion - keyframes are already undistorted by cv2.remap)
- **Intrinsics**: Scaled to SLAM resolution (e.g., 512x448 for 16:14 aspect ratio)
- **LichtFeld Flag**: No `--gut` flag needed (PINHOLE has no distortion parameters)
- **Advantages**: Simpler, faster, no need for original images
- **Disadvantages**: Lower resolution limits detail in final splats

### Mode B: Splat with raw High-Resolution images (`use_highres_for_splatting: true`)
**Best for:** In theory should give maximum quality as it will use the original high-res gopro images rather than the keyframes output by MASt3R-SLAM which are downsamples.

- **Images Used**: Original high-resolution images (e.g., 5568x4872 from GoPro)
- **Camera Model**: Original model from COLMAP (OPENCV or OPENCV_FISHEYE with distortion)
- **Intrinsics**: Scaled from low-res to high-res resolution (e.g., 1600x1400 → 5568x4872)
- **LichtFeld Flag**: Must use `--gut` flag (distortion parameters present) #TODO: check this
- **Advantages**: Maximum resolution and detail in final splats
- **Disadvantages**: Requires original images, longer training time, more memory
- **Requirements**: Must set `paths.original_images_path` in config
- **Automatic Pipeline**: When enabled, steps 5b and 5c run automatically to update poses and copy high-res images

**Technical Note**: MASt3R-SLAM internally undistorts images using `cv2.remap()` with the OPENCV model. The output keyframes are distortion-free, which is why we can use PINHOLE for Mode A. However, the original high-res images retain the lens distortion, so Mode B must use the full OPENCV model with distortion coefficients.

**Pipeline Automation**: Setting `use_highres_for_splatting: true` automatically triggers:
- Step 5b: Updates COLMAP images.txt/bin with high-res filenames
- Step 5c: Copies high-res keyframe images to splatting directory


## Pipeline Steps

The pipeline executes these steps in order (with timing reported for each):

### 1. COLMAP Intrinsics Estimation (`estimate_intrinsics.py`)
- **What**: Calibrates camera intrinsics from first N raw images using COLMAP
- **Why**: Provides accurate focal length, principal point, and distortion parameters
- **Typical Duration**: 30s - 2min (depends on num_images and image size)
- **Outputs**: 
  - `colmap_outputs/cameras.txt` - Camera parameters in COLMAP format
  - `colmap_outputs/calibration_summary.txt` - Summary with registration percentage
  - `colmap_outputs/sparse/0/` - Full COLMAP reconstruction (cameras, images, points)

### 2. Intrinsics Conversion (`shuttle_intrinsics.py` + optional `convert_intrinsics.py`)
- **What**: Converts COLMAP intrinsics into formats needed by downstream tools
- **Why**: MASt3R-SLAM needs OPENCV camera model at raw resolution; LichtFeld needs a camera model matching image type
- **Typical Duration**: <1s (shuttle_intrinsics) + 1-2s (convert_intrinsics if needed)
- **Two Modes** (controlled by `use_highres_for_splatting` in config):
  
  **Mode A: Splat with M-SLAM Keyframes** (`use_highres_for_splatting: false`, default)
  - Uses MASt3R-SLAM's undistorted keyframes (~512px width)
  - Converts to PINHOLE model (no distortion needed - keyframes already undistorted)
  - Outputs: `intrinsics.yaml` (OPENCV), `for_splat/sparse/0/cameras.txt/bin` (PINHOLE)
  
  **Mode B: Splat with raw High-Resolution images** (`use_highres_for_splatting: true`)
  - Uses original high-resolution images (retain lens distortion)
  - Keeps original camera model (OPENCV/FISHEYE with distortion)
  - Runs `convert_intrinsics.py` to scale intrinsics from low-res to high-res resolution
  - Outputs: `intrinsics.yaml` (OPENCV), `for_splat/sparse/0/cameras.txt/bin` (OPENCV scaled to high-res)
  
- **Key Detail**: Automatically scales intrinsics using MASt3R-SLAM's resize transformation, and optionally scales to high-res

### 3. MASt3R-SLAM
- **What**: Runs visual SLAM on full image sequence
- **Why**: Estimates camera poses and builds sparse 3D point cloud
- **Typical Duration**: This is a slow step. On the lab 3090 I found approx. 2min for a dataset of 500 images with the default keyframe settings.
- **Outputs**:
  - `keyframes/` - Undistorted keyframe images selected by SLAM
  - `{dataset_name}.txt` - Camera poses in TUM format (timestamp tx ty tz qx qy qz qw)
  - `{dataset_name}.ply` - Sparse 3D point cloud
- **Notes**: 
  - Runs from MASt3R-SLAM directory (required for checkpoint loading)
  - Set `enable_visualization: false` in config for faster automated runs
  - Pipeline waits for SLAM to complete before continuing

### 4. Move MASt3R-SLAM Outputs
- **What**: Moves SLAM outputs from `MASt3R-SLAM/logs/` to run directory
- **Why**: Keeps MASt3R-SLAM repo clean for version control and future runs
- **Typical Duration**: 1-5s (file copy/move operations)
- **Outputs**: Files moved to `{run_name}/mslam_logs/`
- **Key Detail**: Auto-detects dataset name from images directory, then renames all files to run_name for consistency

### 5. Pose/Keyframe Conversion (`cam_pose_keyframes_shuttle.py`)
- **What**: Converts TUM poses to COLMAP format and prepares images
- **Why**: LichtFeld-Studio requires COLMAP format (images.bin, images.txt)
- **Typical Duration**: 1-10s (depends on number of keyframes)
- **Outputs**:
  - `for_splat/images/` - Keyframe images (copied or symlinked)
  - `for_splat/sparse/0/cameras.bin` - Camera model in COLMAP binary format
  - `for_splat/sparse/0/images.txt` - Poses in COLMAP text format
- **Key Detail**: Converts from camera→world (TUM) to world→camera (COLMAP) transformation
- **Auto-triggers**: If `use_highres_for_splatting: true`, automatically runs steps 5b and 5c after completion

### 5b. Update High-Res Poses (`get_highres_poses.py`) - Auto if `use_highres_for_splatting: true`
- **What**: Updates COLMAP images.txt/bin to use high-res image filenames
- **Why**: Enables splatting with full-resolution images while keeping SLAM poses
- **Typical Duration**: <1s
- **Outputs**:
  - `for_splat/sparse/0/images_lowres.txt/bin` - Backup of original
  - `for_splat/sparse/0/images.txt/bin` - Updated with high-res filenames
  - `mslam_logs/keyframe_mapping_full.txt` - Extended mapping
- **Key Detail**: Only the NAME field changes; all pose data remains identical

### 5c. Copy High-Res Images (`copy_highres_keyframes.py`) - Auto if `use_highres_for_splatting: true`
- **What**: Copies high-resolution keyframe images to splatting directory
- **Why**: Provides the actual high-res images for Gaussian splatting
- **Typical Duration**: 5-30s (depends on image count and size)
- **Outputs**:
  - `for_splat/images/` - High-res keyframe images with original filenames
- **Key Detail**: Preserves original filenames including spaces (e.g., `2019A_GP_Left (7).JPG`)

### 6. PLY to points3D Conversion (`mslam_ply_to_points3d.py`)
- **What**: Converts MASt3R point cloud to COLMAP points3D.bin
- **Why**: Provides better initialization than random points for Gaussian splatting
- **Typical Duration**: 2-10s (depends on point cloud size and sample percentage)
- **Outputs**: `for_splat/sparse/0/points3D.bin`
- **Key Detail**: Samples 10% of points by default (~500K-1M points from 5-10M original)

### 7. Gaussian Splatting Training (`train_splat.py`)
- **What**: Trains 3D Gaussian Splatting model with LichtFeld-Studio
- **Why**: Creates final splat representation for novel view synthesis
- **Typical Duration**: This is a slow step. On the lab 3090 I found approx. 2min for a dataset of 500 images with the default keyframe settings output by mast3r-slam.
- **Outputs**:
  - `splats/splat_*.ply` - Trained Gaussian splat models (checkpoints)
  - `splats/run.log` - Full training output
  - `splats/run_report.txt` - Concise summary with training progress
- **Key Detail**: Uses PINHOLE camera model (no distortion) with initialized points from SLAM

## Output Structure

All outputs for a run are organized under `/intermediate_data/{run_name}/`:

```
/intermediate_data/{run_name}/
├── pipeline_config.yaml           # Configuration used (saved for reproducibility)
├── pipeline.log                   # Structured log with step info and timing
├── terminal_output.log            # Full terminal output from all commands (appends across runs)
│
├── colmap_outputs/                # Step 1: COLMAP intrinsics calibration
│   ├── cameras.txt                # Camera intrinsics (OPENCV model)
│   ├── calibration_summary.txt    # Registration stats, reprojection error
│   ├── database.db                # COLMAP database
│   ├── images_subset/             # Symlinks to first N images used for calibration
│   └── sparse/0/                  # Full COLMAP reconstruction
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
│
├── intrinsics.yaml                # Step 2: MASt3R-SLAM intrinsics (OPENCV with distortion)
│
├── mslam_logs/                    # Steps 3-4: MASt3R-SLAM outputs (moved from MASt3R-SLAM/logs/)
│   ├── keyframes/                 # Undistorted keyframe images
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── {run_name}.txt             # Camera poses (TUM format)
│   └── {run_name}.ply             # Sparse point cloud (5-10M points)
│
├── for_splat/                     # Steps 5-6: COLMAP format for LichtFeld-Studio
│   ├── images/                    # Keyframes (low-res PNGs or high-res originals depending on mode)
│   │   ├── 000000.png             # Mode A: downsampled keyframes
│   │   └── 2019A_GP_Left (7).JPG  # Mode B: high-res originals (if use_highres_for_splatting=true)
│   │   └── ...
│   └── sparse/0/
│       ├── cameras.bin            # Camera model (PINHOLE for keyframes, OPENCV for high-res)
│       ├── cameras.txt            # Scaled to appropriate resolution
│       ├── cameras_lowres.txt     # Backup (if convert_intrinsics.py was run)
│       ├── cameras_lowres.bin     # Backup (if convert_intrinsics.py was run)
│       ├── images.bin             # Camera poses (COLMAP format)
│       ├── images.txt             # Uses high-res filenames if Mode B
│       ├── images_lowres.txt      # Backup with low-res names (if use_highres_for_splatting=true)
│       ├── images_lowres.bin      # Backup with low-res names (if use_highres_for_splatting=true)
│       └── points3D.bin           # Sampled point cloud for initialization (~500K-1M points)
│
├── splats/                        # Step 7: Gaussian splatting outputs (first run)
│   ├── splat_25000.ply            # Trained splat model (final checkpoint)
│   ├── splat_*.ply                # Intermediate checkpoints (if saved)
│   ├── run.log                    # Full LichtFeld-Studio output
│   └── run_report.txt             # Concise summary with training progress
│
├── splats1/                       # Step 7 re-run with different params (e.g., different max-cap)
│   ├── splat_25000.ply
│   └── ...
│
└── splats2/                       # Step 7 another re-run
    └── ...
```

**Note on Splats Versioning**: When you re-run step 7 (e.g., with `--only 7` after changing parameters like `max_cap` in the config), the pipeline automatically creates `splats1/`, `splats2/`, etc. instead of overwriting `splats/`. This allows you to experiment with different splatting parameters without losing previous results.

### Logging Details

The pipeline creates two log files:

1. **`pipeline.log`** - Structured execution log:
   - Step start/end times
   - Commands executed
   - Skip detection messages
   - Timing for each step (e.g., "Step 1 took 1m 23s")
   - Total elapsed time

2. **`terminal_output.log`** - Full subprocess output (appends across runs):
   - Each run adds a header: `# NEW PIPELINE RUN` with timestamp and metadata
   - Each step adds: command, timestamp, full stdout/stderr
   - Useful for debugging failures or checking detailed progress
   - **Note**: This file appends, so it grows with each run of the same config

## Advanced Usage

### Resume from Specific Step

If a step fails, you can resume from that point:

```bash
python run_pipeline.py --config my_config.yaml --start-from 3
```

### Run Single Step

To re-run a specific step (e.g., with different parameters):

```bash
# Edit config with new parameters
python run_pipeline.py --config my_config.yaml --only 7
```

### Re-run Splatting with Different Parameters

Step 7 (Gaussian Splatting) supports automatic versioning and **command-line parameter overrides**, allowing you to run multiple splatting experiments within the same run folder **without editing the config file**. Each splat run creates sequential output folders (splats/, splats1/, splats2/, etc.), the command used is stored in splat/run_report.txt.

```bash
# First run with original config (if not done already in a full pipeline run)
python run_pipeline.py --config my_config.yaml
# Creates: /intermediate_data/pipeline_test3/splats/

# Re-run with different max-cap (no config edit needed!)
python run_pipeline.py --config my_config.yaml --only 7 --max-cap 200000
# Creates: /intermediate_data/pipeline_test3/splats1/

# Try with different iterations and max-cap
python run_pipeline.py --config my_config.yaml --only 7 -i 50000 --max-cap 2000000
# Creates: /intermediate_data/pipeline_test3/splats2/
```

**Available Step 7 command-line overrides:**
- `-i, --iterations N` - Number of training iterations
- `--max-cap N` - Maximum splat count after densification  
- `--headless` / `--no-headless` - Run with/without GUI
- `--splat-extra-args ARG1 ARG2 ...` - Additional LichtFeld-Studio arguments

This allows you to compare results from different splatting parameters without:
- Editing the config file repeatedly
- Creating a whole new run directory
- Overwriting previous splat results
- Re-running the expensive SLAM steps (1-6)

### Interactive Mode

Enable step-by-step confirmation:

```yaml
pipeline:
  interactive: true
```

### Skip Existing Outputs

By default, completed steps are skipped if outputs exist:

```yaml
pipeline:
  skip_existing: true  # Set to false to force re-run
```

## Configuration Reference

See `pipeline_config_template.yaml` for full documentation of all parameters.

Key sections:
- **paths**: Input images, output directory, tool locations
- **intrinsics_estimation**: COLMAP calibration settings
- **mast3r_slam**: SLAM configuration
- **gaussian_splatting**: Training parameters

## Examples

- `example_reef_soneva.yaml`: Configuration for reef dataset
- Copy and modify for your own datasets

## Monitoring Execution

The pipeline provides real-time feedback on progress:

```
[14:32:15] Pipeline started: 2025-10-23 14:32:15
[14:32:15] Run name: pipeline_test3
[14:32:15] Dataset name (from images path): LHS_downsampled_png
...
[14:33:42] ✅ Step 1 completed: COLMAP intrinsics estimation
[14:33:42] ⏱️  This step took: 1m 27s
[14:33:42] ⏱️  Total elapsed time: 1m 27s
```

**Watch for**:
- Registration percentage in Step 1 (should be >90% for good calibration)
- Number of keyframes selected by MASt3R-SLAM
- Point cloud size after Step 6 (sometimes number of points is still larger than number of splats given in step 7).
- Training loss decrease in Step 7 (check run_report.txt for progress)

**Tail logs during execution**:
```bash
# Structured log with timing
tail -f /intermediate_data/{run_name}/pipeline.log

# Full terminal output
tail -f /intermediate_data/{run_name}/terminal_output.log
```

## Troubleshooting

### Common Issues

**Problem**: MASt3R-SLAM fails with "FileNotFoundError" on checkpoint  
**Solution**: Pipeline handles this automatically by running from MASt3R-SLAM directory. If it still fails, verify checkpoint exists at `MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`

**Problem**: Step already completed but want to re-run  
**Solution**: Use `--only` flag or set `skip_existing: false` in config, or manually delete output subdirectory for that step

**Problem**: Need different parameters for one step  
**Solution**: Edit config, then use `--start-from N` or `--only N` to re-run from/only that step

**Problem**: COLMAP calibration has low registration (<25%)  
**Solution**: Try increasing `num_images` or verifying image quality (blur, exposure, sufficient overlap). Could give it images from elsewhere in the sequence.

**Problem**: COLMAP bundle adjustment crashes with SQLite or SIGABRT errors  
**Solution**: Make sure you're using **system COLMAP** (via apt), not conda COLMAP. Check `which colmap` shows `/usr/bin/colmap`. Conda COLMAP 3.13.0 has known bugs.

**Problem**: MASt3R-SLAM reconstruction looks wrong  
**Solution**: Check intrinsics.yaml values are reasonable. Try different COLMAP camera_model (OPENCV vs OPENCV_FISHEYE). Visualize outputs manually.

**Problem**: Splatting training diverges or produces artifacts  
**Solution**: Verify pose accuracy (check images.txt), try different initialization (adjust sample_percentage), or run with fewer iterations first

### Debugging Strategy

1. **Check logs first**: `pipeline.log` shows which step failed, `terminal_output.log` has detailed error messages
2. **Run step manually**: Copy command from logs and run directly in terminal for better error visibility
3. **Verify intermediate outputs**: Check file sizes, image quality, point cloud in CloudCompare/Rerun
4. **Fallback to manual workflow**: See `notes.txt` for step-by-step manual commands
5. **Start fresh**: Delete run directory or change `run_name` to avoid stale outputs

## Design Notes & Assumptions

### Architecture Decisions

- **No MASt3R-SLAM Code or Logs Modifications**: We move outputs post-run to keep MASt3R-SLAM repo pristine for git updates
- **Config Saved**: Each run saves its config to output directory for reproducibility
- **Backward Compatible**: Individual scripts still work standalone with `--mslam_logs_dir` parameter
- **Working Directory Management**: MASt3R-SLAM must run from its root directory (for checkpoint loading), so we temporarily `chdir()` for Step 3
- **File Naming**: Pipeline auto-detects `dataset_name` from images directory but uses `run_name` for all outputs to support multiple runs on same dataset

### Key Assumptions (Watch These!)

1. **Image Format**: Expects PNG images in input directory (JPEG conversion not automated)
2. **Image Resolution**: Assumes all images are same resolution (no mixed-size handling)
3. **Sequential Naming**: Keyframes named sequentially (000000.png, 000001.png, ...) by MASt3R-SLAM
4. **Camera Model**: Assumes single camera (no multi-camera rig support)
5. **Distortion**: Assumes MASt3R-SLAM undistorts keyframes (hence PINHOLE for splatting)
6. **Conda Environments**: Assumes `mast3r-slam` and `ben-splat-env` exist and are configured
7. **Binary Paths**: Hardcoded paths in config (COLMAP assumed in PATH, LichtFeld path required)
8. **Disk Space**: No cleanup of intermediate files (could take up a lot of space for large datasets).

### Limitations

- No multi-sequence support (one image directory per run)
- No automatic quality checks (you must verify outputs manually)
- Basic error handling (some failures may leave partial outputs)
- No automatic parameter tuning (requires manual config adjustment)
- Skip detection based on file existence only (doesn't verify correctness)

### Why This Architecture?

The pipeline glues together 4 independent tools (COLMAP, MASt3R-SLAM, wildflow, LichtFeld-Studio). Each tool has different input/output formats and conventions. Rather than forking and modifying these tools, we:

1. Keep tools unmodified, so we can pull these libraries again on new machines or update with newer versions.
2. So instead, we add conversion scripts to take outputs from one tool and modify these to work as the input for the next tool (shuttle_intrinsics.py, cam_pose_keyframes_shuttle.py, etc.)
3. The whole pipeline is orchestrated with run_pipeline.py (handles paths, sequencing, skip logic)

This makes the code more fragile (format changes break us) but easier to maintain (no custom tool forks).
