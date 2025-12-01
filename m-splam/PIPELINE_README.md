# Automated MASt3R-SLAM → Gaussian Splatting Pipeline

This directory contains an automated pipeline that orchestrates the complete workflow from raw images to trained Gaussian splats.

## TODO
- Integrate conversion of JPG's to PNGS's and any downsampling applied to images (used for colmap intrinsics estimate).
- Clean whitespace from images names right at the very start.
- m-slam outputs keyframe filenames with timestamps, not the original image name. This has been fixed for the high-res image option, but not for standard (which uses the raw keyframes output by m-slam).
- Integrate in the interpolation of camera poses between keyframes and optimisation of these using 

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
- Downsample all images (raw GoPro images are huge 4MB+ files). use `downsample_img.sh`
- If using stereo cameras, they may have a tiny difference in pixel count (e.g., I previously found: 1600x1399 vs 1600x1397). Use `crop_images_uniform.py` to crop to the smaller of the sizes. This will shave the excess pixels off the larger images.
- **Image format**: Pipeline supports both PNG and JPG formats. M-SLAM handles both, but converting to PNG (using `jpeg2png.py`) can avoid any potential compression artifacts during SLAM tracking.

## Splatting Modes: Keyframe vs High-Resolution

The pipeline supports two modes for Gaussian splatting, controlled by the `use_highres_for_splatting` flag in your config:

### Mode A: Splat with M-SLAM Keyframes (Default, `use_highres_for_splatting: false`)
- **Images Used**: MASt3R-SLAM's undistorted keyframes (~512px width, resized to maintain aspect ratio)
- **Camera Model**: PINHOLE (no distortion - keyframes are already undistorted by cv2.remap)
- **Intrinsics**: Scaled to SLAM resolution (e.g., 512x448 for 16:14 aspect ratio)
- **LichtFeld Flag**: No `--gut` flag needed (PINHOLE has no distortion parameters)
- **Advantages**: Simpler, faster, no need for original images
- **Disadvantages**: Lower resolution limits detail in final splats

### Mode B: Splat with High-Resolution images (`use_highres_for_splatting: true`)
**Best for:** Maximum quality splatting - uses original high-res images (e.g., 5568x4872 from GoPro) instead of the downsampled keyframes (~512px) that M-SLAM outputs.

- **Images Used**: Original high-resolution images, **undistorted and cropped** to match M-SLAM preprocessing
- **Camera Model**: PINHOLE (no distortion - images are undistorted in Step 5c using cv2.remap)
- **Intrinsics**: Scaled from low-res to high-res, then adjusted for undistortion and crop
- **LichtFeld Flag**: **No `--gut` flag needed** (PINHOLE has no distortion parameters)
- **Advantages**: Maximum resolution and detail in final splats
- **Disadvantages**: Requires original images, longer training time (more pixels), more GPU memory
- **Requirements**: Must set `paths.original_images_path` in config
- **Automatic Pipeline**: When enabled, steps 5b and 5c run automatically after Step 5

**Technical Details**: 
1. M-SLAM tracks on low-res images (e.g., 1600x1400) and outputs undistorted keyframes (~512px)
2. Step 5c applies the **same undistortion and cropping** to high-res source images that M-SLAM applied to low-res
3. This ensures geometric consistency: high-res images have identical geometry to M-SLAM keyframes, just at higher resolution
4. Final camera model is PINHOLE because all distortion has been removed via cv2.remap()

**Pipeline Automation**: Setting `use_highres_for_splatting: true` automatically triggers after Step 5:
- Step 5b: Updates COLMAP images.txt/bin with high-res filenames (pose data unchanged)
- Step 5c: Undistorts, crops, and copies high-res keyframe images + writes PINHOLE cameras.txt/bin


## Pipeline Steps

The pipeline executes these steps in order (with timing reported for each):

### 1. COLMAP Intrinsics Estimation (`estimate_intrinsics.py`)
- **What**: Calibrates camera intrinsics from first N downsampled images using COLMAP (colmap fails on raw high res images so have been using the 1600x1400 images then scaling the intrinsics in step 2)
- **Why**: Provides accurate focal length, principal point, and distortion parameters for MASt3R-SLAM
- **Typical Duration**: 30s - 2min (depends on num_images and image size)
- **Outputs**: 
  - `colmap_outputs/cameras.txt` - Camera parameters in COLMAP format
  - `colmap_outputs/calibration_summary.txt` - Summary with registration percentage
  - `colmap_outputs/sparse/0/` - Full COLMAP reconstruction (cameras, images, points)
- **Calibration Usage**: If `mast3r_slam.use_calibration: true` in config, these intrinsics will be passed to MASt3R-SLAM

### 2. Intrinsics Conversion (`shuttle_intrinsics.py`)
- **What**: Converts COLMAP intrinsics into `intrinsics.yaml` for MASt3R-SLAM and optionally generates COLMAP cameras.txt/bin for splatting
- **Why**: MASt3R-SLAM needs OPENCV camera model with inline list format `[fx, fy, cx, cy, k1, k2, p1, p2]`
- **Typical Duration**: <1s
- **Always Outputs**: `intrinsics.yaml` (OPENCV format with distortion, used by M-SLAM for tracking on low-res images)
- **Two Modes** (controlled by `use_highres_for_splatting` in config):
  
  **Mode A: Low-Res Splatting** (`use_highres_for_splatting: false`, default)
  - Also outputs: `for_splat/sparse/0/cameras.txt/bin` (PINHOLE, scaled to ~512px M-SLAM resolution)
  - Uses MASt3R-SLAM's `resize_img()` transformation to calculate cropped resolution and scaled intrinsics
  - Ready for splatting with M-SLAM's undistorted keyframes
  
  **Mode B: High-Res Splatting** (`use_highres_for_splatting: true`)
  - Does **NOT** output cameras.txt/bin (deferred to Step 5c which needs high-res image dimensions)
  - Only creates intrinsics.yaml for M-SLAM tracking
  - Step 5c (`prepare_highres_splat.py`) will later handle undistortion, cropping, and PINHOLE camera generation
  
- **Key Detail**: Automatically applies M-SLAM's crop-aware scaling using `resize_img()` transformation (accounts for aspect ratio padding and center cropping)

### 3. MASt3R-SLAM
- **What**: Runs visual SLAM on full image sequence
- **Why**: Estimates camera poses and builds sparse 3D point cloud
- **Typical Duration**: This is a slow step. On the lab 3090 I found approx. 2min for a dataset of 500 images with the default keyframe settings.
- **Calibration Modes**:
  - If `use_calibration: true`: Uses intrinsics.yaml from Step 2 (passed via `--calib` flag)
  - If `use_calibration: false`: MASt3R-SLAM estimates intrinsics internally
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
- **What**: Converts TUM poses to COLMAP format and copies/symlinks low-res keyframe images
- **Why**: LichtFeld-Studio requires COLMAP format (images.bin, images.txt)
- **Typical Duration**: 1-10s (depends on number of keyframes)
- **Outputs**:
  - `for_splat/images/` - M-SLAM keyframe images (~512px, copied or symlinked)
  - `for_splat/sparse/0/images.txt/bin` - Camera poses in COLMAP format
- **Key Detail**: Converts from camera→world (TUM) to world→camera (COLMAP) transformation
- **High-Res Mode Cleanup**: If `use_highres_for_splatting: true`, automatically:
  1. Deletes the low-res keyframes just copied (to make room for high-res)
  2. Triggers Steps 5b and 5c to prepare high-res images and update poses

### 5b. Update High-Res Poses (`get_highres_poses.py`) - Auto if `use_highres_for_splatting: true`
- **What**: Updates COLMAP images.txt/bin to reference original high-res filenames instead of M-SLAM's timestamp-based keyframe names
- **Why**: Enables using high-res source images while keeping SLAM-estimated poses
- **Typical Duration**: <1s
- **Outputs**:
  - `for_splat/sparse/0/images_lowres.txt/bin` - Backup of original (timestamp-based names)
  - `for_splat/sparse/0/images.txt/bin` - Updated with high-res filenames (e.g., "2019A_GP_Left (7).JPG")
  - `mslam_logs/keyframe_mapping_full.txt` - Extended mapping with both name formats
- **Key Detail**: Only the NAME field changes; all pose data (rotation, translation) remains identical

### 5c. Prepare High-Res Images (`prepare_highres_splat.py`) - Auto if `use_highres_for_splatting: true`
- **What**: Undistorts, crops, and copies high-resolution keyframe images + writes PINHOLE camera model
- **Why**: Applies the same geometric preprocessing to high-res images that M-SLAM applied to low-res tracking images
- **Typical Duration**: approx. 1 sec per image (depends on resolution and I/O)
- **Process**:
  1. Reads low-res intrinsics.yaml (OPENCV with distortion parameters)
  2. Scales intrinsics to high-res dimensions (e.g., 1600x1400 → 5568x4872)
  3. Computes undistortion maps using `cv2.getOptimalNewCameraMatrix()` and `cv2.initUndistortRectifyMap()`
  4. Undistorts each high-res image using `cv2.remap()` (removes lens distortion)
  5. Applies M-SLAM's center-crop logic to match aspect ratio requirements
  6. Writes final PINHOLE cameras.txt/bin (no distortion parameters - images now distortion-free)
- **Outputs**:
  - `for_splat/images/` - High-res keyframe images, undistorted and cropped (e.g., 5568x4872 → 5568x4872 undistorted → 5568x4872 cropped)
  - `for_splat/sparse/0/cameras.txt/bin` - PINHOLE model with adjusted intrinsics (overwrites any existing)
- **Key Detail**: Final resolution and intrinsics account for both undistortion (using cv2's optimal new camera matrix) and M-SLAM's center cropping, ensuring geometric consistency with low-res tracking

### 6. PLY to points3D Conversion (`mslam_ply_to_points3d.py`)
- **What**: Converts MASt3R point cloud to COLMAP points3D.bin
- **Why**: Provides better initialization than random points for Gaussian splatting
- **Typical Duration**: 2-10s (depends on point cloud size and sample percentage)
- **Outputs**: `for_splat/sparse/0/points3D.bin`
- **Key Detail**: Samples 10% of points by default (~500K-1M points from 5-10M original)

### 7. Gaussian Splatting Training (`train_splat.py`)
- **What**: Trains 3D Gaussian Splatting model with LichtFeld-Studio
- **Why**: Creates final splat representation for novel view synthesis
- **Typical Duration**: Varies significantly with image count, resolution, and iterations:
  - Low-res mode (~512px, 500 images, 20k iters): ~30-45 min on RTX 3090
  - High-res mode (~5568px, 500 images, 20k iters): ~2-4 hours on RTX 3090
- **Outputs**:
  - `splats/splat_*.ply` - Trained Gaussian splat models (checkpoints)
  - `splats/run.log` - Full LichtFeld-Studio training output
  - `splats/run_report.txt` - Concise summary with training progress and final command used
- **Key Detail**: Always uses PINHOLE camera model (no distortion) because images are undistorted in earlier steps. Initialized with SLAM point cloud for faster convergence.

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
│   ├── images/                    # Keyframes (undistorted, format depends on mode)
│   │   ├── 000000.png             # Mode A: M-SLAM's low-res keyframes (~512px)
│   │   └── 2019A_GP_Left (7).JPG  # Mode B: High-res images, undistorted & cropped (if use_highres_for_splatting=true)
│   │   └── ...
│   └── sparse/0/
│       ├── cameras.bin            # Camera model - ALWAYS PINHOLE (images are undistorted)
│       ├── cameras.txt            # fx, fy, cx, cy scaled to final image resolution
│       ├── images.bin             # Camera poses (COLMAP format)
│       ├── images.txt             # Contains timestamp-based names (Mode A) or original filenames (Mode B)
│       ├── images_lowres.txt      # Backup with timestamp names (only if use_highres_for_splatting=true)
│       ├── images_lowres.bin      # Backup with timestamp names (only if use_highres_for_splatting=true)
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
