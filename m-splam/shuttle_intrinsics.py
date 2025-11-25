#!/usr/bin/env python3
"""
shuttle_intrinsics.py

Convert and move COLMAP calibration intrinsics for use in:
1. MASt3R-SLAM (intrinsics.yaml: original model with distortion, scaled to SLAM resolution 512x448)
2. LichtFeld splat (cameras.txt + cameras.bin):
   - If use_highres_for_splatting=False: PINHOLE model at SLAM resolution (keyframes already undistorted)
   - If use_highres_for_splatting=True: OPENCV model at high-res resolution (original images have distortion)

Key behavior:
- Always scales intrinsics from calibration resolution (e.g., 1600x1400) to SLAM resolution (512x448) for MASt3R-SLAM
- For splatting:
  * Low-res mode: Scales 1600→512, converts to PINHOLE (no distortion)
  * High-res mode: Scales 1600→5568 (auto-detected), keeps OPENCV (with distortion)

MUST RUN WITH: conda activate mast3r-slam

Usage:
    # Low-res splatting mode (default)
    python shuttle_intrinsics.py --dataset reef_soneva
    
    # High-res splatting mode
    python shuttle_intrinsics.py --dataset reef_soneva --use-highres-for-splatting \
        --highres-images-path /path/to/highres/images
"""
import argparse
import yaml
import numpy as np
import struct
import glob
import os
from pathlib import Path
from PIL import Image

# Import MASt3R-SLAM resize function for intrinsics adjustment
from mast3r_slam.dataloader import resize_img

# Dataset
INTERMEDIATE_DATA_ROOT = Path('/home/ben/encode/data/intermediate_data')
# The keyframe size parameter for mast3r-slam (single int, aspect ratio preserved)
SLAM_SIZE = 512


def get_resolution(directory):
    """Finds the first image in a directory and returns (width, height)."""
    #TODO: Would be safe to check a few. Sometimes images are a few pixels off and/or
    # may even add multiple cameras so would need to handle that.
    # Search for common extensions
    types = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    files = []
    for t in types:
        files.extend(glob.glob(os.path.join(directory, t)))
    
    if not files:
        raise FileNotFoundError(f"No images found in {directory}")
    
    with Image.open(files[0]) as img:
        return img.size  # returns (width, height)


def read_colmap_cameras_txt(cameras_txt_path):
    """Read COLMAP cameras.txt and extract intrinsics."""
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            return {
                'camera_id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    raise ValueError(f"No camera found in {cameras_txt_path}")


def scale_intrinsics(K, raw_w, raw_h, target_size):
    """
    Scale intrinsics from raw resolution to target resolution using
    MASt3R-SLAM's resize_img transformation.
    
    Args:
        K: 3x3 intrinsic matrix
        raw_w, raw_h: Raw image dimensions
        target_size: Target size (512 for MASt3R-SLAM) - will maintain aspect ratio
    
    Returns:
        K_scaled: Adjusted 3x3 intrinsic matrix
        scaled_width, scaled_height: New dimensions
    """
    _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
        np.zeros((raw_h, raw_w, 3)), target_size, return_transformation=True
    )
    
    K_scaled = K.copy()
    K_scaled[0, 0] = K[0, 0] / scale_w
    K_scaled[1, 1] = K[1, 1] / scale_h
    K_scaled[0, 2] = K[0, 2] / scale_w - half_crop_w
    K_scaled[1, 2] = K[1, 2] / scale_h - half_crop_h
    
    # Calculate new dimensions
    scaled_width = int(raw_w / scale_w)
    scaled_height = int(raw_h / scale_h)
    
    return K_scaled, scaled_width, scaled_height


def params_to_matrix(params):
    """Convert camera params to 3x3 intrinsic matrix K (assumes first 4 are fx, fy, cx, cy)."""
    fx, fy, cx, cy = params[:4]
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return K


def write_mast3r_yaml(output_path, width, height, K, distortion_params, model_name):
    """
    Write intrinsics.yaml for MASt3R-SLAM with original model (includes distortion if present).
    
    Args:
        output_path: Path to save intrinsics.yaml
        width, height: Raw image dimensions
        K: 3x3 intrinsic matrix
        distortion_params: Distortion coefficients (model-dependent)
        model_name: Original camera model name
    """
    # Build calibration list: [fx, fy, cx, cy] + distortion params if present
    calibration = [
        float(K[0, 0]),  # fx
        float(K[1, 1]),  # fy
        float(K[0, 2]),  # cx
        float(K[1, 2])   # cy
    ]
    
    # Check if distortion is significant (not all zeros/near-zeros)
    has_distortion = any(abs(d) > 1e-6 for d in distortion_params)
    
    if has_distortion:
        # Add distortion coefficients
        calibration.extend([float(d) for d in distortion_params])
    
    intrinsics_dict = {
        'width': int(width),
        'height': int(height),
        'calibration': calibration
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(intrinsics_dict, f, default_flow_style=False, sort_keys=False)
    
    distortion_str = ', '.join([f'{d:.6f}' for d in distortion_params]) if has_distortion else 'none'
    print(f"✓ Saved MASt3R-SLAM intrinsics ({model_name}): {output_path}")
    print(f"  - Resolution: {width}x{height} (raw)")
    print(f"  - Distortion: {distortion_str}")


def write_colmap_cameras_txt(output_path, camera_id, model, width, height, params):
    """Write COLMAP cameras.txt format."""
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        
        params_str = ' '.join([str(p) for p in params])
        f.write(f"{camera_id} {model} {width} {height} {params_str}\n")
    
    print(f"✓ Saved cameras.txt ({model}): {output_path}")


def write_colmap_cameras_bin(output_path, camera_id, model, width, height, params):
    """
    Write COLMAP cameras.bin in binary format.
    
    Binary format:
        num_cameras (uint64)
        For each camera:
            camera_id (int32)
            model_id (int32)
            width (uint64)
            height (uint64)
            params (double[num_params]) - num_params is implicit from model_id, NOT written
    """
    # COLMAP model name to ID mapping
    MODEL_NAME_TO_ID = {
        'SIMPLE_PINHOLE': 0,
        'PINHOLE': 1,
        'SIMPLE_RADIAL': 2,
        'RADIAL': 3,
        'OPENCV': 4,
        'OPENCV_FISHEYE': 5,
        'FULL_OPENCV': 6,
        'FOV': 7,
        'SIMPLE_RADIAL_FISHEYE': 8,
        'RADIAL_FISHEYE': 9,
        'THIN_PRISM_FISHEYE': 10
    }
    
    model_id = MODEL_NAME_TO_ID.get(model)
    if model_id is None:
        raise ValueError(f"Unknown camera model: {model}")
    
    with open(output_path, 'wb') as f:
        # Write number of cameras
        f.write(struct.pack('<Q', 1))
        
        # Write camera data
        f.write(struct.pack('<i', camera_id))
        f.write(struct.pack('<i', model_id))
        f.write(struct.pack('<Q', width))
        f.write(struct.pack('<Q', height))
        # Note: COLMAP cameras.bin does NOT include num_params field
        # The number of params is determined implicitly by the model_id
        
        # Write parameters
        for param in params:
            f.write(struct.pack('<d', param))
    
    print(f"✓ Saved cameras.bin ({model}): {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP intrinsics for MASt3R-SLAM (original model) and LichtFeld"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., reef_soneva)'
    )
    parser.add_argument(
        '--use-highres-for-splatting',
        action='store_true',
        help='Scale intrinsics for high-res images instead of SLAM resolution (requires --highres-images-path)'
    )
    parser.add_argument(
        '--highres-images-path',
        type=str,
        default=None,
        help='Path to high-resolution images directory (required if --use-highres-for-splatting is set)'
    )
    parser.add_argument(
        '--keep-original',
        action='store_true',
        help='Also save original model as cameras_{MODEL}.bin/txt'
    )
    
    args = parser.parse_args()
    
    # Validate high-res arguments
    if args.use_highres_for_splatting and not args.highres_images_path:
        parser.error("--highres-images-path is required when --use-highres-for-splatting is set")
    
    # Construct paths from dataset name
    dataset_root = INTERMEDIATE_DATA_ROOT / args.dataset
    colmap_cameras = dataset_root / 'colmap_outputs' / 'cameras.txt'
    
    if not colmap_cameras.exists():
        print(f"Error: COLMAP cameras.txt not found: {colmap_cameras}")
        print("Run estimate_intrinsics.sh first!")
        return
    
    # Read COLMAP intrinsics
    print(f"\n{'='*70}")
    print(f"Processing intrinsics for: {args.dataset}")
    print(f"{'='*70}")
    print(f"Reading COLMAP cameras.txt: {colmap_cameras}")
    cam = read_colmap_cameras_txt(colmap_cameras)
    
    print(f"  Camera ID: {cam['camera_id']}")
    print(f"  Model: {cam['model']}")
    print(f"  Resolution: {cam['width']}x{cam['height']}")
    print(f"  Params: {cam['params']}")
    
    # Extract intrinsics
    raw_width = cam['width']
    raw_height = cam['height']
    original_model = cam['model']
    fx, fy, cx, cy = cam['params'][:4]
    distortion = cam['params'][4:] if len(cam['params']) > 4 else []
    
    K_raw = params_to_matrix(cam['params'])
    
    # Setup output paths
    yaml_output = dataset_root / 'intrinsics.yaml'
    splat_dir = dataset_root / 'for_splat' / 'sparse' / '0'
    splat_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. MASt3R-SLAM intrinsics.yaml (original model with distortion, raw resolution)
    # =========================================================================
    print(f"\n[1/3] Generating MASt3R-SLAM intrinsics.yaml ({original_model})...")
    write_mast3r_yaml(yaml_output, raw_width, raw_height, K_raw, distortion, original_model)
    
    # =========================================================================
    # 2. Adjusted intrinsics for splat (SLAM resolution)
    # =========================================================================
    print(f"\n[2/3] Adjusting intrinsics for SLAM resolution ({SLAM_SIZE} with aspect ratio)...")
    K_slam, slam_width, slam_height = scale_intrinsics(K_raw, raw_width, raw_height, SLAM_SIZE)
    
    print(f"  Scaled resolution: {slam_width}x{slam_height}")
    print(f"  Scaled intrinsics:")
    print(f"    fx={K_slam[0,0]:.2f}, fy={K_slam[1,1]:.2f}")
    print(f"    cx={K_slam[0,2]:.2f}, cy={K_slam[1,2]:.2f}")
    
    # =========================================================================
    # 3. Determine target resolution for splatting
    # =========================================================================
    if args.use_highres_for_splatting:
        # Detect high-res image dimensions
        print(f"\n[3/4] Detecting high-res image resolution...")
        print(f"  Scanning: {args.highres_images_path}")
        target_width, target_height = get_resolution(args.highres_images_path)
        print(f"  Detected resolution: {target_width}x{target_height}")
        print(f"  Will scale intrinsics from calibration ({raw_width}x{raw_height}) → high-res")
    else:
        # Use SLAM resolution for undistorted keyframes
        print(f"\n[3/4] Using SLAM resolution for undistorted keyframes...")
        target_width, target_height = slam_width, slam_height
        print(f"  Target resolution: {target_width}x{target_height}")
    
    # =========================================================================
    # 4. Save LichtFeld cameras with correct scaling
    # =========================================================================
    if args.use_highres_for_splatting:
        print(f"\n[4/4] Saving LichtFeld cameras ({original_model} - for high-res splatting)...")
        
        # Scale intrinsics directly from raw calibration to high-res
        # Can't use resize_img() here because it only accepts 224 or 512
        # Instead, manually compute scale factors
        scale_x = target_width / raw_width
        scale_y = target_height / raw_height
        
        K_target = K_raw.copy()
        K_target[0, 0] = K_raw[0, 0] * scale_x  # fx
        K_target[1, 1] = K_raw[1, 1] * scale_y  # fy
        K_target[0, 2] = K_raw[0, 2] * scale_x  # cx
        K_target[1, 2] = K_raw[1, 2] * scale_y  # cy
        
        # Keep original model with distortion
        splat_model = original_model
        splat_params = [K_target[0,0], K_target[1,1], K_target[0,2], K_target[1,2]]
        if distortion:
            splat_params.extend(distortion)
        
        print(f"  Scaled intrinsics from {raw_width}x{raw_height} → {target_width}x{target_height}:")
        print(f"    Scale factors: {scale_x:.4f}x, {scale_y:.4f}x")
        print(f"    fx={K_target[0,0]:.2f}, fy={K_target[1,1]:.2f}")
        print(f"    cx={K_target[0,2]:.2f}, cy={K_target[1,2]:.2f}")
    else:
        print(f"\n[4/4] Saving LichtFeld cameras (PINHOLE - no distortion, for undistorted keyframes)...")
        
        # Use SLAM-scaled intrinsics
        target_width, target_height = slam_width, slam_height
        K_target = K_slam
        
        # PINHOLE parameters (only fx, fy, cx, cy) - keyframes are already undistorted
        splat_model = 'PINHOLE'
        splat_params = [K_target[0,0], K_target[1,1], K_target[0,2], K_target[1,2]]
    
    # Default output: cameras.bin/txt
    cameras_bin_path = splat_dir / 'cameras.bin'
    cameras_txt_path = splat_dir / 'cameras.txt'
    
    write_colmap_cameras_bin(
        cameras_bin_path,
        cam['camera_id'],
        splat_model,
        target_width,
        target_height,
        splat_params
    )
    
    write_colmap_cameras_txt(
        cameras_txt_path,
        cam['camera_id'],
        splat_model,
        target_width,
        target_height,
        splat_params
    )
    
    # Optional: also save original model version
    if args.keep_original:
        print(f"\n[3b] Also saving original model ({original_model})...")
        
        # Original model parameters (scaled intrinsics + distortion)
        original_params = [K_slam[0,0], K_slam[1,1], K_slam[0,2], K_slam[1,2]]
        if distortion:
            original_params.extend(distortion)
        
        original_bin_path = splat_dir / f'cameras_{original_model}.bin'
        original_txt_path = splat_dir / f'cameras_{original_model}.txt'
        
        write_colmap_cameras_bin(
            original_bin_path,
            cam['camera_id'],
            original_model,
            slam_width,
            slam_height,
            original_params
        )
        
        write_colmap_cameras_txt(
            original_txt_path,
            cam['camera_id'],
            original_model,
            slam_width,
            slam_height,
            original_params
        )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"  1. MASt3R-SLAM:")
    print(f"     - {yaml_output}")
    print(f"     - Format: {original_model} (raw resolution {raw_width}x{raw_height})")
    print(f"\n  2. LichtFeld-Studio:")
    print(f"     - {cameras_bin_path}")
    print(f"     - {cameras_txt_path}")
    if args.use_highres_for_splatting:
        print(f"     - Format: {splat_model} with distortion")
        print(f"     - Resolution: {target_width}x{target_height} (high-res)")
        print(f"     - Mode: High-res splatting (scaled directly from calibration)")
    else:
        print(f"     - Format: {splat_model} no distortion")
        print(f"     - Resolution: {target_width}x{target_height} (SLAM resolution)")
        print(f"     - Mode: Keyframe splatting (undistorted images from MASt3R-SLAM)")
    
    if args.keep_original:
        print(f"\n  3. LichtFeld-Studio (original model backup):")
        print(f"     - {splat_dir / f'cameras_{original_model}.bin'}")
        print(f"     - {splat_dir / f'cameras_{original_model}.txt'}")
        print(f"     - Format: {original_model} (SLAM resolution)")
    
    print(f"\nKey design decisions:")
    print(f"  ✓ MASt3R-SLAM uses {original_model} model (needs distortion for cv2.remap)")
    print(f"  ✓ MASt3R-SLAM keyframes are undistorted internally via cv2.remap()")
    if args.use_highres_for_splatting:
        print(f"  ✓ LichtFeld will use {splat_model} model (for distorted high-res images)")
        print(f"  ✓ Intrinsics scaled directly from calibration {raw_width}x{raw_height} → {target_width}x{target_height}")
        print(f"  ✓ Use --gut flag in LichtFeld ({splat_model} has distortion params)")
    else:
        print(f"  ✓ LichtFeld uses PINHOLE model (keyframes already undistorted)")
        print(f"  ✓ No --gut flag needed with PINHOLE (no distortion to model)")
    
    print(f"\nNext steps:")
    print(f"  1. Run MASt3R-SLAM with intrinsics.yaml")
    print(f"  2. Run cam_pose_keyframes_shuttle.py to convert poses")
    if args.use_highres_for_splatting:
        print(f"  3. Run convert_intrinsics.py to scale intrinsics to high-res")
        print(f"  4. Run mslam_ply_to_points3d.py to convert point cloud")
        print(f"  5. Run LichtFeld WITH --gut flag ({splat_model} model)")
    else:
        print(f"  3. Run mslam_ply_to_points3d.py to convert point cloud")
        print(f"  4. Run LichtFeld WITHOUT --gut flag (PINHOLE model)")
    print()


if __name__ == '__main__':
    main()
