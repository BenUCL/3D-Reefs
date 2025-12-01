#!/usr/bin/env python3
"""
prepare_highres_splat.py

PURPOSE: Apply the same undistortion and cropping to high-res images that M-SLAM
         applied to low-res images, ensuring geometric consistency.

Modes:
  --mode keyframes: Process only keyframe images (from keyframe_mapping.txt)
  --mode all:       Process ALL images in original_images_path

Inputs:
  - High-Res Raw Images (with lens distortion, e.g., 5568x4872 GoPro JPG/PNG)
  - Low-Res intrinsics.yaml (OPENCV model with distortion, from M-SLAM input)
  - keyframe_mapping.txt (for keyframes mode - maps timestamps to original filenames)

Actions:
  1. Scales Low-Res Intrinsics → High-Res (e.g., 1600x1400 → 5568x4872)
  2. Undistorts High-Res Images using cv2.remap() (removes lens distortion)
  3. Crops High-Res Images using M-SLAM's center-crop logic
  4. Outputs High-Res PINHOLE cameras.txt (no distortion - images now distortion-free)

Result: High-res images with identical geometry to M-SLAM keyframes, just higher resolution.
        Final camera model is PINHOLE because all distortion has been removed.

Usage:
  # Process only keyframes (Mode B - high-res keyframes only)
  python prepare_highres_splat.py --dataset my_run --highres_dir /path/to/images --intrinsics intrinsics.yaml --mode keyframes
  
  # Process all images (Mode C - interpolation + all images)
  python prepare_highres_splat.py --dataset my_run --highres_dir /path/to/images --intrinsics intrinsics.yaml --mode all
"""
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

INTERMEDIATE_DATA_ROOT = Path('/home/ben/encode/data/intermediate_data')
SLAM_SIZE = 512

def get_mslam_crop_ratio(w_rectified, h_rectified):
    scale = SLAM_SIZE / max(w_rectified, h_rectified)
    w_new, h_new = int(round(w_rectified * scale)), int(round(h_rectified * scale))
    cx, cy = w_new // 2, h_new // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    crop_l, crop_t = cx - halfw, cy - halfh
    crop_r, crop_b = cx + halfw, cy + halfh
    return (crop_l / w_new, crop_t / h_new, crop_r / w_new, crop_b / h_new)

def write_colmap_pinhole(output_dir, width, height, fx, fy, cx, cy, cam_id=1):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"{cam_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    import struct
    with open(output_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack('<Q', 1) + struct.pack('<i', cam_id) + struct.pack('<i', 1) + 
                struct.pack('<Q', width) + struct.pack('<Q', height) + 
                struct.pack('<d', fx) + struct.pack('<d', fy) + 
                struct.pack('<d', cx) + struct.pack('<d', cy))
    print(f"✓ Saved PINHOLE cameras.txt/bin")

def natural_sort_key(filename):
    """Natural sorting for filenames with embedded numbers."""
    import re
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', str(filename))]


def get_images_to_process(highres_dir, mode, mapping_file=None):
    """
    Determine which images to process based on mode.
    
    Args:
        highres_dir: Path to high-res images directory
        mode: "keyframes" or "all"
        mapping_file: Path to keyframe_mapping.txt (required for keyframes mode)
    
    Returns:
        List of image filenames to process (sorted naturally)
    """
    if mode == "keyframes":
        if mapping_file is None or not mapping_file.exists():
            raise FileNotFoundError(f"Keyframes mode requires keyframe_mapping.txt: {mapping_file}")
        
        # Read keyframe mapping to get list of keyframe images
        keyframe_images = []
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "original_filename" in line:
                    continue
                
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    original_filename = parts[2].strip('"')
                    keyframe_images.append(original_filename)
        
        print(f"Mode: keyframes - Processing {len(keyframe_images)} keyframe images")
        return keyframe_images
    
    elif mode == "all":
        # Get ALL images from directory
        image_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(highres_dir.glob(f'*{ext}')))
        
        # Sort naturally and return just filenames
        all_images_sorted = sorted([img.name for img in all_images], key=natural_sort_key)
        print(f"Mode: all - Processing {len(all_images_sorted)} total images")
        return all_images_sorted
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'keyframes' or 'all'")


def main():
    parser = argparse.ArgumentParser(
        description="Process high-res images with M-SLAM undistortion and cropping"
    )
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--highres_dir', required=True, type=Path, help='High-res images directory')
    parser.add_argument('--intrinsics', required=True, type=Path, help='Low-res intrinsics.yaml')
    parser.add_argument(
        '--mode', 
        type=str, 
        default='keyframes',
        choices=['keyframes', 'all'],
        help='Processing mode: keyframes (only keyframes) or all (all images)'
    )
    args = parser.parse_args()

    run_dir = INTERMEDIATE_DATA_ROOT / args.dataset
    mapping_file = run_dir / 'mslam_logs' / 'keyframe_mapping.txt'
    output_img_dir = run_dir / 'for_splat' / 'images'
    output_sparse_dir = run_dir / 'for_splat' / 'sparse' / '0'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing High-Resolution Images")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"High-res dir: {args.highres_dir}")
    print(f"Mode: {args.mode}")
    print()

    # 1. Load Low-Res Intrinsics
    print("[1/8] Loading low-res intrinsics...")
    with open(args.intrinsics, 'r') as f:
        calib = yaml.safe_load(f)
    
    w_low, h_low = calib['width'], calib['height']
    if isinstance(calib['calibration'], str):
        c_vals = calib['calibration'].strip('[]').split(',')
        calibration = [float(x) for x in c_vals]
    else:
        calibration = calib['calibration']
    fx_low, fy_low, cx_low, cy_low = calibration[:4]
    dist = np.array(calibration[4:]) if len(calibration) > 4 else np.zeros(4)
    
    print(f"  Low-res: {w_low}x{h_low}")
    print(f"  fx={fx_low:.2f}, fy={fy_low:.2f}, cx={cx_low:.2f}, cy={cy_low:.2f}")
    print(f"  Distortion: {dist}")

    # 2. Detect High-Res Size
    print("\n[2/8] Detecting high-res resolution...")
    first_img_path = next(args.highres_dir.glob("*.[jJpP]*"))
    img_test = cv2.imread(str(first_img_path))
    h_high, w_high = img_test.shape[:2]
    
    # 3. Calculate Scale & High-Res Matrix
    scale = w_high / w_low
    print(f"  High-res: {w_high}x{h_high}")
    print(f"  Scale factor: {scale:.4f}")
    
    K_high = np.array([
        [fx_low * scale, 0, cx_low * scale],
        [0, fy_low * scale, cy_low * scale],
        [0, 0, 1]
    ])
    
    # 4. Compute Undistortion & Rectification
    print("\n[3/8] Computing undistortion maps...")
    K_rect, roi = cv2.getOptimalNewCameraMatrix(
        K_high, dist, (w_high, h_high), 0, (w_high, h_high), centerPrincipalPoint=True
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        K_high, dist, None, K_rect, (w_high, h_high), cv2.CV_32FC1
    )
    
    # 5. Compute Crop Logic (on High-Res dimensions)
    print("\n[4/8] Computing crop parameters...")
    l_pct, t_pct, r_pct, b_pct = get_mslam_crop_ratio(w_high, h_high)
    crop_l = int(l_pct * w_high)
    crop_t = int(t_pct * h_high)
    crop_r = int(r_pct * w_high)
    crop_b = int(b_pct * h_high)
    
    final_w, final_h = crop_r - crop_l, crop_b - crop_t
    print(f"  Rectified size: {w_high}x{h_high}")
    print(f"  Crop region: ({crop_l}, {crop_t}) to ({crop_r}, {crop_b})")
    print(f"  Final size: {final_w}x{final_h}")

    # 6. Save PINHOLE Intrinsics (Shifted by crop)
    print("\n[5/8] Writing PINHOLE cameras.txt/bin...")
    fx_final, fy_final = K_rect[0,0], K_rect[1,1]
    cx_final, cy_final = K_rect[0,2] - crop_l, K_rect[1,2] - crop_t
    write_colmap_pinhole(output_sparse_dir, final_w, final_h, fx_final, fy_final, cx_final, cy_final)
    
    print("\n[6/8] Output directories prepared...")
    print(f"  Images: {output_img_dir}")
    print(f"  Sparse: {output_sparse_dir}")

    # 7. Determine which images to process
    print("\n[7/8] Determining images to process...")
    images_to_process = get_images_to_process(
        args.highres_dir, 
        args.mode, 
        mapping_file if args.mode == 'keyframes' else None
    )
    
    # 8. Process Images
    print(f"\n[8/8] Processing {len(images_to_process)} images...")
    processed = 0
    skipped = 0
    
    for img_filename in tqdm(images_to_process, desc="Processing"):
        src_path = args.highres_dir / img_filename
        dst_path = output_img_dir / img_filename
        
        # Handle extension mismatches (e.g., file is .JPG but mapping says .jpg)
        if not src_path.exists():
            tried_extensions = ['.JPG', '.jpg', '.PNG', '.png', '.JPEG', '.jpeg']
            base_name = src_path.stem
            for ext in tried_extensions:
                alt_src = args.highres_dir / f"{base_name}{ext}"
                if alt_src.exists():
                    src_path = alt_src
                    img_filename = src_path.name  # Update filename to actual
                    dst_path = output_img_dir / img_filename
                    break
        
        if not src_path.exists():
            print(f"\n  WARNING: Could not find {img_filename}, skipping")
            skipped += 1
            continue
        
        # Read, undistort, crop, and save
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"\n  WARNING: Could not read {img_filename}, skipping")
            skipped += 1
            continue
            
        img_rect = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        img_final = img_rect[crop_t:crop_b, crop_l:crop_r]
        cv2.imwrite(str(dst_path), img_final)
        processed += 1

    print(f"\n{'='*70}")
    print(f"✅ Successfully processed {processed} images!")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} images (not found or read errors)")
    print(f"{'='*70}")
    print(f"Output images: {output_img_dir}")
    print(f"Output cameras: {output_sparse_dir}")
    print()

if __name__ == "__main__":
    main()