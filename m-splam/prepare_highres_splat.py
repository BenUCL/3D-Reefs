#!/usr/bin/env python3
"""
prepare_highres_splat.py

PURPOSE: Apply the same undistortion and cropping to high-res images that M-SLAM
         applied to low-res images, ensuring geometric consistency.

Inputs:
  - High-Res Raw Images (with lens distortion, e.g., 5568x4872 GoPro JPG/PNG)
  - Low-Res intrinsics.yaml (OPENCV model with distortion, from M-SLAM input)
  - keyframe_mapping.txt (maps timestamps to original filenames)

Actions:
  1. Scales Low-Res Intrinsics → High-Res (e.g., 1600x1400 → 5568x4872)
  2. Undistorts High-Res Images using cv2.remap() (removes lens distortion)
  3. Crops High-Res Images using M-SLAM's center-crop logic
  4. Outputs High-Res PINHOLE cameras.txt (no distortion - images now distortion-free)

Result: High-res images with identical geometry to M-SLAM keyframes, just higher resolution.
        Final camera model is PINHOLE because all distortion has been removed.
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--highres_dir', required=True, type=Path)
    parser.add_argument('--intrinsics', required=True, type=Path, help="The LOW-RES YAML used by M-SLAM")
    args = parser.parse_args()

    run_dir = INTERMEDIATE_DATA_ROOT / args.dataset
    mapping_file = run_dir / 'mslam_logs' / 'keyframe_mapping.txt'
    output_img_dir = run_dir / 'for_splat' / 'images'
    output_sparse_dir = run_dir / 'for_splat' / 'sparse' / '0'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_sparse_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Low-Res Intrinsics
    print(f"Loading Low-Res Intrinsics: {args.intrinsics}")
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

    # 2. Detect High-Res Size
    first_img_path = next(args.highres_dir.glob("*.[jJpP]*"))
    img_test = cv2.imread(str(first_img_path))
    h_high, w_high = img_test.shape[:2]
    
    # 3. Calculate Scale & High-Res Matrix
    scale = w_high / w_low
    print(f"Scaling Intrinsics: {w_low}x{h_low} -> {w_high}x{h_high} (Factor: {scale:.4f})")
    
    K_high = np.array([
        [fx_low * scale, 0, cx_low * scale],
        [0, fy_low * scale, cy_low * scale],
        [0, 0, 1]
    ])
    
    # 4. Compute Undistortion & Rectification
    K_rect, roi = cv2.getOptimalNewCameraMatrix(
        K_high, dist, (w_high, h_high), 0, (w_high, h_high), centerPrincipalPoint=True
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        K_high, dist, None, K_rect, (w_high, h_high), cv2.CV_32FC1
    )
    
    # 5. Compute Crop Logic (on High-Res dimensions)
    l_pct, t_pct, r_pct, b_pct = get_mslam_crop_ratio(w_high, h_high)
    crop_l = int(l_pct * w_high)
    crop_t = int(t_pct * h_high)
    crop_r = int(r_pct * w_high)
    crop_b = int(b_pct * h_high)
    
    final_w, final_h = crop_r - crop_l, crop_b - crop_t
    print(f"Final Splat Resolution: {final_w}x{final_h}")

    # 6. Save PINHOLE Intrinsics (Shifted by crop)
    fx_final, fy_final = K_rect[0,0], K_rect[1,1]
    cx_final, cy_final = K_rect[0,2] - crop_l, K_rect[1,2] - crop_t
    write_colmap_pinhole(output_sparse_dir, final_w, final_h, fx_final, fy_final, cx_final, cy_final)

    # 7. Process Images
    print("\nProcessing Images...")
    with open(mapping_file, 'r') as f:
        lines = [l for l in f.readlines() if not l.startswith("#") and "original_filename" not in l]

    for line in tqdm(lines):
        parts = line.strip().split(None, 2)
        if len(parts) < 3: continue
        original_name = parts[2].strip('"')
        
        src_path = args.highres_dir / original_name
        dst_path = output_img_dir / original_name
        
        # Handle extension mismatches
        if not src_path.exists():
            for ext in ['.JPG', '.jpg', '.PNG', '.png']:
                if src_path.with_suffix(ext).exists():
                    src_path = src_path.with_suffix(ext)
                    break
        
        if not src_path.exists(): continue
            
        img = cv2.imread(str(src_path))
        img_rect = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        img_final = img_rect[crop_t:crop_b, crop_l:crop_r]
        cv2.imwrite(str(dst_path), img_final)

    print("\n✅ Done!")

if __name__ == "__main__":
    main()