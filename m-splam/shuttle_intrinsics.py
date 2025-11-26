#!/usr/bin/env python3
"""
shuttle_intrinsics.py

Modes:
- Low-Res Mode (use_highres_for_splatting=False):
  * Outputs intrinsics.yaml (Low-Res) for M-SLAM.
  * Outputs cameras.txt (Low-Res PINHOLE) for Splatting.

- High-Res Mode (use_highres_for_splatting=True):
  * Outputs intrinsics.yaml (Low-Res) for M-SLAM.
  * DOES NOT output cameras.txt (Handled by prepare_highres_splat.py).
"""
import argparse
import yaml
import numpy as np
import struct
import glob
import os
from pathlib import Path
from PIL import Image
from mast3r_slam.dataloader import resize_img

INTERMEDIATE_DATA_ROOT = Path('/home/ben/encode/data/intermediate_data')
SLAM_SIZE = 512

def read_colmap_cameras_txt(cameras_txt_path):
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            return {
                'camera_id': int(parts[0]),
                'model': parts[1],
                'width': int(parts[2]),
                'height': int(parts[3]),
                'params': [float(p) for p in parts[4:]]
            }
    raise ValueError(f"No camera found in {cameras_txt_path}")

def scale_intrinsics_crop_aware(K, raw_w, raw_h, target_size):
    _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
        np.zeros((raw_h, raw_w, 3)), target_size, return_transformation=True
    )
    K_scaled = K.copy()
    K_scaled[0, 0] = K[0, 0] / scale_w
    K_scaled[1, 1] = K[1, 1] / scale_h
    K_scaled[0, 2] = K[0, 2] / scale_w - half_crop_w
    K_scaled[1, 2] = K[1, 2] / scale_h - half_crop_h
    return K_scaled, int(raw_w / scale_w), int(raw_h / scale_h)

def params_to_matrix(params):
    fx, fy, cx, cy = params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

def write_mast3r_yaml(output_path, width, height, K, distortion_params):
    calibration = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
    if any(abs(d) > 1e-6 for d in distortion_params):
        calibration.extend([float(d) for d in distortion_params])
    
    with open(output_path, 'w') as f:
        f.write(f"width: {int(width)}\n")
        f.write(f"height: {int(height)}\n")
        calib_str = '[' + ', '.join([f'{x:.6g}' for x in calibration]) + ']'
        f.write(f"calibration: {calib_str}\n")
    print(f"✓ Saved MASt3R-SLAM intrinsics (Low-Res): {output_path}")

def write_colmap_cameras_txt(output_path, camera_id, model, width, height, params):
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        params_str = ' '.join([str(p) for p in params])
        f.write(f"{camera_id} {model} {width} {height} {params_str}\n")
    print(f"✓ Saved cameras.txt: {output_path}")

def write_colmap_cameras_bin(output_path, camera_id, model, width, height, params):
    # Mapping omitted for brevity, identical to previous script
    MODEL_NAME_TO_ID = {'PINHOLE': 1} # Simplified
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<i', camera_id))
        f.write(struct.pack('<i', 1)) # PINHOLE
        f.write(struct.pack('<Q', width))
        f.write(struct.pack('<Q', height))
        for param in params: f.write(struct.pack('<d', param))
    print(f"✓ Saved cameras.bin: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--use-highres-for-splatting', action='store_true')
    parser.add_argument('--highres-images-path', type=str) # Kept for arg compatibility
    parser.add_argument('--keep-original', action='store_true')
    args = parser.parse_args()
    
    dataset_root = INTERMEDIATE_DATA_ROOT / args.dataset
    colmap_cameras = dataset_root / 'colmap_outputs' / 'cameras.txt'
    
    if not colmap_cameras.exists():
        print(f"Error: {colmap_cameras} not found")
        return
    
    cam = read_colmap_cameras_txt(colmap_cameras)
    low_w, low_h = cam['width'], cam['height']
    distortion = cam['params'][4:] if len(cam['params']) > 4 else []
    K_low = params_to_matrix(cam['params'])
    
    yaml_output = dataset_root / 'intrinsics.yaml'
    splat_dir = dataset_root / 'for_splat' / 'sparse' / '0'
    splat_dir.mkdir(parents=True, exist_ok=True)
    
    # ALWAYS generate Low-Res YAML for M-SLAM
    write_mast3r_yaml(yaml_output, low_w, low_h, K_low, distortion)
    
    if args.use_highres_for_splatting:
        print("\n[High-Res Mode Enabled]")
        print("1. Generated Low-Res intrinsics.yaml for M-SLAM.")
        print("2. Skipping cameras.txt generation (prepare_highres_splat.py will handle it).")
    else:
        print("\n[Low-Res Mode Enabled]")
        K_slam, slam_w, slam_h = scale_intrinsics_crop_aware(K_low, low_w, low_h, SLAM_SIZE)
        splat_params = [K_slam[0,0], K_slam[1,1], K_slam[0,2], K_slam[1,2]]
        write_colmap_cameras_txt(splat_dir / 'cameras.txt', cam['camera_id'], 'PINHOLE', slam_w, slam_h, splat_params)
        write_colmap_cameras_bin(splat_dir / 'cameras.bin', cam['camera_id'], 'PINHOLE', slam_w, slam_h, splat_params)

if __name__ == '__main__':
    main()