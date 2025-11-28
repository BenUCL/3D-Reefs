#TODO: this script is so slow. Can we make it faster?
"""
Undistort images from COLMAP OPENCV camera model to PINHOLE model.
This script reads the distorted images and their intrinsics, undistorts them,
and overwrites the images in place. It also writes a new camera file with
the updated PINHOLE intrinsics to OUTPUT_CAMERAS_TXT.
"""

import cv2
import numpy as np
import os
import glob

# --- CONFIGURATION ---
# Folder containing the raw distorted images (Will be overwritten!)
IMAGES_DIR = "/home/ben/encode/data/intermediate_data/pycusfm1/for_splat/images"

# The current distorted camera file (OPENCV model)
INPUT_CAMERAS_TXT = "/home/ben/encode/data/intermediate_data/pycusfm1/cameras_5568x4872.txt"

# The output location for the new PINHOLE camera file
OUTPUT_CAMERAS_TXT = "/home/ben/encode/data/intermediate_data/pycusfm1/for_pycusfm/cameras.txt"

def read_opencv_intrinsics(path):
    print(f"Reading intrinsics from: {path}")
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            # COLMAP OPENCV format: ID MODEL W H fx fy cx cy k1 k2 p1 p2
            w = int(parts[2])
            h = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            dist = np.array(params[4:]) # k1, k2, p1, p2...
            
            return w, h, fx, fy, cx, cy, dist
    raise ValueError("Could not find valid camera line in file.")

def main():
    # 1. Get current distorted parameters
    W, H, fx, fy, cx, cy, dist = read_opencv_intrinsics(INPUT_CAMERAS_TXT)
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    print(f"Original Matrix (OPENCV):\n{K}")
    print(f"Distortion Coeffs: {dist}")

    # 2. Calculate New Optimal Pinhole Matrix
    # 'alpha=0' means crop all black pixels (valid ROI).
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W,H), 0, (W,H))
    
    # 3. Undistort Images In-Place
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    print(f"Undistorting {len(image_files)} images in: {IMAGES_DIR}")
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        # Undistort
        dst = cv2.undistort(img, K, dist, None, new_K)
        
        # Crop to valid ROI (to match the new matrix logic)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Overwrite
        cv2.imwrite(img_path, dst)
        
        if i % 20 == 0:
            print(f"Processed {i}/{len(image_files)}...")

    # 4. Write New PINHOLE cameras.txt
    # Update dimensions based on crop
    new_W, new_H = w, h 
    
    # Extract new params
    n_fx = new_K[0,0]
    n_fy = new_K[1,1]
    n_cx = new_K[0,2] - x # Adjust principal point for the crop offset
    n_cy = new_K[1,2] - y
    
    print(f"\nNew Dimensions: {new_W} x {new_H}")
    print(f"New Intrinsics: fx={n_fx:.2f}, fy={n_fy:.2f}, cx={n_cx:.2f}, cy={n_cy:.2f}")
    
    os.makedirs(os.path.dirname(OUTPUT_CAMERAS_TXT), exist_ok=True)
    with open(OUTPUT_CAMERAS_TXT, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        # COLMAP PINHOLE format: fx, fy, cx, cy
        f.write(f"1 PINHOLE {new_W} {new_H} {n_fx} {n_fy} {n_cx} {n_cy}\n")
        
    print(f"Saved new PINHOLE intrinsics to: {OUTPUT_CAMERAS_TXT}")

if __name__ == "__main__":
    main()