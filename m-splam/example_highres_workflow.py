#!/usr/bin/env python3
"""
Example: How to use the high-res images for Gaussian Splatting

This example shows the workflow for using full-resolution images with poses
estimated from downsampled images by MASt3R-SLAM.
"""

# ==============================================================================
# STEP 1: Run the full pipeline with generate_original_colmap enabled
# ==============================================================================

# Edit your config file (e.g., slam_splat_config.yaml):
"""
generate_highres_colmap:
  enabled: true
paths:
  original_images_path: "/home/ben/encode/data/mars_johns/left"
  extension: ".JPG"
"""

# Run pipeline:
# python run_pipeline.py --config slam_splat_config.yaml

# This will create:
# - /intermediate_data/mars_johns_1/for_splat/sparse/0/images.txt (downsampled keyframes)
# - /intermediate_data/mars_johns_1/for_splat/sparse/0/images_highres.txt (high-res names)
# - /intermediate_data/mars_johns_1/mslam_logs/keyframe_mapping.txt (timestamp→frame_id)
# - /intermediate_data/mars_johns_1/mslam_logs/keyframe_mapping_full.txt (complete mapping)


# ==============================================================================
# STEP 2: Copy/convert original high-res images to splatting directory
# ==============================================================================

import shutil
from pathlib import Path

# Read the full mapping file
mapping_file = Path("/home/ben/encode/data/intermediate_data/mars_johns_1/mslam_logs/keyframe_mapping_full.txt")
original_images_dir = Path("/home/ben/encode/data/mars_johns/left")
target_dir = Path("/home/ben/encode/data/intermediate_data/mars_johns_1/for_splat/images_highres")
target_dir.mkdir(exist_ok=True, parents=True)

# Parse mapping and copy images
with open(mapping_file, 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        
        parts = line.strip().split()
        if len(parts) >= 4:
            timestamp, frame_id, keyframe_name, original_name = parts[0], parts[1], parts[2], parts[3]
            
            # Source: original high-res image
            src = original_images_dir / original_name
            
            # Target: use original filename but convert to PNG if needed
            dst = target_dir / original_name.replace('.JPG', '.png').replace('.jpg', '.png')
            
            # Option A: Copy and convert JPG→PNG
            from PIL import Image
            img = Image.open(src)
            img.save(dst)
            
            # Option B: Just copy if already PNG
            # shutil.copy2(src, dst)
            
            print(f"Processed: {original_name} → {dst.name}")


# ==============================================================================
# STEP 3: Update cameras.txt to use high-res dimensions
# ==============================================================================

# The cameras.txt currently has dimensions from downsampled images (e.g., 512xH)
# You need to create cameras_highres.txt with full-res dimensions

from PIL import Image

# Read one high-res image to get dimensions
sample_image = next(target_dir.glob("*.png"))
img = Image.open(sample_image)
width_highres, height_highres = img.size

# Read original cameras.txt
cameras_txt = Path("/home/ben/encode/data/intermediate_data/mars_johns_1/for_splat/sparse/0/cameras.txt")
cameras_highres_txt = cameras_txt.parent / "cameras_highres.txt"

with open(cameras_txt, 'r') as f_in, open(cameras_highres_txt, 'w') as f_out:
    for line in f_in:
        if line.startswith('#'):
            f_out.write(line)
        else:
            # Parse camera line: camera_id model width height fx fy cx cy
            parts = line.split()
            camera_id, model = parts[0], parts[1]
            # Keep other params but scale for high-res
            width_old, height_old = int(parts[2]), int(parts[3])
            fx, fy, cx, cy = map(float, parts[4:8])
            
            # Scale intrinsics to high-res
            scale_x = width_highres / width_old
            scale_y = height_highres / height_old
            fx_new = fx * scale_x
            fy_new = fy * scale_y
            cx_new = cx * scale_x
            cy_new = cy * scale_y
            
            f_out.write(f"{camera_id} {model} {width_highres} {height_highres} ")
            f_out.write(f"{fx_new} {fy_new} {cx_new} {cy_new}\n")

print(f"Created high-res cameras: {cameras_highres_txt}")


# ==============================================================================
# STEP 4: Run Gaussian Splatting with high-res images
# ==============================================================================

# Now you have:
# - images_highres/ with full-resolution images
# - images_highres.txt/bin with correct filenames
# - cameras_highres.txt with correct dimensions and intrinsics

# Manually run LichtFeld-Studio:
"""
cd /home/ben/encode/code/lichtfeld-studio/build

./LichtFeld-Studio \\
    --dataset /home/ben/encode/data/intermediate_data/mars_johns_1/for_splat_highres \\
    --output /home/ben/encode/data/intermediate_data/mars_johns_1/splats_highres \\
    --iterations 25000 \\
    --max-cap 1000000 \\
    --headless

Where for_splat_highres/ contains:
  - images/ → symlink to images_highres/
  - sparse/0/images.txt → copy of images_original.txt
  - sparse/0/images.bin → copy of images_original.bin
  - sparse/0/cameras.txt → copy of cameras_highres.txt
  - sparse/0/points3D.bin → same as before
"""

# Or create a wrapper directory structure:
highres_splat_dir = Path("/home/ben/encode/data/intermediate_data/mars_johns_1/for_splat_highres")
(highres_splat_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)

# Symlink images
(highres_splat_dir / "images").symlink_to(target_dir)

# Copy COLMAP files
shutil.copy2(
    "/home/ben/encode/data/intermediate_data/mars_johns_1/for_splat/sparse/0/images_highres.txt",
    highres_splat_dir / "sparse" / "0" / "images.txt"
)
shutil.copy2(
    "/home/ben/encode/data/intermediate_data/mars_johns_1/for_splat/sparse/0/images_highres.bin",
    highres_splat_dir / "sparse" / "0" / "images.bin"
)
shutil.copy2(
    cameras_highres_txt,
    highres_splat_dir / "sparse" / "0" / "cameras.txt"
)
shutil.copy2(
    "/home/ben/encode/data/intermediate_data/mars_johns_1/for_splat/sparse/0/points3D.bin",
    highres_splat_dir / "sparse" / "0" / "points3D.bin"
)

print(f"High-res splatting directory ready: {highres_splat_dir}")
print("Now run LichtFeld-Studio with this directory!")
