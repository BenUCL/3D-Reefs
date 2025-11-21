#!/usr/bin/env python3
"""
Copy high-resolution keyframe images based on keyframe_mapping.txt

This script reads the keyframe mapping file from MASt3R-SLAM and copies
the corresponding high-resolution images to the splatting directory.
"""

import shutil
from pathlib import Path
from natsort import natsorted

# ================= CONFIGURATION =================
KEYFRAME_MAPPING = "/home/ben/encode/data/intermediate_data/highres_m-slam_MarsJohnS/mslam_logs/keyframe_mapping.txt"
HIGHRES_IMAGES_DIR = "/home/ben/encode/data/mars_johns/left"
OUTPUT_DIR = "/home/ben/encode/data/intermediate_data/highres_m-slam_MarsJohnS/for_splat/images"
EXTENSION = ".JPG"  # Extension of high-res images
# =================================================


def get_highres_filenames(images_dir, extension):
    """Get sorted list of high-res image filenames using natural sort."""
    images = list(Path(images_dir).glob(f"*{extension}"))
    return natsorted(images)


def main():
    # 1. Read keyframe mapping
    print(f"Reading keyframe mapping from: {KEYFRAME_MAPPING}")
    keyframe_data = []
    with open(KEYFRAME_MAPPING, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamp = parts[0]
                frame_id = int(parts[1])
                keyframe_data.append((timestamp, frame_id))
    
    print(f"Found {len(keyframe_data)} keyframes")
    
    # 2. Get sorted list of high-res images
    print(f"\nScanning high-res images in: {HIGHRES_IMAGES_DIR}")
    highres_images = get_highres_filenames(HIGHRES_IMAGES_DIR, EXTENSION)
    print(f"Found {len(highres_images)} high-res images")
    
    if not highres_images:
        print(f"ERROR: No images with extension {EXTENSION} found!")
        return
    
    # 3. Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # 4. Copy images
    print(f"\nCopying keyframe images...")
    copied = 0
    for timestamp, frame_id in keyframe_data:
        if frame_id >= len(highres_images):
            print(f"WARNING: frame_id {frame_id} out of range (only {len(highres_images)} images)")
            continue
        
        src = highres_images[frame_id]
        # Keep the original high-res filename
        dst = output_dir / src.name
        
        shutil.copy2(src, dst)
        copied += 1
        print(f"  [{copied}/{len(keyframe_data)}] {src.name} → {dst.name}")
    
    print(f"\n✅ Done! Copied {copied} images to {output_dir}")


if __name__ == "__main__":
    main()
