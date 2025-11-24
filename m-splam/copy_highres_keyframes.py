#!/usr/bin/env python3
"""
Copy high-resolution keyframe images based on keyframe_mapping.txt

This script reads the keyframe mapping file from MASt3R-SLAM and copies
the corresponding high-resolution images to the splatting directory.
"""

import shutil
from pathlib import Path
from natsort import natsorted


# TODO: integrate into pipeline so doesn't have to be run manually with hard coded paths
# ================= CONFIGURATION =================
KEYFRAME_MAPPING = "/home/ben/encode/data/intermediate_data/highres_Mars/mslam_logs/keyframe_mapping.txt"
HIGHRES_IMAGES_DIR = "/home/ben/encode/data/mars_johns/left"
OUTPUT_DIR = "/home/ben/encode/data/intermediate_data/highres_Mars/for_splat/images"
# =================================================


def main():
    # 1. Read keyframe mapping
    print(f"Reading keyframe mapping from: {KEYFRAME_MAPPING}")
    keyframe_data = []
    with open(KEYFRAME_MAPPING, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue
            # Skip header line
            if line.strip().startswith('m-slam_file'):
                continue
            
            # Parse: timestamp frame_id "filename with spaces.ext"
            # Split only on first two spaces to preserve filename with spaces
            parts = line.strip().split(None, 2)  # Split on whitespace, max 2 splits
            if len(parts) >= 3:
                timestamp = parts[0]
                frame_id = int(parts[1])
                # Remove quotes from filename
                original_filename = parts[2].strip('"')
                keyframe_data.append((timestamp, frame_id, original_filename))
    
    print(f"Found {len(keyframe_data)} keyframes")
    
    # 2. Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # 3. Copy images using the original filenames from mapping
    print(f"\nCopying keyframe images from: {HIGHRES_IMAGES_DIR}")
    copied = 0
    for timestamp, frame_id, original_filename in keyframe_data:
        # Construct source path from original filename
        src = Path(HIGHRES_IMAGES_DIR) / original_filename
        
        # If file doesn't exist, try with different common extensions
        if not src.exists():
            # Try common extension variations (png/PNG, jpg/JPG, jpeg/JPEG)
            base_name = src.stem
            tried_extensions = ['.JPG', '.jpg', '.PNG', '.png', '.JPEG', '.jpeg']
            for ext in tried_extensions:
                alt_src = Path(HIGHRES_IMAGES_DIR) / f"{base_name}{ext}"
                if alt_src.exists():
                    src = alt_src
                    original_filename = src.name  # Update to actual filename
                    break
            else:
                print(f"WARNING: Source file not found: {original_filename} (tried multiple extensions)")
                continue
        
        # Keep the original high-res filename
        dst = output_dir / original_filename
        
        shutil.copy2(src, dst)
        copied += 1
        print(f"  [{copied}/{len(keyframe_data)}] {original_filename}")
    
    print(f"\n✅ Done! Copied {copied} images to {output_dir}")


if __name__ == "__main__":
    main()
