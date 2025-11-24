#!/usr/bin/env python3
"""
Copy high-resolution keyframe images based on keyframe_mapping.txt

This script reads the keyframe mapping file from MASt3R-SLAM and copies
the corresponding high-resolution images to the splatting directory.

Usage:
  python copy_highres_keyframes.py --dataset highres_Mars \
      --highres-images /path/to/highres/images \
      --output-dir /path/to/output
"""

import argparse
import shutil
from pathlib import Path

INTERMEDIATE_DATA_ROOT = Path('/home/ben/encode/data/intermediate_data')


def main():
    parser = argparse.ArgumentParser(
        description="Copy high-res keyframe images based on keyframe_mapping.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (run directory in intermediate_data)'
    )
    parser.add_argument(
        '--highres-images',
        type=str,
        required=True,
        help='Path to directory containing high-resolution images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: {dataset}/for_splat/images)'
    )
    parser.add_argument(
        '--mslam-logs-dir',
        type=str,
        default=None,
        help='Path to MASt3R-SLAM logs directory (default: {dataset}/mslam_logs)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    run_dir = INTERMEDIATE_DATA_ROOT / args.dataset
    
    if args.mslam_logs_dir:
        mslam_logs = Path(args.mslam_logs_dir)
    else:
        mslam_logs = run_dir / 'mslam_logs'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_dir / 'for_splat' / 'images'
    
    keyframe_mapping = mslam_logs / 'keyframe_mapping.txt'
    highres_images_dir = Path(args.highres_images)
    
    # Validate inputs
    if not keyframe_mapping.exists():
        raise FileNotFoundError(f"Keyframe mapping not found: {keyframe_mapping}")
    
    if not highres_images_dir.exists():
        raise FileNotFoundError(f"High-res images directory not found: {highres_images_dir}")
    
    print(f"\n{'='*70}")
    print(f"Copying high-resolution keyframe images")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"High-res images: {highres_images_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # 1. Read keyframe mapping
    print("[1/3] Reading keyframe mapping...")
    keyframe_data = []
    with open(keyframe_mapping, 'r') as f:
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
    
    print(f"  ✓ Found {len(keyframe_data)} keyframes")
    
    # 2. Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/3] Output directory: {output_dir}")
    
    # 3. Copy images using the original filenames from mapping
    print(f"\n[3/3] Copying keyframe images...")
    copied = 0
    for timestamp, frame_id, original_filename in keyframe_data:
        # Construct source path from original filename
        src = highres_images_dir / original_filename
        
        # If file doesn't exist, try with different common extensions
        if not src.exists():
            # Try common extension variations (png/PNG, jpg/JPG, jpeg/JPEG)
            base_name = src.stem
            tried_extensions = ['.JPG', '.jpg', '.PNG', '.png', '.JPEG', '.jpeg']
            for ext in tried_extensions:
                alt_src = highres_images_dir / f"{base_name}{ext}"
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
        if copied <= 5 or copied == len(keyframe_data):
            print(f"  [{copied}/{len(keyframe_data)}] {original_filename}")
        elif copied == 6:
            print(f"  ... copying remaining images ...")
    
    print(f"\n{'='*70}")
    print(f"✅ Successfully copied {copied}/{len(keyframe_data)} images!")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print()


if __name__ == "__main__":
    main()
