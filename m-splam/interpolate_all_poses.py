#!/usr/bin/env python3
"""
interpolate_all_poses.py

Interpolate camera poses for ALL images based on M-SLAM keyframe poses.

This script:
1. Reads keyframe poses from images.txt/bin (COLMAP format)
2. Reads keyframe_mapping.txt to identify which images were keyframes
3. Lists ALL images from original_images_path directory
4. Interpolates poses for non-keyframe images using SLERP (rotation) and linear interpolation (translation)
5. Backs up original keyframe-only images.txt/bin to keyframe_poses/
6. Writes new images.txt/bin with all images and interpolated poses

The interpolated poses provide good initialization for pose optimization during splatting.

Usage:
  python interpolate_all_poses.py --dataset my_run
  python interpolate_all_poses.py --dataset my_run --original-images /path/to/images
"""
import argparse
import shutil
import struct
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

INTERMEDIATE_DATA_ROOT = Path('/home/ben/encode/data/intermediate_data')


def natural_sort_key(filename):
    """
    Natural sorting: handles embedded numbers correctly (e.g., img1, img2, ..., img10).
    Converts '2019A_GP_Left (123).png' → ['2019A_GP_Left (', 123, ').png']
    """
    import re
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', str(filename))]


def read_colmap_images_txt(images_txt):
    """
    Read COLMAP images.txt and return dict: {image_name: {pose data}}.
    
    Format:
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        (empty line - no 2D points)
    """
    images = {}
    
    with open(images_txt, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            i += 1
            continue
        
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        
        image_id = int(parts[0])
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])  # QW QX QY QZ
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])  # TX TY TZ
        camera_id = int(parts[8])
        name = " ".join(parts[9:])  # Handle filenames with spaces
        
        images[name] = {
            'image_id': image_id,
            'qvec': qvec,
            'tvec': tvec,
            'camera_id': camera_id,
            'name': name
        }
        
        i += 2  # Skip the empty second line (no 2D points for our case)
    
    return images


def read_keyframe_mapping(mapping_file):
    """
    Read keyframe_mapping.txt to get list of keyframe filenames.
    
    Format: timestamp frame_id "original_filename.ext"
    Returns: set of filenames that are keyframes
    """
    keyframes = set()
    
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments, empty lines, and header
            if not line or line.startswith('#') or 'original_filename' in line:
                continue
            
            # Parse: timestamp frame_id "filename"
            parts = line.split(None, 2)
            if len(parts) >= 3:
                filename = parts[2].strip('"')
                keyframes.add(filename)
    
    return keyframes


def get_all_images(original_images_dir):
    """
    Get all image files from directory, sorted naturally.
    Returns list of filenames (not full paths).
    """
    image_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(original_images_dir.glob(f'*{ext}'))
    
    # Sort naturally and return just filenames
    all_images_sorted = sorted([img.name for img in all_images], key=natural_sort_key)
    
    return all_images_sorted


def interpolate_poses(keyframe_poses, keyframe_names, all_image_names):
    """
    Interpolate poses for all images based on keyframe poses.
    
    Args:
        keyframe_poses: dict {filename: pose_data} for keyframes
        keyframe_names: set of keyframe filenames
        all_image_names: list of ALL image filenames in sequence order
    
    Returns:
        list of dicts with interpolated pose data for all images
    """
    # Build ordered list of keyframe indices
    keyframe_indices = []
    for idx, img_name in enumerate(all_image_names):
        if img_name in keyframe_names:
            keyframe_indices.append(idx)
    
    if len(keyframe_indices) == 0:
        raise ValueError("No keyframes found in image sequence!")
    
    print(f"Found {len(keyframe_indices)} keyframes in sequence of {len(all_image_names)} images")
    print(f"First keyframe at index {keyframe_indices[0]}: {all_image_names[keyframe_indices[0]]}")
    print(f"Last keyframe at index {keyframe_indices[-1]}: {all_image_names[keyframe_indices[-1]]}")
    
    # Get reference camera ID from first keyframe
    first_kf_name = all_image_names[keyframe_indices[0]]
    ref_camera_id = keyframe_poses[first_kf_name]['camera_id']
    
    interpolated_images = []
    
    for idx, img_name in enumerate(all_image_names):
        # Case 1: This is a keyframe - use exact pose
        if img_name in keyframe_names:
            pose = keyframe_poses[img_name]
            interpolated_images.append({
                'id': idx + 1,
                'qvec': pose['qvec'],
                'tvec': pose['tvec'],
                'camera_id': pose['camera_id'],
                'name': img_name
            })
            continue
        
        # Case 2: Interpolate between keyframes
        # Find bracketing keyframe indices
        prev_kf_idx = None
        next_kf_idx = None
        
        for kf_idx in keyframe_indices:
            if kf_idx < idx:
                prev_kf_idx = kf_idx
            if kf_idx > idx and next_kf_idx is None:
                next_kf_idx = kf_idx
                break
        
        # Handle edge cases (before first or after last keyframe)
        if prev_kf_idx is None:
            # Before first keyframe - clamp to first
            target_name = all_image_names[next_kf_idx]
            target_pose = keyframe_poses[target_name]
            interpolated_images.append({
                'id': idx + 1,
                'qvec': target_pose['qvec'],
                'tvec': target_pose['tvec'],
                'camera_id': ref_camera_id,
                'name': img_name
            })
            continue
        
        if next_kf_idx is None:
            # After last keyframe - clamp to last
            target_name = all_image_names[prev_kf_idx]
            target_pose = keyframe_poses[target_name]
            interpolated_images.append({
                'id': idx + 1,
                'qvec': target_pose['qvec'],
                'tvec': target_pose['tvec'],
                'camera_id': ref_camera_id,
                'name': img_name
            })
            continue
        
        # Interpolate between prev_kf and next_kf
        prev_name = all_image_names[prev_kf_idx]
        next_name = all_image_names[next_kf_idx]
        
        pose_a = keyframe_poses[prev_name]
        pose_b = keyframe_poses[next_name]
        
        # Calculate interpolation parameter alpha
        total_dist = next_kf_idx - prev_kf_idx
        curr_dist = idx - prev_kf_idx
        alpha = curr_dist / float(total_dist)
        
        # Linear interpolation for translation
        t_interp = (1 - alpha) * pose_a['tvec'] + alpha * pose_b['tvec']
        
        # SLERP for rotation (quaternion interpolation)
        # COLMAP quaternion format: QW, QX, QY, QZ
        # scipy expects: QX, QY, QZ, QW
        qa_scipy = [pose_a['qvec'][1], pose_a['qvec'][2], pose_a['qvec'][3], pose_a['qvec'][0]]
        qb_scipy = [pose_b['qvec'][1], pose_b['qvec'][2], pose_b['qvec'][3], pose_b['qvec'][0]]
        
        key_rots = R.from_quat([qa_scipy, qb_scipy])
        slerp = Slerp([0, 1], key_rots)
        q_interp_scipy = slerp([alpha])[0].as_quat()  # Returns [QX, QY, QZ, QW]
        
        # Convert back to COLMAP format [QW, QX, QY, QZ]
        q_interp = np.array([q_interp_scipy[3], q_interp_scipy[0], q_interp_scipy[1], q_interp_scipy[2]])
        
        interpolated_images.append({
            'id': idx + 1,
            'qvec': q_interp,
            'tvec': t_interp,
            'camera_id': ref_camera_id,
            'name': img_name
        })
    
    return interpolated_images


def write_colmap_images_txt(output_path, images):
    """
    Write COLMAP images.txt format.
    
    Format:
        # Header
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        (empty line - no 2D points)
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        
        for img in images:
            q = img['qvec']
            t = img['tvec']
            
            # Write image line
            f.write(f"{img['id']} ")
            f.write(f"{q[0]} {q[1]} {q[2]} {q[3]} ")  # QW QX QY QZ
            f.write(f"{t[0]} {t[1]} {t[2]} ")          # TX TY TZ
            f.write(f"{img['camera_id']} {img['name']}\n")
            
            # Write empty second line (no 2D points)
            f.write("\n")
    
    print(f"✓ Wrote {len(images)} images to {output_path}")


def write_colmap_images_bin(output_path, images):
    """
    Write COLMAP images.bin in binary format.
    
    Binary format:
        num_images (uint64)
        For each image:
            image_id (uint64)
            qw, qx, qy, qz (double[4])
            tx, ty, tz (double[3])
            camera_id (uint64)
            name (null-terminated string)
            num_points2D (uint64) = 0
    """
    with open(output_path, 'wb') as f:
        # Write number of images
        f.write(struct.pack('<Q', len(images)))
        
        for img in images:
            # Image ID (uint64)
            f.write(struct.pack('<Q', img['id']))
            
            # Quaternion (4 doubles): QW, QX, QY, QZ
            q = img['qvec']
            f.write(struct.pack('<dddd', q[0], q[1], q[2], q[3]))
            
            # Translation (3 doubles): TX, TY, TZ
            t = img['tvec']
            f.write(struct.pack('<ddd', t[0], t[1], t[2]))
            
            # Camera ID (uint64)
            f.write(struct.pack('<Q', img['camera_id']))
            
            # Name (null-terminated string)
            name_bytes = img['name'].encode('utf-8') + b'\x00'
            f.write(name_bytes)
            
            # Number of 2D points (uint64) = 0
            f.write(struct.pack('<Q', 0))
    
    print(f"✓ Wrote {len(images)} images to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate camera poses for all images based on keyframe poses"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (run directory in intermediate_data)'
    )
    parser.add_argument(
        '--original-images',
        type=str,
        default=None,
        help='Path to original images directory (default: reads from pipeline config if available)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    run_dir = INTERMEDIATE_DATA_ROOT / args.dataset
    sparse_dir = run_dir / 'for_splat' / 'sparse' / '0'
    mslam_logs = run_dir / 'mslam_logs'
    
    images_txt = sparse_dir / 'images.txt'
    images_bin = sparse_dir / 'images.bin'
    mapping_file = mslam_logs / 'keyframe_mapping.txt'
    
    # Validate inputs
    if not images_txt.exists():
        raise FileNotFoundError(f"Keyframe poses not found: {images_txt}")
    if not mapping_file.exists():
        raise FileNotFoundError(f"Keyframe mapping not found: {mapping_file}")
    
    # Determine original images path
    if args.original_images:
        original_images_dir = Path(args.original_images)
    else:
        # Try to read from pipeline config
        config_file = run_dir / 'pipeline_config.yaml'
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            original_images_dir = Path(config['paths']['original_images_path'])
        else:
            raise ValueError("Must provide --original-images or have pipeline_config.yaml in run directory")
    
    if not original_images_dir.exists():
        raise FileNotFoundError(f"Original images directory not found: {original_images_dir}")
    
    print(f"\n{'='*70}")
    print(f"Interpolating Poses for All Images")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Original images: {original_images_dir}")
    print(f"Sparse directory: {sparse_dir}")
    print()
    
    # 1. Read keyframe poses
    print("[1/5] Reading keyframe poses...")
    keyframe_poses = read_colmap_images_txt(images_txt)
    print(f"  ✓ Loaded {len(keyframe_poses)} keyframe poses")
    
    # 2. Read keyframe mapping
    print("\n[2/5] Reading keyframe mapping...")
    keyframe_names = read_keyframe_mapping(mapping_file)
    print(f"  ✓ Identified {len(keyframe_names)} keyframe filenames")
    
    # 3. Get all images
    print("\n[3/5] Scanning original images...")
    all_images = get_all_images(original_images_dir)
    print(f"  ✓ Found {len(all_images)} total images")
    
    # 4. Interpolate poses
    print("\n[4/5] Interpolating poses...")
    interpolated = interpolate_poses(keyframe_poses, keyframe_names, all_images)
    print(f"  ✓ Generated {len(interpolated)} poses ({len(keyframe_names)} exact, {len(interpolated) - len(keyframe_names)} interpolated)")
    
    # 5. Backup and write new files
    print("\n[5/5] Writing output files...")
    
    # Create backup directory
    backup_dir = sparse_dir / 'keyframe_poses'
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup original files
    shutil.copy2(images_txt, backup_dir / 'images.txt')
    shutil.copy2(images_bin, backup_dir / 'images.bin')
    print(f"  ✓ Backed up keyframe-only poses to: {backup_dir}")
    
    # Write new files with all images
    write_colmap_images_txt(images_txt, interpolated)
    write_colmap_images_bin(images_bin, interpolated)
    
    print(f"\n{'='*70}")
    print(f"✅ Successfully interpolated poses for {len(interpolated)} images!")
    print(f"{'='*70}")
    print(f"Keyframes: {len(keyframe_names)}")
    print(f"Interpolated: {len(interpolated) - len(keyframe_names)}")
    print(f"Total: {len(interpolated)}")
    print(f"\nOriginal keyframe-only files backed up to:")
    print(f"  {backup_dir}")
    print()


if __name__ == "__main__":
    main()
