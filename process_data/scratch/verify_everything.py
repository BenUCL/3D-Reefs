#!/usr/bin/env python3
"""
Complete verification of image dimensions and COLMAP intrinsics.
Check EVERY step in the pipeline.
"""

import sys
from pathlib import Path
from PIL import Image
import pycolmap
from collections import Counter
import yaml

print("="*70)
print("COMPLETE SYSTEM VERIFICATION")
print("="*70)

# Load config
config_path = Path("/home/ben/encode/code/3D-Reefs/process_data/splat_config.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)

images_dir = Path(config['paths']['images_dir'])
sparse_dir = Path(config['paths']['sparse_dir'])
patches_dir = Path(config['paths']['patches_dir'])

print(f"\nConfig paths:")
print(f"  images_dir:  {images_dir}")
print(f"  sparse_dir:  {sparse_dir}")
print(f"  patches_dir: {patches_dir}")

# STEP 1: Check actual images on disk
print("\n" + "="*70)
print("STEP 1: Verify actual image dimensions")
print("="*70)

left_dir = images_dir / "left"
right_dir = images_dir / "right"

print(f"\nChecking {left_dir}...")
left_dims = []
for img_path in sorted(left_dir.glob("*.png"))[:20]:
    with Image.open(img_path) as img:
        left_dims.append(img.size)

print(f"Sampled {len(left_dims)} left images:")
left_counter = Counter(left_dims)
for dims, count in left_counter.most_common():
    print(f"  {dims[0]}×{dims[1]}: {count} images")

print(f"\nChecking {right_dir}...")
right_dims = []
for img_path in sorted(right_dir.glob("*.png"))[:20]:
    with Image.open(img_path) as img:
        right_dims.append(img.size)

print(f"Sampled {len(right_dims)} right images:")
right_counter = Counter(right_dims)
for dims, count in right_counter.most_common():
    print(f"  {dims[0]}×{dims[1]}: {count} images")

# Determine target dimensions
if len(left_counter) == 1 and len(right_counter) == 1:
    left_dim = list(left_counter.keys())[0]
    right_dim = list(right_counter.keys())[0]
    if left_dim == right_dim:
        target_dims = left_dim
        print(f"\n✓ All images are: {target_dims[0]}×{target_dims[1]}")
    else:
        print(f"\n❌ ERROR: Left and right have different dimensions!")
        print(f"   Left: {left_dim}, Right: {right_dim}")
        sys.exit(1)
else:
    print(f"\n❌ ERROR: Multiple dimensions found!")
    sys.exit(1)

# STEP 2: Check original COLMAP reconstruction
print("\n" + "="*70)
print("STEP 2: Verify original COLMAP reconstruction")
print("="*70)

print(f"\nLoading {sparse_dir}...")
reconstruction = pycolmap.Reconstruction(str(sparse_dir))

print(f"\nCameras in reconstruction: {len(reconstruction.cameras)}")
cameras_match = True
for cam_id, camera in reconstruction.cameras.items():
    expected = (camera.width, camera.height)
    match = "✓" if expected == target_dims else "❌"
    print(f"  {match} Camera {cam_id}: {camera.width}×{camera.height}")
    if expected != target_dims:
        cameras_match = False
        print(f"      Expected: {target_dims[0]}×{target_dims[1]}")
        print(f"      Diff: width={expected[0]-target_dims[0]}, height={expected[1]-target_dims[1]}")

if cameras_match:
    print("\n✓ Original COLMAP cameras match image dimensions")
else:
    print("\n❌ Original COLMAP cameras DO NOT match image dimensions")

# STEP 3: Check patch intrinsics
print("\n" + "="*70)
print("STEP 3: Verify patch camera intrinsics")
print("="*70)

patch_dirs = sorted([d for d in patches_dir.glob("p*") if d.is_dir()], 
                   key=lambda x: int(x.name[1:]))

print(f"\nFound {len(patch_dirs)} patches to check")

patches_match = True
for patch_dir in patch_dirs[:3]:  # Check first 3 patches
    patch_sparse = patch_dir / "sparse" / "0"
    if not patch_sparse.exists():
        print(f"\n{patch_dir.name}: No sparse/0 directory")
        continue
    
    patch_recon = pycolmap.Reconstruction(str(patch_sparse))
    print(f"\n{patch_dir.name}:")
    
    patch_match = True
    for cam_id, camera in patch_recon.cameras.items():
        expected = (camera.width, camera.height)
        match = "✓" if expected == target_dims else "❌"
        print(f"  {match} Camera {cam_id}: {camera.width}×{camera.height}")
        if expected != target_dims:
            patch_match = False
            patches_match = False
            print(f"      Expected: {target_dims[0]}×{target_dims[1]}")

# STEP 4: Detailed check of specific images referenced in patches
print("\n" + "="*70)
print("STEP 4: Verify images referenced in patches")
print("="*70)

# Check p5 specifically since it failed
p5_dir = patches_dir / "p5" / "sparse" / "0"
if p5_dir.exists():
    print(f"\nChecking patch p5 (failed during training)...")
    p5_recon = pycolmap.Reconstruction(str(p5_dir))
    
    # Check first 5 images
    print(f"  Total images in p5: {len(p5_recon.images)}")
    print(f"  Checking first 5 images:")
    
    for i, (img_id, image) in enumerate(p5_recon.images.items()):
        if i >= 5:
            break
        
        img_path = images_dir / image.name
        camera = p5_recon.cameras[image.camera_id]
        expected = (camera.width, camera.height)
        
        if img_path.exists():
            with Image.open(img_path) as img:
                actual = img.size
                match = "✓" if actual == expected else "❌"
                print(f"    {match} {image.name}")
                print(f"       Camera {image.camera_id} expects: {expected}")
                print(f"       Image file actual: {actual}")
        else:
            print(f"    ❌ {image.name} - FILE NOT FOUND")

# SUMMARY
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n1. Image dimensions: {target_dims[0]}×{target_dims[1]}")
print(f"2. Original COLMAP matches: {'✓ YES' if cameras_match else '❌ NO'}")
print(f"3. Patch intrinsics match: {'✓ YES' if patches_match else '❌ NO'}")

if cameras_match and patches_match:
    print("\n✓ ALL CHECKS PASSED - Dimensions should be correct")
else:
    print("\n❌ CHECKS FAILED - Dimension mismatch exists")
    print("\nRECOMMENDATION:")
    if not cameras_match:
        print("  1. Original COLMAP needs fixing - run fix_original_colmap_intrinsics.py")
    if not patches_match:
        print("  2. Patches need regenerating - delete patches and re-run patch_colmap_data.py")
