#!/usr/bin/env python3
"""Check the original COLMAP reconstruction before patching."""

import pycolmap
from pathlib import Path
from PIL import Image
from collections import Counter

sparse_dir = Path("/home/ben/encode/data/intermediate_data/colmap5/sparse")
images_dir = Path("/home/ben/encode/data/intermediate_data/colmap5/images")

print("="*70)
print("Original COLMAP Reconstruction Analysis")
print("="*70)

reconstruction = pycolmap.Reconstruction(str(sparse_dir))

print(f"\nCameras: {len(reconstruction.cameras)}")
for cam_id, camera in reconstruction.cameras.items():
    print(f"  Camera {cam_id}: {camera.model.name}, {camera.width}x{camera.height}")
    print(f"    Params: {camera.params}")

print(f"\nTotal images: {len(reconstruction.images)}")

# Check actual image dimensions
left_dims = []
right_dims = []

print("\nChecking actual image dimensions...")
for img_id, image in list(reconstruction.images.items())[:10]:  # Sample first 10
    img_path = images_dir / image.name
    if img_path.exists():
        with Image.open(img_path) as img:
            actual_dims = img.size
            camera = reconstruction.cameras[image.camera_id]
            expected = (camera.width, camera.height)
            
            folder = image.name.split('/')[0]
            if folder == 'left':
                left_dims.append(actual_dims)
            else:
                right_dims.append(actual_dims)
            
            match = "✓" if actual_dims == expected else "❌"
            print(f"  {match} {image.name}: expected {expected}, actual {actual_dims}")

# Sample more images to get full picture
print("\nSampling 100 images from each camera...")
left_sample = []
right_sample = []
count = 0
for img_id, image in reconstruction.images.items():
    img_path = images_dir / image.name
    if not img_path.exists():
        continue
    
    folder = image.name.split('/')[0]
    if folder == 'left' and len(left_sample) < 100:
        with Image.open(img_path) as img:
            left_sample.append(img.size)
    elif folder == 'right' and len(right_sample) < 100:
        with Image.open(img_path) as img:
            right_sample.append(img.size)
    
    if len(left_sample) >= 100 and len(right_sample) >= 100:
        break

print(f"\nLeft camera dimensions (n={len(left_sample)}):")
left_counter = Counter(left_sample)
for dims, count in left_counter.most_common():
    print(f"  {dims}: {count} images")

print(f"\nRight camera dimensions (n={len(right_sample)}):")
right_counter = Counter(right_sample)
for dims, count in right_counter.most_common():
    print(f"  {dims}: {count} images")
