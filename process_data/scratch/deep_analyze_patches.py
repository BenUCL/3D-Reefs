#!/usr/bin/env python3
"""
Deep analysis of p5 and p6 to find what makes them different from successful patches.
"""

import sys
from pathlib import Path
from PIL import Image
import pycolmap
import numpy as np
from collections import Counter
import json

patches_dir = Path("/home/ben/encode/data/intermediate_data/colmap5/sparse_patches")
images_dir = Path("/home/ben/encode/data/intermediate_data/colmap5/images")

# Patches that consistently fail
FAILED = ["p5", "p6"]
# Patches that consistently succeed
SUCCESS = ["p0", "p1", "p2", "p3", "p4", "p7"]

print("="*70)
print("DEEP ANALYSIS: What makes p5 and p6 different?")
print("="*70)

def analyze_patch_deeply(patch_name):
    """Comprehensive analysis of a patch."""
    sparse_dir = patches_dir / patch_name / "sparse" / "0"
    
    if not sparse_dir.exists():
        return None
    
    recon = pycolmap.Reconstruction(str(sparse_dir))
    
    result = {
        "name": patch_name,
        "num_images": len(recon.images),
        "num_cameras": len(recon.cameras),
        "num_points": len(recon.points3D),
    }
    
    # Camera analysis
    for cam_id, camera in recon.cameras.items():
        result[f"cam{cam_id}_dims"] = (camera.width, camera.height)
        result[f"cam{cam_id}_model"] = camera.model.name
        result[f"cam{cam_id}_params"] = list(camera.params)
    
    # Image analysis
    image_names = [img.name for img in recon.images.values()]
    left_count = sum(1 for n in image_names if n.startswith("left/"))
    right_count = sum(1 for n in image_names if n.startswith("right/"))
    result["left_images"] = left_count
    result["right_images"] = right_count
    result["left_right_ratio"] = left_count / right_count if right_count > 0 else float('inf')
    
    # Camera ID distribution
    cam_ids = [img.camera_id for img in recon.images.values()]
    result["cam_id_distribution"] = dict(Counter(cam_ids))
    
    # Check for any images with unusual properties
    unusual_images = []
    for img_id, image in recon.images.items():
        # Check if image file exists
        img_path = images_dir / image.name
        if not img_path.exists():
            unusual_images.append({"name": image.name, "issue": "missing"})
            continue
        
        # Check actual dimensions
        with Image.open(img_path) as img:
            actual_dims = img.size
            camera = recon.cameras[image.camera_id]
            expected = (camera.width, camera.height)
            if actual_dims != expected:
                unusual_images.append({
                    "name": image.name,
                    "issue": f"dim mismatch: {actual_dims} vs {expected}"
                })
    
    result["unusual_images"] = unusual_images
    result["unusual_count"] = len(unusual_images)
    
    # Point cloud statistics
    if recon.points3D:
        points = np.array([p.xyz for p in recon.points3D.values()])
        result["points_mean"] = points.mean(axis=0).tolist()
        result["points_std"] = points.std(axis=0).tolist()
        result["points_min"] = points.min(axis=0).tolist()
        result["points_max"] = points.max(axis=0).tolist()
    
    # Check patch metadata
    metadata_file = patches_dir / patch_name / "patch_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        result["bbox"] = {
            "min_x": metadata.get("min_x"),
            "max_x": metadata.get("max_x"),
            "min_y": metadata.get("min_y"),
            "max_y": metadata.get("max_y"),
        }
        result["bbox_size"] = {
            "x": metadata.get("max_x", 0) - metadata.get("min_x", 0),
            "y": metadata.get("max_y", 0) - metadata.get("min_y", 0),
        }
    
    return result


print("\n" + "="*70)
print("ANALYZING FAILED PATCHES")
print("="*70)

failed_data = []
for patch in FAILED:
    print(f"\n--- {patch} ---")
    data = analyze_patch_deeply(patch)
    if data:
        failed_data.append(data)
        print(f"  Images: {data['num_images']} (left: {data['left_images']}, right: {data['right_images']})")
        print(f"  Points: {data['num_points']}")
        print(f"  Camera distribution: {data['cam_id_distribution']}")
        print(f"  Unusual images: {data['unusual_count']}")
        if data['unusual_images']:
            for u in data['unusual_images'][:5]:
                print(f"    - {u['name']}: {u['issue']}")
        if 'bbox_size' in data:
            print(f"  Bbox size: {data['bbox_size']['x']:.2f} x {data['bbox_size']['y']:.2f}")

print("\n" + "="*70)
print("ANALYZING SUCCESSFUL PATCHES")
print("="*70)

success_data = []
for patch in SUCCESS:
    print(f"\n--- {patch} ---")
    data = analyze_patch_deeply(patch)
    if data:
        success_data.append(data)
        print(f"  Images: {data['num_images']} (left: {data['left_images']}, right: {data['right_images']})")
        print(f"  Points: {data['num_points']}")
        print(f"  Camera distribution: {data['cam_id_distribution']}")
        print(f"  Unusual images: {data['unusual_count']}")
        if 'bbox_size' in data:
            print(f"  Bbox size: {data['bbox_size']['x']:.2f} x {data['bbox_size']['y']:.2f}")


print("\n" + "="*70)
print("COMPARISON: What's different about failed patches?")
print("="*70)

# Compare camera params
print("\nCamera parameters comparison:")
for data in failed_data + success_data:
    status = "FAIL" if data['name'] in FAILED else "OK"
    print(f"\n{data['name']} [{status}]:")
    for key in ['cam1_params', 'cam2_params']:
        if key in data:
            params = data[key]
            print(f"  {key}: fx={params[0]:.2f}, fy={params[1]:.2f}, cx={params[2]:.2f}, cy={params[3]:.2f}")

# Check for patterns
print("\n" + "="*70)
print("LOOKING FOR PATTERNS")
print("="*70)

failed_points = [d['num_points'] for d in failed_data]
success_points = [d['num_points'] for d in success_data]
failed_images = [d['num_images'] for d in failed_data]
success_images = [d['num_images'] for d in success_data]

print(f"\nFailed patches point counts:  {failed_points}")
print(f"Success patches point counts: {success_points}")
print(f"Failed patches image counts:  {failed_images}")
print(f"Success patches image counts: {success_images}")

# Check if failed patches have specific image ranges
print("\n\nImage name ranges in failed patches:")
for data in failed_data:
    sparse_dir = patches_dir / data['name'] / "sparse" / "0"
    recon = pycolmap.Reconstruction(str(sparse_dir))
    image_names = sorted([img.name for img in recon.images.values()])
    print(f"\n{data['name']}:")
    print(f"  First 3: {image_names[:3]}")
    print(f"  Last 3:  {image_names[-3:]}")
