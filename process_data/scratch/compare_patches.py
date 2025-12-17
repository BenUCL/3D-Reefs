#!/usr/bin/env python3
"""
Compare successful vs failed patches to find patterns.
"""

from pathlib import Path
import pycolmap

patches_dir = Path("/home/ben/encode/data/intermediate_data/colmap5/sparse_patches")

# Known from log
successful = ["p0", "p1", "p2", "p3", "p4", "p7", "p9", "p11", "p13"]
failed = ["p5", "p6", "p8", "p10", "p12"]

print("="*70)
print("Comparing Successful vs Failed Patches")
print("="*70)

def analyze_patch(patch_name):
    sparse_dir = patches_dir / patch_name / "sparse" / "0"
    if not sparse_dir.exists():
        return None
    
    recon = pycolmap.Reconstruction(str(sparse_dir))
    
    return {
        "name": patch_name,
        "num_images": len(recon.images),
        "num_cameras": len(recon.cameras),
        "num_points": len(recon.points3D),
        "cameras": {cam_id: (cam.width, cam.height) for cam_id, cam in recon.cameras.items()}
    }

print("\nSUCCESSFUL PATCHES:")
print("-" * 70)
success_data = []
for patch in successful:
    data = analyze_patch(patch)
    if data:
        success_data.append(data)
        print(f"{data['name']}: {data['num_images']} images, {data['num_points']} points")

print("\nFAILED PATCHES:")
print("-" * 70)
failed_data = []
for patch in failed:
    data = analyze_patch(patch)
    if data:
        failed_data.append(data)
        print(f"{data['name']}: {data['num_images']} images, {data['num_points']} points")

# Statistics
success_img_counts = [d['num_images'] for d in success_data]
failed_img_counts = [d['num_images'] for d in failed_data]
success_pt_counts = [d['num_points'] for d in success_data]
failed_pt_counts = [d['num_points'] for d in failed_data]

print("\nSTATISTICS:")
print("-" * 70)
print(f"Successful patches:")
print(f"  Images: min={min(success_img_counts)}, max={max(success_img_counts)}, avg={sum(success_img_counts)/len(success_img_counts):.1f}")
print(f"  Points: min={min(success_pt_counts)}, max={max(success_pt_counts)}, avg={sum(success_pt_counts)/len(success_pt_counts):.1f}")

print(f"\nFailed patches:")
print(f"  Images: min={min(failed_img_counts)}, max={max(failed_img_counts)}, avg={sum(failed_img_counts)/len(failed_img_counts):.1f}")
print(f"  Points: min={min(failed_pt_counts)}, max={max(failed_pt_counts)}, avg={sum(failed_pt_counts)/len(failed_pt_counts):.1f}")

# Check camera dimensions in failed patches
print("\nCAMERA DIMENSIONS IN FAILED PATCHES:")
print("-" * 70)
for data in failed_data:
    print(f"\n{data['name']}:")
    for cam_id, dims in data['cameras'].items():
        print(f"  Camera {cam_id}: {dims[0]}×{dims[1]}")
