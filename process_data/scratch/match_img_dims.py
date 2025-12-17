#!/usr/bin/env python3
"""
match_img_dims.py

Resize all images to consistent dimensions and update COLMAP camera intrinsics accordingly.
Handles multi-camera setups where different cameras have different image dimensions.

This fixes tensor dimension mismatch errors in LFS during splatting.

#TODO: in future, just do resizing before running colmap to prevent this messy issue.
"""

import shutil
from pathlib import Path
from PIL import Image
import pycolmap
import numpy as np


# CONFIGURATION
IMAGES_DIR = Path("/home/ben/encode/data/intermediate_data/colmap5/images")
PATCHES_DIR = Path("/home/ben/encode/data/intermediate_data/colmap5/sparse_patches")


# IMPLEMENTATION
def get_all_image_dimensions(images_dir: Path) -> dict[str, tuple[int, int]]:
    """
    Scan all images and return {subfolder: (width, height)} for each camera.
    """
    dims = {}
    for subfolder in sorted(images_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        
        # Find first image in this subfolder
        for img_path in subfolder.glob("*.[pP][nN][gG]"):
            with Image.open(img_path) as img:
                dims[subfolder.name] = img.size  # (width, height)
            break
        
        # Also try jpg
        if subfolder.name not in dims:
            for img_path in subfolder.glob("*.[jJ][pP][gG]"):
                with Image.open(img_path) as img:
                    dims[subfolder.name] = img.size
                break
    
    return dims


def compute_target_dimensions(dims: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """
    Compute target dimensions as minimum width and minimum height across all cameras.
    """
    widths = [w for w, h in dims.values()]
    heights = [h for w, h in dims.values()]
    return (min(widths), min(heights))


def resize_images(images_dir: Path, backup_dir: Path, target_size: tuple[int, int]):
    """
    Resize all images to target_size. Backs up original directory first.
    """
    if backup_dir.exists():
        print(f"⚠️  Backup already exists: {backup_dir}")
        response = input("Delete and continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            exit(1)
        shutil.rmtree(backup_dir)
    
    # Backup original
    print(f"Backing up: {images_dir} → {backup_dir}")
    shutil.copytree(images_dir, backup_dir)
    
    # Remove original and recreate
    shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)
    
    target_w, target_h = target_size
    
    # Resize all images
    for subfolder in sorted(backup_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        
        output_folder = images_dir / subfolder.name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        img_files = list(subfolder.glob("*.[pP][nN][gG]")) + list(subfolder.glob("*.[jJ][pP][gG]"))
        print(f"\nResizing {len(img_files)} images in {subfolder.name}/ to {target_w}×{target_h}...")
        
        for i, img_path in enumerate(img_files, 1):
            with Image.open(img_path) as img:
                orig_size = img.size
                resized = img.resize((target_w, target_h), Image.LANCZOS)
                
                output_path = output_folder / img_path.name
                resized.save(output_path)
            
            if i % 100 == 0:
                print(f"  Processed {i}/{len(img_files)}...")
        
        print(f"  ✓ Completed {subfolder.name}/")


def update_cameras_for_patch(patch_dir: Path, scale_factors: dict[int, tuple[float, float]]):
    """
    Update cameras.bin and cameras.txt with rescaled intrinsics.
    
    Args:
        patch_dir: Path to sparse/0 directory containing cameras.bin
        scale_factors: {camera_id: (scale_x, scale_y)}
    """
    cameras_bin = patch_dir / "cameras.bin"
    cameras_txt = patch_dir / "cameras.txt"
    
    if not cameras_bin.exists():
        print(f"  ⚠️  No cameras.bin found in {patch_dir}")
        return
    
    # Read reconstruction
    reconstruction = pycolmap.Reconstruction(str(patch_dir))
    
    print(f"\n  Updating cameras in {patch_dir.parent.parent.name}:")
    
    # Update each camera
    for cam_id, camera in reconstruction.cameras.items():
        if cam_id not in scale_factors:
            print(f"    ⚠️  Camera {cam_id} not in scale_factors, skipping")
            continue
        
        scale_x, scale_y = scale_factors[cam_id]
        
        # Store original
        orig_width = camera.width
        orig_height = camera.height
        orig_params = camera.params.copy()
        
        # Update dimensions
        new_width = int(orig_width * scale_x)
        new_height = int(orig_height * scale_y)
        
        # Rescale intrinsics
        model_name = camera.model.name
        
        if model_name == "PINHOLE":
            # PINHOLE: fx, fy, cx, cy
            fx, fy, cx, cy = camera.params
            new_params = np.array([
                fx * scale_x,
                fy * scale_y,
                cx * scale_x,
                cy * scale_y
            ])
        elif model_name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            # SIMPLE_PINHOLE: f, cx, cy
            # SIMPLE_RADIAL: f, cx, cy, k
            f, cx, cy = camera.params[:3]
            new_params = camera.params.copy()
            new_params[0] = f * (scale_x + scale_y) / 2  # Average scale for focal length
            new_params[1] = cx * scale_x
            new_params[2] = cy * scale_y
        elif model_name in ["RADIAL", "OPENCV"]:
            # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            fx, fy, cx, cy = camera.params[:4]
            new_params = camera.params.copy()
            new_params[0] = fx * scale_x
            new_params[1] = fy * scale_y
            new_params[2] = cx * scale_x
            new_params[3] = cy * scale_y
            # Distortion params (k1, k2, p1, p2, ...) remain unchanged
        else:
            print(f"    ⚠️  Unknown camera model: {model_name}, skipping")
            continue
        
        # Update camera
        camera.width = new_width
        camera.height = new_height
        camera.params = new_params
        
        # Print changes
        print(f"    Camera {cam_id} ({model_name}):")
        print(f"      Dimensions: {orig_width}×{orig_height} → {new_width}×{new_height}")
        print(f"      Scale: x={scale_x:.6f}, y={scale_y:.6f}")
        if model_name == "PINHOLE":
            print(f"      fx: {orig_params[0]:.2f} → {new_params[0]:.2f}")
            print(f"      fy: {orig_params[1]:.2f} → {new_params[1]:.2f}")
            print(f"      cx: {orig_params[2]:.2f} → {new_params[2]:.2f}")
            print(f"      cy: {orig_params[3]:.2f} → {new_params[3]:.2f}")
    
    # Write updated reconstruction
    reconstruction.write_binary(str(patch_dir))
    reconstruction.write_text(str(patch_dir))
    
    print(f"  ✓ Updated {patch_dir.parent.parent.name}")


def main():
    print("="*70)
    print("Image Dimension Matcher - COLMAP Intrinsics Updater")
    print("="*70)
    
    # 1. Analyze current image dimensions
    print("\n[1/4] Analyzing image dimensions...")
    backup_images = IMAGES_DIR.parent / (IMAGES_DIR.name + "_bad_size")
    
    # Check if we already resized images
    if backup_images.exists() and IMAGES_DIR.exists():
        print(f"✓ Found existing backup: {backup_images}")
        print(f"  Images appear to be already resized. Checking dimensions...")
        orig_dims = get_all_image_dimensions(IMAGES_DIR)
        
        if not orig_dims:
            print("❌ No images found!")
            return
        
        unique_dims = set(orig_dims.values())
        if len(unique_dims) == 1:
            target_w, target_h = list(orig_dims.values())[0]
            print(f"✓ All images have consistent dimensions: {target_w}×{target_h}")
            print("  Skipping image resize, proceeding to update intrinsics...")
            skip_resize = True
        else:
            print(f"⚠️  Images still have inconsistent dimensions!")
            for folder, (w, h) in sorted(orig_dims.items()):
                print(f"  {folder}: {w}×{h}")
            return
    else:
        skip_resize = False
        orig_dims = get_all_image_dimensions(IMAGES_DIR)
        
        if not orig_dims:
            print("❌ No images found!")
            return
        
        print(f"\nFound {len(orig_dims)} camera folders:")
        for folder, (w, h) in sorted(orig_dims.items()):
            print(f"  {folder}: {w}×{h}")
        
        # Check if all dimensions are already consistent
        unique_dims = set(orig_dims.values())
        if len(unique_dims) == 1:
            print("\n✓ All images already have consistent dimensions. Nothing to do!")
            return
    
    # 2. Compute target dimensions
    if not skip_resize:
        target_w, target_h = compute_target_dimensions(orig_dims)
        print(f"\n[2/4] Target dimensions: {target_w}×{target_h}")
        print(f"      (minimum width and height across all cameras)")
        
        # Compute scale factors for each camera folder
        folder_scales = {}
        for folder, (orig_w, orig_h) in orig_dims.items():
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            folder_scales[folder] = (scale_x, scale_y)
            print(f"  {folder}: scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")
        
        # 3. Resize images
        print(f"\n[3/4] Resizing images...")
        resize_images(IMAGES_DIR, backup_images, (target_w, target_h))
    else:
        # Get scale factors from backup
        print(f"\n[2-3/4] Computing scale factors from backup...")
        backup_dims = get_all_image_dimensions(backup_images)
        target_w, target_h = list(orig_dims.values())[0]
        
        folder_scales = {}
        for folder, (orig_w, orig_h) in backup_dims.items():
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            folder_scales[folder] = (scale_x, scale_y)
            print(f"  {folder}: {orig_w}×{orig_h} → {target_w}×{target_h}")
            print(f"           scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")

    
    # 4. Update camera intrinsics in all patches
    print(f"\n[4/4] Updating camera intrinsics in patches...")
    
    # First, determine camera_id → folder mapping from any patch
    sample_patch = sorted(PATCHES_DIR.glob("p*/sparse/0"))[0]
    reconstruction = pycolmap.Reconstruction(str(sample_patch))
    
    # Map camera_id to folder by checking first image for each camera
    cam_id_to_folder = {}
    for image_id, image in reconstruction.images.items():
        camera_id = image.camera_id
        if camera_id not in cam_id_to_folder:
            # Extract folder from image name (e.g., "left/image.png" → "left")
            img_name = image.name
            if '/' in img_name:
                folder = img_name.split('/')[0]
                cam_id_to_folder[camera_id] = folder
    
    print(f"\nCamera ID → Folder mapping:")
    for cam_id, folder in sorted(cam_id_to_folder.items()):
        print(f"  Camera {cam_id} → {folder}/")
    
    # Convert to camera_id → scale_factors
    cam_scale_factors = {}
    for cam_id, folder in cam_id_to_folder.items():
        if folder in folder_scales:
            cam_scale_factors[cam_id] = folder_scales[folder]
    
    # Backup patches directory
    backup_patches = PATCHES_DIR.parent / (PATCHES_DIR.name + "_bad_sizes")
    if backup_patches.exists():
        print(f"\n⚠️  Backup already exists: {backup_patches}")
        response = input("Delete and continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(backup_patches)
    
    print(f"\nBacking up: {PATCHES_DIR} → {backup_patches}")
    shutil.copytree(PATCHES_DIR, backup_patches)
    
    # Update all patches
    patch_dirs = sorted(PATCHES_DIR.glob("p*/sparse/0"))
    print(f"\nUpdating {len(patch_dirs)} patches...")
    
    for patch_dir in patch_dirs:
        update_cameras_for_patch(patch_dir, cam_scale_factors)
    
    print("\n" + "="*70)
    print("✓ COMPLETED")
    print("="*70)
    print(f"\nBackups created:")
    print(f"  Images:  {backup_images}")
    print(f"  Patches: {backup_patches}")
    print(f"\nAll images resized to: {target_w}×{target_h}")
    print(f"All camera intrinsics updated in {len(patch_dirs)} patches.")
    print("\n⚠️  If anything goes wrong, restore from backups:")
    print(f"  rm -rf {IMAGES_DIR} {PATCHES_DIR}")
    print(f"  mv {backup_images} {IMAGES_DIR}")
    print(f"  mv {backup_patches} {PATCHES_DIR}")


if __name__ == '__main__':
    main()
