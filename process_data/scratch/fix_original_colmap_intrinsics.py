#!/usr/bin/env python3
"""
Fix the root cause: Update the ORIGINAL COLMAP reconstruction's camera intrinsics
to match the resized images, then re-run patching.

The issue is:
1. Images were resized to 1548x1357 (all cameras)
2. match_img_dims.py updated PATCHES but not the original sparse/ directory
3. When patching was re-run with different max_cameras, it read from the
   ORIGINAL sparse/ which still had old dimensions (1550x1357 for cam2, 1548x1360 for cam1)
4. This created new patches with incorrect intrinsics
"""

import pycolmap
from pathlib import Path
from PIL import Image
import shutil

# Paths
SPARSE_DIR = Path("/home/ben/encode/data/intermediate_data/colmap5/sparse")
IMAGES_DIR = Path("/home/ben/encode/data/intermediate_data/colmap5/images")

def main():
    print("="*70)
    print("Fix Original COLMAP Reconstruction Intrinsics")
    print("="*70)
    
    # 1. Check actual image dimensions
    print("\nStep 1: Verifying current image dimensions...")
    sample_left = IMAGES_DIR / "left"
    sample_right = IMAGES_DIR / "right"
    
    left_img = next(sample_left.glob("*.png"))
    right_img = next(sample_right.glob("*.png"))
    
    with Image.open(left_img) as img:
        left_dims = img.size
    with Image.open(right_img) as img:
        right_dims = img.size
    
    print(f"  Left camera images:  {left_dims[0]}×{left_dims[1]}")
    print(f"  Right camera images: {right_dims[0]}×{right_dims[1]}")
    
    if left_dims != right_dims:
        print("\n❌ ERROR: Left and right cameras have different dimensions!")
        print("   Run match_img_dims.py first to standardize image sizes.")
        return
    
    target_width, target_height = left_dims
    print(f"\n✓ All images are: {target_width}×{target_height}")
    
    # 2. Load current COLMAP reconstruction
    print("\nStep 2: Loading current COLMAP reconstruction...")
    reconstruction = pycolmap.Reconstruction(str(SPARSE_DIR))
    
    print(f"  Cameras: {len(reconstruction.cameras)}")
    for cam_id, camera in reconstruction.cameras.items():
        print(f"    Camera {cam_id}: {camera.width}×{camera.height}")
    
    # 3. Check if update is needed
    needs_update = False
    for cam_id, camera in reconstruction.cameras.items():
        if camera.width != target_width or camera.height != target_height:
            needs_update = True
            break
    
    if not needs_update:
        print("\n✓ Camera intrinsics already match image dimensions!")
        return
    
    # 4. Backup original
    print("\nStep 3: Creating backup...")
    backup_dir = SPARSE_DIR.parent / "sparse_original_backup"
    if backup_dir.exists():
        print(f"  Backup already exists: {backup_dir}")
        response = input("  Overwrite backup? [y/N]: ")
        if response.lower() != 'y':
            print("  Aborted.")
            return
        shutil.rmtree(backup_dir)
    
    shutil.copytree(SPARSE_DIR, backup_dir)
    print(f"  ✓ Backed up to: {backup_dir}")
    
    # 5. Update cameras
    print("\nStep 4: Updating camera intrinsics...")
    for cam_id, camera in reconstruction.cameras.items():
        old_width = camera.width
        old_height = camera.height
        
        # Calculate scale factors
        scale_x = target_width / old_width
        scale_y = target_height / old_height
        
        print(f"  Camera {cam_id}:")
        print(f"    Old: {old_width}×{old_height}")
        print(f"    New: {target_width}×{target_height}")
        print(f"    Scale: {scale_x:.6f}×{scale_y:.6f}")
        
        # Update camera parameters
        new_params = camera.params.copy()
        new_params[0] *= scale_x  # fx
        new_params[1] *= scale_y  # fy
        new_params[2] *= scale_x  # cx
        new_params[3] *= scale_y  # cy
        
        print(f"    Old params: {camera.params}")
        print(f"    New params: {new_params}")
        
        # Create new camera with updated parameters
        new_camera = pycolmap.Camera(
            model=camera.model,
            width=target_width,
            height=target_height,
            params=new_params,
            camera_id=cam_id
        )
        
        reconstruction.cameras[cam_id] = new_camera
    
    # 6. Save updated reconstruction
    print("\nStep 5: Saving updated reconstruction...")
    reconstruction.write_binary(str(SPARSE_DIR))
    reconstruction.write_text(str(SPARSE_DIR))
    
    print("\n" + "="*70)
    print("✓ SUCCESS")
    print("="*70)
    print("\nOriginal COLMAP reconstruction updated!")
    print(f"Backup saved to: {backup_dir}")
    print("\nNext steps:")
    print("1. Delete current patches: rm -rf /home/ben/encode/data/intermediate_data/colmap5/sparse_patches")
    print("2. Re-run patching: python patch_colmap_data.py --config splat_config.yml")
    print("3. Resume training: ./batch_train_splat.sh")

if __name__ == "__main__":
    main()
