#!/usr/bin/env python3
"""
Diagnostic script to check image dimensions across all patches.
Compares COLMAP expected dimensions with actual image files.
"""

import sys
from pathlib import Path
from PIL import Image
import pycolmap
from collections import defaultdict
import json

def check_patch_images(patch_dir, images_base_dir):
    """Check all images referenced in a patch's COLMAP reconstruction."""
    sparse_dir = patch_dir / "sparse" / "0"
    
    if not sparse_dir.exists():
        return None
    
    # Load COLMAP reconstruction
    try:
        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    except Exception as e:
        return {"error": f"Failed to load reconstruction: {e}"}
    
    results = {
        "patch_name": patch_dir.name,
        "total_images": len(reconstruction.images),
        "cameras": {},
        "mismatches": [],
        "unique_dimensions": set()
    }
    
    # Check each camera's expected dimensions
    for cam_id, camera in reconstruction.cameras.items():
        results["cameras"][cam_id] = {
            "model": camera.model.name,
            "expected_width": camera.width,
            "expected_height": camera.height
        }
    
    # Check each image
    mismatch_count = 0
    for img_id, image in reconstruction.images.items():
        # Get expected dimensions from camera
        camera = reconstruction.cameras[image.camera_id]
        expected_dims = (camera.width, camera.height)
        
        # Get actual image file
        img_path = images_base_dir / image.name
        
        if not img_path.exists():
            results["mismatches"].append({
                "image_name": image.name,
                "camera_id": image.camera_id,
                "error": "File not found"
            })
            continue
        
        # Check actual dimensions
        try:
            with Image.open(img_path) as img:
                actual_dims = img.size  # (width, height)
                results["unique_dimensions"].add(actual_dims)
                
                if actual_dims != expected_dims:
                    mismatch_count += 1
                    results["mismatches"].append({
                        "image_name": image.name,
                        "camera_id": image.camera_id,
                        "expected": expected_dims,
                        "actual": actual_dims,
                        "diff": (actual_dims[0] - expected_dims[0], 
                                actual_dims[1] - expected_dims[1])
                    })
        except Exception as e:
            results["mismatches"].append({
                "image_name": image.name,
                "camera_id": image.camera_id,
                "error": f"Failed to read image: {e}"
            })
    
    results["mismatch_count"] = mismatch_count
    results["unique_dimensions"] = list(results["unique_dimensions"])
    
    return results


def main():
    patches_dir = Path("/home/ben/encode/data/intermediate_data/colmap5/sparse_patches")
    images_dir = Path("/home/ben/encode/data/intermediate_data/colmap5/images")
    
    print("="*70)
    print("Image Dimension Diagnostic Report")
    print("="*70)
    print(f"Patches directory: {patches_dir}")
    print(f"Images directory:  {images_dir}")
    print()
    
    # Find all patch directories
    patch_dirs = sorted([d for d in patches_dir.glob("p*") if d.is_dir()], 
                       key=lambda x: int(x.name[1:]))
    
    print(f"Found {len(patch_dirs)} patches to check")
    print()
    
    all_results = []
    failed_patches = []
    total_mismatches = 0
    
    for patch_dir in patch_dirs:
        print(f"Checking {patch_dir.name}...", end=" ", flush=True)
        
        results = check_patch_images(patch_dir, images_dir)
        
        if results is None:
            print("SKIPPED (no sparse/0)")
            continue
        
        if "error" in results:
            print(f"ERROR: {results['error']}")
            continue
        
        all_results.append(results)
        
        if results["mismatch_count"] > 0:
            failed_patches.append(patch_dir.name)
            total_mismatches += results["mismatch_count"]
            print(f"❌ {results['mismatch_count']} mismatches")
        else:
            print("✓ All match")
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total patches checked: {len(all_results)}")
    print(f"Patches with mismatches: {len(failed_patches)}")
    print(f"Total image mismatches: {total_mismatches}")
    print()
    
    if failed_patches:
        print(f"Failed patches: {', '.join(failed_patches)}")
        print()
        
        # Detailed mismatch report
        print("="*70)
        print("DETAILED MISMATCH REPORT")
        print("="*70)
        
        for result in all_results:
            if result["mismatch_count"] == 0:
                continue
            
            print(f"\n{result['patch_name']}:")
            print(f"  Total images: {result['total_images']}")
            print(f"  Mismatches: {result['mismatch_count']}")
            print(f"  Unique dimensions found: {result['unique_dimensions']}")
            print(f"  Expected cameras:")
            for cam_id, cam_info in result["cameras"].items():
                print(f"    Camera {cam_id}: {cam_info['expected_width']}x{cam_info['expected_height']}")
            
            # Show first few mismatches
            print(f"  Sample mismatches:")
            for mismatch in result["mismatches"][:5]:
                if "error" in mismatch:
                    print(f"    {mismatch['image_name']}: {mismatch['error']}")
                else:
                    print(f"    {mismatch['image_name']} (cam {mismatch['camera_id']}): "
                          f"expected {mismatch['expected']}, got {mismatch['actual']}, "
                          f"diff {mismatch['diff']}")
            
            if len(result["mismatches"]) > 5:
                print(f"    ... and {len(result['mismatches']) - 5} more")
    else:
        print("✓ All patches have matching dimensions!")
    
    # Save detailed results to JSON
    output_file = patches_dir / "dimension_diagnostic.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print()
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
