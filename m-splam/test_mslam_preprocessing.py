#!/usr/bin/env python3
"""
test_mslam_preprocessing.py

Debug script to compare our preprocessing with M-SLAM's actual output.
Tests the first high-res image to see if our crop/resize matches M-SLAM's keyframe output.

Usage:
    python test_mslam_preprocessing.py
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

# Import M-SLAM's resize function
sys.path.insert(0, '/home/ben/encode/code/MASt3R-SLAM')
from mast3r_slam.mast3r_utils import resize_img

# Paths
HIGHRES_DIR = Path('/home/ben/encode/data/mars_johns/left')
DOWNSAMPLED_DIR = Path('/home/ben/encode/data/mars_johns/left_downsampled_png')
OUTPUT_DIR = Path('/home/ben/encode/data/intermediate_data/test_images')

# Try to find the most recent M-SLAM keyframe
MSLAM_KEYFRAME_CANDIDATES = [
    Path('/home/ben/encode/data/intermediate_data/highres_Mars/mslam_logs/keyframes/0.0.png'),
    OUTPUT_DIR / '0.0.png',  # Fallback to user-provided
]

MSLAM_KEYFRAME = None
for candidate in MSLAM_KEYFRAME_CANDIDATES:
    if candidate.exists():
        MSLAM_KEYFRAME = candidate
        break

if not MSLAM_KEYFRAME:
    print("❌ No M-SLAM keyframe found! Please run M-SLAM first or copy 0.0.png to test_images/")
    sys.exit(1)


def load_first_highres_image():
    """Find and load the first high-res image."""
    # Find all images
    all_images = []
    for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
        all_images.extend(HIGHRES_DIR.glob(ext))
    
    all_images = sorted(all_images)
    
    if not all_images:
        raise FileNotFoundError(f"No images found in {HIGHRES_DIR}")
    
    first_image = all_images[0]
    print(f"First high-res image: {first_image.name}")
    print(f"  Path: {first_image}")
    
    return first_image


def preprocess_with_mslam_resize(img_path, output_path, img_size=512):
    """
    Apply M-SLAM's resize_img() function exactly as M-SLAM dataloader does.
    
    Matches dataloader.py:
    - Line 44: img = cv2.imread(self.rgb_files[idx])
    - Line 45: return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    - Line 48: return img.astype(self.dtype) / 255.0
    
    Args:
        img_path: Path to input image
        output_path: Path to save preprocessed image
        img_size: Target size (default 512)
    """
    print(f"\n{'='*70}")
    print("Preprocessing with M-SLAM's resize_img()")
    print(f"{'='*70}")
    print(f"Input: {img_path}")
    
    # Load image EXACTLY as M-SLAM dataloader does (dataloader.py:44-45)
    img_bgr = cv2.imread(str(img_path))
    print(f"After cv2.imread (BGR):")
    print(f"  Shape: {img_bgr.shape}, Dtype: {img_bgr.dtype}")
    print(f"  Top-left pixel (BGR): {img_bgr[0, 0]}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"After BGR→RGB:")
    print(f"  Shape: {img_rgb.shape}, Dtype: {img_rgb.dtype}")
    print(f"  Top-left pixel (RGB): {img_rgb[0, 0]}")
    
    # Convert to float32 [0,1] as dataloader does (dataloader.py:48)
    img_array = img_rgb.astype(np.float32) / 255.0
    print(f"After astype(float32) / 255:")
    print(f"  Shape: {img_array.shape}, Dtype: {img_array.dtype}")
    print(f"  Range: [{img_array.min():.6f}, {img_array.max():.6f}]")
    print(f"  Top-left pixel: {img_array[0, 0]}")
    
    # Apply M-SLAM's resize_img
    print(f"\nApplying resize_img(img, {img_size})...")
    result_dict = resize_img(img_array, img_size)
    
    # Extract the unnormalized image (uint8 [0,255])
    unnormalized_uint8 = result_dict['unnormalized_img']
    print(f"After resize_img:")
    print(f"  Shape: {unnormalized_uint8.shape} (H x W x C)")
    print(f"  Dtype: {unnormalized_uint8.dtype}, Range: [{unnormalized_uint8.min()}, {unnormalized_uint8.max()}]")
    
    # CRITICAL: M-SLAM does an extra float conversion in create_frame (frame.py:126)
    # uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
    # Then when saving keyframes, converts back: (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
    print(f"\nApplying M-SLAM's float conversion (create_frame logic):")
    uimg_float = unnormalized_uint8.astype(np.float32) / 255.0
    print(f"  After /255: dtype={uimg_float.dtype}, range=[{uimg_float.min():.6f}, {uimg_float.max():.6f}]")
    
    # Then convert back to uint8 as M-SLAM does when saving
    resized_uint8 = (uimg_float * 255).astype(np.uint8)
    print(f"  After *255: dtype={resized_uint8.dtype}, range=[{resized_uint8.min()}, {resized_uint8.max()}]")
    
    # Save as PNG (matching M-SLAM's save_keyframes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(resized_uint8, cv2.COLOR_RGB2BGR))
    print(f"\n✓ Saved: {output_path}")
    print(f"  Size: {resized_uint8.shape[1]}x{resized_uint8.shape[0]} (W x H)")
    
    return resized_uint8


def compare_images(img1_path, img2_path):
    """
    Compare two images pixel-by-pixel.
    
    Args:
        img1_path: Path to first image (our preprocessed)
        img2_path: Path to second image (M-SLAM keyframe)
    
    Returns:
        dict with comparison metrics
    """
    print(f"\n{'='*70}")
    print("Comparing Images")
    print(f"{'='*70}")
    
    if not img1_path.exists():
        print(f"❌ Image 1 not found: {img1_path}")
        return None
    
    if not img2_path.exists():
        print(f"❌ Image 2 not found: {img2_path}")
        return None
    
    # Load images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    print(f"\nImage 1 (our preprocessed): {img1_path.name}")
    print(f"  Shape: {img1.shape} (H x W x C)")
    print(f"  Dtype: {img1.dtype}")
    print(f"  Sample pixels (top-left 3x3, B,G,R):")
    print(f"    {img1[:3, :3, :]}")
    
    print(f"\nImage 2 (M-SLAM keyframe): {img2_path.name}")
    print(f"  Shape: {img2.shape} (H x W x C)")
    print(f"  Dtype: {img2.dtype}")
    print(f"  Sample pixels (top-left 3x3, B,G,R):")
    print(f"    {img2[:3, :3, :]}")
    
    # Check if shapes match
    if img1.shape != img2.shape:
        print(f"\n❌ SHAPES DON'T MATCH!")
        print(f"   Image 1: {img1.shape}")
        print(f"   Image 2: {img2.shape}")
        return {
            'identical': False,
            'shapes_match': False,
            'shape1': img1.shape,
            'shape2': img2.shape
        }
    
    # Pixel-by-pixel comparison
    identical = np.array_equal(img1, img2)
    
    if identical:
        print(f"\n✅ IMAGES ARE IDENTICAL!")
        print(f"   All {img1.size} pixels match exactly")
        return {
            'identical': True,
            'shapes_match': True,
            'mean_abs_diff': 0.0,
            'max_abs_diff': 0,
            'num_different_pixels': 0
        }
    else:
        # Calculate differences
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        mean_abs_diff = diff.mean()
        max_abs_diff = diff.max()
        num_different = np.sum(diff > 0) // 3  # Divide by 3 for RGB channels
        total_pixels = img1.shape[0] * img1.shape[1]
        pct_different = (num_different / total_pixels) * 100
        
        print(f"\n❌ IMAGES ARE DIFFERENT!")
        print(f"   Mean absolute difference: {mean_abs_diff:.4f}")
        print(f"   Max absolute difference: {max_abs_diff:.0f}")
        print(f"   Different pixels: {num_different:,} / {total_pixels:,} ({pct_different:.2f}%)")
        
        # Save difference map
        diff_map = (diff / diff.max() * 255).astype(np.uint8)
        diff_path = img1_path.parent / f"diff_{img1_path.stem}_vs_{img2_path.stem}.png"
        cv2.imwrite(str(diff_path), diff_map)
        print(f"\n💾 Difference map saved: {diff_path}")
        print(f"   (brighter = more different)")
        
        return {
            'identical': False,
            'shapes_match': True,
            'mean_abs_diff': float(mean_abs_diff),
            'max_abs_diff': int(max_abs_diff),
            'num_different_pixels': int(num_different),
            'total_pixels': int(total_pixels),
            'pct_different': float(pct_different),
            'diff_map_path': str(diff_path)
        }


def main():
    print(f"\n{'='*70}")
    print("M-SLAM Preprocessing Test")
    print(f"{'='*70}")
    print(f"Testing if our preprocessing matches M-SLAM's keyframe output")
    print()
    
    # 1. Find first high-res image
    first_highres = load_first_highres_image()
    
    # 2. Test with HIGH-RES image
    print(f"\n{'='*70}")
    print("TEST 1: High-res image")
    print(f"{'='*70}")
    output_path_highres = OUTPUT_DIR / f"preprocessed_highres_{first_highres.stem}.png"
    preprocessed_highres = preprocess_with_mslam_resize(first_highres, output_path_highres)
    result_highres = compare_images(output_path_highres, MSLAM_KEYFRAME)
    
    # 3. Find corresponding downsampled image
    # The downsampled images should have same name but possibly different extension
    downsampled_name = first_highres.stem + '.png'  # Downsampled are PNG
    downsampled_path = DOWNSAMPLED_DIR / downsampled_name
    
    if not downsampled_path.exists():
        print(f"\n⚠️  Downsampled image not found: {downsampled_path}")
        print(f"   Skipping downsampled test")
        return
    
    print(f"\nDownsampled image found: {downsampled_path}")
    print(f"  File size: {downsampled_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 4. Test with DOWNSAMPLED image
    print(f"\n{'='*70}")
    print("TEST 2: Downsampled image (what M-SLAM actually uses)")
    print(f"{'='*70}")
    output_path_downsampled = OUTPUT_DIR / f"preprocessed_downsampled_{downsampled_path.stem}.png"
    preprocessed_downsampled = preprocess_with_mslam_resize(downsampled_path, output_path_downsampled)
    
    # 5. Compare downsampled with M-SLAM's actual keyframe
    result_downsampled = compare_images(output_path_downsampled, MSLAM_KEYFRAME)
    
    # 6. Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"High-res input:        {first_highres.name}")
    print(f"Downsampled input:     {downsampled_path.name}")
    print(f"M-SLAM keyframe:       {MSLAM_KEYFRAME.name}")
    print()
    
    print("TEST 1: High-res → M-SLAM preprocessing")
    if result_highres and result_highres['identical']:
        print(f"  ✅ IDENTICAL")
    elif result_highres:
        print(f"  ❌ DIFFERENT: {result_highres.get('pct_different', 0):.2f}% pixels differ")
    
    print()
    print("TEST 2: Downsampled → M-SLAM preprocessing")
    if result_downsampled and result_downsampled['identical']:
        print(f"  ✅ IDENTICAL - This is what M-SLAM uses!")
    elif result_downsampled:
        print(f"  ❌ DIFFERENT: {result_downsampled.get('pct_different', 0):.2f}% pixels differ")
    
    print()
    print("CONCLUSION:")
    if result_downsampled and result_downsampled['identical']:
        print("  ✅ M-SLAM processes the DOWNSAMPLED images (1600x1400)")
        print("  ✅ Our preprocessing matches perfectly when using downsampled input")
        print()
        print("  ⚠️  This means:")
        print("     - M-SLAM never sees the high-res images directly")
        print("     - The 'slither' difference comes from downsampling artifacts")
        print("     - To use high-res for splatting, we need to account for this")
    else:
        print("  ⚠️  Neither high-res nor downsampled match M-SLAM keyframe")
        print("     - Something else is going on with M-SLAM's preprocessing")
        print("     - May need to check M-SLAM's dataloader for additional processing")
    
    print(f"\n{'='*70}")
    print(f"Test complete. Check images in: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
