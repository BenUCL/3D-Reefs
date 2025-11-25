"""
Test M-SLAM's exact save conversion logic to identify where pixel differences come from
"""
import sys
sys.path.insert(0, '/home/ben/encode/code/MASt3R-SLAM')

from mast3r_slam.config import set_global_config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
import cv2
import numpy as np
from pathlib import Path

# Configure M-SLAM
import yaml
config_path = Path("/home/ben/encode/code/MASt3R-SLAM/config/base.yaml")
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
set_global_config(cfg)

# Load dataset
dataset = load_dataset("/home/ben/encode/data/mars_johns/left_downsampled_png")
print(f"Dataset loaded: {len(dataset)} images")

# Get first image
timestamp, img = dataset[0]
print(f"\nOriginal dataloader output:")
print(f"  Shape: {img.shape}, dtype: {img.dtype}")
print(f"  Top-left pixel (RGB float): {img[0, 0]}")

# Create frame
frame = create_frame(0, img, None, img_size=512)
print(f"\nFrame.uimg:")
print(f"  Shape: {frame.uimg.shape}, dtype: {frame.uimg.dtype}")
print(f"  Top-left pixel (RGB float): {frame.uimg[0, 0].cpu().numpy()}")

# Test different conversion paths
print(f"\n=== Testing Conversion Paths ===")

# Path 1: Direct numpy + uint8
uimg_np = frame.uimg.cpu().numpy()
method1 = (uimg_np * 255).astype(np.uint8)
print(f"\nMethod 1: (uimg * 255).astype(uint8)")
print(f"  Top-left: {method1[0, 0]}")

# Path 2: clip then astype
method2 = (uimg_np * 255).clip(0, 255).astype(np.uint8)
print(f"\nMethod 2: (uimg * 255).clip(0,255).astype(uint8)")
print(f"  Top-left: {method2[0, 0]}")

# Path 3: round then astype
method3 = np.round(uimg_np * 255).astype(np.uint8)
print(f"\nMethod 3: round(uimg * 255).astype(uint8)")
print(f"  Top-left: {method3[0, 0]}")

# Path 4: M-SLAM's exact code from evaluate.py line 99-101
keyframe_img = (frame.uimg.cpu().numpy() * 255).astype(np.uint8)
keyframe_bgr = cv2.cvtColor(keyframe_img, cv2.COLOR_RGB2BGR)
print(f"\nMethod 4: M-SLAM's exact save code")
print(f"  RGB top-left: {keyframe_img[0, 0]}")
print(f"  BGR top-left: {keyframe_bgr[0, 0]}")

# Save with cv2.imwrite (exactly as M-SLAM does)
output_path = "/home/ben/encode/data/intermediate_data/test_images/test_save_output.png"
cv2.imwrite(output_path, keyframe_bgr)
print(f"\n✓ Saved with cv2.imwrite")

# Read it back
readback = cv2.imread(output_path)
print(f"\nRead back from disk:")
print(f"  BGR top-left: {readback[0, 0]}")

# Compare with actual M-SLAM keyframe
mslam_keyframe = cv2.imread("/home/ben/encode/data/intermediate_data/test_images/0.0_fresh.png")
print(f"\nM-SLAM keyframe (fresh from Mars_lowres):")
print(f"  BGR top-left: {mslam_keyframe[0, 0]}")

print(f"\n=== Pixel Difference Analysis ===")
print(f"Our save:     BGR {readback[0, 0]}")
print(f"M-SLAM save:  BGR {mslam_keyframe[0, 0]}")
print(f"Difference:   {mslam_keyframe[0, 0] - readback[0, 0]}")
