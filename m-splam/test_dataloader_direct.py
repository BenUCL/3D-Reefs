#!/usr/bin/env python3
"""
test_dataloader_direct.py

Directly test M-SLAM's dataloader to see what it produces.
This bypasses all our preprocessing and calls M-SLAM's code directly.
"""

import sys
sys.path.insert(0, '/home/ben/encode/code/MASt3R-SLAM')

from mast3r_slam.config import set_global_config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
import cv2
import numpy as np
from pathlib import Path

# Initialize config
import yaml
config_path = "/home/ben/encode/code/MASt3R-SLAM/config/base.yaml"
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
set_global_config(cfg)

# Load dataset
dataset_path = "/home/ben/encode/data/mars_johns/left_downsampled_png"
dataset = load_dataset(dataset_path)

print(f"Dataset loaded: {len(dataset)} images")
print(f"First image path: {dataset.rgb_files[0]}")

# Get first image through dataloader
timestamp, img = dataset[0]
print(f"\nDataloader output for image 0:")
print(f"  Timestamp: {timestamp}")
print(f"  img dtype: {img.dtype}")
print(f"  img shape: {img.shape}")
print(f"  img range: [{img.min():.6f}, {img.max():.6f}]")
print(f"  Top-left pixel (RGB, float32): {img[0, 0]}")

# Also manually load and check
import cv2
manual_bgr = cv2.imread(str(dataset.rgb_files[0]))
manual_rgb = cv2.cvtColor(manual_bgr, cv2.COLOR_BGR2RGB)
manual_float = manual_rgb.astype(np.float32) / 255.0
print(f"\nManual load verification:")
print(f"  BGR top-left: {manual_bgr[0, 0]}")
print(f"  RGB top-left: {manual_rgb[0, 0]}")
print(f"  Float top-left: {manual_float[0, 0]}")
print(f"  Match? {np.array_equal(img, manual_float)}")

# Create frame as M-SLAM does
frame = create_frame(0, img, None, img_size=512)

# Extract uimg (what gets saved as keyframe)
uimg = frame.uimg.cpu().numpy()
print(f"\nFrame.uimg (what becomes keyframe):")
print(f"  dtype: {uimg.dtype}")
print(f"  shape: {uimg.shape}")
print(f"  range: [{uimg.min():.6f}, {uimg.max():.6f}]")

# Convert to uint8 as save_keyframes does
uimg_uint8 = (uimg * 255).astype('uint8')
print(f"\nAfter *255 and uint8 (ready to save):")
print(f"  dtype: {uimg_uint8.dtype}")
print(f"  range: [{uimg_uint8.min()}, {uimg_uint8.max()}]")
print(f"  Top-left 3x3 pixels (RGB):")
print(uimg_uint8[:3, :3, :])

# Save it
output_path = Path('/home/ben/encode/data/intermediate_data/test_images/dataloader_direct_output.png')
cv2.imwrite(str(output_path), cv2.cvtColor(uimg_uint8, cv2.COLOR_RGB2BGR))
print(f"\n✓ Saved to: {output_path}")
print(f"\nNow compare this with M-SLAM's actual 0.0.png keyframe!")
