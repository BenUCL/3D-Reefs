#!/usr/bin/env python3
"""
Test training on a single patch with CUDA debugging enabled.
"""

import os
import subprocess
import sys
from pathlib import Path

# Set CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

patch = "p5"
config = "/home/ben/encode/code/3D-Reefs/process_data/splat_config.yml"

print("="*70)
print(f"Testing training on {patch} with CUDA debugging")
print("="*70)
print(f"CUDA_LAUNCH_BLOCKING=1")
print(f"TORCH_USE_CUDA_DSA=1")
print()

cmd = [
    "python",
    "/home/ben/encode/code/3D-Reefs/process_data/train_splat.py",
    "--config", config,
    "--patch", patch
]

result = subprocess.run(cmd, capture_output=False)
sys.exit(result.returncode)
