#!/usr/bin/env python3
"""
Copy files containing 'Left' in their name from raw_download to left directory.
"""

import os
import shutil
from pathlib import Path

# Define source and destination directories
source_dir = Path("/home/ben/encode/data/mars_johns/raw_download")
dest_dir = Path("/home/ben/encode/data/mars_johns/left")

# Create destination directory if it doesn't exist
dest_dir.mkdir(parents=True, exist_ok=True)

# Counter for copied files
copied_count = 0

# Iterate through files in source directory
for file_path in source_dir.iterdir():
    if file_path.is_file() and "Left" in file_path.name:
        dest_path = dest_dir / file_path.name
        shutil.copy2(file_path, dest_path)
        print(f"Copied: {file_path.name}")
        copied_count += 1

print(f"\nTotal files copied: {copied_count}")
