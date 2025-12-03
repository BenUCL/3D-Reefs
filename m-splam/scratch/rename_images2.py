# For only a single Gopro image set
#!/usr/bin/env python3
import os
import re
from pathlib import Path

# --- CONFIGURATION ---
# The folder containing your images
# Will rename these images in place
INPUT_DIR = "/home/ben/encode/data/intermediate_data/colmap3/left_950_png_downsampled"

def get_frame_number(filename):
    """
    Extracts the number inside the brackets. 
    e.g., "Image (10).png" -> returns 10
    """
    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))
    return 0

def main():
    directory = Path(INPUT_DIR)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return

    # 1. Get all png files
    files = [f for f in directory.iterdir() if f.suffix.lower() == '.png']
    
    # 2. Sort them numerically based on the number in brackets
    # This solves the "1, 10, 2" fear by forcing integer comparison
    files.sort(key=lambda f: get_frame_number(f.name))

    print(f"Found {len(files)} images. Starting rename...\n")

    count = 0
    for file in files:
        original_name = file.name
        
        # Get the number (e.g., 1, 2, 10)
        num = get_frame_number(original_name)
        
        # 3. Create clean name
        # Strip out the " (N)" part and spaces
        clean_base = re.sub(r'\s*\(\d+\)', '', file.stem).replace(' ', '_')
        
        # Create new name: 0001_2019A_GP_Left.png
        # The :04d ensures it adds zeros: 1 -> 0001, 10 -> 0010
        new_name = f"{num:04d}_{clean_base}{file.suffix}"
        
        # Rename in place
        old_path = file
        new_path = directory / new_name
        
        # Don't overwrite if it already exists (safety check)
        if not new_path.exists():
            os.rename(old_path, new_path)
            count += 1
            # Print first few to show user it's working
            if count <= 5:
                print(f"Renamed: {original_name} -> {new_name}")
        else:
            print(f"Skipped: {new_name} already exists.")

    print(f"\nSuccess! Renamed {count} images.")
    print(f"Files are now strictly ordered (0001, 0002... 0010).")

if __name__ == "__main__":
    main()