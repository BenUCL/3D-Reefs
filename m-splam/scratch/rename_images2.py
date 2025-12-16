# Rename images with optional stereo camera support
#!/usr/bin/env python3
import os
import re
from pathlib import Path

# --- CONFIGURATION ---
# The folder containing your images
# Will rename these images in place
INPUT_DIR = "/home/ben/encode/data/intermediate_data/colmap5/4559_downsampled_png"
# Set to True to enable stereo camera renaming (left/right pairs)
STEREO_MODE = True

def get_frame_number(filename):
    """
    Extracts the number inside the brackets. 
    e.g., "Image (10).png" -> returns 10
    """
    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))
    return 0

def is_left_camera(filename):
    """Check if filename contains 'left' (case insensitive)"""
    return 'left' in filename.lower()

def is_right_camera(filename):
    """Check if filename contains 'right' (case insensitive)"""
    return 'right' in filename.lower()

def rename_stereo_files(directory, files):
    """Rename stereo camera files with sequential ordering"""
    left_files = [f for f in files if is_left_camera(f.name)]
    right_files = [f for f in files if is_right_camera(f.name)]
    
    # Sort by frame number
    left_files.sort(key=lambda f: get_frame_number(f.name))
    right_files.sort(key=lambda f: get_frame_number(f.name))
    
    if len(left_files) != len(right_files):
        print(f"Warning: Found {len(left_files)} left images but {len(right_files)} right images")
        print("Proceeding anyway...")
    
    # Create pairs based on position in sorted list
    max_pairs = max(len(left_files), len(right_files))
    
    rename_plan = []
    for i in range(max_pairs):
        frame_num = i + 1
        
        if i < len(left_files):
            left_file = left_files[i]
            # Remove spaces and keep original name structure
            clean_name = left_file.name.replace(' ', '')
            new_name = f"{frame_num:04d}L_{clean_name}"
            rename_plan.append((left_file, new_name))
        
        if i < len(right_files):
            right_file = right_files[i]
            # Remove spaces and keep original name structure
            clean_name = right_file.name.replace(' ', '')
            new_name = f"{frame_num:04d}R_{clean_name}"
            rename_plan.append((right_file, new_name))
    
    # Show first 5 examples
    print(f"\nFound {len(left_files)} left and {len(right_files)} right images")
    print("\nRename preview (first 5):")
    for old_file, new_name in rename_plan[:5]:
        print(f"  {old_file.name} -> {new_name}")
    
    if len(rename_plan) > 5:
        print(f"  ... and {len(rename_plan) - 5} more")
    
    response = input("\nProceed with renaming? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Execute renames
    count = 0
    for old_path, new_name in rename_plan:
        new_path = directory / new_name
        if not new_path.exists():
            os.rename(old_path, new_path)
            count += 1
        else:
            print(f"Skipped: {new_name} already exists.")
    
    print(f"\nSuccess! Renamed {count} images in stereo mode.")

def rename_single_camera_files(directory, files):
    """Original single camera renaming logic"""
    files.sort(key=lambda f: get_frame_number(f.name))
    
    print(f"Found {len(files)} images. Starting rename...\n")
    
    count = 0
    for file in files:
        original_name = file.name
        num = get_frame_number(original_name)
        clean_base = re.sub(r'\s*\(\d+\)', '', file.stem).replace(' ', '_')
        new_name = f"{num:04d}_{clean_base}{file.suffix}"
        
        old_path = file
        new_path = directory / new_name
        
        if not new_path.exists():
            os.rename(old_path, new_path)
            count += 1
            if count <= 5:
                print(f"Renamed: {original_name} -> {new_name}")
        else:
            print(f"Skipped: {new_name} already exists.")
    
    print(f"\nSuccess! Renamed {count} images.")
    print(f"Files are now strictly ordered (0001, 0002... 0010).")

def main():
    directory = Path(INPUT_DIR)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return

    # Get all png files
    files = [f for f in directory.iterdir() if f.suffix.lower() == '.png']
    
    if len(files) == 0:
        print("No PNG files found in directory.")
        return
    
    if STEREO_MODE:
        # Check for left/right in filenames
        has_left = any(is_left_camera(f.name) for f in files)
        has_right = any(is_right_camera(f.name) for f in files)
        
        if has_left and has_right:
            print("STEREO MODE: Detected left and right camera files")
            rename_stereo_files(directory, files)
        else:
            print("STEREO MODE is True but no left/right camera files detected.")
            print(f"Found: left={has_left}, right={has_right}\n")
            
            # Show 5 example renames for single camera mode
            preview_files = sorted(files, key=lambda f: get_frame_number(f.name))[:5]
            print("Will rename using single camera mode. Examples:")
            for file in preview_files:
                num = get_frame_number(file.name)
                clean_base = re.sub(r'\s*\(\d+\)', '', file.stem).replace(' ', '_')
                new_name = f"{num:04d}_{clean_base}{file.suffix}"
                print(f"  {file.name} -> {new_name}")
            
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
            
            response = input("\nProceed with single camera renaming? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return
            
            rename_single_camera_files(directory, files)
    else:
        # Single camera mode (original behavior)
        rename_single_camera_files(directory, files)

if __name__ == "__main__":
    main()