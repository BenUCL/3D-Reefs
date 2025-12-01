#TODO: the input images.txt has loads of extra numbers after each images vector. Might be source of issue.

import os
import shutil

# --- CONFIGURATION ---
# 1. Where the raw data is coming FROM (PyCuSFM Output)
SOURCE_TXT = "/home/ben/encode/data/intermediate_data/pycusfm1/output/sparse/images.txt"

# 2. Where the clean data is going TO (LichtFeld Studio Input)
BASE_DIR = "/home/ben/encode/data/intermediate_data/pycusfm1/for_splat"
DEST_SPARSE_DIR = os.path.join(BASE_DIR, "sparse/0")
DEST_TXT = os.path.join(DEST_SPARSE_DIR, "images.txt")

# 3. Where the images are located/managed
IMAGES_DIR = os.path.join(BASE_DIR, "images")
REMOVED_DIR = os.path.join(BASE_DIR, "removed")

def main():
    print(f"--- Master Cleanup ---")
    print(f"Source: {SOURCE_TXT}")
    print(f"Target: {DEST_TXT}")
    
    if not os.path.exists(SOURCE_TXT):
        print(f"ERROR: Source file not found: {SOURCE_TXT}")
        return
    
    # Ensure destination directories exist
    if not os.path.exists(DEST_SPARSE_DIR):
        os.makedirs(DEST_SPARSE_DIR, exist_ok=True)
    if not os.path.exists(REMOVED_DIR):
        os.makedirs(REMOVED_DIR, exist_ok=True)

    # --- PART 1: PARSE & CLEAN TEXT ---
    print("\nParsing source text file...")
    with open(SOURCE_TXT, 'r') as f:
        raw_lines = f.readlines()

    # Find start of data (skip headers)
    data_start_idx = 0
    for i, line in enumerate(raw_lines):
        if not line.startswith("#") and line.strip():
            data_start_idx = i
            break
    
    data_lines = [l.strip() for l in raw_lines[data_start_idx:] if l.strip()]
    
    if len(data_lines) % 2 != 0:
        print(f"Warning: Odd number of data lines. Dropping last line.")
        data_lines = data_lines[:-1]
    
    num_images = len(data_lines) // 2
    print(f"Parsed {num_images} image entries.")

    new_txt_lines = []
    valid_clean_filenames = set()
    
    # Add Standard Header
    new_txt_lines.append("# Image list with two lines of data per image:\n")
    new_txt_lines.append("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    new_txt_lines.append("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    new_txt_lines.append(f"# Number of images: {num_images}, mean observations per image: 4000\n")
    
    for i in range(0, len(data_lines), 2):
        pose_line = data_lines[i]
        points_line = data_lines[i+1]
        
        parts = pose_line.split()
        
        # Extract old filename (handle spaces)
        old_name_raw = " ".join(parts[9:])
        base_name = os.path.basename(old_name_raw)
        
        # Clean the filename (Remove spaces)
        clean_base_name = base_name.replace(" ", "")
        
        # Reconstruct Pose Line (No directory prefix, just filename)
        new_pose_line = " ".join(parts[:9] + [clean_base_name])
        
        new_txt_lines.append(new_pose_line + "\n")
        new_txt_lines.append(points_line + "\n")
        
        valid_clean_filenames.add(clean_base_name)

    # Write to DESTINATION (The Fix)
    print(f"Writing clean file to: {DEST_TXT}")
    with open(DEST_TXT, 'w') as f:
        f.writelines(new_txt_lines)

    # --- PART 2: SYNC IMAGE FILES ---
    print("\n--- Syncing Image Directory ---")
    if os.path.exists(IMAGES_DIR):
        disk_files = sorted(os.listdir(IMAGES_DIR))
        renamed_count = 0
        moved_count = 0
        kept_count = 0
        
        for filename in disk_files:
            file_path = os.path.join(IMAGES_DIR, filename)
            if os.path.isdir(file_path): continue
            
            # Calculate what this file SHOULD be named
            clean_version = filename.replace(" ", "")
            
            if clean_version in valid_clean_filenames:
                # It belongs in the set
                if filename != clean_version:
                    # Needs rename
                    new_path = os.path.join(IMAGES_DIR, clean_version)
                    if not os.path.exists(new_path):
                        os.rename(file_path, new_path)
                        renamed_count += 1
                kept_count += 1
            else:
                # Does not belong
                print(f"Pruning unused: {filename}")
                shutil.move(file_path, os.path.join(REMOVED_DIR, filename))
                moved_count += 1
                
        print("\n--- Summary ---")
        print(f"Images in Text File: {num_images}")
        print(f"Images on Disk:      {kept_count}")
        print(f"Renamed:             {renamed_count}")
        print(f"Pruned:              {moved_count}")
        print(f"Dataset Ready at:    {BASE_DIR}")
    else:
        print(f"Error: Images directory not found at {IMAGES_DIR}")

if __name__ == "__main__":
    main()