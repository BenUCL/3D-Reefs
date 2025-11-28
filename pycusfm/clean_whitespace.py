#TODO: In the future just remove whitespace from names of incoming images
# as very first step
import os
import shutil

# --- CONFIGURATION ---
BASE_DIR = "/home/ben/encode/data/intermediate_data/pycusfm1/for_splat"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
SPARSE_DIR = os.path.join(BASE_DIR, "sparse/0")
INPUT_TXT = os.path.join(SPARSE_DIR, "images.txt")
BACKUP_TXT = os.path.join(SPARSE_DIR, "images.txt.bak")
REMOVED_DIR = os.path.join(BASE_DIR, "removed")

def main():
    if not os.path.exists(INPUT_TXT):
        print(f"Error: Cannot find {INPUT_TXT}")
        return

    print(f"Processing: {BASE_DIR}")
    
    # 1. Setup Directories
    if not os.path.exists(REMOVED_DIR):
        os.makedirs(REMOVED_DIR, exist_ok=True)

    # 2. Backup (Only if fresh run, otherwise read existing)
    if not os.path.exists(BACKUP_TXT):
        shutil.copy(INPUT_TXT, BACKUP_TXT)
        print(f"Backed up original text file to: {BACKUP_TXT}")
    
    # 3. Read Data
    # We always read from the BACKUP to ensure we are processing the source of truth
    with open(BACKUP_TXT, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    valid_filenames = set() # Clean names we want to keep
    rename_map = {}         # Old Name on Disk -> New Name on Disk
    
    iterator = iter(lines)

    print("--- Step 1: Sanitizing images.txt ---")

    try:
        while True:
            line = next(iterator)
            
            # FIX HEADER (Dummy value 4000)
            if line.startswith("# Number of images"):
                parts = line.strip().split(":")
                if len(parts) < 3 or not parts[2].strip():
                    line = f"{parts[0]}: {parts[1].strip()}, mean observations per image: 4000\n"
                new_lines.append(line)
                continue
            
            if line.startswith("#") or not line.strip():
                new_lines.append(line)
                continue

            # PROCESS DATA LINE
            parts = line.split()
            
            # 1. Get the full path string from text file (e.g. "images/2019A GP Left (1).png")
            raw_path_in_txt = " ".join(parts[9:])
            
            # 2. Extract just the filename (e.g. "2019A GP Left (1).png")
            old_basename = os.path.basename(raw_path_in_txt)
            
            # 3. Create clean name (Remove spaces: "2019A_GP_Left(1).png")
            new_basename = old_basename.replace(" ", "")
            
            # 4. Reconstruct Line with ONLY the clean filename (No "images/" prefix)
            clean_parts = parts[:9] + [new_basename]
            new_lines.append(" ".join(clean_parts) + "\n")
            
            # Track this file
            valid_filenames.add(new_basename)
            if old_basename != new_basename:
                rename_map[old_basename] = new_basename
            
            # Points Line (Copy as is)
            points_line = next(iterator)
            new_lines.append(points_line)

    except StopIteration:
        pass

    # Write the fixed images.txt
    with open(INPUT_TXT, 'w') as f:
        f.writelines(new_lines)
    print(f"Successfully updated: {INPUT_TXT}")

    # 4. Rename Files on Disk
    print("\n--- Step 2: Renaming files on disk ---")
    renamed_count = 0
    
    # Handle map renames
    for old, new in rename_map.items():
        old_path = os.path.join(IMAGES_DIR, old)
        new_path = os.path.join(IMAGES_DIR, new)
        
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            renamed_count += 1
        elif os.path.exists(new_path):
            # Already renamed, ignore
            pass
        else:
            print(f"Warning: Expected image {old} not found on disk.")
    
    print(f"Renamed {renamed_count} files.")

    # 5. Cleanup (Pruning)
    print("\n--- Step 3: Pruning unused images ---")
    moved_files = []
    
    if os.path.exists(IMAGES_DIR):
        all_files_on_disk = os.listdir(IMAGES_DIR)
        
        for filename in all_files_on_disk:
            file_path = os.path.join(IMAGES_DIR, filename)
            
            if os.path.isdir(file_path): continue
            
            # If the file on disk (which might already be cleaned) is not in our valid list
            if filename not in valid_filenames:
                target_path = os.path.join(REMOVED_DIR, filename)
                shutil.move(file_path, target_path)
                moved_files.append(filename)

    # 6. Report
    print("\n" + "="*40)
    print("CLEANUP REPORT")
    print("="*40)
    print(f"Valid Images Retained: {len(valid_filenames)}")
    print(f"Images Pruned:         {len(moved_files)}")
    print("-" * 40)
    print(f"Clean dataset ready at: {BASE_DIR}")

if __name__ == "__main__":
    main()