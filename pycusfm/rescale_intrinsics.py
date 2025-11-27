import os
import cv2
import glob
import sys

# --- CONFIGURATION ---
INPUT_CAM_FILE = "/home/ben/encode/data/intermediate_data/pycusfm1/cameras_1600x1400.txt"
IMAGE_DIR = "/home/ben/encode/data/intermediate_data/pycusfm1/images"
OUTPUT_CAM_FILE = "/home/ben/encode/data/intermediate_data/pycusfm1/cameras_5568x4872_.txt"

def get_image_resolution(image_dir):
    """
    Scans all PNGs in the directory to find resolution.
    Ensures consistency across the dataset.
    """
    print(f"Scanning images in {image_dir}...")
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_files:
        print("Error: No .png images found in directory.")
        sys.exit(1)

    # Read the first image to set the baseline
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Error: Could not read {image_files[0]}")
        sys.exit(1)
        
    ref_h, ref_w = first_img.shape[:2]
    print(f"Reference Resolution found: {ref_w}x{ref_h}")

    # Optional: Verify all images match (Good for data hygiene)
    # We check first 10 to save time, or all if critical
    for img_path in image_files[:20]: 
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if w != ref_w or h != ref_h:
            print(f"Error: Resolution mismatch! {img_path} is {w}x{h}, expected {ref_w}x{ref_h}")
            sys.exit(1)
            
    return ref_w, ref_h

def scale_camera_file(input_path, output_path, target_w, target_h):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    output_lines = []

    for line in lines:
        stripped = line.strip()
        
        # Pass comments through
        if stripped.startswith("#") or not stripped:
            output_lines.append(line)
            continue

        # Parse Data Line
        # Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
        parts = stripped.split()
        
        cam_id = parts[0]
        model = parts[1]
        old_w = int(parts[2])
        old_h = int(parts[3])
        params = [float(p) for p in parts[4:]]

        # Calculate Scale Factors
        scale_x = target_w / old_w
        scale_y = target_h / old_h
        
        print(f"Scaling Camera {cam_id} ({model}):")
        print(f"  Old: {old_w}x{old_h}")
        print(f"  New: {target_w}x{target_h}")
        print(f"  Scale Factors: X={scale_x:.4f}, Y={scale_y:.4f}")

        # --- SCALING LOGIC BASED ON MODEL ---
        # COLMAP 'OPENCV' Model Params: fx, fy, cx, cy, k1, k2, p1, p2
        # COLMAP 'PINHOLE' Model Params: fx, fy, cx, cy
        
        new_params = []
        
        if model in ["OPENCV", "PINHOLE", "FULL_OPENCV"]:
            # 1. Scale Focal Lengths (fx, fy)
            fx = params[0] * scale_x
            fy = params[1] * scale_y
            
            # 2. Scale Principal Points (cx, cy)
            cx = params[2] * scale_x
            cy = params[3] * scale_y
            
            new_params = [fx, fy, cx, cy]
            
            # 3. Copy Distortion Coefficients EXACTLY (Do not scale!)
            # Append the rest of the params (k1, k2, etc...)
            new_params.extend(params[4:])
            
        elif model == "SIMPLE_RADIAL":
            # Params: f, cx, cy, k
            f = params[0] * scale_x # Assuming square pixels usually
            cx = params[1] * scale_x
            cy = params[2] * scale_y
            k = params[3] # Distortion
            new_params = [f, cx, cy, k]
            
        else:
            print(f"Warning: Model {model} logic not explicitly defined. Assuming first 4 params are f/c and rest are distortion.")
            # Generic fallback for standard 4+ param models
            new_params = [
                params[0] * scale_x, 
                params[1] * scale_y, 
                params[2] * scale_x, 
                params[3] * scale_y
            ] + params[4:]

        # Reconstruct Line
        # Join params with spaces
        param_str = " ".join([str(p) for p in new_params])
        new_line = f"{cam_id} {model} {target_w} {target_h} {param_str}\n"
        output_lines.append(new_line)

    # Write Output
    with open(output_path, 'w') as f:
        f.writelines(output_lines)
    
    print(f"Successfully wrote scaled intrinsics to: {output_path}")

if __name__ == "__main__":
    # 1. Detect Target Resolution
    target_w, target_h = get_image_resolution(IMAGE_DIR)
    
    # 2. Process File
    scale_camera_file(INPUT_CAM_FILE, OUTPUT_CAM_FILE, target_w, target_h)