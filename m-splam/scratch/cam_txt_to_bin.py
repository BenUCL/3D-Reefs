import struct
import os

# --- CONFIGURATION ---
# The folder where your cameras.txt is located
INPUT_DIR = "/home/ben/scratch/soneva_cam_txt" 
# The folder where you want the .bin output
OUTPUT_DIR = "/home/ben/scratch/soneva_cam_bin" 
# ---------------------

# COLMAP Model Name -> ID mapping
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3,
    "OPENCV": 4, "OPENCV_FISHEYE": 5, "FULL_OPENCV": 6, "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8, "RADIAL_FISHEYE": 9, "THIN_PRISM_FISHEYE": 10
}

def convert():
    input_path = os.path.join(INPUT_DIR, "cameras.txt")
    output_path = os.path.join(OUTPUT_DIR, "cameras.bin")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cameras = []

    # 1. Read Text File
    print(f"Reading {input_path}...")
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Parse line: ID MODEL WIDTH HEIGHT PARAMS...
            parts = line.split()
            cam_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            model_id = CAMERA_MODELS.get(model_name)
            if model_id is None:
                print(f"Error: Unknown camera model {model_name}")
                return

            cameras.append((cam_id, model_id, width, height, params))

    # 2. Write Binary File
    print(f"Writing {len(cameras)} cameras to {output_path}...")
    with open(output_path, "wb") as fid:
        # Write number of cameras (uint64)
        fid.write(struct.pack("<Q", len(cameras)))
        
        for cam in cameras:
            cam_id, model_id, width, height, params = cam
            
            # Write Camera ID (int32)
            fid.write(struct.pack("<i", cam_id))
            # Write Model ID (int32)
            fid.write(struct.pack("<i", model_id))
            # Write Width (uint64)
            fid.write(struct.pack("<Q", width))
            # Write Height (uint64)
            fid.write(struct.pack("<Q", height))
            
            # Write Parameters (doubles)
            for param in params:
                fid.write(struct.pack("<d", param))

    print("Done.")

if __name__ == "__main__":
    convert()