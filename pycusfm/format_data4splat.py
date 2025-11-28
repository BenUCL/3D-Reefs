import os
import shutil
import struct
import numpy as np

# --- CONFIGURATION ---
# Source (Text Files from PyCuSFM)
SRC_SPARSE_DIR = "/home/ben/encode/data/intermediate_data/pycusfm1/output/sparse"
SRC_IMAGES_TXT = os.path.join(SRC_SPARSE_DIR, "images.txt")
SRC_POINTS_TXT = os.path.join(SRC_SPARSE_DIR, "points3D.txt")

# Note: We use your CUSTOM cameras.txt (the one you rescaled/undistorted), NOT the PyCuSFM one.
# Assuming you put the correct cameras.txt here:
SRC_CAMERAS_TXT = "/home/ben/encode/data/intermediate_data/pycusfm1/for_splat/sparse/0/cameras.txt"

# Destination (Binary Files for Splatting)
BASE_DIR = "/home/ben/encode/data/intermediate_data/pycusfm1/for_splat"
DEST_SPARSE_DIR = os.path.join(BASE_DIR, "sparse/0")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
REMOVED_DIR = os.path.join(BASE_DIR, "removed")

def read_cameras_txt(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = (model, width, height, params)
    return cameras

def read_points3D_txt(path):
    points = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            point_id = int(parts[0])
            xyz = [float(x) for x in parts[1:4]]
            rgb = [int(x) for x in parts[4:7]]
            error = float(parts[7])
            track = [float(x) for x in parts[8:]] # Track list
            points[point_id] = (xyz, rgb, error, track)
    return points

def read_images_txt_and_clean(path):
    images = []
    with open(path, "r") as f:
        lines = f.readlines()
    
    # Skip headers
    data_start = 0
    for i, line in enumerate(lines):
        if not line.startswith("#") and line.strip():
            data_start = i
            break
            
    lines = [l.strip() for l in lines[data_start:] if l.strip()]
    
    for i in range(0, len(lines), 2):
        if i+1 >= len(lines): break
        
        pose_line = lines[i]
        points_line = lines[i+1]
        
        parts = pose_line.split()
        img_id = int(parts[0])
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]
        cam_id = int(parts[8])
        
        # CLEAN NAME: Remove directory prefix and spaces
        raw_name = " ".join(parts[9:])
        clean_name = os.path.basename(raw_name).replace(" ", "")
        
        # Points Line
        points_data = points_line.split()
        # Parse points data: [x, y, p3d_id, ...]
        points2D = []
        if points_data:
             # COLMAP points line is groups of 3: X, Y, POINT3D_ID
             # We need to keep them as floats/ints to write binary correctly
             points2D = [float(x) for x in points_data]

        images.append({
            "id": img_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": clean_name,
            "points": points2D
        })
        
    return images

# --- BINARY WRITERS ---
# COLMAP Binary Format Specs: https://colmap.github.io/format.html#binary-file-format

def write_cameras_bin(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(cameras)))
        for cam_id, (model_name, w, h, params) in cameras.items():
            # Model ID mapping (simplified for common models)
            # 0=SIMPLE_PINHOLE, 1=PINHOLE, 2=SIMPLE_RADIAL, 3=RADIAL, 4=OPENCV
            model_map = {"PINHOLE": 1, "OPENCV": 4, "SIMPLE_RADIAL": 2, "SIMPLE_PINHOLE": 0}
            model_id = model_map.get(model_name, 1) # Default to PINHOLE
            
            f.write(struct.pack("iiQQ", cam_id, model_id, w, h))
            for p in params:
                f.write(struct.pack("d", p))

def write_images_bin(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(images)))
        for img in images:
            f.write(struct.pack("I", img['id']))
            f.write(struct.pack("dddd", *img['qvec']))
            f.write(struct.pack("ddd", *img['tvec']))
            f.write(struct.pack("I", img['camera_id']))
            
            name_bytes = img['name'].encode("utf-8") + b"\x00"
            f.write(name_bytes)
            
            # Points2D
            # The list is flat: x, y, id, x, y, id...
            # Number of points = len / 3
            num_points = len(img['points']) // 3
            f.write(struct.pack("Q", num_points))
            
            for i in range(num_points):
                x = img['points'][3*i]
                y = img['points'][3*i+1]
                p3d_id = int(img['points'][3*i+2])
                f.write(struct.pack("ddQ", x, y, p3d_id))

def write_points3D_bin(points, path):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(points)))
        for p_id, (xyz, rgb, error, track) in points.items():
            f.write(struct.pack("Q", p_id))
            f.write(struct.pack("ddd", *xyz))
            f.write(struct.pack("BBB", *rgb))
            f.write(struct.pack("d", error))
            
            # Track list is [img_id, point2d_idx, img_id, point2d_idx...]
            track_len = len(track) // 2
            f.write(struct.pack("Q", track_len))
            for i in range(track_len):
                img_id = int(track[2*i])
                idx = int(track[2*i+1])
                f.write(struct.pack("II", img_id, idx))

def main():
    print(f"--- Converting to Binary ---")
    
    # 1. Load Data
    print(f"Reading Cameras from: {SRC_CAMERAS_TXT}")
    cameras = read_cameras_txt(SRC_CAMERAS_TXT)
    
    print(f"Reading Images from: {SRC_IMAGES_TXT}")
    images = read_images_txt_and_clean(SRC_IMAGES_TXT)
    
    print(f"Reading Points from: {SRC_POINTS_TXT}")
    points = read_points3D_txt(SRC_POINTS_TXT)
    
    # 2. Write Binary
    if not os.path.exists(DEST_SPARSE_DIR):
        os.makedirs(DEST_SPARSE_DIR)
        
    print(f"Writing binaries to: {DEST_SPARSE_DIR}")
    write_cameras_bin(cameras, os.path.join(DEST_SPARSE_DIR, "cameras.bin"))
    write_images_bin(images, os.path.join(DEST_SPARSE_DIR, "images.bin"))
    write_points3D_bin(points, os.path.join(DEST_SPARSE_DIR, "points3D.bin"))
    
    # 3. Sync Files
    print("Syncing Image Directory...")
    if not os.path.exists(REMOVED_DIR): os.makedirs(REMOVED_DIR)
    
    valid_names = set(img['name'] for img in images)
    files_on_disk = sorted(os.listdir(IMAGES_DIR))
    
    count_renamed = 0
    count_removed = 0
    
    for filename in files_on_disk:
        path = os.path.join(IMAGES_DIR, filename)
        if os.path.isdir(path): continue
        
        clean_name = filename.replace(" ", "")
        
        if clean_name in valid_names:
            if filename != clean_name:
                os.rename(path, os.path.join(IMAGES_DIR, clean_name))
                count_renamed += 1
        else:
            shutil.move(path, os.path.join(REMOVED_DIR, filename))
            count_removed += 1

    print("Done.")
    print(f"Images Renamed: {count_renamed}")
    print(f"Images Removed: {count_removed}")
    print("You can now run Splatting pointing to the binary files.")

if __name__ == "__main__":
    main()