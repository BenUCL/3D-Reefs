import os
import re
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# --- CONFIGURATION ---
EXISTING_SPARSE_DIR = "/home/ben/encode/data/intermediate_data/fix_intrinsics5/for_splat/sparse/0"
ALL_IMAGES_DIR = "/home/ben/encode/data/intermediate_data/pycusfm1/images"

# Outputs
OUTPUT_TXT_PATH = "/home/ben/encode/data/intermediate_data/pycusfm1/images_interpolated.txt"
OUTPUT_BIN_PATH = "/home/ben/encode/data/intermediate_data/pycusfm1/images_interpolated.bin"

def natural_keys(text):
    """Sorts strings with embedded numbers naturally (1, 2, ... 10)."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def get_image_index(filename):
    """Extracts '1' from '2019A_GP_Left (1).png'."""
    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))
    return -1

def read_images_txt(path):
    """Reads COLMAP images.txt into a dictionary keyed by image index."""
    images = {} 
    
    with open(path, "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or not line:
            i += 1
            continue
        
        parts = line.split()
        img_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]]) # w, x, y, z
        tvec = np.array([float(x) for x in parts[5:8]])
        cam_id = int(parts[8])
        name = " ".join(parts[9:])
        
        idx = get_image_index(name)
        if idx != -1:
            images[idx] = {
                "qvec": qvec, 
                "tvec": tvec, 
                "camera_id": cam_id, 
                "name": name, 
                "colmap_id": img_id
            }
        
        i += 2 # Skip points line
    return images

def interpolate_missing(known_poses, all_filenames):
    interpolated_images = []
    
    # Sort filenames by index
    all_filenames.sort(key=natural_keys)
    
    # Get sorted list of indices that we HAVE poses for
    known_indices = sorted(known_poses.keys())
    
    if not known_indices:
        print("Error: No known poses found via regex matching!")
        return []

    print(f"Found {len(known_indices)} keyframes. Interpolating to cover {len(all_filenames)} total images...")

    ref_cam_id = known_poses[known_indices[0]]["camera_id"]

    for i in range(len(all_filenames)):
        current_filename = all_filenames[i]
        current_idx = get_image_index(current_filename)
        
        # Case 1: Exact match
        if current_idx in known_poses:
            pose = known_poses[current_idx]
            interpolated_images.append({
                "id": i + 1,
                "qvec": pose["qvec"],
                "tvec": pose["tvec"],
                "camera_id": pose["camera_id"],
                "name": pose["name"]
            })
            continue

        # Case 2: Interpolate
        prev_idx = -1
        next_idx = -1
        
        for k in reversed(known_indices):
            if k < current_idx:
                prev_idx = k
                break
        
        for k in known_indices:
            if k > current_idx:
                next_idx = k
                break
                
        if prev_idx == -1: # Clamp to start
            target = known_poses[next_idx]
            interpolated_images.append({
                "id": i + 1, "qvec": target["qvec"], "tvec": target["tvec"], 
                "camera_id": ref_cam_id, "name": current_filename
            })
            continue
            
        if next_idx == -1: # Clamp to end
            target = known_poses[prev_idx]
            interpolated_images.append({
                "id": i + 1, "qvec": target["qvec"], "tvec": target["tvec"], 
                "camera_id": ref_cam_id, "name": current_filename
            })
            continue

        # Math
        pose_a = known_poses[prev_idx]
        pose_b = known_poses[next_idx]
        
        total_dist = next_idx - prev_idx
        curr_dist = current_idx - prev_idx
        alpha = curr_dist / float(total_dist)
        
        t_interp = (1 - alpha) * pose_a["tvec"] + alpha * pose_b["tvec"]
        
        qa = [pose_a["qvec"][1], pose_a["qvec"][2], pose_a["qvec"][3], pose_a["qvec"][0]]
        qb = [pose_b["qvec"][1], pose_b["qvec"][2], pose_b["qvec"][3], pose_b["qvec"][0]]
        
        key_rots = R.from_quat([qa, qb])
        slerp = Slerp([0, 1], key_rots)
        q_interp_scipy = slerp([alpha])[0].as_quat()
        
        q_interp = np.array([q_interp_scipy[3], q_interp_scipy[0], q_interp_scipy[1], q_interp_scipy[2]])
        
        interpolated_images.append({
            "id": i + 1,
            "qvec": q_interp,
            "tvec": t_interp,
            "camera_id": ref_cam_id,
            "name": current_filename
        })

    return interpolated_images

def write_images_txt(images, path):
    print(f"Writing TXT: {path}")
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        
        for img in images:
            q = img['qvec']
            t = img['tvec']
            line = f"{img['id']} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {img['camera_id']} {img['name']}\n"
            f.write(line)
            f.write("\n")

def write_images_bin(images, path):
    print(f"Writing BIN: {path}")
    with open(path, "wb") as f:
        # 1. Number of images (uint64)
        f.write(struct.pack("Q", len(images)))
        
        for img in images:
            # 2. Image ID (uint32)
            f.write(struct.pack("I", img['id']))
            
            # 3. Qvec (4 doubles)
            q = img['qvec']
            f.write(struct.pack("dddd", q[0], q[1], q[2], q[3]))
            
            # 4. Tvec (3 doubles)
            t = img['tvec']
            f.write(struct.pack("ddd", t[0], t[1], t[2]))
            
            # 5. Camera ID (uint32)
            f.write(struct.pack("I", img['camera_id']))
            
            # 6. Name (Null-terminated string)
            name_bytes = img['name'].encode("utf-8") + b"\x00"
            f.write(name_bytes)
            
            # 7. Number of 2D points (uint64) - We have 0 for priors
            f.write(struct.pack("Q", 0))

def main():
    print("--- 1. Loading Existing Keyframes ---")
    known_poses = read_images_txt(os.path.join(EXISTING_SPARSE_DIR, "images.txt"))
    
    print("--- 2. Scanning Full Image Set ---")
    all_files = [f for f in os.listdir(ALL_IMAGES_DIR) if f.endswith(".png")]
    
    print("--- 3. Interpolating ---")
    final_list = interpolate_missing(known_poses, all_files)
    
    print("--- 4. Saving Outputs ---")
    write_images_txt(final_list, OUTPUT_TXT_PATH)
    write_images_bin(final_list, OUTPUT_BIN_PATH)
    
    print("Done.")

if __name__ == "__main__":
    main()