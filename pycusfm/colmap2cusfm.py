""""
Converts COLMAP formatted outputs form MASt3R-SLAM to PyCuSFM JSON format.
Expected inputs:
  - sparse/0/cameras.txt
  - sparse/0/images.txt
  - images/ (image files)
Outputs:
    - frames_meta.json (PyCuSFM format)
"""

import os
import json
import re
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
INPUT_SPARSE_DIR = "/home/ben/encode/data/intermediate_data/fix_intrinsics5/for_splat/sparse/0"
INPUT_IMAGES_DIR = "/home/ben/encode/data/intermediate_data/pycusfm1/images"
OUTPUT_JSON = "/home/ben/encode/data/intermediate_data/pycusfm1/frames_meta.json"

# --- HELPERS ---
def read_cameras(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            # PARAMS depend on model. PINHOLE = f, cx, cy (or fx, fy, cx, cy)
            # Your example: 1 PINHOLE 5568 4872 3289.68... 3289.10... 2783.5 2435.5
            # That looks like: fx, fy, cx, cy
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {"model": model, "w": width, "h": height, "params": params}
    return cameras

def read_images(path):
    images = []
    with open(path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or not line:
            i += 1
            continue
        parts = line.split()
        img_id = parts[0]
        # COLMAP Image format: QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        qvec = np.array([float(x) for x in parts[1:5]]) # w, x, y, z
        tvec = np.array([float(x) for x in parts[5:8]])
        cam_id = int(parts[8])
        
        # Image name may contain spaces, so join the rest
        name = " ".join(parts[9:])
        
        images.append({
            "id": img_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": name
        })
        i += 2 # Skip points line
    return images

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def main():
    print(f"Reading data from {INPUT_SPARSE_DIR}...")
    cams = read_cameras(os.path.join(INPUT_SPARSE_DIR, "cameras.txt"))
    imgs = read_images(os.path.join(INPUT_SPARSE_DIR, "images.txt"))
    
    # CRITICAL FIX: Natural Sort ensures (1), (2), (3)... instead of (1), (10), (100)
    imgs.sort(key=lambda x: natural_keys(x['name']))

    json_out = {
        "keyframes_metadata": [],
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_session_name": {},
        "camera_params_id_to_camera_params": {}
    }

    # 1. PROCESS CAMERAS
    for cam_id, cam in cams.items():
        str_id = str(cam_id)
        json_out["camera_params_id_to_session_name"][str_id] = "session_0"
        
        p = cam['params']
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        
        # Build 3x3 Intrinsic Matrix (K)
        k_matrix = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        
        # Build 3x4 Projection Matrix (P = K[I|0])
        # [fx, 0, cx, 0]
        # [0, fy, cy, 0]
        # [0,  0,  1, 0]
        p_matrix = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        
        # Handle Distortion (Fill with 0.0 if empty to be safe)
        dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0] 

        cam_entry = {
            "sensor_meta_data": {
                "sensor_id": cam_id,
                "sensor_type": "CAMERA",
                "sensor_name": "gopro_left",
                "frequency": 2.0,
                "sensor_to_vehicle_transform": {
                    "axis_angle": {"x":0.0, "y":0.0, "z":0.0, "angle_degrees":0.0},
                    "translation": {"x":0.0, "y":0.0, "z":0.0}
                }
            },
            "calibration_parameters": {
                "image_width": cam['w'],
                "image_height": cam['h'],
                "camera_matrix": {
                    "data": k_matrix
                },
                "distortion_coefficients": {
                    "data": dist_coeffs
                },
                "projection_matrix": {
                    "data": p_matrix
                }
            }
        }
        json_out["camera_params_id_to_camera_params"][str_id] = cam_entry

    # 2. PROCESS IMAGES (POSES)
    for img in imgs:
        # COLMAP = World-to-Camera (T_cw)
        # PyCuSFM = Camera-to-World (T_wc)
        
        # Quaternion (w,x,y,z) -> Rotation Matrix
        # Scipy uses (x,y,z,w)
        q_scipy = [img['qvec'][1], img['qvec'][2], img['qvec'][3], img['qvec'][0]]
        R_cw = R.from_quat(q_scipy).as_matrix()
        t_cw = img['tvec']
        
        # Invert to get T_wc
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        
        # Convert to Axis-Angle
        r_obj = R.from_matrix(R_wc)
        rot_vec = r_obj.as_rotvec()
        angle_rad = np.linalg.norm(rot_vec)
        
        if angle_rad < 1e-6:
            axis = [1,0,0]
            angle_deg = 0.0
        else:
            axis = rot_vec / angle_rad
            angle_deg = np.degrees(angle_rad)
            
        # Image path must be relative to the JSON file
        rel_path = os.path.join("images", img['name'])

        frame_entry = {
            "id": img['id'],
            "camera_params_id": str(img['camera_id']),
            "timestamp_microseconds": str(int(img['id']) * 500000), # Fake timestamps (2fps = 0.5s)
            "image_name": rel_path,
            "camera_to_world": {
                "axis_angle": {
                    "x": axis[0], "y": axis[1], "z": axis[2],
                    "angle_degrees": angle_deg
                },
                "translation": {
                    "x": t_wc[0], "y": t_wc[1], "z": t_wc[2]
                }
            }
        }
        json_out["keyframes_metadata"].append(frame_entry)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"DONE: Generated {OUTPUT_JSON}")

if __name__ == "__main__":
    main()