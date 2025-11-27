import os
import json
import re
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
# 1. Use the INTERPOLATED IMAGES (The list of 274 frames)
INPUT_IMAGES_TXT = "/home/ben/encode/data/intermediate_data/pycusfm1/images_interpolated.txt"

# 2. Use the SCALED INTRINSICS (The file we created with the correct resolution and distortion)
INPUT_CAMERAS_TXT = "/home/ben/encode/data/intermediate_data/pycusfm1/cameras_5568x4872.txt"

# 3. Output Location
OUTPUT_JSON = "/home/ben/encode/data/intermediate_data/pycusfm1/frames_meta.json"

# --- HELPERS ---
def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def read_cameras(path):
    print(f"Reading Cameras from: {path}")
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            # Capture ALL parameters (Focals, Centers, AND Distortion)
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {"model": model, "w": width, "h": height, "params": params}
    return cameras

def read_images(path):
    print(f"Reading Images from: {path}")
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
        qvec = np.array([float(x) for x in parts[1:5]])
        tvec = np.array([float(x) for x in parts[5:8]])
        cam_id = int(parts[8])
        name = " ".join(parts[9:])
        
        images.append({
            "id": img_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": name
        })
        i += 2 
    return images

def main():
    cams = read_cameras(INPUT_CAMERAS_TXT)
    imgs = read_images(INPUT_IMAGES_TXT)
    
    # Sort to ensure sequential order
    imgs.sort(key=lambda x: natural_keys(x['name']))

    json_out = {
        "keyframes_metadata": [],
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_session_name": {},
        "camera_params_id_to_camera_params": {}
    }

    # 1. PROCESS CAMERAS (With Distortion!)
    for cam_id, cam in cams.items():
        str_id = str(cam_id)
        json_out["camera_params_id_to_session_name"][str_id] = "session_0"
        
        params = cam['params']
        # COLMAP OPENCV/PINHOLE format: fx, fy, cx, cy, k1, k2, p1, p2...
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        
        # Extract Distortion Coefficients (Everything after the first 4 params)
        if len(params) > 4:
            dist_coeffs = params[4:]
        else:
            dist_coeffs = [0.0, 0.0, 0.0, 0.0] # Fallback if actually Pinhole
            
        print(f"Camera {cam_id}: Distortion Coeffs found: {dist_coeffs}")
        
        k_matrix = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        p_matrix = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

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
                "camera_matrix": {"data": k_matrix},
                "distortion_coefficients": {"data": dist_coeffs}, # NOW POPULATED
                "projection_matrix": {"data": p_matrix}
            }
        }
        json_out["camera_params_id_to_camera_params"][str_id] = cam_entry

    # 2. PROCESS IMAGES
    for img in imgs:
        q_scipy = [img['qvec'][1], img['qvec'][2], img['qvec'][3], img['qvec'][0]]
        R_cw = R.from_quat(q_scipy).as_matrix()
        t_cw = img['tvec']
        
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        
        r_obj = R.from_matrix(R_wc)
        rot_vec = r_obj.as_rotvec()
        angle_rad = np.linalg.norm(rot_vec)
        
        if angle_rad < 1e-6:
            axis = [1,0,0]
            angle_deg = 0.0
        else:
            axis = rot_vec / angle_rad
            angle_deg = np.degrees(angle_rad)
            
        rel_path = os.path.join("images", img['name'])

        frame_entry = {
            "id": img['id'],
            "camera_params_id": str(img['camera_id']),
            "timestamp_microseconds": str(int(img['id']) * 500000), 
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
    print(f"DONE: Generated {OUTPUT_JSON} with {len(imgs)} frames.")

if __name__ == "__main__":
    main()