# COLMAP struggles to calculate intrinsics for high-res images, but has no issue on low res.
# What we can do is take the intrinsics files form the low-res and convert them what these would
# be for the high-res by scaling them.


import os
import glob
import struct
from PIL import Image
from pathlib import Path

# ================= CONFIGURATION =================
HIGH_RES_IMAGES_DIR = "/home/ben/encode/data/mars_johns/left"
LOW_RES_IMAGES_DIR  = "/home/ben/encode/data/mars_johns/left_downsampled_png"
# Path to directory containing cameras.txt and cameras.bin
INPUT_INTRINSICS    = "/home/ben/encode/data/intermediate_data/highres_m-slam_MarsJohnS/colmap_outputs"
# Output directory (will be created if needed)
OUTPUT_INTRINSICS   = "/home/ben/encode/data/intermediate_data/highres_m-slam_MarsJohnS/highres_intrinsics"
# =================================================

def get_resolution(directory):
    """Finds the first image in a directory and returns (width, height)."""
    # Search for common extensions
    types = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    files = []
    for t in types:
        files.extend(glob.glob(os.path.join(directory, t)))
    
    if not files:
        raise FileNotFoundError(f"No images found in {directory}")
    
    with Image.open(files[0]) as img:
        return img.size # returns (width, height)

def read_cameras_binary(path):
    """Read COLMAP cameras.bin file and return list of camera dicts."""
    cameras = []
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('I', f.read(4))[0]
            model_id = struct.unpack('i', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]
            num_params = struct.unpack('Q', f.read(8))[0]
            params = struct.unpack(f'{num_params}d', f.read(8 * num_params))
            
            # Map model_id to model_name (COLMAP convention)
            model_names = {0: 'SIMPLE_PINHOLE', 1: 'PINHOLE', 2: 'SIMPLE_RADIAL',
                          3: 'RADIAL', 4: 'OPENCV', 5: 'OPENCV_FISHEYE',
                          6: 'FULL_OPENCV', 7: 'FOV', 8: 'SIMPLE_RADIAL_FISHEYE',
                          9: 'RADIAL_FISHEYE', 10: 'THIN_PRISM_FISHEYE'}
            model_name = model_names.get(model_id, 'UNKNOWN')
            
            cameras.append({
                'camera_id': camera_id,
                'model_id': model_id,
                'model_name': model_name,
                'width': width,
                'height': height,
                'params': list(params)
            })
    return cameras

def write_cameras_binary(path, cameras):
    """Write COLMAP cameras.bin file."""
    with open(path, 'wb') as f:
        f.write(struct.pack('Q', len(cameras)))
        for cam in cameras:
            f.write(struct.pack('I', cam['camera_id']))
            f.write(struct.pack('i', cam['model_id']))
            f.write(struct.pack('Q', cam['width']))
            f.write(struct.pack('Q', cam['height']))
            f.write(struct.pack('Q', len(cam['params'])))
            f.write(struct.pack(f'{len(cam["params"])}d', *cam['params']))

def scale_camera_params(params, scale_x, scale_y):
    """Scale camera intrinsic parameters (fx, fy, cx, cy are first 4 params)."""
    scaled_params = params.copy()
    if len(scaled_params) >= 4:
        scaled_params[0] *= scale_x  # fx
        scaled_params[1] *= scale_y  # fy
        scaled_params[2] *= scale_x  # cx
        scaled_params[3] *= scale_y  # cy
    # Distortion parameters (index 4+) remain unchanged
    return scaled_params

def process_intrinsics():
    # 1. Get Resolutions and Calculate Scale
    w_hr, h_hr = get_resolution(HIGH_RES_IMAGES_DIR)
    w_lr, h_lr = get_resolution(LOW_RES_IMAGES_DIR)
    
    scale_x = w_hr / w_lr
    scale_y = h_hr / h_lr
    
    print(f"High Res: {w_hr}x{h_hr}")
    print(f"Low Res:  {w_lr}x{h_lr}")
    print(f"Scaling:  x={scale_x:.4f}, y={scale_y:.4f}")

    # 2. Setup paths
    input_dir = Path(INPUT_INTRINSICS)
    output_dir = Path(OUTPUT_INTRINSICS)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_txt = input_dir / 'cameras.txt'
    input_bin = input_dir / 'cameras.bin'
    output_txt = output_dir / 'cameras.txt'
    output_bin = output_dir / 'cameras.bin'
    
    # 3. Process cameras.txt if it exists
    if input_txt.exists():
        print(f"\n📄 Processing {input_txt}...")
        with open(input_txt, 'r') as f_in, open(output_txt, 'w') as f_out:
            for line in f_in:
                line_stripped = line.strip()
                
                # Copy comments directly
                if not line_stripped or line_stripped.startswith('#'):
                    f_out.write(line)
                    continue
                
                parts = line_stripped.split()
                
                # COLMAP FORMAT: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
                cam_id = parts[0]
                model = parts[1]
                
                # Scale parameters
                params = [float(p) for p in parts[4:]]
                params = scale_camera_params(params, scale_x, scale_y)
                
                # Reconstruct line with high-res dimensions
                params_str = " ".join(f"{p:.15f}" for p in params)
                new_line = f"{cam_id} {model} {w_hr} {h_hr} {params_str}\n"
                f_out.write(new_line)
        
        print(f"✅ Saved: {output_txt}")
    else:
        print(f"⚠️  {input_txt} not found, skipping")
    
    # 4. Process cameras.bin if it exists
    if input_bin.exists():
        print(f"\n📦 Processing {input_bin}...")
        cameras = read_cameras_binary(input_bin)
        
        # Scale each camera
        for cam in cameras:
            cam['width'] = w_hr
            cam['height'] = h_hr
            cam['params'] = scale_camera_params(cam['params'], scale_x, scale_y)
        
        write_cameras_binary(output_bin, cameras)
        print(f"✅ Saved: {output_bin}")
    else:
        print(f"⚠️  {input_bin} not found, skipping")
    
    print(f"\n✅ All conversions complete! Output directory: {output_dir}")

if __name__ == "__main__":
    process_intrinsics()