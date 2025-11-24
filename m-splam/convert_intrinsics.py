# COLMAP struggles to calculate intrinsics for high-res images, but has no issue on low res.
# What we can do is take the intrinsics files form the low-res and convert them what these would
# be for the high-res by scaling them.


import os
import glob
import struct
import argparse
from PIL import Image
from pathlib import Path

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
    # Map model_id to (model_name, num_params) - COLMAP convention
    model_info = {
        0: ('SIMPLE_PINHOLE', 3),      # fx, cx, cy
        1: ('PINHOLE', 4),              # fx, fy, cx, cy
        2: ('SIMPLE_RADIAL', 4),        # fx, cx, cy, k1
        3: ('RADIAL', 5),               # fx, cx, cy, k1, k2
        4: ('OPENCV', 8),               # fx, fy, cx, cy, k1, k2, p1, p2
        5: ('OPENCV_FISHEYE', 8),       # fx, fy, cx, cy, k1, k2, k3, k4
        6: ('FULL_OPENCV', 12),         # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        7: ('FOV', 5),                  # fx, fy, cx, cy, omega
        8: ('SIMPLE_RADIAL_FISHEYE', 4),
        9: ('RADIAL_FISHEYE', 5),
        10: ('THIN_PRISM_FISHEYE', 12)
    }
    
    cameras = []
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for i in range(num_cameras):
            # Read camera properties: camera_id (int), model_id (int), width (uint64), height (uint64)
            # Format: iiQQ = 4 + 4 + 8 + 8 = 24 bytes
            camera_id, model_id, width, height = struct.unpack('<iiQQ', f.read(24))
            
            # Infer num_params from model_id (NOT stored in binary format)
            if model_id not in model_info:
                raise ValueError(f"Unknown model_id={model_id} for camera {i+1}")
            
            model_name, num_params = model_info[model_id]
            params = struct.unpack(f'<{num_params}d', f.read(8 * num_params))
            
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
        f.write(struct.pack('<Q', len(cameras)))
        for cam in cameras:
            # Write camera properties: camera_id (int), model_id (int), width (uint64), height (uint64)
            # Format: iiQQ = 24 bytes (num_params is NOT stored, it's inferred from model_id)
            f.write(struct.pack('<iiQQ', cam['camera_id'], cam['model_id'], 
                              cam['width'], cam['height']))
            # Write params as doubles
            f.write(struct.pack(f'<{len(cam["params"])}d', *cam['params']))

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

def process_intrinsics(highres_images_dir, lowres_images_dir, intrinsics_dir):
    # 1. Get Resolutions and Calculate Scale
    w_hr, h_hr = get_resolution(highres_images_dir)
    w_lr, h_lr = get_resolution(lowres_images_dir)
    
    scale_x = w_hr / w_lr
    scale_y = h_hr / h_lr
    
    print(f"High Res: {w_hr}x{h_hr}")
    print(f"Low Res:  {w_lr}x{h_lr}")
    print(f"Scaling:  x={scale_x:.4f}, y={scale_y:.4f}")

    # 2. Setup paths
    input_dir = Path(intrinsics_dir)
    
    cameras_txt = input_dir / 'cameras.txt'
    cameras_bin = input_dir / 'cameras.bin'
    backup_txt = input_dir / 'cameras_lowres.txt'
    backup_bin = input_dir / 'cameras_lowres.bin'
    
    # 3. Process cameras.txt if it exists
    if cameras_txt.exists():
        print(f"\n📄 Processing {cameras_txt}...")
        
        # Backup original
        import shutil
        shutil.copy2(cameras_txt, backup_txt)
        print(f"📋 Backed up to: {backup_txt}")
        
        # Read, scale, and write back in place
        scaled_lines = []
        with open(cameras_txt, 'r') as f_in:
            for line in f_in:
                line_stripped = line.strip()
                
                # Copy comments directly
                if not line_stripped or line_stripped.startswith('#'):
                    scaled_lines.append(line)
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
                scaled_lines.append(new_line)
        
        # Write scaled version
        with open(cameras_txt, 'w') as f_out:
            f_out.writelines(scaled_lines)
        
        print(f"✅ Scaled version saved to: {cameras_txt}")
    else:
        print(f"⚠️  {cameras_txt} not found, skipping")
    
    # 4. Process cameras.bin if it exists
    if cameras_bin.exists():
        print(f"\n📦 Processing {cameras_bin}...")
        
        # Backup original
        import shutil
        shutil.copy2(cameras_bin, backup_bin)
        print(f"📋 Backed up to: {backup_bin}")
        
        # Read and scale binary file
        cameras = read_cameras_binary(cameras_bin)
        for cam in cameras:
            cam['width'] = w_hr
            cam['height'] = h_hr
            cam['params'] = scale_camera_params(cam['params'], scale_x, scale_y)
        
        # Write scaled version back
        write_cameras_binary(cameras_bin, cameras)
        print(f"✅ Scaled version saved to: {cameras_bin}")
    else:
        print(f"⚠️  {cameras_bin} not found, skipping")
    
    print(f"\n✅ All conversions complete!")
    print(f"   Original low-res files backed up with '_lowres' suffix")
    print(f"   Scaled high-res files now in: {input_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Scale camera intrinsics from low-res to high-res resolution"
    )
    parser.add_argument(
        '--highres-images',
        type=str,
        required=True,
        help='Directory containing high-resolution images'
    )
    parser.add_argument(
        '--lowres-images',
        type=str,
        required=True,
        help='Directory containing low-resolution images'
    )
    parser.add_argument(
        '--intrinsics-dir',
        type=str,
        required=True,
        help='Directory containing cameras.txt and cameras.bin to scale (modified in place)'
    )
    
    args = parser.parse_args()
    
    process_intrinsics(
        args.highres_images,
        args.lowres_images,
        args.intrinsics_dir
    )

if __name__ == "__main__":
    main()