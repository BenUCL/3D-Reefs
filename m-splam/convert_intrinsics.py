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
    cameras = []
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for i in range(num_cameras):
            camera_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            num_params = struct.unpack('<Q', f.read(8))[0]  # Unsigned, matching write format
            
            # Validate num_params is reasonable
            if num_params > 20:
                raise ValueError(f"Invalid num_params={num_params} for camera {i+1}. File may be corrupted.")
            
            params = struct.unpack(f'<{num_params}d', f.read(8 * num_params))
            
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
        f.write(struct.pack('<Q', len(cameras)))
        for cam in cameras:
            f.write(struct.pack('<I', cam['camera_id']))
            f.write(struct.pack('<i', cam['model_id']))
            f.write(struct.pack('<Q', cam['width']))
            f.write(struct.pack('<Q', cam['height']))
            f.write(struct.pack('<Q', len(cam['params'])))
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