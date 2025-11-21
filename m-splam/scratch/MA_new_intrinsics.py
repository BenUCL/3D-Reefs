import os
import struct
import shutil

# ===============================================================================
# CONFIGURATION
# ===============================================================================

# Path to your high-quality single camera text file (from the standard COLMAP run)
HQ_CAMERAS_TXT = "/home/ben/encode/data/intermediate_data/reef_soneva/colmap_outputs/cameras.txt"

# Path to the MapAnything sparse folder (containing the '0' subfolder)
# The script will look for: TARGET_SPARSE_DIR/0/images.bin
TARGET_SPARSE_DIR = "/home/ben/encode/data/intermediate_data/MA_soneva/for_splat/sparse"

# ===============================================================================
# LOGIC
# ===============================================================================

# COLMAP Camera Model ID map
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3,
    "OPENCV": 4, "OPENCV_FISHEYE": 5, "FULL_OPENCV": 6, "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8, "RADIAL_FISHEYE": 9, "THIN_PRISM_FISHEYE": 10
}

def read_next_bytes(fid, num_bytes, format_char_sequence):
    """Helper to read and unpack bytes."""
    data = fid.read(num_bytes)
    return struct.unpack(format_char_sequence, data)

def read_string(fid):
    """Read a zero-terminated string from the binary stream."""
    chars = []
    while True:
        c = fid.read(1)
        if c == b'\x00':
            return b"".join(chars).decode("utf-8")
        chars.append(c)

def main():
    # 1. Define paths
    src_model_path = os.path.join(TARGET_SPARSE_DIR, "0")
    images_bin_path = os.path.join(src_model_path, "images.bin")
    cameras_bin_out = os.path.join(src_model_path, "cameras.bin")
    
    # 2. Create Backup
    backup_dir = os.path.join(os.path.dirname(TARGET_SPARSE_DIR), "MA_sparse")
    if os.path.exists(backup_dir):
        print(f"Warning: Backup dir {backup_dir} already exists. Skipping backup to avoid overwriting previous backup.")
    else:
        print(f"Creating backup of '{TARGET_SPARSE_DIR}' to '{backup_dir}'...")
        shutil.copytree(TARGET_SPARSE_DIR, backup_dir)
        print("Backup complete.")

    # 3. Parse the HQ Camera TXT to get the Target Camera ID and Params
    print(f"Reading High-Quality Camera from: {HQ_CAMERAS_TXT}")
    target_cam_id = 1 # Default
    cam_data = None
    
    with open(HQ_CAMERAS_TXT, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # Parse: ID MODEL WIDTH HEIGHT PARAMS...
            c_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            model_id = CAMERA_MODELS.get(model_name)
            if model_id is None:
                raise ValueError(f"Unknown camera model: {model_name}")
            
            # Store data for binary writing
            cam_data = (c_id, model_id, width, height, params)
            target_cam_id = c_id
            print(f"Loaded Camera ID: {target_cam_id} | Model: {model_name}")
            break # Assume only one camera in the file
            
    if not cam_data:
        raise ValueError("Could not find camera data in text file.")

    # 4. Write cameras.bin
    print(f"Writing new {cameras_bin_out}...")
    with open(cameras_bin_out, "wb") as fid:
        # Write num_cameras (1)
        fid.write(struct.pack("<Q", 1))
        # Write camera data
        cid, mid, w, h, p = cam_data
        fid.write(struct.pack("<i", cid))
        fid.write(struct.pack("<i", mid))
        fid.write(struct.pack("<Q", w))
        fid.write(struct.pack("<Q", h))
        for param in p:
            fid.write(struct.pack("<d", param))

    # 5. Patch images.bin
    # We need to read the existing bin, change the camera_id field, and write to a buffer
    print(f"Patching {images_bin_path} to point all images to Camera ID {target_cam_id}...")
    
    # We will write to a temporary file first
    temp_images_bin = images_bin_path + ".temp"
    
    with open(images_bin_path, "rb") as fin, open(temp_images_bin, "wb") as fout:
        # Read/Write Header (Number of images)
        num_reg_images = struct.unpack("<Q", fin.read(8))[0]
        fout.write(struct.pack("<Q", num_reg_images))
        
        print(f"Processing {num_reg_images} images...")
        
        for _ in range(num_reg_images):
            # --- READ ---
            binary_image_id = struct.unpack("<I", fin.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", fin.read(32))
            tx, ty, tz = struct.unpack("<ddd", fin.read(24))
            old_camera_id = struct.unpack("<I", fin.read(4))[0]
            
            # Read Name (char by char)
            name_bytes = b""
            while True:
                char = fin.read(1)
                name_bytes += char
                if char == b'\x00':
                    break
            
            # Read Points
            num_points2D = struct.unpack("<Q", fin.read(8))[0]
            # Each point is (x, y, p3d_id) -> 2 doubles + 1 uint64 = 16 + 8 = 24 bytes
            points_data = fin.read(num_points2D * 24)
            
            # --- WRITE ---
            fout.write(struct.pack("<I", binary_image_id))
            fout.write(struct.pack("<dddd", qw, qx, qy, qz))
            fout.write(struct.pack("<ddd", tx, ty, tz))
            
            # ** THE FIX: Write the Target Camera ID instead of the old one **
            fout.write(struct.pack("<I", target_cam_id))
            
            fout.write(name_bytes)
            fout.write(struct.pack("<Q", num_points2D))
            fout.write(points_data)

    # Replace original images.bin with the patched one
    os.replace(temp_images_bin, images_bin_path)
    print("✅ Success! images.bin patched and cameras.bin replaced.")
    print(f"Original data backed up to: {backup_dir}")

if __name__ == "__main__":
    main()