import argparse
import cv2
import numpy as np
import yaml
import PIL.Image
from pathlib import Path

# --- Helper from mast3r_utils.py ---
def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)

# --- Logic to replicate M-SLAM crop ---
def mslam_crop_logic(pil_img, size=512):
    W, H = pil_img.size
    cx, cy = W // 2, H // 2
    
    # Logic from mast3r_utils.py:resize_img
    # Ensures dimensions are multiples of 16
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    
    # Calculate crop box (left, upper, right, lower)
    box = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
    img_cropped = pil_img.crop(box)
    return img_cropped, box

# --- Logic to replicate M-SLAM Undistortion ---
def get_undistort_map(w, h, calib_data):
    fx, fy, cx, cy = calib_data['calibration'][:4]
    distortion = np.array(calib_data['calibration'][4:]) if len(calib_data['calibration']) > 4 else np.zeros(4)
    
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    
    # M-SLAM default config uses centerPrincipalPoint=True and alpha=0
    # Alpha=0 means "crop to valid pixels" (Zoom in to remove black borders)
    K_opt, _ = cv2.getOptimalNewCameraMatrix(
        K, distortion, (w, h), 0, (w, h), centerPrincipalPoint=True
    )
    
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, distortion, None, K_opt, (w, h), cv2.CV_32FC1
    )
    return mapx, mapy

def main():
    parser = argparse.ArgumentParser(description="Replicate M-SLAM Keyframe Generation")
    parser.add_argument("--image", required=True, help="Path to 1600x1400 Input Image")
    parser.add_argument("--intrinsics", required=True, help="Path to intrinsics.yaml")
    parser.add_argument("--output_dir", default="verification_output", help="Where to save results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load Image (OpenCV BGR -> RGB)
    print(f"Loading {args.image}...")
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_raw, w_raw = img_rgb.shape[:2]
    print(f"Input Dimensions: {w_raw}x{h_raw}")

    # 2. Undistort (Critical Step!)
    print("Applying Undistortion (cv2.remap)...")
    with open(args.intrinsics, 'r') as f:
        calib = yaml.safe_load(f)
    
    mapx, mapy = get_undistort_map(w_raw, h_raw, calib)
    img_undistorted = cv2.remap(img_rgb, mapx, mapy, cv2.INTER_LINEAR)
    
    # Save undistorted intermediate
    cv2.imwrite(str(out_dir / "step1_undistorted_1600.jpg"), cv2.cvtColor(img_undistorted, cv2.COLOR_RGB2BGR))

    # 3. Resize & Crop (M-SLAM Logic)
    pil_img = PIL.Image.fromarray(img_undistorted)
    pil_resized = _resize_pil_image(pil_img, 512)
    pil_cropped, crop_box = mslam_crop_logic(pil_resized)
    
    # Save Final Keyframe
    pil_cropped.save(out_dir / "replicated_keyframe.png")
    print(f"\n[SUCCESS] Replicated Keyframe saved to: {out_dir / 'replicated_keyframe.png'}")
    print("Compare this image with your actual M-SLAM keyframe. They should be identical.")

    # 4. Calculate High-Res Equivalents
    # This logic assumes your High Res is 5568x4872 and Low Res is 1600x1400
    SCALE_FACTOR = 5568 / 1600  # Approx 3.48
    
    print("\n" + "="*60)
    print("HOW TO FIX YOUR PIPELINE:")
    print("="*60)
    print("If the image above matches, M-SLAM is using an UNDISTORTED image.")
    print("You CANNOT apply a simple rectangular crop to your DISTORTED High-Res image.")
    print("You must UNDISTORT your High-Res image first!")
    print("\nCorrect Pipeline for 'copy_highres_keyframes.py':")
    print("1. Load High-Res Image (5568x4872).")
    print("2. UNDISTORT it using scaled intrinsics (cv2.remap).")
    print(f"3. Crop the undistorted High-Res image.")
    print(f"   (Use the crop box from step 3 scaled by {SCALE_FACTOR:.4f})")

if __name__ == "__main__":
    main()