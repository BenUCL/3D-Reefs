
#python debug_image_preprocessing.py --image high_res.png --intrinsics intrinsics.yaml --output_dir debug_output

import argparse
import cv2
import numpy as np
import yaml
import PIL.Image
from pathlib import Path

def _resize_pil_image(img, long_edge_size):
    """Exact copy from mast3r_utils.py"""
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)

def mslam_crop_logic(pil_img, size=512):
    """
    Replicates the cropping logic from mast3r_utils.py:resize_img
    Returns the cropped image AND the crop boundaries for analysis.
    """
    W, H = pil_img.size
    cx, cy = W // 2, H // 2
    
    # Logic from mast3r_utils.py
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    
    # Calculate crop box (left, upper, right, lower)
    box = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
    
    img_cropped = pil_img.crop(box)
    return img_cropped, box

def get_undistort_map(w, h, calib_data):
    """Replicates Intrinsics.from_calib logic to build remap tables"""
    fx, fy, cx, cy = calib_data['calibration'][:4]
    distortion = np.array(calib_data['calibration'][4:]) if len(calib_data['calibration']) > 4 else np.zeros(4)
    
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    
    # M-SLAM optimization logic
    # center = config["dataset"]["center_principle_point"] -> Defaults to True in base.yaml usually
    # We will assume True for now as it's standard
    K_opt, _ = cv2.getOptimalNewCameraMatrix(
        K, distortion, (w, h), 0, (w, h), centerPrincipalPoint=True
    )
    
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, distortion, None, K_opt, (w, h), cv2.CV_32FC1
    )
    return mapx, mapy

def main():
    parser = argparse.ArgumentParser(description="Replicate M-SLAM Keyframe Generation")
    parser.add_argument("--image", required=True, help="Path to High Res Raw Image")
    parser.add_argument("--intrinsics", default=None, help="Path to intrinsics.yaml (if used in M-SLAM)")
    parser.add_argument("--output_dir", default="verification_output", help="Where to save results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load Image (OpenCV BGR -> RGB)
    print(f"Loading {args.image}...")
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise ValueError("Could not load image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_raw, w_raw = img_rgb.shape[:2]
    print(f"Original Dimensions: {w_raw}x{h_raw}")

    # 2. Undistort (If intrinsics provided)
    if args.intrinsics:
        print("Applying Undistortion (cv2.remap)...")
        with open(args.intrinsics, 'r') as f:
            calib = yaml.safe_load(f)
        
        mapx, mapy = get_undistort_map(w_raw, h_raw, calib)
        img_rgb = cv2.remap(img_rgb, mapx, mapy, cv2.INTER_LINEAR)
        
        # Save intermediate for debugging
        cv2.imwrite(str(out_dir / "step2_undistorted.jpg"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # 3. Convert to PIL for Resizing
    pil_img = PIL.Image.fromarray(img_rgb)

    # 4. Resize (Long edge -> 512)
    pil_resized = _resize_pil_image(pil_img, 512)
    print(f"Resized Dimensions: {pil_resized.size}")

    # 5. Crop (Center multiples of 16)
    pil_cropped, crop_box = mslam_crop_logic(pil_resized)
    print(f"Crop Box applied to Resized Image: {crop_box}")
    print(f"Final Keyframe Dimensions: {pil_cropped.size}")

    # Save Final Result
    pil_cropped.save(out_dir / "replicated_keyframe.png")
    print(f"Saved replicated keyframe to {out_dir / 'replicated_keyframe.png'}")

    # =========================================================
    # CALCULATE HIGH-RES CROP EQUIVALENT
    # =========================================================
    # We work backwards from the crop box on the *resized* image
    # to find the crop box on the *original* image.
    
    # Scale factors
    w_resized, h_resized = pil_resized.size
    scale_w = w_resized / w_raw
    scale_h = h_resized / h_raw
    
    # The crop box is (left, upper, right, lower) in Resized Space
    left, upper, right, lower = crop_box
    
    # Convert to Raw Space
    # Note: We use floor/ceil to be safe, or round. M-SLAM uses integer division.
    # Since we want to replicate the VIEW, we project the box back.
    raw_left = int(left / scale_w)
    raw_upper = int(upper / scale_h)
    raw_right = int(right / scale_w)
    raw_lower = int(lower / scale_h)
    
    print("\n" + "="*50)
    print("CALCULATED HIGH-RES CROP:")
    print("="*50)
    print(f"To match the M-SLAM FOV, crop the High Res image to:")
    print(f"Left: {raw_left}")
    print(f"Top:  {raw_upper}")
    print(f"Width:  {raw_right - raw_left}")
    print(f"Height: {raw_lower - raw_upper}")
    
    # Generate the High-Res Cropped Verification Image
    # (This is what you will eventually feed to Splatting)
    high_res_pil = PIL.Image.fromarray(img_rgb) # Note: this is the Undistorted version if calib used!
    high_res_cropped = high_res_pil.crop((raw_left, raw_upper, raw_right, raw_lower))
    high_res_cropped.save(out_dir / "high_res_cropped_verification.png")
    print(f"Saved High-Res cropped verification to {out_dir / 'high_res_cropped_verification.png'}")

if __name__ == "__main__":
    main()