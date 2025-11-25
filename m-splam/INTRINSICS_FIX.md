# Intrinsics Scaling Fix

## Problem Discovered

After running the full pipeline with `use_highres_for_splatting=true`, Gaussian splatting training plateaued at loss=0.55 throughout the entire training period. Blue dots were visible in the correct shape, but no actual render appeared when hiding splats.

### Root Cause

The intrinsics scaling chain was **fundamentally broken** due to a compounding error:

1. **Step 1**: `shuttle_intrinsics.py` reads COLMAP calibration at **1600x1400** resolution
2. **Step 2**: `shuttle_intrinsics.py` scales intrinsics to **512x448** (SLAM resolution) and saves to `for_splat/sparse/0/`
3. **Step 3**: `convert_intrinsics.py` reads these **512x448** intrinsics but **assumes they're 1600x1400**
4. **Step 4**: `convert_intrinsics.py` detects scale from image directories: 1600→5568 = **3.48x**
5. **Step 5**: `convert_intrinsics.py` applies 3.48x to **already-scaled** 512x448 intrinsics

**Result**: Intrinsics represent ~1782x1555 resolution instead of 5568x4872 (**3.12x too small**)

### Mathematical Analysis

- Correct scaling: 512 × 10.875 = 5568 ✓
- Actual scaling: 512 × 3.48 = 1782 ✗
- Error factor: 5568 / 1782 = 3.12x underestimation

This explains why:
- Training loss plateaued (wrong focal length = wrong perspective projection)
- Blue dots visible (poses and points3D were correct)
- No actual render (intrinsics too small for actual image resolution)

## Solution Implemented

### Design Principles

1. **Single-step scaling**: Scale intrinsics directly from calibration resolution to target resolution (no intermediate steps)
2. **Resolution auto-detection**: Detect high-res image dimensions automatically using `get_resolution()`
3. **Eliminate buggy step**: Remove `convert_intrinsics.py` from pipeline (contained the compounding error)
4. **Preserve SLAM correctness**: M-SLAM still gets 512x448 intrinsics (matches its internal resize)

### Implementation

#### Modified Files

1. **`shuttle_intrinsics.py`** - Comprehensive rewrite:
   - Added `get_resolution()` function to detect image dimensions
   - Added `--highres-images-path` argument
   - Added validation: requires `--highres-images-path` when `--use-highres-for-splatting` is set
   - Modified main logic to:
     * Always create `intrinsics.yaml` at 512x448 for M-SLAM ✓
     * If low-res mode: Create `cameras.bin/txt` at 512x448 PINHOLE ✓
     * If high-res mode: Detect high-res size, scale from 1600x1400→5568x4872 directly ✓

2. **`run_pipeline.py`** - Simplified step 2:
   - Pass `--highres-images-path` to `shuttle_intrinsics.py` when `use_highres_for_splatting=true`
   - Removed step 2b (no longer calls `convert_intrinsics.py`)
   - Updated docstring to reflect single-step scaling

3. **`convert_intrinsics.py`** - TO BE DEPRECATED:
   - Contains the bug (reads 512x448 data but thinks it's 1600x1400)
   - No longer called by pipeline
   - Can be deleted after validation

### Scaling Summary

| Component | Input Resolution | Output Resolution | Camera Model | Distortion |
|-----------|------------------|-------------------|--------------|------------|
| **COLMAP Calibration** | 1600x1400 | - | OPENCV | Yes (k1,k2,p1,p2) |
| **M-SLAM intrinsics.yaml** | 1600x1400 | 512x448 | OPENCV | Yes |
| **Low-res mode** | 1600x1400 | 512x448 | PINHOLE | No |
| **High-res mode** | 1600x1400 | 5568x4872 | OPENCV | Yes |

### Key Insights

1. **M-SLAM's internal resize**: M-SLAM resizes images to 512px width internally using `resize_img()`, so `intrinsics.yaml` must match the resized resolution (512x448), not the input image resolution.

2. **Single source of truth**: Always scale from the original COLMAP calibration resolution (1600x1400) to avoid compounding errors.

3. **Resolution detection**: Use `get_resolution()` to auto-detect high-res dimensions instead of hardcoding, making the pipeline flexible for different camera setups.

## Validation Plan

1. Run full pipeline with `use_highres_for_splatting=true`
2. Verify `cameras.bin` contains correct resolution and intrinsics:
   ```bash
   # Check resolution in cameras.txt
   cat runs/reef_soneva/for_splat/sparse/0/cameras.txt
   # Should show: 1 OPENCV 5568 4872 [fx fy cx cy k1 k2 p1 p2]
   ```
3. Train Gaussian splats and confirm:
   - Loss decreases properly (not plateau at 0.55)
   - Actual scene renders correctly (not just blue dots)
4. Compare with low-res mode for validation

## References

- Original bug report: User notes showing loss plateau and blue dots issue
- Mathematical analysis: Confirmed 3.12x error matches observed symptoms
- MASt3R-SLAM resize behavior: `resize_img()` function resizes to 512px width
- COLMAP binary format: `cameras.bin` structure with width/height/params
