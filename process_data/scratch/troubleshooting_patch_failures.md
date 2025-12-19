# Troubleshooting Patch Training Failures: The p7 Saga

## Initial Issue

Some patche consistently crashed during MCMC training with:

```
adam_step_cu kernel launch failed: invalid configuration argument
```
It was always the same patches on every run. When I split to 14 patches it was about 3 or 4, when I split to 11 it was only patch 7, most of the troubleshooting is on this later one.

Tried updating LichtFeld-Studio from v0.2.0 to v0.3.0, the team just released this today and suggested I do so in respsone to my [github issue](https://github.com/MrNeRF/LichtFeld-Studio/issues/552).

The crash occurred around iteration 20 (~0.1% into training), though not always the same iteration. Other patches (p0, p5, p6, p8) trained successfully with v0.3.0.

## Investigation Timeline

### Phase 1: Environment & Data Validation
- Verified CUDA 12.8.61 and RTX PRO 6000 Blackwell GPU
- Confirmed LFS v0.3.0 built correctly (libtorch removed)
- Checked p7's COLMAP data integrity - all files valid

### Phase 2: Data Comparison (p7 vs p0)
Compared working patch (p0) against failing patch (p7):

| Metric | p0 (works) | p7 (fails) |
|--------|------------|------------|
| Cameras | 450 | 450 |
| Points | 54,086 | 58,116 |
| XYZ range | 0.85 | 1.31 |
| Mean scale | 0.0015 | 0.0021 |

**Finding**: p7 has ~50% larger spatial extent than p0. Though it was a bit smaller than some succesful patches.

### Phase 3: Isolation Testing

#### Camera & Point Cloud Swapping
Created hybrid datasets to isolate whether the issue was in cameras or points:
- Tested p7's XYZ with p0's cameras → **CRASH**
- Tested p0's XYZ with p7's cameras → **SUCCESS**
- Tested subsets of p7's points → Still crashed

**Finding**: The issue was in p7's point cloud characteristics, not cameras.

#### World Origin Shifting
Shifted p7's entire coordinate system to test if absolute position mattered:
- Recentered points to origin (subtracted mean XYZ) → **CRASH**
- Shifted to match p0's coordinate range → **CRASH**

**Finding**: Absolute world position doesn't matter; it's the relative scale/distribution.

#### Visibility & Duplicate Analysis
- Checked for duplicate points in points3D.bin → None found
- Analyzed visibility counts (how many cameras see each point) → Normal distribution
- Compared point density patterns → p7 slightly sparser but within normal range

**Finding**: No data corruption or anomalies in the point cloud structure.

#### Camera Pose Analysis
- Extracted and compared camera poses between p0 and p7
- Checked for outlier cameras with extreme positions → None found
- Verified camera intrinsics matched expected values

**Finding**: Camera configurations were valid and similar across patches.

### Phase 4: Iteration Analysis
- Crash consistently occurred at around iteration ~20
- Not related to specific point indices or visibility masks
- Happened before any densification/pruning

**Finding**: Crash occurs during early optimization, not data loading.

### Phase 5: Parameter Ablation
Systematically tested `init_scaling` and `means_lr`:

| init_scaling | means_lr | Result |
|-------------|----------|--------|
| 0.1 (default) | 0.00016 | ❌ CRASH |
| 0.1 | 0.00008 | ❌ CRASH |
| 0.2 | 0.00016 | ❌ CRASH |
| 0.3 | 0.00016 | ✅ SUCCESS |
| 0.4 | 0.00016 | ✅ SUCCESS |

**Finding**: `init_scaling=0.3` is the minimum working value.

## Root Cause

The default `init_scaling=0.1` initializes Gaussians too small for p7's larger scene scale. This causes:

1. Very small Gaussians relative to scene extent
2. Large gradients during early optimization  
3. Numerical instability in the Adam optimizer
4. CUDA kernel failure due to invalid tensor configurations

The 50% larger spatial extent of p7 (1.31 vs 0.85) means the same `init_scaling` produces relatively smaller Gaussians, amplifying gradient magnitudes.

## Solution

Set `init_scaling=0.3` (3x default) for patches with larger scene scales. In fact I tried going up in 0.1 increments, 0.2 got a 100 iters in then failed, 0.3 does the job. But, putting it to high (like 1.0) causes splat to explode.

**Implementation**: Created `lfs_configs/increase_init_scaling.json` with the fix, selectable via `splat_config.yml`:

```yaml
training:
  lfs_config: increase_init_scaling.json
```

## Validation

- p7 trained successfully for 10,000 iterations with `init_scaling=0.3`
- Visual inspection confirmed good splat quality
- No degradation in reconstruction fidelity

## Lessons Learned

1. Scene scale affects optimal initialization parameters
2. CUDA kernel errors can stem from numerical instability, not just memory
3. Systematic ablation testing is essential for isolating root causes
4. Default parameters may not generalize across all scene types
