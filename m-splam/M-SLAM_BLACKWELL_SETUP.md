# MASt3R-SLAM Setup for Blackwell GPU (Nov 19, 2025)

This should be a comprehensive guide on how to set up a machine with a Blackwell RTX 6000 PRO (and potentially related Blackwell GPU's) that will succesfully run MASt3R-SLAM. I have it working and have used copilot to create this doc as I go, so be aware these notes are slightly vibe coded so despite my careful checks there could be issues.

## 📋 Quick Reference

**Status:** ✅ FULLY WORKING - All components built and tested successfully

**Target Hardware:**
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, compute capability 10.0)
- Also works for: RTX 50xx series and any other Blackwell-based GPUs

**Critical Requirements:**
1. CUDA Toolkit 12.8+ (for native sm_120 support)
2. Use `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0"` for ALL CUDA extension builds
3. Use `--no-deps` flag when building CUDA extensions to prevent dependency chaos
4. PyTorch 2.8.0+cu128 (verify sm_120 in arch list)
5. System GCC only (NO conda GCC - it corrupts the environment)

**Total Setup Time:** ~30-45 minutes (including downloads)

**TL;DR for Experienced Users:**
```bash
# 1. Install CUDA 12.8, create conda env with Python 3.11 (system GCC only - NO conda GCC!)
#    MANUAL: Set up activation script (see Step 2)
# 2. pip install torch==2.8.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
# 3. git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git && cd MASt3R-SLAM
#    MANUAL: Download 3 checkpoint files (see Step 4.1)
#    MANUAL: Apply 4 PyTorch 2.6+ patches to .cu and .py files (see Step 4.2 - REQUIRED!)
# 4. cd /tmp && git clone https://github.com/hectorpiteau/lietorch.git && cd lietorch
#    MANUAL: Fix pyproject.toml duplicate keys (see Step 5.1 - has complete fixed file)
#    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0" pip install --no-build-isolation -e .
# 5. cd MASt3R-SLAM/thirdparty/mast3r/dust3r/croco/models/curope
#    rm -rf build/ *.egg-info/
#    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0" pip install --no-build-isolation --no-deps -e .
# 6. cd MASt3R-SLAM && pip install --no-build-isolation -e thirdparty/mast3r --no-deps
#    pip install numpy==1.26.4
# 7. pip install -e thirdparty/in3d
# 8. TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0" pip install --no-build-isolation -e .
#    MANUAL: Add PyTorch lib path to activation script (see Step 6.7)
# 9. sudo ln -sf /lib/x86_64-linux-gnu/libEGL.so.1 /lib/x86_64-linux-gnu/libEGL.so
#    sudo ln -sf /lib/x86_64-linux-gnu/libGL.so.1 /lib/x86_64-linux-gnu/libGL.so
# 10. Verify: cuobjdump --list-text $(python -c "import curope, os; print(os.path.join(os.path.dirname(curope.__file__), 'curope.cpython-311-x86_64-linux-gnu.so'))") | grep sm_120
```

**Key Manual Steps You Cannot Skip:**
- **Step 2:** Activation script setup (prevents environment corruption)
- **Step 4.2:** Apply 4 patches to .cu and .py files (PyTorch 2.8.0 compatibility)
- **Step 5.1:** Fix lietorch pyproject.toml (duplicate keys error)
- **Step 6.7:** Add PyTorch lib path to activation script (permanent LD_LIBRARY_PATH fix)

---

## System Configuration
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, compute capability 10.0)
- Driver: NVIDIA 580.95.05
- CUDA Toolkit: 12.8.61 (native sm_120 support)
- System GCC: 13.3.0 (Ubuntu 24.04)
- Python: 3.11 (conda environment)
- PyTorch: 2.8.0+cu128

## Steps Completed

### 1. Install CUDA Toolkit 12.8
```bash
# Download and install
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_*.deb
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda-toolkit-12-8

# Set as system default
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.8 /usr/local/cuda

# Update .bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc
```

### 2. Create Conda Environment with CUDA 12.8
```bash
# Create environment (Python 3.11 only, NO conda GCC!)
conda create -n mast3r-slam-blackwell python=3.11 -y
conda activate mast3r-slam-blackwell

# Set CUDA 12.8 paths using activation script (with duplicate prevention)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.8
if [[ ":$PATH:" != *":/usr/local/cuda-12.8/bin:"* ]]; then
    export PATH=/usr/local/cuda-12.8/bin:$PATH
fi
if [[ ":$LD_LIBRARY_PATH:" != *":/usr/local/cuda-12.8/lib64:"* ]]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
fi
EOF

# Reactivate to apply
conda deactivate
conda activate mast3r-slam-blackwell

# Verify
nvcc --version  # Should show: release 12.8, V12.8.61
gcc --version   # Should show: gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
echo $CUDA_HOME # Should show: /usr/local/cuda-12.8
```

**Critical Lessons Learned:**
1. ❌ **DO NOT** install `conda install gcc_linux-64` - it corrupts the environment with ~100 broken env vars
2. ✅ **DO** use system GCC 13.3.0 - works perfectly with CUDA 12.8
3. ❌ **DO NOT** use `conda env config vars set PATH=...` - causes conda activation bugs
4. ✅ **DO** use activation script with duplicate-check conditionals
5. If environment gets corrupted: close terminal, `rm -rf ~/miniconda3/envs/mast3r-slam-blackwell`, start fresh

### 3. Install PyTorch 2.8.0 with CUDA 12.8
```bash
# Install PyTorch 2.8.0 stable (let pip find compatible torchvision/torchaudio)
pip install torch==2.8.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# Verify installation and sm_120 support
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Arch list:', torch.cuda.get_arch_list())"
```

**Expected output:**
```
PyTorch: 2.8.0+cu128
CUDA available: True
Arch list: ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

✅ **Critical:** Confirm `sm_120` is in the arch list - this means native Blackwell support!

**Note:** Don't specify exact torchvision/torchaudio versions - let pip resolve compatible ones automatically.

### 4. Clone MASt3R-SLAM and Apply Patches

**Step 4.1: Clone Repository**
```bash
cd /home/ben/encode/code
git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git
cd MASt3R-SLAM

# Download model checkpoints (~2.8GB)
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

**Step 4.2: Apply PyTorch 2.6+ Compatibility Patches (PR #86)**

These 4 patches are **REQUIRED** for PyTorch 2.8.0 compatibility. You need to manually edit these files:

**Patch 1: thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu (line ~101)**

Open the file and find this line:
```cpp
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type().scalarType(), "rope_2d_cuda", ([&] {
```

Replace it with:
```cpp
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.scalar_type(), "rope_2d_cuda", ([&] {
```

**Patch 2: mast3r_slam/backend/src/matching_kernels.cu (line ~29)**

Open the file and find this line in the `triangulation_kernel` function:
```cpp
at::ScalarType scalar_type = D11.type().scalarType();
```

Replace it with:
```cpp
at::ScalarType scalar_type = D11.scalar_type();
```

**Patch 3: mast3r_slam/backend/src/gn_kernels.cu (3 occurrences: lines ~802, ~1219, ~1629)**

Open the file and find ALL 3 occurrences of lines like:
```cpp
auto norm = torch::linalg::linalg_norm(dx, 2, 0, false, c10::nullopt);
```
or
```cpp
delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
```

Replace each with the simplified form:
```cpp
auto norm = dx.flatten().norm();
```
or
```cpp
delta_norm = dx.flatten().norm();
```

**Patch 4: thirdparty/mast3r/mast3r/model.py (line ~24)**

Open the file and find this line in the `load_model` function:
```python
ckpt = torch.load(model_path, map_location='cpu')
```

Replace it with:
```python
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
```

**Why These Patches Are Needed:**
- `.scalar_type()`: PyTorch 2.6+ deprecated `.type().scalarType()` API
- `.norm()`: Simplified replacement for deprecated `torch::linalg::linalg_norm()`
- `weights_only=False`: PyTorch 2.6+ changed default to True (breaks checkpoint loading)

**Verification:**
```bash
# Check patches applied correctly
grep "scalar_type()" thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu
grep "scalar_type()" mast3r_slam/backend/src/matching_kernels.cu
grep "\.flatten()\.norm()" mast3r_slam/backend/src/gn_kernels.cu
grep "weights_only=False" thirdparty/mast3r/mast3r/model.py
```

### 5. Build lietorch with sm_120 Support

**Step 5.1: Clone and fix hectorpiteau/lietorch**
```bash
# Clone lietorch (PyTorch 2.6+ compatible fork)
cd /tmp
rm -rf lietorch
git clone https://github.com/hectorpiteau/lietorch.git
cd lietorch
git submodule update --init --recursive

# Fix duplicate keys in pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools", "torch>=2.6.0"]
build-backend = "setuptools.build_meta"

[project]
name = "lietorch"
version = "0.2" 
description = "Lie Groups for PyTorch"
authors = [
    { name="teedrz", email="zachteed@gmail.com" }
]
license = { text = "BSD-3-Clause" }
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.6.0",
    "numpy<2",
]
EOF
```

**Step 5.2: Build lietorch with CUDA 12.8 and sm_120**
```bash
cd /tmp/lietorch
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0" pip install --no-build-isolation -e .
```

**Architecture list explanation:**
- `7.5` = Turing (RTX 20xx series)
- `8.0` = Ampere (A100)
- `8.6` = Ampere (RTX 30xx series)
- `9.0` = Hopper (H100)
- `12.0` = Blackwell (RTX 50xx, RTX PRO 6000) ← **Our target**

**Note:** We use `12.0` instead of `10.0` for sm_120 architecture.

**Step 5.3: Verify lietorch**
```bash
conda activate mast3r-slam-blackwell
python -c "import lietorch; import torch; print('lietorch imported successfully'); print('PyTorch:', torch.__version__); x = lietorch.SE3.Identity(1, device='cuda'); print('lietorch CUDA operations work')"
```

**Expected output:**
```
lietorch imported successfully
PyTorch: 2.8.0+cu128
lietorch CUDA operations work
```

### 6. Build MASt3R-SLAM Components

**CRITICAL LESSON LEARNED:** The setup.py smart detection does NOT work reliably. You MUST use `TORCH_CUDA_ARCH_LIST` explicitly, AND you MUST use `--no-deps` to prevent pip from upgrading PyTorch.

**Why `--no-deps` is critical:**
- mast3r has `huggingface-hub[torch]>=0.22` in its dependencies
- The `[torch]` extra causes huggingface-hub to install the latest PyTorch (2.9.1 nightly)
- Using `--no-deps` prevents this dependency resolution nightmare
- We install dependencies separately after building the CUDA extension

**Step 6.1: Build curope CUDA extension WITH explicit architecture list**
```bash
cd /home/ben/encode/code/MASt3R-SLAM/thirdparty/mast3r/dust3r/croco/models/curope

# Clean any old build artifacts
rm -rf build/ *.egg-info/

# Set LD_LIBRARY_PATH for PyTorch libs
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# Build with explicit TORCH_CUDA_ARCH_LIST and --no-deps
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0" pip install --no-build-isolation --no-deps -e .
```

**Step 6.2: Install mast3r package metadata (without rebuilding)**
```bash
cd /home/ben/encode/code/MASt3R-SLAM
pip install --no-build-isolation -e thirdparty/mast3r --no-deps
```

**Step 6.3: Fix numpy version conflict**
```bash
# mast3r wants numpy 2.x but lietorch requires numpy<2
pip install numpy==1.26.4
```

**Note:** opencv-python will complain about numpy>=2 requirement, but it works fine with 1.26.4.

**Step 6.4: Verify curope has sm_120 compiled**
```bash
# Check that sm_120 kernels are present in the compiled binary
cuobjdump --list-text $(python -c "import curope, os; print(os.path.join(os.path.dirname(curope.__file__), 'curope.cpython-311-x86_64-linux-gnu.so'))") | grep sm_120

# Should see output like:
# SASS text section 16 : ...sm_120.elf.bin
# SASS text section 17 : ...sm_120.elf.bin
# SASS text section 18 : ...sm_120.elf.bin
```

**Step 6.5: Build thirdparty/in3d (no CUDA compilation)**
```bash
cd /home/ben/encode/code/MASt3R-SLAM
pip install -e thirdparty/in3d
```

**Step 6.6: Build main mast3r_slam package** ✅ **COMPLETED**
```bash
cd /home/ben/encode/code/MASt3R-SLAM

# Ensure LD_LIBRARY_PATH includes PyTorch libs
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# Build main package (also use TORCH_CUDA_ARCH_LIST for consistency)
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0" pip install --no-build-isolation -e . 2>&1 | tee /tmp/mast3r_slam_build.log
```

**Result:** Successfully built MAST3R-SLAM-0.0.1 with all dependencies.

**Critical Notes:**
- ALWAYS use TORCH_CUDA_ARCH_LIST for CUDA extension builds (don't rely on setup.py smart detection)
- The setup.py has nvcc detection but PyTorch's build system can override it
- Using TORCH_CUDA_ARCH_LIST ensures sm_120 is actually compiled into the binary
- The LD_LIBRARY_PATH export is needed for CUDA extension imports to find libc10.so
- Build takes ~2-5 minutes

**Step 6.7: Fix LD_LIBRARY_PATH permanently in activation script** ✅ **COMPLETED**
```bash
# Update activation script to include PyTorch lib path
cat >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'

# Add PyTorch lib path for CUDA extension imports
TORCH_LIB_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")
if [ -n "$TORCH_LIB_PATH" ] && [[ ":$LD_LIBRARY_PATH:" != *":$TORCH_LIB_PATH:"* ]]; then
    export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH
fi
EOF

# Test by reactivating environment
conda deactivate
conda activate mast3r-slam-blackwell
```

**Result:** Verified - imports work after reactivation without manual LD_LIBRARY_PATH export.

### 7. Verification - All Imports Working ✅ **COMPLETED**

```bash
python -c "
import torch
print('=== PyTorch ===')
print(f'Version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'sm_120 in arch list: {\"sm_120\" in torch.cuda.get_arch_list()}')
print()

import lietorch
print('✓ lietorch imported')

import curope
print('✓ curope imported')

import mast3r
print('✓ mast3r imported')

import mast3r_slam
print('✓ mast3r_slam imported')

import in3d
print('✓ in3d imported')

print()
print('=== ALL IMPORTS SUCCESSFUL ===')
"
```

**Expected Output:**
```
=== PyTorch ===
Version: 2.8.0+cu128
CUDA available: True
CUDA device: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
sm_120 in arch list: True

✓ lietorch imported
✓ curope imported
✓ mast3r imported
✓ mast3r_slam imported
✓ in3d imported

=== ALL IMPORTS SUCCESSFUL ===
```

### 8. Fix Visualization (OpenGL Libraries) ✅ **COMPLETED**

**Problem:** The real-time 3D visualization window fails with:
```
OSError: libEGL.so: cannot open shared object file: No such file or directory
OSError: libGL.so: cannot open shared object file: No such file or directory
```

**Root Cause:** The system has `libEGL.so.1` and `libGL.so.1`, but `moderngl` looks for files without the `.1` suffix.

**Fix: Create symlinks**
```bash
sudo ln -sf /lib/x86_64-linux-gnu/libEGL.so.1 /lib/x86_64-linux-gnu/libEGL.so
sudo ln -sf /lib/x86_64-linux-gnu/libGL.so.1 /lib/x86_64-linux-gnu/libGL.so
```

**Verify:**
```bash
python -c "import moderngl; print('✓ moderngl can import OpenGL libraries')"
```

**Expected output:** `✓ moderngl can import OpenGL libraries`

**Note:** The visualization error is NOT fatal - MASt3R-SLAM will still process successfully without the visualization window, but the real-time 3D view is very useful for monitoring progress.

## Summary: Deep Research Instructions vs. Our Approach

### ✅ Alignment with Deep Research Guide (CORRECTED AFTER DEBUGGING)

We successfully followed the deep research instructions (Steps 1-7) with an important lesson learned:

**Steps 1-6:** Followed exactly as specified
- ✅ CUDA 12.8 toolkit installed
- ✅ Python 3.11 conda environment (system GCC 13.3.0, NO conda GCC)
- ✅ PyTorch 2.8.0+cu128 with verified sm_120 support
- ✅ Cloned repository with checkpoints
- ✅ Applied all 4 PyTorch 2.6+ API patches (PR #86)
- ✅ Built hectorpiteau/lietorch with TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0"

**Step 7 (Compile Extensions) - CRITICAL LESSON LEARNED:**

**Initial approach (WRONG):**
- ❌ Built mast3r/curope WITHOUT TORCH_CUDA_ARCH_LIST (trusted setup.py smart detection)
- ❌ Built main package WITHOUT TORCH_CUDA_ARCH_LIST
- ❌ Result: curope only had sm_70, sm_75, sm_80, sm_86, sm_90 - NO sm_120!
- ❌ Runtime error: "CUDA error: no kernel image is available for execution on the device"

**Why it failed:**
- Setup.py's nvcc detection correctly identifies CUDA 12.8
- BUT PyTorch's build system overrides the gencode flags from setup.py
- The smart detection doesn't actually make it into the final compiled binary
- TORCH_CUDA_ARCH_LIST is the ONLY reliable way to ensure architectures are compiled

**Second attempt (ALSO WRONG):**
- ❌ Used `pip install --force-reinstall -e thirdparty/mast3r` with TORCH_CUDA_ARCH_LIST
- ❌ Result: PyTorch upgraded from 2.8.0 to 2.9.1 (nightly) due to `huggingface-hub[torch]` dependency
- ❌ Broke entire environment with version mismatches

**Final approach (CORRECT):**
- ✅ Use TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0" for ALL CUDA extension builds
- ✅ Use `--no-deps` flag to prevent pip from upgrading dependencies
- ✅ Build curope directly in its directory first
- ✅ Then install mast3r package metadata separately with `--no-deps`
- ✅ Result: curope has sm_120 kernels, verified with cuobjdump

**The Deep Research Guide Was RIGHT:** Always use TORCH_CUDA_ARCH_LIST for all builds (Method A).

### 📋 Installation Checklist

- [x] Step 1: CUDA 12.8 toolkit installed (nvcc 12.8.61)
- [x] Step 2: Conda environment (Python 3.11, system GCC 13.3.0)
- [x] Step 3: PyTorch 2.8.0+cu128 (sm_120 verified)
- [x] Step 4: Repository cloned with checkpoints
- [x] Step 5: All 4 API patches applied
- [x] Step 6: lietorch built with sm_120 (TORCH_CUDA_ARCH_LIST)
- [x] Step 7a: curope built with sm_120 (TORCH_CUDA_ARCH_LIST + --no-deps)
- [x] Step 7b: curope sm_120 kernels verified with cuobjdump
- [x] Step 7c: in3d built
- [x] Step 7d: main package built (TORCH_CUDA_ARCH_LIST)
- [x] Step 7e: All imports verified
- [x] Step 8: OpenGL libraries fixed for visualization
- [x] Step 9: Checkpoints downloaded (done in Step 4)
- [x] Step 10: Test run completed successfully ✅
- [x] Step 11: Visualization window working ✅

### 🔑 Key Success Factors

1. **ALWAYS use TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0"** for ALL CUDA extension builds (don't trust setup.py smart detection)
2. **ALWAYS use --no-deps** when building CUDA extensions to prevent pip from upgrading dependencies
3. **Verify with cuobjdump** that sm_120 kernels are actually in the compiled binary
4. **PyTorch version must stay at 2.8.0** - huggingface-hub[torch] dependency will try to upgrade it to 2.9.1
5. **numpy must be 1.26.4** - lietorch requires numpy<2 (opencv warnings about numpy>=2 are harmless)
6. **Create OpenGL symlinks** for visualization - moderngl needs libEGL.so and libGL.so (without .1 suffix)

## Running MASt3R-SLAM ✅ **TESTED AND WORKING**

### Test Run (Completed Successfully)
```bash
cd /home/ben/encode/code/MASt3R-SLAM
python main.py --dataset <path/to/dataset> --config config/base.yaml
```

**Results:**
- ✅ No "kernel image available" errors - sm_120 kernels working perfectly
- ✅ Processing completed in ~1 minute for test dataset  
- ✅ Visualization window displays real-time 3D reconstruction
- ✅ GPU utilization confirmed with `nvtop`
- ✅ Output saved to dataset directory

### Performance Notes
- Blackwell GPU (sm_120) provides native acceleration
- Real-time processing at ~10-15 FPS expected (depends on image resolution)
- Much faster than previous generation GPUs due to native sm_120 support

### Troubleshooting

**If you see "CUDA error: no kernel image is available":**
1. Check curope has sm_120: `cuobjdump --list-text <path_to_curope.so> | grep sm_120`
2. If missing sm_120, rebuild curope with `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0"` and `--no-deps`
3. Verify PyTorch version: `python -c "import torch; print(torch.__version__)"` (must be 2.8.0+cu128)

**If visualization window fails:**
1. Check symlinks exist: `ls -la /lib/x86_64-linux-gnu/libEGL.so /lib/x86_64-linux-gnu/libGL.so`
2. If missing, create them: `sudo ln -sf /lib/x86_64-linux-gnu/libEGL.so.1 /lib/x86_64-linux-gnu/libEGL.so`
3. Note: MASt3R-SLAM will still work without visualization (headless mode)

---

## ⚠️ Common Pitfalls and How to Avoid Them

### 1. **Installing conda GCC (DO NOT DO THIS)**
- ❌ **Problem:** Running `conda install gcc_linux-64` corrupts environment with ~100 broken environment variables
- ✅ **Solution:** Use system GCC only (Ubuntu's gcc 13.3.0 works perfectly with CUDA 12.8)
- **Recovery:** If already installed, delete environment and start fresh

### 2. **Building CUDA Extensions Without TORCH_CUDA_ARCH_LIST**
- ❌ **Problem:** Setup.py smart detection doesn't work - PyTorch build system overrides it
- ❌ **Result:** Binary missing sm_120 kernels → "no kernel image available" runtime error
- ✅ **Solution:** ALWAYS use `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0"` for ALL CUDA extension builds

### 3. **Using --force-reinstall Without --no-deps**
- ❌ **Problem:** `pip install --force-reinstall -e thirdparty/mast3r` upgrades PyTorch 2.8.0 → 2.9.1 (nightly)
- ❌ **Cause:** mast3r has `huggingface-hub[torch]>=0.22` dependency which pulls latest PyTorch
- ✅ **Solution:** ALWAYS use `--no-deps` when building/rebuilding CUDA extensions

### 4. **Not Verifying sm_120 Kernels After Build**
- ❌ **Problem:** Build completes without errors, but sm_120 kernels aren't actually in the binary
- ✅ **Solution:** Always verify with cuobjdump: `cuobjdump --list-text curope.so | grep sm_120`
- **Expected:** Should see sections 16-18 with sm_120.elf.bin

### 5. **Forgetting LD_LIBRARY_PATH for PyTorch Libs**
- ❌ **Problem:** Import errors for CUDA extensions: "cannot find libc10.so"
- ✅ **Solution:** Add PyTorch lib path to activation script (Step 6.7) - then it works permanently

### 6. **Missing OpenGL Symlinks**
- ❌ **Problem:** Visualization fails with "libEGL.so: cannot open shared object file"
- ✅ **Solution:** Create symlinks (Step 8) - moderngl needs libEGL.so not libEGL.so.1

### 7. **Using conda env config vars for PATH**
- ❌ **Problem:** `conda env config vars set PATH=...` causes conda activation bugs
- ✅ **Solution:** Use activation script with duplicate-check conditionals (Step 2)

---

## 📝 Complete Environment Setup Summary

**What We Built:**
1. CUDA 12.8 toolkit with native sm_120 support
2. Python 3.11 conda environment (system GCC 13.3.0)
3. PyTorch 2.8.0+cu128 with verified sm_120 in arch list
4. lietorch (PyTorch 2.6+ compatible fork) with sm_120 kernels
5. curope CUDA extension with sm_120 kernels (verified with cuobjdump)
6. in3d package (no CUDA compilation)
7. Main mast3r_slam package with sm_120 kernels
8. OpenGL libraries for real-time visualization

**Total Build Time:** ~30-45 minutes
- CUDA toolkit install: ~5 minutes
- PyTorch install: ~5 minutes  
- Repository clone + checkpoints: ~10 minutes (depends on internet speed)
- lietorch build: ~3 minutes
- curope build: ~1 minute
- Main package build: ~2-3 minutes
- Patches + verification: ~5 minutes

**Final Result:** 
- ✅ All imports working
- ✅ No CUDA kernel errors
- ✅ Visualization window functional
- ✅ Test run completed successfully
- ✅ Native Blackwell (sm_120) acceleration confirmed

