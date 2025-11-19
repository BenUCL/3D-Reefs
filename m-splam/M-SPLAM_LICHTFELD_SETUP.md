# LichtFeld-Studio Setup for Blackwell RTX PRO 6000 (Nov 19, 2025)

Complete setup guide for building LichtFeld-Studio on Ubuntu 24.04 with Blackwell GPU support.

## 📋 System Information

**Status:** ✅ COMPLETE

**Target Hardware:**
- GPU: NVIDIA RTX PRO 6000 Blackwell (sm_120, compute capability 12.0)
- OS: Ubuntu 24.04.3 LTS
- CUDA: 12.8.61

**Critical Requirements:**
1. GCC 14+ (C++23 support required by LichtFeld)
2. CMake 3.30+
3. CUDA 12.8+ (✅ already installed)
4. LibTorch 2.7.0 with CUDA 12.8
5. vcpkg for dependency management
6. Ninja build system

**Key Finding:** LichtFeld-Studio CMakeLists.txt already includes sm_120 support (line 155)! No patches needed.

---

## Setup Steps

### ✅ Phase 1: System Dependencies (COMPLETED)

**1.1: Install GCC-14**
```bash
sudo apt update
sudo apt install -y gcc-14 g++-14 gfortran-14

# Set up alternatives (GCC-14 priority 60, GCC-13 priority 50)
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 60 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-14 \
    --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-14

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 50 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
    --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-13

# Select GCC-14 (choose option 0 or press Enter)
sudo update-alternatives --config gcc
```

**Verification:**
```bash
gcc --version   # Should show: gcc (Ubuntu 14.x.x)
g++ --version   # Should show: g++ (Ubuntu 14.x.x)
```

**Note:** System GCC-14 is only used for compilation. If you have existing conda environments with compiled packages, they won't be affected since they use their own runtime libraries.

---

### ✅ Phase 2: CMake 4.1.3 & Ninja (COMPLETED)

Ubuntu 24.04 apt repos only have CMake 3.27, which is too old. Installed from Kitware's official repository:

```bash
# Add Kitware APT repository
sudo apt install -y software-properties-common wget
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

# Install CMake
sudo apt update
sudo apt install -y cmake

# Verify
cmake --version  # Should show 3.30+
```

**Install Ninja + X11 dev libs:**
```bash
sudo apt install -y ninja-build libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config
ninja --version  # Shows 1.11.1
```

**Verification:**
```bash
cmake --version   # Showed: cmake version 4.1.3
ninja --version   # Showed: 1.11.1
```

---

### ✅ Phase 3: vcpkg (COMPLETED)

```bash
cd ~
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh -disableMetrics

# Add to ~/.bashrc
echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.bashrc
source ~/.bashrc

# Verify
echo $VCPKG_ROOT  # Should show: /home/ben/vcpkg
```

**Verification:** vcpkg installed at `/home/ben/vcpkg`, `$VCPKG_ROOT` set correctly.

---

### ✅ Phase 4: LibTorch 2.7.0 with CUDA 12.8 (COMPLETED)

**Note:** LibTorch was already present in `external/libtorch/` (5.9GB) from previous setup. Verified contents:

```bash
ls -lh /home/ben/encode/code/lichtfeld-studio/external/libtorch/lib/
# Output shows libtorch.so, libtorch_cuda.so, etc. totaling 5.9GB
```

**If starting fresh, use:**
```bash
cd /home/ben/encode/code/lichtfeld-studio
mkdir -p external

# Download LibTorch (cxx11 ABI version, ~2.5GB)
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip

# Extract
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip -d external/

# Cleanup
rm libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip
```

---

### ✅ Phase 5: Build LichtFeld-Studio (COMPLETED)

```bash
cd /home/ben/encode/code/lichtfeld-studio

# Configure build (auto-detects RTX PRO 6000 as sm_120)
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja

# Compile (10-15 minutes)
cmake --build build -- -j$(nproc)

# Verify binary exists
ls -lh build/LichtFeld-Studio
```

**Actual build output:**
- ✅ CMake detected: `Detected GPU compute capability: 12.0`
- ✅ CUDA arch set: `TORCH_CUDA_ARCH_LIST: 12.0`
- ✅ NVCC flags: `Added CUDA NVCC flags for: -gencode;arch=compute_120,code=sm_120`
- ✅ CUDA-OpenGL interop: `ENABLED`
- ✅ vcpkg built 17 packages: glfw3, imgui, nlohmann-json, spdlog, etc.
- ✅ Compilation: 223 compilation units, ~10 minutes with 46 cores
- ✅ Binary: 33MB at `/home/ben/encode/code/lichtfeld-studio/build/LichtFeld-Studio`

**Runtime requirement:**
```bash
export LD_LIBRARY_PATH="/home/ben/encode/code/lichtfeld-studio/external/libtorch/lib:${LD_LIBRARY_PATH}"
```

**Warnings:** Minor GCC-14 stringop-overflow warnings in std::vector<bool> (false positives, safe to ignore).

---

### ✅ Phase 6: Download Truck Dataset (COMPLETED)

```bash
cd /home/ben/encode/code/lichtfeld-studio
mkdir -p data

# Download Tanks & Trains dataset (651MB)
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip

# Extract
unzip tandt_db.zip

# Verify truck dataset exists
ls -lh data/tandt/truck/
```

**Verification:** Dataset extracted with 251 images and COLMAP reconstruction (cameras.bin, images.bin, points3D.bin).

---

### ✅ Phase 7: Test Basic Run (COMPLETED)

```bash
cd /home/ben/encode/code/lichtfeld-studio

# CRITICAL: Set library path before running
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${LD_LIBRARY_PATH}"

# Quick test run (1000 iterations)
./build/LichtFeld-Studio \
    -d data/tandt/truck \
    -o output/truck_test \
    --eval \
    --headless \
    -i 1000
```

**Actual test results:**
- ✅ **Training completed:** 1000 iterations in 2.6 seconds (384 iter/s)
- ✅ **Blackwell sm_120:** No kernel errors - GPU fully supported
- ✅ **Gaussian splats:** 136k initial points → 173k final splats
- ✅ **Loss convergence:** 0.31 → 0.18
- ✅ **Output files:** `metrics_report.txt`, `metrics.csv`, splat PLY files

**Blackwell support CONFIRMED!** Ready to integrate with MASt3R-SLAM pipeline.

---

## Integration with MASt3R-SLAM Pipeline

Once LichtFeld-Studio is working, update pipeline config:

```yaml
# In slam_splat_config.yaml
paths:
  lichtfeld_binary: "/home/ben/encode/code/lichtfeld-studio/build/LichtFeld-Studio"
```

The pipeline's `train_splat.py` wrapper will handle:
- Setting `LD_LIBRARY_PATH` automatically
- Passing correct arguments to LichtFeld-Studio
- Capturing training progress
- Creating run reports

---

## Troubleshooting

### Common Issues

**Problem:** CMake version too old  
**Solution:** Install from Kitware repo (see Phase 2)

**Problem:** "no kernel image available for function" error  
**Solution:** Check CMake output shows sm_120 in architecture list

**Problem:** LichtFeld-Studio crashes with segfault  
**Solution:** Verify `LD_LIBRARY_PATH` includes `external/libtorch/lib`

**Problem:** GCC-14 breaks conda environment  
**Solution:** This shouldn't happen - conda uses its own runtime libraries. If it does, rebuild specific packages: `conda install libstdcxx-ng libgcc-ng`

---

## Architecture Notes

### Why This Setup Works:

1. **Standalone Binary:**
   - LichtFeld-Studio is a C++23 binary with LibTorch 2.7.0+cu128 statically linked
   - No Python dependencies or conda environment required
   - Runs independently from MASt3R-SLAM pipeline

2. **GCC-14 Requirement:**
   - LichtFeld requires C++23 features (std::expected, import std, etc.)
   - System GCC-14 is used only at compile time
   - Existing conda environments remain unaffected

3. **Blackwell Support:**
   - LichtFeld-Studio has sm_120 in fallback architecture list
   - Auto-detection via `nvidia-smi` should work
   - No manual patches required (unlike MASt3R-SLAM which needed 4 patches)

---

## Timeline

- **Start:** 2025-11-19 16:00
- **Phase 1 Complete:** 2025-11-19 16:15 (GCC-14 installed)
- **Phase 2 Complete:** 2025-11-19 16:20 (CMake 4.1.3, Ninja, X11 libs)
- **Phase 3 Complete:** 2025-11-19 16:25 (vcpkg installed)
- **Phase 4 Complete:** 2025-11-19 16:30 (LibTorch verified present)
- **Phase 5 Complete:** 2025-11-19 16:55 (Binary built successfully, sm_120 confirmed)
- **Phase 6 Complete:** 2025-11-19 16:58 (Truck dataset downloaded - 651MB, 251 images)
- **Phase 7 Complete:** 2025-11-19 17:00 (Test run successful - 384 iter/s, sm_120 working)
- **End:** 2025-11-19 17:00 (All phases complete, Blackwell support verified)
