#!/usr/bin/env python3
"""
patch_colmap_data.py

Split large COLMAP reconstructions into smaller patches for gaussian splatting training.


Usage:
    # Using COLMAP sparse points (from points3D.bin)
    python patch_colmap_data.py \
        --sparse /path/to/sparse/0 \
        --images /path/to/images \
        --output /path/to/output \
        --use-colmap-points \
        --sample-percentage 10.0 \
        --max-cameras 400 \
        --buffer 0.1

    # Or use, separate dense point cloud (PLY)
    python patch_colmap_data.py \
        --sparse /path/to/sparse/0 \
        --images /path/to/images \
        --output /path/to/output \
        --pointcloud /path/to/dense.ply \
        --sample-percentage 10.0 \
        --max-cameras 400 \
        --buffer 0.1
"""

import argparse
import os
import subprocess
import pycolmap
from pathlib import Path
from wildflow import splat
from typing import Dict, Any, List


class PatchConfig:
    """Configuration for patching COLMAP data."""
    
    def __init__(self, args):
        # Input paths
        self.sparse_path = Path(args.sparse)
        self.images_path = Path(args.images)
        self.pointcloud_path = Path(args.pointcloud) if args.pointcloud else None
        
        # Output path
        self.output_path = Path(args.output)
        
        # Patching settings
        self.max_cameras = args.max_cameras
        self.buffer_meters = args.buffer
        self.sample_percentage = args.sample_percentage
        self.use_colmap_points = args.use_colmap_points
        
        # Validate inputs
        self._validate()
    
    def _validate(self):
        """Validate that required input files exist."""
        if not self.sparse_path.exists():
            raise FileNotFoundError(f"Sparse directory not found: {self.sparse_path}")
        
        cameras_bin = self.sparse_path / "cameras.bin"
        images_bin = self.sparse_path / "images.bin"
        points3d_bin = self.sparse_path / "points3D.bin"
        
        if not cameras_bin.exists():
            raise FileNotFoundError(f"cameras.bin not found: {cameras_bin}")
        if not images_bin.exists():
            raise FileNotFoundError(f"images.bin not found: {images_bin}")
        if not points3d_bin.exists():
            raise FileNotFoundError(f"points3D.bin not found: {points3d_bin}")
        
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        
        # Validate point cloud options
        if self.use_colmap_points and self.pointcloud_path:
            raise ValueError("Cannot use both --use-colmap-points and --pointcloud. Choose one.")
        
        if self.pointcloud_path and not self.pointcloud_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {self.pointcloud_path}")


def step1_create_patches(config: PatchConfig):
    """
    Create patches based on camera positions.
    
    Analyzes camera positions and creates overlapping rectangular patches
    that each contain at most max_cameras cameras.
    """
    print(f"\n{'='*70}")
    print("STEP 1: Creating Patches")
    print(f"{'='*70}")
    
    model = pycolmap.Reconstruction(str(config.sparse_path))
    camera_poses = [img.projection_center() for img in model.images.values()]
    camera_z_values = [pos[2] for pos in camera_poses]
    cameras_2d = [(pos[0], pos[1]) for pos in camera_poses]
    
    print(f"Total cameras: {len(cameras_2d)}")
    
    # Add some Z buffer for points above/below camera heights
    min_z = min(camera_z_values) - 2.0
    max_z = max(camera_z_values) + 0.5
    
    patches_list = splat.patches(
        cameras_2d, 
        max_cameras=config.max_cameras, 
        buffer_meters=config.buffer_meters
    )

    print(f"✓ Created {len(patches_list)} patches")
    print(f"  Max cameras per patch: {config.max_cameras}")
    print(f"  Buffer between patches: {config.buffer_meters}m")
    print(f"  Z range: [{min_z:.1f}, {max_z:.1f}]")
    
    return patches_list, min_z, max_z


def step2_split_cameras(config: PatchConfig, patches_list: List[Dict], min_z: float, max_z: float):
    """
    Split cameras.bin and images.bin into patches.
    
    Each patch gets its own sparse/0 directory with cameras.bin and images.bin.
    If use_colmap_points=True, also splits points3D.bin from COLMAP reconstruction.
    """
    print(f"\n{'='*70}")
    print("STEP 2: Splitting Cameras & Images")
    if config.use_colmap_points:
        print("         (including COLMAP points3D.bin)")
    print(f"{'='*70}")
    
    result = splat.split_cameras({
        "input_path": str(config.sparse_path),
        "min_z": min_z,
        "max_z": max_z,
        "save_points3d": config.use_colmap_points,  # Split COLMAP points if requested
        "patches": [
            {**patch, "output_path": str(config.output_path / f"p{i}" / "sparse" / "0")}
            for i, patch in enumerate(patches_list)
        ]
    })
    
    print(f"✓ Split cameras: {result['total_cameras_written']} cameras, {result['total_images_written']} images")
    print(f"  across {len(patches_list)} patches")
    
    # Export to .txt format as well for compatibility
    print(f"\nExporting patches to .txt format...")
    for i in range(len(patches_list)):
        patch_sparse = config.output_path / f"p{i}" / "sparse" / "0"
        if patch_sparse.exists():
            try:
                reconstruction = pycolmap.Reconstruction(str(patch_sparse))
                reconstruction.write_text(str(patch_sparse))
                #TODO: I think the images.txt is be export in wrong format. Way too many entries per row.
                if config.use_colmap_points:
                    print(f"  ✓ p{i}: cameras.txt, images.txt, points3D.txt exported")
                else:
                    print(f"  ✓ p{i}: cameras.txt, images.txt exported")
            except Exception as e:
                print(f"  ✗ p{i}: Failed to export txt - {e}")
    
    return {"result": result, "patches_count": len(patches_list)}


def step3_split_pointcloud(config: PatchConfig, patches_list: List[Dict], min_z: float, max_z: float):
    """
    Split dense point cloud into patches.
    
    Takes a PLY point cloud and splits it into patches, saving as points3D.bin.
    Each patch only gets points within its boundaries.
    Uses sample_percentage to downsample the point cloud (fewer points = faster training).
    
    Note: This is only used if --pointcloud is provided. If --use-colmap-points is used,
    the points3D.bin splitting happens in step2_split_cameras instead.
    """
    if config.use_colmap_points:
        print(f"\n{'='*70}")
        print("STEP 3: Splitting Point Cloud")
        print(f"{'='*70}")
        print("✓ Using COLMAP points3D.bin (already split in Step 2)")
        return {"skipped": False, "used_colmap_points": True}
    
    if not config.pointcloud_path:
        print(f"\n{'='*70}")
        print("STEP 3: Splitting Point Cloud")
        print(f"{'='*70}")
        print("⚠️  No point cloud provided - skipping")
        print("   Patches will be created without points3D.bin")
        print("   You can train with random initialization or add points later")
        return {"skipped": True}
    
    print(f"\n{'='*70}")
    print("STEP 3: Splitting Point Cloud")
    print(f"{'='*70}")
    print(f"Input: {config.pointcloud_path}")
    print(f"Sample percentage: {config.sample_percentage}%")
    
    coords = lambda p: {k: p[k] for k in ('min_x', 'max_x', 'min_y', 'max_y')}
    
    result = splat.split_point_cloud({
        "input_file": str(config.pointcloud_path),
        "min_z": min_z,
        "max_z": max_z,
        "sample_percentage": config.sample_percentage,
        "patches": [
            {**coords(patch), "output_file": str(config.output_path / f"p{i}" / "sparse" / "0" / "points3D.bin")}
            for i, patch in enumerate(patches_list)
        ]
    })
    
    print(f"✓ Split point cloud: {result['points_loaded']:,} → {result['total_points_written']:,} points")
    print(f"  ({config.sample_percentage}% sampling)")
    
    # Export to .txt format as well
    print(f"\nExporting point clouds to .txt format...")
    for i in range(len(patches_list)):
        patch_sparse = config.output_path / f"p{i}" / "sparse" / "0"
        points3d_bin = patch_sparse / "points3D.bin"
        if points3d_bin.exists():
            try:
                reconstruction = pycolmap.Reconstruction(str(patch_sparse))
                reconstruction.write_text(str(patch_sparse))
                print(f"  ✓ p{i}: points3D.txt exported ({len(reconstruction.points3D)} points)")
            except Exception as e:
                print(f"  ✗ p{i}: Failed to export txt - {e}")
    
    return {"result": result}


# TODO: Re-enable training step with LichtFeld-Studio
# 
# The original script used Postshot CLI for training. Below is the commented-out
# training code for reference. When ready to add LichtFeld-Studio training:
#
# 1. Replace postshot_exe with path to LichtFeld-Studio executable
# 2. Update command line arguments for LichtFeld-Studio format
# 3. Adjust output paths and log parsing for LichtFeld-Studio
# 4. Consider using run_lichtfeld.sh scripts if available
#
# Example LichtFeld-Studio command structure:
#   ./lichtfeld-studio \
#       --sparse /path/to/sparse/0 \
#       --images /path/to/images \
#       --output /path/to/output.ply \
#       --iterations 30000 \
#       --gpu 0
#
# Original Postshot training code:
"""
def train_patch(patch_idx, gpu_id, config):
    paths = {
        "sparse": config.output_path / f"p{patch_idx}" / "sparse" / "0",
        "output": config.output_path / f"p{patch_idx}" / f"raw-p{patch_idx}.ply",
        "log": config.output_path / f"p{patch_idx}" / f"train_p{patch_idx}_gpu{gpu_id}.log"
    }
    
    cmd = [
        config.postshot_exe, "train",
        "-i", str(paths["sparse"] / "cameras.bin"),
        "-i", str(paths["sparse"] / "images.bin"), 
        "-i", str(paths["sparse"] / "points3D.bin"),
        "-i", str(config.images_path),
        "-p", "Splat3",
        "--gpu", str(gpu_id),
        "--train-steps-limit", str(config.train_steps),
        "--show-train-error",
        "--export-splat-ply", str(paths["output"])
    ]
    
    # Training execution code...
    # See patch_models.py for full implementation
"""


def print_summary(config: PatchConfig, patches_list: List[Dict]):
    """Print summary of what was created."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Created {len(patches_list)} patches in: {config.output_path}")
    print()
    print("Each patch contains:")
    print("  sparse/0/")
    print("    ├── cameras.bin + cameras.txt")
    print("    ├── images.bin + images.txt")
    print("    └── points3D.bin + points3D.txt (if point cloud provided)")

def main():
    parser = argparse.ArgumentParser(
        description="Split COLMAP reconstruction into patches for gaussian splatting training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using COLMAP sparse points (fastest, no PLY needed)
  python patch_colmap_data.py \\
      --sparse /data/colmap/sparse/0 \\
      --images /data/images \\
      --output /data/patches \\
      --use-colmap-points

  # Using dense point cloud for better initialization
  python patch_colmap_data.py \\
      --sparse /data/colmap/sparse/0 \\
      --images /data/images \\
      --output /data/patches \\
      --pointcloud /data/dense_pointcloud.ply \\
      --sample-percentage 8.0

  # Custom patching settings
  python patch_colmap_data.py \\
      --sparse /data/colmap/sparse/0 \\
      --images /data/images \\
      --output /data/patches \\
      --use-colmap-points \\
      --max-cameras 1000 \\
      --buffer 1.5
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--sparse',
        type=str,
        required=True,
        help='Path to COLMAP sparse/0 directory (contains cameras.bin, images.bin, points3D.bin)'
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to directory containing source images'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for patches (will create p0/, p1/, etc.)'
    )
    
    # Optional arguments - Point cloud source (choose one)
    point_cloud_group = parser.add_mutually_exclusive_group()
    point_cloud_group.add_argument(
        '--use-colmap-points',
        action='store_true',
        help='Use points3D.bin from COLMAP sparse reconstruction (default if no --pointcloud)'
    )
    point_cloud_group.add_argument(
        '--pointcloud',
        type=str,
        default=None,
        help='Path to dense point cloud PLY file (for denser initialization than COLMAP points)'
    )
    parser.add_argument(
        '--max-cameras',
        type=int,
        default=1200,
        help='Maximum cameras per patch (default: 1200)'
    )
    parser.add_argument(
        '--buffer',
        type=float,
        default=0.8,
        help='Buffer overlap between patches IN METERS (default: 0.8). Creates overlap for smooth merging. Typical: 0.5-2.0m'
    )
    parser.add_argument(
        '--sample-percentage',
        type=float,
        default=5.0,
        help='Percentage of point cloud to use when using --pointcloud (default: 5.0). Ignored with --use-colmap-points'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PatchConfig(args)
    
    print("="*70)
    print("COLMAP Patching Workflow")
    print("="*70)
    print(f"Input sparse:    {config.sparse_path}")
    print(f"Input images:    {config.images_path}")
    
    if config.use_colmap_points:
        print(f"Point cloud:     COLMAP points3D.bin (from sparse reconstruction)")
    elif config.pointcloud_path:
        print(f"Point cloud:     {config.pointcloud_path} (dense PLY)")
        print(f"Sample %:        {config.sample_percentage}%")
    else:
        print(f"Point cloud:     None (will train with random initialization)")
    
    print(f"Output:          {config.output_path}")
    print(f"Max cameras:     {config.max_cameras}")
    print(f"Buffer:          {config.buffer_meters}m (overlap between patches)")
    
    # Ensure output directory exists
    config.output_path.mkdir(parents=True, exist_ok=True)
    
    # Run workflow steps
    patches_list, min_z, max_z = step1_create_patches(config)
    step2_split_cameras(config, patches_list, min_z, max_z)
    step3_split_pointcloud(config, patches_list, min_z, max_z)
    
    # TODO: Add training step here when LichtFeld-Studio integration is ready
    # step4_train_patches(config, patches_list)
    
    # TODO: Add cleanup step (see patch_models.py step5_cleanup_splats)
    # This removes outlier splats outside patch boundaries and with bad properties
    # step5_cleanup_splats(config, patches_list, min_z, max_z)
    
    # TODO: Add merge step (see patch_models.py merge_clean_ply_files)
    # This combines all cleaned patches into one full-model.ply
    # step6_merge_patches(config)
    
    print_summary(config, patches_list)
    
    print("Patching completed successfully!")


if __name__ == "__main__":
    main()
