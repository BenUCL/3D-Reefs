#!/usr/bin/env python3
"""
patch_colmap_data.py

Split large COLMAP reconstructions into smaller spatial patches for gaussian splatting.

This script analyzes camera positions and creates overlapping rectangular patches,
splitting cameras.bin, images.bin, and optionally points3D.bin into separate directories.

Usage:
    python patch_colmap_data.py --config splat_config.yml

Configuration is read from splat_config.yml which contains:
    - Paths (sparse, images, output)
    - Patching parameters (max_cameras, buffer_meters)
    - Point cloud options (use COLMAP points or external PLY)
"""

import argparse
import os
import subprocess
import pycolmap
from pathlib import Path
from wildflow import splat
from typing import Dict, Any, List
import yaml


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class PatchConfig:
    """Configuration for patching COLMAP data."""
    
    def __init__(self, config: dict):
        # Extract patching config section
        patch_cfg = config.get('patching', {})
        
        # Input paths
        self.sparse_path = Path(patch_cfg['sparse_dir'])
        self.images_path = Path(patch_cfg['images_dir'])
        self.pointcloud_path = Path(patch_cfg['pointcloud_path']) if patch_cfg.get('pointcloud_path') else None
        
        # Output path
        self.output_path = Path(config['paths']['patches_dir'])
        
        # Patching settings
        self.max_cameras = patch_cfg.get('max_cameras', 1200)
        self.buffer_meters = patch_cfg.get('buffer_meters', 0.8)
        self.sample_percentage = patch_cfg.get('sample_percentage', 5.0)
        self.use_colmap_points = patch_cfg.get('use_colmap_points', True)
        
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
        description="Split COLMAP reconstruction into spatial patches for gaussian splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python patch_colmap_data.py --config splat_config.yml

Configuration is read from splat_config.yml which should contain:
  - patching.sparse_dir: Path to COLMAP sparse/0 directory
  - patching.images_dir: Path to images directory
  - paths.patches_dir: Output directory for patches
  - patching.max_cameras: Maximum cameras per patch (default: 1200)
  - patching.buffer_meters: Overlap between patches in meters (default: 0.8)
        """
    )
    
    parser.add_argument('--config', required=True,
                       help='Path to splat_config.yml configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    config = load_config(config_path)
    
    # Create configuration object
    patch_config = PatchConfig(config)
    
    print("="*70)
    print("COLMAP Patching Workflow")
    print("="*70)
    print(f"Input sparse:    {patch_config.sparse_path}")
    print(f"Input images:    {patch_config.images_path}")
    
    if patch_config.use_colmap_points:
        print(f"Point cloud:     COLMAP points3D.bin (from sparse reconstruction)")
    elif patch_config.pointcloud_path:
        print(f"Point cloud:     {patch_config.pointcloud_path} (dense PLY)")
        print(f"Sample %:        {patch_config.sample_percentage}%")
    else:
        print(f"Point cloud:     None (will train with random initialization)")
    
    print(f"Output:          {patch_config.output_path}")
    print(f"Max cameras:     {patch_config.max_cameras}")
    print(f"Buffer:          {patch_config.buffer_meters}m (overlap between patches)")
    
    # Ensure output directory exists
    patch_config.output_path.mkdir(parents=True, exist_ok=True)
    
    # Run workflow steps
    patches_list, min_z, max_z = step1_create_patches(patch_config)
    step2_split_cameras(patch_config, patches_list, min_z, max_z)
    step3_split_pointcloud(patch_config, patches_list, min_z, max_z)
    
    print_summary(patch_config, patches_list)
    
    print("Patching completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
