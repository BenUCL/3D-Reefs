#!/usr/bin/env python3
"""
clean_splats.py

Clean gaussian splat models by filtering splats based on quality criteria.
Uses wildflow.splat.cleanup_splats to remove:
  - Splats outside patch boundaries
  - Oversized splats (high area)
  - Isolated splats (few neighbors)

The script finds the highest iteration splat file (e.g., splat_20000.ply) and
creates a cleaned version with '_clean' suffix (e.g., splat_20000_clean.ply).

Usage:
  python clean_splats.py --config splat_config.yml --patch p0
"""

import argparse
import sys
from pathlib import Path
import yaml
from wildflow import splat
import re
import json


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_highest_iteration_splat(splat_dir: Path, patch_name: str) -> Path:
    """
    Find the splat PLY file with the highest iteration number.
    
    Looks for files like p0_splat_10000.ply, p0_splat_20000.ply, etc.
    Also supports legacy filenames without prefix: splat_10000.ply, splat_20000.ply
    Returns the one with the highest iteration number.
    """
    # Try prefixed files first (e.g., p0_splat_10000.ply)
    splat_files = list(splat_dir.glob(f'{patch_name}_splat_*.ply'))
    
    # Fall back to legacy non-prefixed files if none found
    if not splat_files:
        splat_files = list(splat_dir.glob('splat_*.ply'))
    
    if not splat_files:
        raise FileNotFoundError(f"No splat_*.ply files found in {splat_dir}")
    
    # Extract iteration numbers and find max
    # Support both formats: p0_splat_10000.ply and splat_10000.ply
    pattern_prefixed = re.compile(rf'{re.escape(patch_name)}_splat_(\d+)\.ply')
    pattern_legacy = re.compile(r'splat_(\d+)\.ply')
    max_iter = -1
    max_file = None
    
    for f in splat_files:
        match = pattern_prefixed.match(f.name)
        if not match:
            match = pattern_legacy.match(f.name)
        
        if match:
            iteration = int(match.group(1))
            if iteration > max_iter:
                max_iter = iteration
                max_file = f
    
    if max_file is None:
        raise FileNotFoundError(f"No valid splat_*.ply files found in {splat_dir}")
    
    return max_file


def get_patch_boundaries(patch_dir: Path, buffer_meters: float) -> dict:
    """
    Get patch boundaries from patch metadata and shrink by buffer.
    
    Reads patch_metadata.json created during patching, which contains
    the original patch boundaries. Shrinks boundaries inward by
    buffer_meters to exclude the overlap zone.
    """
    metadata_file = patch_dir / "patch_metadata.json"
    
    if not metadata_file.exists():
        # Throw error
        print(f" patch_metadata.json not found, cannot apply boundary filtering")
        print(f" Re-run patch_colmap_data.py to generate metadata")
        raise FileNotFoundError(f"patch_metadata.json not found in {patch_dir}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Shrink boundaries by buffer amount to exclude overlap zone
    boundaries = {
        "min_x": metadata['min_x'] + buffer_meters,
        "max_x": metadata['max_x'] - buffer_meters,
        "min_y": metadata['min_y'] + buffer_meters,
        "max_y": metadata['max_y'] - buffer_meters,
        "min_z": metadata['min_z'],
        "max_z": metadata['max_z']
    }
    
    return boundaries


def main():
    parser = argparse.ArgumentParser(
        description='Clean gaussian splat models using quality criteria',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python clean_splats.py --config splat_config.yml --patch p0
  python clean_splats.py --config splat_config.yml --patch p1

Configuration file should contain cleanup parameters and paths.
        """
    )
    
    parser.add_argument('--config', required=True,
                       help='Path to splat_config.yml configuration file')
    parser.add_argument('--patch', required=True,
                       help='Patch name to clean (e.g., p0, p1, p2)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(2)
    
    config = load_config(config_path)
    
    # Extract paths and settings from config
    patches_dir = Path(config['paths']['patches_dir']).expanduser()
    cleanup_config = config['cleanup']
    patching_config = config.get('patching', {})
    
    # Build paths for this specific patch
    patch_name = args.patch
    patch_dir = patches_dir / patch_name
    splat_dir = patch_dir / "sparse" / "splat"
    
    # Validate paths
    if not patch_dir.exists():
        print(f"ERROR: Patch directory not found: {patch_dir}")
        sys.exit(2)
    
    if not splat_dir.exists():
        print(f"ERROR: Splat directory not found: {splat_dir}")
        print(f"       Train the splat first using train_splat.py")
        sys.exit(2)
    
    print("="*70)
    print("Gaussian Splat Cleanup")
    print("="*70)
    print(f"Patch:       {patch_name}")
    print(f"Splat dir:   {splat_dir}")
    print()
    
    # Find highest iteration splat file
    try:
        input_file = find_highest_iteration_splat(splat_dir, patch_name)
        print(f"Found splat file: {input_file.name}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(2)
    
    # Create output filename with '_clean' suffix
    base_name = input_file.stem  # e.g., 'splat_20000'
    output_file = input_file.parent / f"{base_name}_clean.ply"
    
    # Check if already cleaned
    if output_file.exists():
        print(f"\nWARNING: Cleaned file already exists: {output_file.name}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping cleanup")
            sys.exit(0)
    
    print()
    print("Cleanup parameters:")
    print(f"  max_area:       {cleanup_config['max_area']}")
    print(f"  min_neighbors:  {cleanup_config['min_neighbors']}")
    print(f"  radius:         {cleanup_config['radius']}m")
    if cleanup_config.get('filter_boundaries', False):
        print(f"  boundary filter: enabled (buffer: {cleanup_config.get('boundary_buffer', 0.0)}m)")
    print()
    
    # Get patch boundaries from patching config if boundary filtering enabled
    boundaries = {}
    if cleanup_config.get('filter_boundaries', False):
        boundary_buffer = cleanup_config.get('boundary_buffer', 0.0)
        boundaries = get_patch_boundaries(patch_dir, boundary_buffer)
        if boundaries:
            print(f"Applying boundary filtering (shrunk by {boundary_buffer}m):")
            print(f"  X: [{boundaries['min_x']:.2f}, {boundaries['max_x']:.2f}]")
            print(f"  Y: [{boundaries['min_y']:.2f}, {boundaries['max_y']:.2f}]")
            print(f"  Z: [{boundaries['min_z']:.2f}, {boundaries['max_z']:.2f}]")
            print()
    
    # Build cleanup configuration
    # TODO: Add disposed_file parameter to save filtered-out splats for quality verification
    #       This would allow examining which splats were removed to validate cleanup parameters
    cleanup_params = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "max_area": cleanup_config['max_area'],
        "min_neighbors": cleanup_config['min_neighbors'],
        "radius": cleanup_config['radius']
    }
    
    # Add spatial boundaries if enabled and available
    if boundaries:
        cleanup_params.update(boundaries)
    
    print("Running wildflow.splat.cleanup_splats...")
    try:
        splat.cleanup_splats(cleanup_params)
        print()
        print(f"SUCCESS: Cleaned splat saved to: {output_file.name}")
        
        # Show file sizes for comparison
        input_size_mb = input_file.stat().st_size / (1024 * 1024)
        output_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print()
        print("File sizes:")
        print(f"  Original:  {input_size_mb:.1f} MB")
        print(f"  Cleaned:   {output_size_mb:.1f} MB")
        print(f"  Removed:   {input_size_mb - output_size_mb:.1f} MB ({100 * (1 - output_size_mb/input_size_mb):.1f}%)")
        
    except Exception as e:
        print(f"\nERROR: Cleanup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
