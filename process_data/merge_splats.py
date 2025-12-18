#!/usr/bin/env python3
"""
merge_splats.py

Merge all cleaned gaussian splat patches into a single unified PLY file.
Uses wildflow.splat.merge_ply_files to concatenate vertex data from all patches.

The script finds all cleaned splat files (*_clean.ply) or raw splat files
and merges them into a single output file.

Usage:
  python merge_splats.py --config splat_config.yml
"""

import argparse
import sys
from pathlib import Path
import yaml
from wildflow import splat
import re


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_highest_iteration_splat(splat_dir: Path, patch_name: str, prefer_cleaned: bool = True) -> Path:
    """
    Find the splat PLY file with the highest iteration number.
    
    If prefer_cleaned=True, looks for cleaned files (*_clean.ply) first.
    Falls back to raw splat files if cleaned versions don't exist.
    
    Returns the path to the best available splat file.
    """
    # Pattern for prefixed files (e.g., p0_splat_10000.ply, p0_splat_10000_clean.ply)
    pattern_prefixed = re.compile(rf'{re.escape(patch_name)}_splat_(\d+)(_clean)?\.ply')
    # Pattern for legacy files (e.g., splat_10000.ply, splat_10000_clean.ply)
    pattern_legacy = re.compile(r'splat_(\d+)(_clean)?\.ply')
    
    # Collect all splat files with their iteration numbers
    splat_files = []
    
    for f in splat_dir.glob('*.ply'):
        match = pattern_prefixed.match(f.name)
        if not match:
            match = pattern_legacy.match(f.name)
        
        if match:
            iteration = int(match.group(1))
            is_clean = match.group(2) is not None
            splat_files.append((f, iteration, is_clean))
    
    if not splat_files:
        return None
    
    # Find highest iteration
    max_iter = max(f[1] for f in splat_files)
    
    # Filter to highest iteration files
    max_iter_files = [f for f in splat_files if f[1] == max_iter]
    
    if prefer_cleaned:
        # Prefer cleaned version at highest iteration
        cleaned = [f for f in max_iter_files if f[2]]
        if cleaned:
            return cleaned[0][0]
    
    # Fall back to raw version
    raw = [f for f in max_iter_files if not f[2]]
    if raw:
        return raw[0][0]
    
    # If only cleaned exists and we didn't prefer it, still use it
    if max_iter_files:
        return max_iter_files[0][0]
    
    return None


def find_all_patch_splats(patches_dir: Path, prefer_cleaned: bool = True) -> list:
    """
    Find all patch splat files in the patches directory.
    
    Returns list of tuples: (patch_name, splat_file_path)
    """
    patches = []
    
    # Find all patch directories (p0, p1, p2, ...)
    patch_dirs = sorted(
        [d for d in patches_dir.iterdir() if d.is_dir() and re.match(r'p\d+$', d.name)],
        key=lambda d: int(d.name[1:])  # Sort by patch number
    )
    
    for patch_dir in patch_dirs:
        patch_name = patch_dir.name
        splat_dir = patch_dir / "sparse" / "splat"
        
        if not splat_dir.exists():
            continue
        
        splat_file = find_highest_iteration_splat(splat_dir, patch_name, prefer_cleaned)
        if splat_file:
            patches.append((patch_name, splat_file))
    
    return patches


def main():
    parser = argparse.ArgumentParser(
        description='Merge all gaussian splat patches into a single PLY file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python merge_splats.py --config splat_config.yml

Configuration file should contain merge parameters and paths.
        """
    )
    
    parser.add_argument('--config', required=True,
                       help='Path to splat_config.yml configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(2)
    
    config = load_config(config_path)
    
    # Extract paths and settings from config
    patches_dir = Path(config['paths']['patches_dir']).expanduser()
    merge_config = config.get('merge', {})
    
    # Get output path from paths.merged_splat, default to patches_dir/merged_splat.ply
    output_file = config['paths'].get('merged_splat')
    if output_file:
        output_file = Path(output_file).expanduser()
    else:
        output_file = patches_dir / "merged_splat.ply"
    
    prefer_cleaned = merge_config.get('prefer_cleaned', True)
    
    # Validate patches directory
    if not patches_dir.exists():
        print(f"ERROR: Patches directory not found: {patches_dir}")
        sys.exit(2)
    
    print("="*70)
    print("Merge Gaussian Splats")
    print("="*70)
    print(f"Patches dir:    {patches_dir}")
    print(f"Output file:    {output_file}")
    print(f"Prefer cleaned: {prefer_cleaned}")
    print()
    
    # Find all patch splat files
    patch_splats = find_all_patch_splats(patches_dir, prefer_cleaned)
    
    if not patch_splats:
        print("ERROR: No splat files found in any patches")
        print("       Train and optionally clean splats first")
        sys.exit(2)
    
    print(f"Found {len(patch_splats)} patch splat files to merge:")
    total_size_mb = 0
    for patch_name, splat_file in patch_splats:
        size_mb = splat_file.stat().st_size / (1024 * 1024)
        total_size_mb += size_mb
        clean_marker = " (cleaned)" if "_clean" in splat_file.name else " (raw)"
        print(f"  {patch_name}: {splat_file.name}{clean_marker} ({size_mb:.1f} MB)")
    
    print()
    print(f"Total input size: {total_size_mb:.1f} MB")
    print()
    
    # Check if output already exists
    if output_file.exists():
        print(f"WARNING: Output file already exists: {output_file}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(0)
        print()
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build merge configuration
    merge_params = {
        "input_files": [str(f) for _, f in patch_splats],
        "output_file": str(output_file)
    }
    
    print("Running wildflow.splat.merge_ply_files...")
    try:
        splat.merge_ply_files(merge_params)
        
        print()
        print(f"SUCCESS: Merged splat saved to: {output_file}")
        
        # Show output file size
        output_size_mb = output_file.stat().st_size / (1024 * 1024)
        print()
        print("File sizes:")
        print(f"  Total input:  {total_size_mb:.1f} MB")
        print(f"  Merged output: {output_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"\nERROR: Merge failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
