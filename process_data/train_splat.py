#!/usr/bin/env python3
"""
train_splat.py

Train gaussian splats with LichtFeld-Studio on multi-camera datasets where images 
are stored in separate folders per camera.

This script creates a temporary directory structure that LichtFeld expects:
  temp_dir/
    sparse/0/     (symlink to your sparse folder)
    images/       (directory with symlinks to actual images)

The images.txt file contains paths like "left/image.png" or "right/image.png",
and this script creates the proper directory structure so LichtFeld can find them.

Usage:
  python train_splat.py --config splat_config.yml --patch p0

Configuration:
  - Edit splat_config.yml to configure paths, camera mapping, and training parameters
"""

import argparse
import subprocess
import shlex
import sys
from pathlib import Path
import datetime
import re
import os
import json
import tempfile
import shutil
import yaml


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\|\s*Loss:\s*([0-9.eE+-]+)\s*\|\s*Splats:\s*(\d+)")


def setup_lichtfeld_structure(sparse_dir: Path, images_dir: Path, temp_dir: Path, camera_mapping: dict):
    """
    Create temporary directory structure that LichtFeld expects:
    
    temp_dir/
      sparse/0/       -> symlink to actual sparse/0
      images/
        left/         -> symlink to actual images/left
        right/        -> symlink to actual images/right
    
    This allows LichtFeld to resolve image paths like "left/image.png" correctly.
    """
    print(f"Setting up LichtFeld-compatible directory structure in: {temp_dir}")
    
    # Create temp structure
    temp_sparse = temp_dir / 'sparse'
    temp_sparse.mkdir(parents=True, exist_ok=True)
    temp_images = temp_dir / 'images'
    temp_images.mkdir(parents=True, exist_ok=True)
    
    # Symlink sparse/0 -> actual sparse folder
    sparse_link = temp_sparse / '0'
    if sparse_link.exists():
        sparse_link.unlink()
    sparse_link.symlink_to(sparse_dir.resolve(), target_is_directory=True)
    print(f"  ✓ Created sparse/0 -> {sparse_dir}")
    
    # Symlink each image subfolder
    for subfolder in camera_mapping.keys():
        src_folder = images_dir / subfolder
        if not src_folder.exists():
            print(f"  ⚠️  Warning: {subfolder} not found in {images_dir}")
            continue
        
        dest_folder = temp_images / subfolder
        if dest_folder.exists():
            dest_folder.unlink()
        dest_folder.symlink_to(src_folder.resolve(), target_is_directory=True)
        
        # Count images
        img_count = sum(1 for _ in src_folder.glob('**/*') if _.is_file() and _.suffix.lower() in ['.png', '.jpg', '.jpeg'])
        print(f"  ✓ Created images/{subfolder} -> {src_folder} ({img_count} images)")
    
    print(f"  ✓ LichtFeld dataset structure ready: {temp_dir}")
    return temp_dir


def gather_metadata(sparse_dir: Path, images_dir: Path, camera_mapping: dict):
    """Collect metadata about the dataset."""
    meta = {}
    
    # Count images from each camera folder
    camera_counts = {}
    total_images = 0
    for subfolder, cam_id in camera_mapping.items():
        folder = images_dir / subfolder
        if folder.exists():
            count = sum(1 for _ in folder.glob('**/*') if _.is_file() and _.suffix.lower() in ['.png', '.jpg', '.jpeg'])
            camera_counts[f'camera_{cam_id}_{subfolder}'] = count
            total_images += count
    
    meta['num_images'] = total_images
    meta['camera_counts'] = camera_counts
    meta['images_path'] = str(images_dir)
    meta['sparse_folder'] = str(sparse_dir)
    
    # Check for camera files
    cameras_bin = sparse_dir / 'cameras.bin'
    cameras_txt = sparse_dir / 'cameras.txt'
    meta['cameras_bin'] = str(cameras_bin) if cameras_bin.exists() else ''
    meta['cameras_txt'] = str(cameras_txt) if cameras_txt.exists() else ''
    
    # Count poses from images.txt if available
    imgs_txt = sparse_dir / 'images.txt'
    if imgs_txt.exists():
        try:
            # Count non-comment, non-empty lines, divide by 2 (COLMAP format has 2 lines per image)
            lines = [l for l in imgs_txt.open('r').readlines() if l.strip() and not l.startswith('#')]
            meta['num_poses'] = len(lines) // 2
        except Exception:
            meta['num_poses'] = ''
    else:
        meta['num_poses'] = ''
    
    return meta


def run_lichtfeld(cmd_list, out_dir: Path):
    """Run LichtFeld and capture output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'run.log'
    all_lines = []
    progress = []

    # Set up environment to use LichtFeld's standalone libtorch
    env = os.environ.copy()
    lichtfeld_bin = Path(cmd_list[0])
    if lichtfeld_bin.exists():
        lichtfeld_root = lichtfeld_bin.parent.parent
        libtorch_path = lichtfeld_root / 'external' / 'libtorch' / 'lib'
        if libtorch_path.exists():
            current_ld_path = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = f"{libtorch_path}:{current_ld_path}" if current_ld_path else str(libtorch_path)

    with open(log_path, 'wb', buffering=0) as logfile:
        proc = subprocess.Popen(
            cmd_list, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False,
            env=env
        )

        assert proc.stdout is not None
        buffer = b''
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                if buffer:
                    line = buffer.decode('utf-8', errors='replace')
                    sys.stdout.write(line + '\n')
                    sys.stdout.flush()
                    logfile.write(buffer + b'\n')
                    all_lines.append(line)
                    m = PROGRESS_RE.search(line)
                    if m:
                        progress.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))))
                break
            
            logfile.write(chunk)
            
            if chunk == b'\n':
                line = buffer.decode('utf-8', errors='replace')
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
                all_lines.append(line)
                m = PROGRESS_RE.search(line)
                if m:
                    progress.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))))
                buffer = b''
            elif chunk == b'\r':
                line = buffer.decode('utf-8', errors='replace')
                sys.stdout.write('\r' + line)
                sys.stdout.flush()
                m = PROGRESS_RE.search(line)
                if m:
                    progress.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))))
                buffer = b''
            else:
                buffer += chunk
        
        return_code = proc.wait()
    
    return {'all_lines': all_lines, 'progress': progress, 'log': str(log_path), 'return_code': return_code}


def write_report(out_dir: Path, cmd_str: str, meta: dict, run_result: dict, camera_mapping: dict):
    """Write run report with metadata and progress."""
    report_path = out_dir / 'run_report.txt'
    timestamp = datetime.datetime.now().isoformat()

    header = []
    header.append(f"Timestamp: {timestamp}")
    header.append("Command:")
    header.append(cmd_str)
    header.append("")
    header.append("Metadata:")
    header.append(json.dumps(meta, indent=2))
    header.append("")
    header.append("Camera Folder Mapping:")
    for folder, cam_id in camera_mapping.items():
        header.append(f"  {folder}/ -> Camera ID {cam_id}")
    header.append("")
    header.append("Generated files in output dir:")
    files = [p.name for p in out_dir.glob('*') if p.is_file()]
    header.append('\n'.join(sorted(files)))
    header.append("")
    header.append("--- Training progress (last lines appended below) ---")

    with report_path.open('w') as f:
        for line in header:
            f.write(line + '\n')
        f.write('\n')
        f.write('Progress lines (most recent last):\n')
        for tup in run_result['progress']:
            f.write(f"{tup[0]}/{tup[1]} | Loss: {tup[2]:.6f} | Splats: {tup[3]}\n")

    return str(report_path)


def main():
    parser = argparse.ArgumentParser(
        description='Train gaussian splats with LichtFeld on multi-camera datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python train_splat.py --config splat_config.yml --patch p0
  python train_splat.py --config splat_config.yml --patch p1

Configuration file should contain paths, camera mapping, and training parameters.
        """
    )
    
    parser.add_argument('--config', required=True, 
                       help='Path to splat_config.yml configuration file')
    parser.add_argument('--patch', required=True, 
                       help='Patch name to train (e.g., p0, p1, p2)')

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(2)
    
    config = load_config(config_path)
    
    # Extract paths and settings from config
    lf_bin = Path(config['paths']['lichtfeld_bin']).expanduser()
    patches_dir = Path(config['paths']['patches_dir']).expanduser()
    images_dir = Path(config['paths']['images_dir']).expanduser()
    camera_mapping = config['camera_mapping']
    
    # Build paths for this specific patch
    patch_name = args.patch
    sparse_dir = patches_dir / patch_name / "sparse" / "0"
    output_dir = patches_dir / patch_name / "sparse" / "splat"

    # Validate inputs
    if not lf_bin.exists():
        print(f"ERROR: LichtFeld binary not found: {lf_bin}")
        sys.exit(2)
    
    if not sparse_dir.exists():
        print(f"ERROR: Sparse directory not found: {sparse_dir}")
        sys.exit(2)
    
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(2)
    
    # Check that required COLMAP files exist
    required_files = ['cameras.bin', 'images.bin']
    for fname in required_files:
        if not (sparse_dir / fname).exists():
            print(f"ERROR: Required file not found: {sparse_dir / fname}")
            sys.exit(2)
    
    print("="*70)
    print("LichtFeld-Studio Multi-Camera Training")
    print("="*70)
    print(f"Patch:       {patch_name}")
    print(f"Sparse dir:  {sparse_dir}")
    print(f"Images dir:  {images_dir}")
    print(f"Output dir:  {output_dir}")
    print()
    
    # Gather metadata
    meta = gather_metadata(sparse_dir, images_dir, camera_mapping)
    print(f"Found {meta['num_images']} total images across {len(meta['camera_counts'])} cameras")
    for cam_key, count in meta['camera_counts'].items():
        print(f"  {cam_key}: {count} images")
    print()
    
    # Create temporary directory structure
    temp_dir = Path(tempfile.mkdtemp(prefix='lichtfeld_multicam_'))
    try:
        setup_lichtfeld_structure(sparse_dir, images_dir, temp_dir, camera_mapping)
        print()
        
        # Build LichtFeld command using temp directory and config parameters
        cmd = [str(lf_bin), '-d', str(temp_dir), '-o', str(output_dir)]
        
        # Add training parameters from config
        train_config = config['training']
        if train_config.get('headless', True):
            cmd.append('--headless')
        cmd.extend(['-i', str(train_config.get('iterations', 20000))])
        cmd.extend(['--max-cap', str(train_config.get('max_cap', 1000000))])
        cmd.extend(['--pose-opt', train_config.get('pose_opt', 'direct')])
        
        cmd_str = ' '.join(shlex.quote(c) for c in cmd)
        print(f"Running LichtFeld:")
        print(f"  {cmd_str}")
        print()
        
        # Run LichtFeld
        run_result = run_lichtfeld(cmd, output_dir)
        
        # Rename splat files to include patch name prefix
        print("\nRenaming splat files with patch prefix...")
        renamed_count = 0
        for splat_file in output_dir.glob("splat_*.ply"):
            new_name = f"{patch_name}_{splat_file.name}"
            new_path = splat_file.parent / new_name
            splat_file.rename(new_path)
            print(f"  ✓ {splat_file.name} → {new_name}")
            renamed_count += 1
        
        if renamed_count > 0:
            print(f"Renamed {renamed_count} splat file(s)")
        else:
            print("No splat files found to rename")
        
        # Write report
        report_path = write_report(output_dir, cmd_str, meta, run_result, camera_mapping)
        print()
        print(f"✓ Report written to: {report_path}")
        print(f"✓ Full log: {run_result['log']}")
        
        if run_result['return_code'] != 0:
            print(f"\nERROR: LichtFeld-Studio exited with code {run_result['return_code']}")
            sys.exit(run_result['return_code'])
        else:
            print("\nTraining completed successfully!")
    
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Explicit CUDA cleanup to prevent memory issues in batch training
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("✓ CUDA cache cleared")
        except Exception as e:
            # Don't fail if torch isn't available or CUDA cleanup fails
            pass


if __name__ == '__main__':
    main()
