#!/usr/bin/env python3
"""
run_pipeline.py

Orchestrate the complete MASt3R-SLAM → Gaussian Splatting pipeline.
"""

import argparse
import subprocess
import sys
import shutil
import yaml
from pathlib import Path
from datetime import datetime
import json
import os
import time

class PipelineRunner:
    def __init__(self, config_path, splat_overrides=None):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.run_name = self.config['run_name']
        self.paths = self.config['paths']
        self.splat_overrides = splat_overrides or {}
        
        # Auto-detect dataset name from Low-Res images path
        images_dir = Path(self.paths['images_path'])
        self.dataset_name = images_dir.name
        
        self.run_dir = Path(self.paths['intermediate_data_root']) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        config_copy = self.run_dir / 'pipeline_config.yaml'
        shutil.copy(self.config_path, config_copy)
        print(f"📋 Config saved to: {config_copy}")
        
        self.log_file = self.run_dir / 'pipeline.log'
        self.terminal_log_file = self.run_dir / 'terminal_output.log'
        self.start_time = datetime.now()
        self.pipeline_start_time = time.time()
        self.step_timings = {}
        
        with open(self.terminal_log_file, 'a') as f:
            f.write(f"\n\n{'#'*70}\n# NEW PIPELINE RUN\n{'#'*70}\n")
            f.write(f"Run name: {self.run_name}\n")
            f.write(f"Dataset name: {self.dataset_name}\n")
        
        self.log(f"Pipeline started: {self.start_time}")
        self.log(f"Run name: {self.run_name}")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def format_duration(self, seconds):
        if seconds < 60: return f"{seconds:.1f}s"
        mins = int(seconds // 60)
        return f"{mins}m {int(seconds % 60)}s"
    
    def log_timing(self, step_num, step_name, duration, skipped=False):
        if not skipped:
            self.step_timings[step_num] = duration
            self.log(f"⏱️  Step {step_num} completed in {self.format_duration(duration)}")
        else:
            self.log(f"⏭️  Step {step_num} skipped")
    
    def run_command(self, cmd, description, check=True):
        self.log(f"\n{'='*70}\nStep: {description}\nCommand: {' '.join(cmd)}\n{'='*70}")
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            with open(self.terminal_log_file, 'a') as log_f:
                for line in process.stdout:
                    print(line, end='')
                    log_f.write(line)
                    log_f.flush()
            return_code = process.wait()
            if return_code != 0:
                self.log(f"✗ {description} failed with exit code {return_code}")
                if check: raise subprocess.CalledProcessError(return_code, cmd)
                return False
            self.log(f"✓ {description} completed")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"✗ {description} failed: {e}")
            if check: raise
            return False
    
    def step_1_intrinsics_estimation(self):
        """Step 1: COLMAP on Low-Res images."""
        step_name = "1. COLMAP Intrinsics Estimation"
        step_num = 1
        self.log(f"\n{'#'*70}\n# {step_name}\n{'#'*70}")
        step_start = time.time()
        
        output_cameras = self.run_dir / 'colmap_outputs' / 'cameras.txt'
        if self.config['pipeline'].get('skip_existing', True) and output_cameras.exists():
            self.log(f"⏭️  Skipping - output exists")
            self.log_timing(step_num, step_name, 0, skipped=True)
            return True
        
        cfg = self.config['intrinsics_estimation']
        cmd = [
            'python', str(Path(__file__).parent / 'estimate_intrinsics.py'),
            '--images_path', self.paths['images_path'],
            '--dataset', self.run_name,
            '--num_images', str(cfg['num_images']),
            '--camera_model', cfg['camera_model']
        ]
        if cfg.get('overwrite', False): cmd.append('--overwrite')
        
        result = self.run_command(cmd, step_name)
        self.log_timing(step_num, step_name, time.time() - step_start)
        return result
    
    def step_2_intrinsics_conversion(self):
        """Step 2: Generate Low-Res intrinsics.yaml for M-SLAM."""
        step_name = "2. Intrinsics Conversion"
        step_num = 2
        self.log(f"\n{'#'*70}\n# {step_name}\n{'#'*70}")
        step_start = time.time()
        
        output_yaml = self.run_dir / 'intrinsics.yaml'
        if self.config['pipeline'].get('skip_existing', True) and output_yaml.exists():
            self.log(f"⏭️  Skipping - output exists")
            self.log_timing(step_num, step_name, 0, skipped=True)
            return True
        
        cfg = self.config['intrinsics_conversion']
        cmd = ['python', str(Path(__file__).parent / 'shuttle_intrinsics.py'), '--dataset', self.run_name]
        
        if cfg.get('use_highres_for_splatting', False):
            cmd.append('--use-highres-for-splatting')
            if 'original_images_path' not in self.paths:
                self.log("❌ ERROR: original_images_path not configured!")
                return False
            cmd.extend(['--highres-images-path', self.paths['original_images_path']])
        
        if cfg.get('keep_original', False): cmd.append('--keep-original')
        
        result = self.run_command(cmd, step_name)
        self.log_timing(step_num, step_name, time.time() - step_start)
        return result
    
    def step_3_mast3r_slam(self):
        """Step 3: Run MASt3R-SLAM on Low-Res Images."""
        step_name = "3. MASt3R-SLAM"
        step_num = 3
        self.log(f"\n{'#'*70}\n# {step_name}\n{'#'*70}")
        step_start = time.time()

        mslam_logs = self.run_dir / 'mslam_logs'
        renamed_ply = mslam_logs / f'{self.run_name}.ply'
        
        if self.config['pipeline'].get('skip_existing', True) and renamed_ply.exists():
            self.log(f"⏭️  Skipping - output exists")
            self.log_timing(step_num, step_name, 0, skipped=True)
            return True

        cfg = self.config['mast3r_slam']
        mslam_root = Path(self.paths['mast3r_slam_root'])
        original_cwd = os.getcwd()
        os.chdir(mslam_root)

        config_path = cfg['config']
        if not Path(config_path).is_absolute():
            config_path = mslam_root / config_path

        # Always use Low-Res images for Tracking
        input_dataset = self.paths['images_path']
        self.log(f"📸 Input Dataset: {input_dataset} (Using Low-Res for Tracking)")

        cmd = [
            'python', str(mslam_root / 'main.py'),
            '--dataset', input_dataset,
            '--config', str(config_path)
        ]
        
        if cfg.get('use_calibration', False):
            intrinsics_yaml = self.run_dir / 'intrinsics.yaml'
            if intrinsics_yaml.exists():
                cmd.extend(['--calib', str(intrinsics_yaml)])
            else:
                raise FileNotFoundError(f"intrinsics.yaml not found at {intrinsics_yaml}")
            
        if not cfg.get('enable_visualization', False): cmd.append('--no-viz')
        cmd.extend(cfg.get('extra_args', []))

        result = self.run_command(cmd, step_name)
        os.chdir(original_cwd)
        self.log_timing(step_num, step_name, time.time() - step_start)
        return result
    
    def step_4_move_mslam_outputs(self):
        step_name = "4. Move MASt3R-SLAM Outputs"
        step_num = 4
        self.log(f"\n{'#'*70}\n# {step_name}\n{'#'*70}")
        step_start = time.time()
        
        mslam_root = Path(self.paths['mast3r_slam_root'])
        src_logs = mslam_root / 'logs'
        target_mslam = self.run_dir / 'mslam_logs'
        target_mslam.mkdir(parents=True, exist_ok=True)
        
        # Always use dataset name from Low-Res input path
        images_dir = Path(self.paths['images_path'])
        dataset_name = images_dir.name
        
        self.log(f"Detected dataset name (from input): {dataset_name}")
        
        src_keyframes = src_logs / 'keyframes' / dataset_name
        target_keyframes = target_mslam / 'keyframes'
        if src_keyframes.exists():
            if target_keyframes.exists(): shutil.rmtree(target_keyframes)
            shutil.move(str(src_keyframes), str(target_keyframes))
        
        # Move mapping file
        src_mapping = src_logs / 'keyframes' / 'keyframe_mapping.txt'
        if src_mapping.exists():
            shutil.move(str(src_mapping), str(target_mslam / 'keyframe_mapping.txt'))
            
        # Move PLY/TXT
        for ext in ['.txt', '.ply']:
            src = src_logs / f'{dataset_name}{ext}'
            dst = target_mslam / f'{self.run_name}{ext}'
            if src.exists():
                if dst.exists(): dst.unlink()
                shutil.move(str(src), str(dst))
                self.log(f"✓ Moved & Renamed: {src.name} -> {dst.name}")
        
        self.log_timing(step_num, step_name, time.time() - step_start)
        return True
    
    def step_5_pose_conversion(self):
        step_name = "5. Pose/Keyframe Conversion"
        step_num = 5
        self.log(f"\n{'#'*70}\n# {step_name}\n{'#'*70}")
        step_start = time.time()
        
        # We can't easily skip this step if we are in high-res mode because we need to ensure cleanup happens
        # So we check logic inside
        
        cfg = self.config['pose_conversion']
        cmd = [
            'python', str(Path(__file__).parent / 'cam_pose_keyframes_shuttle.py'),
            '--dataset', self.run_name,
            '--mslam_logs_dir', str(self.run_dir / 'mslam_logs')
        ]
        if cfg.get('link_images', False): cmd.append('--link')
        if cfg.get('camera_id') is not None: cmd.extend(['--camera_id', str(cfg['camera_id'])])
        
        result = self.run_command(cmd, step_name)
        self.log_timing(step_num, step_name, time.time() - step_start)
        
        # HIGH-RES LOGIC: Cleanup and Prepare
        use_highres = self.config.get('intrinsics_conversion', {}).get('use_highres_for_splatting', False)
        if result and use_highres:
            # 1. CLEANUP: Remove the low-res images we just copied
            images_dir = self.run_dir / 'for_splat' / 'images'
            self.log(f"🧹 High-Res Mode: Cleaning up low-res keyframes from {images_dir.name}...")
            # Delete everything in the folder
            if images_dir.exists():
                for file in images_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                self.log(f"   ✓ Directory emptied (ready for high-res images)")

            # 2. Run High-Res Steps
            if not self.step_5b_update_highres_poses(): return False
            if not self.step_5c_prepare_highres_images(): return False
            
        return result
    
    def step_5b_update_highres_poses(self):
        step_name = "5b. Update COLMAP Poses Filenames"
        self.log(f"\n{'#'*70}\n# {step_name}\n{'#'*70}")
        
        cmd = [
            'python', str(Path(__file__).parent / 'get_highres_poses.py'),
            '--dataset', self.run_name,
            '--mslam_logs_dir', str(self.run_dir / 'mslam_logs'),
            '--original_images_dir', self.paths['original_images_path']
        ]
        return self.run_command(cmd, step_name)
    
    def step_5c_prepare_highres_images(self):
        step_name = "5c. Prepare High-Res Images (Undistort & Crop)"
        self.log(f"\n{'#'*70}\n# {step_name}\n{'#'*70}")
        
        cmd = [
            'python', str(Path(__file__).parent / 'prepare_highres_splat.py'),
            '--dataset', self.run_name,
            '--highres_dir', self.paths['original_images_path'],
            '--intrinsics', str(self.run_dir / 'intrinsics.yaml')
        ]
        return self.run_command(cmd, step_name)
    
    def step_6_ply_conversion(self):
        step_name = "6. PLY to points3D Conversion"
        step_num = 6
        self.log(f"\n{'#'*70}\n# {step_name}\n{'#'*70}")
        step_start = time.time()
        
        if (self.run_dir / 'for_splat' / 'sparse' / '0' / 'points3D.bin').exists() and self.config['pipeline'].get('skip_existing', True):
            self.log("⏭️  Skipping")
            return True
            
        cmd = [
            'python', str(Path(__file__).parent / 'mslam_ply_to_points3d.py'),
            '--dataset', self.run_name,
            '--mslam_logs_dir', str(self.run_dir / 'mslam_logs'),
            '--sample', str(self.config['ply_conversion'].get('sample_percentage', 10.0))
        ]
        result = self.run_command(cmd, step_name)
        self.log_timing(step_num, step_name, time.time() - step_start)
        return result
    
    def step_7_gaussian_splatting(self):
        step_name = "7. Gaussian Splatting Training"
        step_num = 7
        self.log(f"\n{'#'*70}\n# {step_name}\n{'#'*70}")
        step_start = time.time()
        
        iterations = self.splat_overrides.get('iterations', self.config["gaussian_splatting"]["iterations"])
        output_dir = self.run_dir / 'splats'
        
        if (output_dir / f'splat_{iterations}.ply').exists() and not self.splat_overrides and self.config['pipeline'].get('skip_existing', True):
             self.log("⏭️  Skipping")
             return True
             
        if output_dir.exists():
            ver = 1
            while (self.run_dir / f'splats{ver}').exists(): ver += 1
            output_dir = self.run_dir / f'splats{ver}'
        
        self.log(f"Output: {output_dir}")
        
        cfg = self.config['gaussian_splatting']
        cmd = [
            'python', str(Path(__file__).parent / 'train_splat.py'),
            '--lichtfeld', self.paths['lichtfeld_binary'],
            '-d', str(self.run_dir / 'for_splat'),
            '-o', str(output_dir),
            '--'
        ]
        
        if self.splat_overrides.get('headless', cfg.get('headless', True)): cmd.append('--headless')
        cmd.extend(['-i', str(iterations)])
        cmd.extend(['--max-cap', str(self.splat_overrides.get('max_cap', cfg['max_cap']))])
        cmd.extend(self.splat_overrides.get('extra_args', cfg.get('extra_args', [])))
        
        result = self.run_command(cmd, step_name)
        self.log_timing(step_num, step_name, time.time() - step_start)
        return result

    def run(self, start_from=1, only=None):
        steps = [
            (1, "Intrinsics Estimation", self.step_1_intrinsics_estimation),
            (2, "Intrinsics Conversion", self.step_2_intrinsics_conversion),
            (3, "MASt3R-SLAM", self.step_3_mast3r_slam),
            (4, "Move Outputs", self.step_4_move_mslam_outputs),
            (5, "Pose Conversion", self.step_5_pose_conversion),
            ("5b", "Update High-Res Poses", self.step_5b_update_highres_poses),
            ("5c", "Prep High-Res Images", self.step_5c_prepare_highres_images),
            (6, "PLY Conversion", self.step_6_ply_conversion),
            (7, "Gaussian Splatting", self.step_7_gaussian_splatting)
        ]
        
        if only:
            steps = [s for s in steps if str(s[0]) == str(only)]
        else:
            steps = [s for s in steps if isinstance(s[0], int) and s[0] >= start_from]
            
        for num, name, func in steps:
            if not func(): return False
            
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--start-from', type=int, default=1)
    parser.add_argument('--only', type=str, default=None)
    parser.add_argument('-i', '--iterations', type=int)
    parser.add_argument('--max-cap', type=int)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--splat-extra-args', nargs='+')
    
    args = parser.parse_args()
    overrides = {}
    if args.iterations: overrides['iterations'] = args.iterations
    if args.max_cap: overrides['max_cap'] = args.max_cap
    if args.headless: overrides['headless'] = True
    if args.splat_extra_args: overrides['extra_args'] = args.splat_extra_args
    
    runner = PipelineRunner(args.config, overrides)
    sys.exit(0 if runner.run(args.start_from, args.only) else 1)

if __name__ == '__main__':
    main()