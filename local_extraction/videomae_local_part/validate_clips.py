#!/usr/bin/env python3
"""
Validate STA clips by sampling and checking video integrity.

This script is idempotent - running it multiple times will produce the same result.

Usage:
    python validate_clips.py [--sample_size N]
    
Input:
    local_lists/sta_clips_all.txt - List of all clip paths
    
Output:
    local_lists/sta_clips_ok.txt - Validated good clips
    logs/bad_clips.txt - Clips that failed validation
    logs/validation_report.txt - Detailed validation report
"""

import argparse
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Try to import cv2 for video validation
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("WARNING: OpenCV (cv2) not found. Install with: pip install opencv-python")


def validate_clip(clip_path: Path) -> Tuple[bool, Dict]:
    """
    Validate a single clip using OpenCV.
    
    Args:
        clip_path: Path to the MP4 file
        
    Returns:
        Tuple of (is_valid, info_dict)
    """
    info = {
        'path': str(clip_path),
        'exists': False,
        'readable': False,
        'num_frames': 0,
        'fps': 0.0,
        'width': 0,
        'height': 0,
        'duration_sec': 0.0,
        'error': None,
    }
    
    # Check file exists
    if not clip_path.exists():
        info['error'] = "File does not exist"
        return False, info
    
    info['exists'] = True
    
    if not HAS_CV2:
        info['error'] = "OpenCV not available"
        return False, info
    
    try:
        cap = cv2.VideoCapture(str(clip_path))
        
        if not cap.isOpened():
            info['error'] = "Cannot open video file"
            return False, info
        
        info['readable'] = True
        
        # Get video properties
        info['num_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if info['fps'] > 0:
            info['duration_sec'] = info['num_frames'] / info['fps']
        
        cap.release()
        
        # Validate properties
        if info['num_frames'] <= 0:
            info['error'] = "Zero or negative frame count"
            return False, info
        
        if info['fps'] <= 0:
            info['error'] = "Invalid FPS"
            return False, info
        
        if info['width'] <= 0 or info['height'] <= 0:
            info['error'] = "Invalid dimensions"
            return False, info
        
        return True, info
        
    except Exception as e:
        info['error'] = f"Exception: {str(e)}"
        return False, info


def load_clip_list(list_file: Path) -> List[str]:
    """Load clip paths from a text file."""
    if not list_file.exists():
        return []
    
    paths = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(line)
    return paths


def save_clip_list(paths: List[str], output_file: Path):
    """Save clip paths to a text file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in paths:
            f.write(path + '\n')


def main():
    parser = argparse.ArgumentParser(description="Validate STA clips")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of clips to sample for validation (default: 10)"
    )
    parser.add_argument(
        "--validate_all",
        action="store_true",
        help="Validate all clips (ignores --sample_size)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    local_lists_dir = script_dir / "local_lists"
    logs_dir = script_dir / "logs"
    
    all_clips_file = local_lists_dir / "sta_clips_all.txt"
    ok_clips_file = local_lists_dir / "sta_clips_ok.txt"
    bad_clips_file = logs_dir / "bad_clips.txt"
    report_file = logs_dir / "validation_report.txt"
    
    # Ensure directories exist
    local_lists_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print("=" * 60)
    print("STA CLIPS VALIDATOR")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Check for OpenCV
    if not HAS_CV2:
        print("\nERROR: OpenCV is required for video validation.")
        print("Install with: pip install opencv-python")
        return 1
    
    # Load all clips
    print(f"\nLoading clip list from: {all_clips_file}")
    all_paths = load_clip_list(all_clips_file)
    
    if not all_paths:
        print("ERROR: No clips found. Run scan_clips.py first.")
        return 1
    
    print(f"Loaded {len(all_paths)} clip paths")
    
    # Determine which clips to validate
    if args.validate_all:
        sample_paths = all_paths
        print(f"\nValidating ALL {len(sample_paths)} clips...")
    else:
        random.seed(args.seed)
        sample_size = min(args.sample_size, len(all_paths))
        sample_paths = random.sample(all_paths, sample_size)
        print(f"\nSampling {sample_size} clips for validation (seed={args.seed})...")
    
    # Validate sampled clips
    print()
    bad_paths = []
    validation_results = []
    
    for i, path_str in enumerate(sample_paths, 1):
        clip_path = Path(path_str)
        print(f"[{i}/{len(sample_paths)}] Validating: {clip_path.name}...", end=" ")
        
        is_valid, info = validate_clip(clip_path)
        validation_results.append(info)
        
        if is_valid:
            print(f"OK ({info['num_frames']} frames, {info['fps']:.1f} fps, {info['width']}x{info['height']})")
        else:
            print(f"BAD - {info['error']}")
            bad_paths.append(path_str)
    
    # Summary
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total sampled: {len(sample_paths)}")
    print(f"Valid clips:   {len(sample_paths) - len(bad_paths)}")
    print(f"Bad clips:     {len(bad_paths)}")
    
    # Write bad clips
    save_clip_list(bad_paths, bad_clips_file)
    print(f"\nBad clips written to: {bad_clips_file}")
    
    # Create OK list (all clips minus bad clips)
    bad_set = set(bad_paths)
    ok_paths = [p for p in all_paths if p not in bad_set]
    save_clip_list(ok_paths, ok_clips_file)
    print(f"OK clips written to: {ok_clips_file} ({len(ok_paths)} clips)")
    
    # Write detailed report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("STA CLIPS VALIDATION REPORT\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total clips in list: {len(all_paths)}\n")
        f.write(f"Clips sampled: {len(sample_paths)}\n")
        f.write(f"Valid clips: {len(sample_paths) - len(bad_paths)}\n")
        f.write(f"Bad clips: {len(bad_paths)}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 60 + "\n\n")
        
        for info in validation_results:
            status = "OK" if info['error'] is None else f"BAD: {info['error']}"
            f.write(f"File: {Path(info['path']).name}\n")
            f.write(f"  Status: {status}\n")
            if info['error'] is None:
                f.write(f"  Frames: {info['num_frames']}\n")
                f.write(f"  FPS: {info['fps']:.2f}\n")
                f.write(f"  Size: {info['width']}x{info['height']}\n")
                f.write(f"  Duration: {info['duration_sec']:.2f}s\n")
            f.write("\n")
        
        if bad_paths:
            f.write("-" * 60 + "\n")
            f.write("BAD CLIPS LIST\n")
            f.write("-" * 60 + "\n\n")
            for p in bad_paths:
                f.write(f"{p}\n")
    
    print(f"Detailed report: {report_file}")
    
    # Final status
    print()
    if len(bad_paths) == 0:
        print("✓ All sampled clips validated successfully!")
    else:
        print(f"⚠ Found {len(bad_paths)} problematic clips (see {bad_clips_file})")
    
    return 0


if __name__ == "__main__":
    exit(main())
