#!/usr/bin/env python3
"""
Colab Runner for TrackB Tensor Builder (Track A StageB Edition)

This script pre-extracts annotation-aligned frame tensors from Track A StageB
manifests for fast VideoMAE training on Colab.

Key features:
- Uses Track A StageB manifests (YOLO detections matched to GT)
- Only ~2,245 unique frames = ~20 GB storage
- ~30x faster training after extraction

Usage on Colab:
---------------
1. Mount Google Drive
2. Run this script:
   
   %cd /content/drive/MyDrive/Ego4d_STA/Ego4d-LiteSTA/local_extraction/trackB
   !python build_trackB_tensors_colab.py

Default paths assume standard Ego4d-LiteSTA Colab setup.
"""

import sys
from pathlib import Path

# Colab paths
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")
COLAB_REPO_ROOT = COLAB_DRIVE_ROOT / "Ego4d_STA" / "Ego4d-LiteSTA"
COLAB_LOCAL_EXTRACTION = COLAB_REPO_ROOT / "local_extraction"

# Track A StageB manifest paths (latest run)
TRACK_A_RUNS = COLAB_LOCAL_EXTRACTION / "runs" / "Track_A"

# Default paths for Colab
DEFAULT_FRAMES_ROOT = COLAB_LOCAL_EXTRACTION / "v2" / "extracted_frames"
DEFAULT_OUTPUT_ROOT = COLAB_DRIVE_ROOT / "Ego4d_STA" / "videomae_trackB_tensors"

# Add local_extraction to path
if str(COLAB_LOCAL_EXTRACTION) not in sys.path:
    sys.path.insert(0, str(COLAB_LOCAL_EXTRACTION))


def find_latest_stageB_run(track_a_runs: Path) -> Path:
    """Find the latest Track A StageB run directory."""
    if not track_a_runs.exists():
        return None
    
    stageB_runs = sorted([
        d for d in track_a_runs.iterdir() 
        if d.is_dir() and 'stageB' in d.name
    ], key=lambda x: x.name, reverse=True)
    
    return stageB_runs[0] if stageB_runs else None


def main():
    """Build tensors from Track A StageB manifests."""
    print("=" * 70)
    print("TrackB Tensor Builder - Track A StageB Edition")
    print("=" * 70)
    print()
    
    # Find latest Track A StageB run
    latest_run = find_latest_stageB_run(TRACK_A_RUNS)
    
    if latest_run is None:
        print(f"ERROR: No Track A StageB runs found in: {TRACK_A_RUNS}")
        print("Make sure you have run Track A StageB first.")
        sys.exit(1)
    
    print(f"Using Track A StageB run: {latest_run.name}")
    
    # Check manifests exist
    train_manifest = latest_run / "head_train.jsonl"
    val_manifest = latest_run / "head_val.jsonl"
    
    if not train_manifest.exists():
        print(f"ERROR: Training manifest not found: {train_manifest}")
        sys.exit(1)
    
    # Check frames exist
    if not DEFAULT_FRAMES_ROOT.exists():
        print(f"ERROR: Frames root not found: {DEFAULT_FRAMES_ROOT}")
        print("Make sure Google Drive is mounted and frames are extracted")
        sys.exit(1)
    
    # Import the main builder
    try:
        from trackB.build_trackB_tensors import build_tensors, DEFAULT_TIME_LEN, DEFAULT_TIME_STRIDE, DEFAULT_IMG_SIZE
    except ImportError:
        print("ERROR: Could not import build_trackB_tensors")
        print("Make sure you're running from the correct directory")
        sys.exit(1)
    
    # Create output directory
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Process training manifest
    print(f"\n{'='*70}")
    print("Processing TRAINING manifest (Track A StageB)")
    print(f"{'='*70}\n")
    
    train_stats = build_tensors(
        manifest_path=train_manifest,
        output_root=DEFAULT_OUTPUT_ROOT,
        frames_root=DEFAULT_FRAMES_ROOT,
        time_len=DEFAULT_TIME_LEN,
        time_stride=DEFAULT_TIME_STRIDE,
        img_size=DEFAULT_IMG_SIZE,
        skip_existing=True,
    )
    
    # Process validation manifest
    if val_manifest.exists():
        print(f"\n{'='*70}")
        print("Processing VALIDATION manifest (Track A StageB)")
        print(f"{'='*70}\n")
        
        val_stats = build_tensors(
            manifest_path=val_manifest,
            output_root=DEFAULT_OUTPUT_ROOT,
            frames_root=DEFAULT_FRAMES_ROOT,
            time_len=DEFAULT_TIME_LEN,
            time_stride=DEFAULT_TIME_STRIDE,
            img_size=DEFAULT_IMG_SIZE,
            skip_existing=True,
        )
    else:
        print(f"WARNING: Validation manifest not found: {val_manifest}")
    
    print()
    print("=" * 70)
    print("DONE!")
    print("=" * 70)
    print()
    print(f"Tensors saved to: {DEFAULT_OUTPUT_ROOT / 'tensors'}")
    print()
    print("To use in training, set tokens_root in your notebook:")
    print(f"  tokens_root = Path('{DEFAULT_OUTPUT_ROOT / 'tensors'}')")
    print()
    print("Expected storage: ~20 GB")
    print("Expected training speedup: ~30x")


if __name__ == '__main__':
    main()

