#!/usr/bin/env python3
"""
Build VideoMAE Precomputed Tensors

Precomputes frame tensors from video clips for faster loading on Colab.
Avoids decoding MP4 on Colab; instead, loads ready (C, T, H, W) tensors.

Tensor Format:
    - T (frames per clip): 16 (configurable)
    - Spatial size (H, W): 224 x 224 (configurable)
    - Tensor shape: (C, T, H, W) where C = 3
    - Value range: [0, 1] (float32)
    - No normalization beyond scaling by 1/255; VideoMAE processor will normalize later

Sampling Strategy:
    - Uniformly sample a random start index (deterministic per clip using UID as seed)
    - Take T consecutive frames from that start position
    - If video has fewer than T frames, repeat last frame to fill

Usage:
    python build_videomae_tensors.py [--manifest PATH] [--out_root PATH] [--target_frames N] [--target_size N]

Input:
    manifests/ego4d_sta_clips.jsonl - Clip manifest

Output:
    tensors/<uid>.pt - One tensor file per clip
    logs/tensors_summary.txt - Build summary
"""

import argparse
import json
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x  # Fallback: no progress bar


# =============================================================================
# Configuration
# =============================================================================

# Default tensor parameters
DEFAULT_TARGET_FRAMES = 16      # T - temporal dimension
DEFAULT_TARGET_SIZE = 224       # H, W - spatial dimensions

# Path mapping: Colab root -> Local root (reverse of manifest)
COLAB_CLIPS_ROOT = "/content/drive/MyDrive/ego4d_data/clip540s"
LOCAL_CLIPS_ROOT = Path(r"I:\My Drive\ego4d_data\clip540s")


# =============================================================================
# Path Conversion
# =============================================================================

def colab_to_local_path(colab_path: str, colab_root: str, local_root: Path) -> Path:
    """
    Convert Colab path back to local path.
    
    Args:
        colab_path: Colab path string
        colab_root: Colab root prefix
        local_root: Local root directory
        
    Returns:
        Local Path object
    """
    if colab_path.startswith(colab_root):
        relative = colab_path[len(colab_root):].lstrip('/')
        return local_root / relative
    else:
        # Fallback: extract filename
        filename = colab_path.split('/')[-1]
        return local_root / filename


def uid_to_seed(uid: str) -> int:
    """
    Convert UID to deterministic random seed.
    
    Args:
        uid: Clip UID string
        
    Returns:
        Integer seed
    """
    return int(hashlib.md5(uid.encode()).hexdigest()[:8], 16)


# =============================================================================
# Frame Extraction
# =============================================================================

def extract_frames(
    video_path: Path,
    target_frames: int,
    target_size: int,
    seed: int,
) -> Optional[np.ndarray]:
    """
    Extract and preprocess frames from video.
    
    Sampling: uniformly sample start index, take T consecutive frames.
    
    Args:
        video_path: Path to video file
        target_frames: Number of frames to extract (T)
        target_size: Spatial size (H, W)
        seed: Random seed for reproducible sampling
        
    Returns:
        NumPy array of shape (T, H, W, C) with float32 values in [0, 1], or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return None
    
    # Determine start index
    np.random.seed(seed)
    
    if total_frames >= target_frames:
        # Can sample T consecutive frames
        max_start = total_frames - target_frames
        start_idx = np.random.randint(0, max_start + 1)
    else:
        # Not enough frames, start from 0 and will repeat last frame
        start_idx = 0
    
    # Seek to start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    frames = []
    last_frame = None
    
    for i in range(target_frames):
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to target size
            frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            last_frame = frame
            frames.append(frame)
        elif last_frame is not None:
            # Repeat last frame if video ends early
            frames.append(last_frame.copy())
        else:
            # No frames read at all
            cap.release()
            return None
    
    cap.release()
    
    # Stack frames: (T, H, W, C)
    frames_array = np.stack(frames, axis=0)
    
    # Convert to float32 and scale to [0, 1]
    frames_array = frames_array.astype(np.float32) / 255.0
    
    return frames_array


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """
    Convert frames array to PyTorch tensor.
    
    Args:
        frames: NumPy array of shape (T, H, W, C)
        
    Returns:
        Tensor of shape (C, T, H, W)
    """
    # (T, H, W, C) -> (C, T, H, W)
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2)
    return tensor


# =============================================================================
# Main Builder
# =============================================================================

def build_tensors(
    manifest_file: Path,
    output_root: Path,
    target_frames: int,
    target_size: int,
    summary_file: Path,
    local_clips_root: Path,
    colab_clips_root: str,
) -> Dict:
    """
    Build tensor files from manifest.
    
    Args:
        manifest_file: Path to JSONL manifest
        output_root: Output directory for tensors
        target_frames: Number of frames per tensor
        target_size: Spatial size (H, W)
        summary_file: Summary log path
        local_clips_root: Local clips directory
        colab_clips_root: Colab clips path prefix
        
    Returns:
        Stats dictionary
    """
    print(f"Loading manifest from: {manifest_file}")
    
    if not manifest_file.exists():
        print(f"ERROR: Manifest not found: {manifest_file}")
        return {'error': 'Manifest not found'}
    
    # Load manifest
    records = []
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} clip records")
    
    # Ensure output directory exists
    output_root.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process clips
    stats = {
        'total': len(records),
        'created': 0,
        'skipped': 0,
        'errors': 0,
        'error_files': [],
        'total_bytes': 0,
    }
    
    print(f"\nBuilding tensors...")
    print(f"  Target frames: {target_frames}")
    print(f"  Target size: {target_size}x{target_size}")
    print(f"  Output shape: (3, {target_frames}, {target_size}, {target_size})")
    print()
    
    # Use tqdm for progress bar
    record_iter = tqdm(records, desc="Building tensors", unit="clip") if HAS_TQDM else records
    
    for i, record in enumerate(record_iter):
        uid = record['uid']
        colab_path = record['path']
        
        # Output tensor path
        tensor_path = output_root / f"{uid}.pt"
        
        # Update tqdm postfix with stats
        if HAS_TQDM and hasattr(record_iter, 'set_postfix'):
            record_iter.set_postfix({
                'created': stats['created'],
                'skipped': stats['skipped'],
                'errors': stats['errors'],
            })
        
        # Skip if already exists (resume-friendly)
        if tensor_path.exists():
            stats['skipped'] += 1
            continue
        
        # Convert Colab path to local path
        local_path = colab_to_local_path(colab_path, colab_clips_root, local_clips_root)
        
        if not local_path.exists():
            if not HAS_TQDM:
                print(f"    SKIP: File not found: {local_path.name}")
            stats['errors'] += 1
            stats['error_files'].append(str(local_path))
            continue
        
        # Extract frames
        seed = uid_to_seed(uid)
        frames = extract_frames(local_path, target_frames, target_size, seed)
        
        if frames is None:
            if not HAS_TQDM:
                print(f"    ERROR: Failed to extract frames: {uid}")
            stats['errors'] += 1
            stats['error_files'].append(str(local_path))
            continue
        
        # Convert to tensor
        tensor = frames_to_tensor(frames)
        
        # Validate shape
        expected_shape = (3, target_frames, target_size, target_size)
        if tensor.shape != expected_shape:
            if not HAS_TQDM:
                print(f"    ERROR: Wrong shape {tensor.shape} for {uid}")
            stats['errors'] += 1
            stats['error_files'].append(str(local_path))
            continue
        
        # Save tensor
        torch.save(tensor, tensor_path)
        stats['created'] += 1
        stats['total_bytes'] += tensor_path.stat().st_size
    
    # Calculate final disk size
    if output_root.exists():
        total_size = sum(f.stat().st_size for f in output_root.glob("*.pt"))
        stats['total_bytes'] = total_size
    
    # Write summary
    print(f"\nWriting summary to: {summary_file}")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VIDEOMAE TENSORS BUILD SUMMARY\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Target frames (T): {target_frames}\n")
        f.write(f"  Target size (H, W): {target_size} x {target_size}\n")
        f.write(f"  Tensor shape: (3, {target_frames}, {target_size}, {target_size})\n")
        f.write(f"  Value range: [0, 1] (float32)\n\n")
        
        f.write("Results:\n")
        f.write(f"  Total clips in manifest: {stats['total']}\n")
        f.write(f"  Tensors created: {stats['created']}\n")
        f.write(f"  Tensors skipped (existing): {stats['skipped']}\n")
        f.write(f"  Errors: {stats['errors']}\n\n")
        
        f.write("Disk Usage:\n")
        f.write(f"  Total size: {stats['total_bytes'] / (1024**3):.2f} GB\n")
        f.write(f"  Per tensor (avg): {stats['total_bytes'] / max(1, stats['created'] + stats['skipped']) / 1024:.1f} KB\n")
        
        if stats['error_files']:
            f.write("\n" + "-" * 60 + "\n")
            f.write("Error Files:\n")
            for err in stats['error_files']:
                f.write(f"  {err}\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TENSOR BUILD COMPLETE")
    print("=" * 60)
    print(f"Total clips: {stats['total']}")
    print(f"Created: {stats['created']}")
    print(f"Skipped (existing): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total disk size: {stats['total_bytes'] / (1024**3):.2f} GB")
    
    return stats


def main():
    # Check dependencies
    if not HAS_CV2:
        print("ERROR: OpenCV required. Install with: pip install opencv-python")
        return 1
    if not HAS_TORCH:
        print("ERROR: PyTorch required. Install with: pip install torch")
        return 1
    if not HAS_NUMPY:
        print("ERROR: NumPy required. Install with: pip install numpy")
        return 1
    
    parser = argparse.ArgumentParser(description="Build VideoMAE precomputed tensors")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to ego4d_sta_clips.jsonl"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output directory for tensors"
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
        help=f"Number of frames per tensor (default: {DEFAULT_TARGET_FRAMES})"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Spatial size H=W (default: {DEFAULT_TARGET_SIZE})"
    )
    parser.add_argument(
        "--local_clips_root",
        type=str,
        default=str(LOCAL_CLIPS_ROOT),
        help="Local clips root directory"
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    
    manifest_file = Path(args.manifest) if args.manifest else script_dir / "manifests" / "ego4d_sta_clips.jsonl"
    output_root = Path(args.out_root) if args.out_root else script_dir / "tensors"
    summary_file = script_dir / "logs" / "tensors_summary.txt"
    
    # Print header
    print("=" * 60)
    print("VIDEOMAE TENSOR BUILDER")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    print(f"Manifest: {manifest_file}")
    print(f"Output root: {output_root}")
    print(f"Target frames: {args.target_frames}")
    print(f"Target size: {args.target_size}")
    print()
    
    # Build tensors
    stats = build_tensors(
        manifest_file=manifest_file,
        output_root=output_root,
        target_frames=args.target_frames,
        target_size=args.target_size,
        summary_file=summary_file,
        local_clips_root=Path(args.local_clips_root),
        colab_clips_root=COLAB_CLIPS_ROOT,
    )
    
    return 0 if stats.get('created', 0) + stats.get('skipped', 0) > 0 else 1


if __name__ == "__main__":
    exit(main())
