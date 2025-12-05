#!/usr/bin/env python3
"""
Build VideoMAE Clip Manifest (JSONL)

Creates a manifest file that Colab can read directly to know clip locations and metadata.

UID Extraction Rule:
    - Filenames are in format: <uid>.mp4
    - UID = basename without extension (e.g., "94797ff8-a780-4f2b-8e0b-0242f8ffee19")
    - UIDs are UUIDs from Ego4D dataset

Usage:
    python build_videomae_manifest.py [--clips_list PATH] [--output PATH]

Input:
    local_lists/sta_clips_ok.txt - List of validated clip paths

Output:
    manifests/ego4d_sta_clips.jsonl - One JSON object per line
    logs/manifest_summary.txt - Summary statistics
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("ERROR: OpenCV (cv2) required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x  # Fallback: no progress bar


# =============================================================================
# Configuration
# =============================================================================

# Path mapping: Local root -> Colab root
LOCAL_CLIPS_ROOT = Path(r"I:\My Drive\ego4d_data\clip540s")
COLAB_CLIPS_ROOT = "/content/drive/MyDrive/ego4d_data/clip540s"

# Future tensor paths (for Colab)
COLAB_TENSORS_ROOT = "/content/drive/MyDrive/ego4d_data/videomae_tensors"


# =============================================================================
# UID Extraction
# =============================================================================

def extract_uid(clip_path: Path) -> str:
    """
    Extract UID from clip filename.
    
    Rule: Filenames are <uid>.mp4, so UID = stem (basename without extension).
    
    Example:
        94797ff8-a780-4f2b-8e0b-0242f8ffee19.mp4 -> 94797ff8-a780-4f2b-8e0b-0242f8ffee19
    
    Args:
        clip_path: Path to the MP4 file
        
    Returns:
        UID string
    """
    return clip_path.stem


def local_to_colab_path(local_path: Path, local_root: Path, colab_root: str) -> str:
    """
    Convert local path to Colab path.
    
    Args:
        local_path: Local file path
        local_root: Local root directory
        colab_root: Colab mount point
        
    Returns:
        Colab path string
    """
    try:
        relative = local_path.relative_to(local_root)
        return f"{colab_root}/{relative.as_posix()}"
    except ValueError:
        # If not relative, just use filename
        return f"{colab_root}/{local_path.name}"


# =============================================================================
# Video Metadata Extraction
# =============================================================================

def get_video_metadata(video_path: Path) -> Optional[Dict]:
    """
    Extract video metadata using OpenCV.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dict with num_frames, fps, width, height, or None if failed
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        metadata = {
            'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        cap.release()
        
        # Validate
        if metadata['num_frames'] <= 0 or metadata['fps'] <= 0:
            return None
        
        return metadata
        
    except Exception as e:
        print(f"  ERROR reading {video_path.name}: {e}")
        return None


# =============================================================================
# Main Builder
# =============================================================================

def build_manifest(
    clips_list_file: Path,
    output_file: Path,
    summary_file: Path,
    local_root: Path,
    colab_root: str,
    colab_tensors_root: str,
) -> int:
    """
    Build JSONL manifest from clip list.
    
    Args:
        clips_list_file: Path to sta_clips_ok.txt
        output_file: Output JSONL path
        summary_file: Summary log path
        local_root: Local clips root
        colab_root: Colab clips root
        colab_tensors_root: Colab tensors root
        
    Returns:
        Number of clips processed
    """
    # Load clip paths
    print(f"Loading clips from: {clips_list_file}")
    
    if not clips_list_file.exists():
        print(f"ERROR: Clips list not found: {clips_list_file}")
        return 0
    
    with open(clips_list_file, 'r', encoding='utf-8') as f:
        clip_paths = [Path(line.strip()) for line in f if line.strip()]
    
    print(f"Found {len(clip_paths)} clips to process")
    
    # Ensure output directories exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process clips
    records = []
    errors = []
    
    # Stats
    all_frames = []
    all_fps = []
    all_widths = []
    all_heights = []
    
    print(f"\nProcessing clips...")
    
    # Use tqdm for progress bar
    clip_iter = tqdm(clip_paths, desc="Building manifest", unit="clip") if HAS_TQDM else clip_paths
    
    for i, clip_path in enumerate(clip_iter):
        
        # Extract UID
        uid = extract_uid(clip_path)
        
        # Get metadata
        metadata = get_video_metadata(clip_path)
        
        if metadata is None:
            errors.append(str(clip_path))
            # Update tqdm postfix
            if HAS_TQDM and hasattr(clip_iter, 'set_postfix'):
                clip_iter.set_postfix({'ok': len(records), 'err': len(errors)})
            continue
        
        # Build Colab paths
        colab_clip_path = local_to_colab_path(clip_path, local_root, colab_root)
        colab_tensor_path = f"{colab_tensors_root}/{uid}.pt"
        
        # Create record
        record = {
            'uid': uid,
            'path': colab_clip_path,
            'num_frames': metadata['num_frames'],
            'fps': metadata['fps'],
            'width': metadata['width'],
            'height': metadata['height'],
            'tensor_path': colab_tensor_path,
        }
        
        records.append(record)
        
        # Collect stats
        all_frames.append(metadata['num_frames'])
        all_fps.append(metadata['fps'])
        all_widths.append(metadata['width'])
        all_heights.append(metadata['height'])
        
        # Update tqdm postfix
        if HAS_TQDM and hasattr(clip_iter, 'set_postfix'):
            clip_iter.set_postfix({'ok': len(records), 'err': len(errors)})
    
    print(f"\nProcessed {len(records)} clips successfully")
    if errors:
        print(f"Errors: {len(errors)} clips failed")
    
    # Write JSONL
    print(f"\nWriting manifest to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    # Calculate statistics
    if all_frames:
        stats = {
            'total_clips': len(records),
            'errors': len(errors),
            'frames': {
                'min': min(all_frames),
                'max': max(all_frames),
                'mean': sum(all_frames) / len(all_frames),
            },
            'fps': {
                'min': min(all_fps),
                'max': max(all_fps),
                'mean': sum(all_fps) / len(all_fps),
            },
            'width': {
                'min': min(all_widths),
                'max': max(all_widths),
                'values': sorted(set(all_widths)),
            },
            'height': {
                'min': min(all_heights),
                'max': max(all_heights),
                'values': sorted(set(all_heights)),
            },
            'resolutions': {},
        }
        
        # Count resolutions
        for w, h in zip(all_widths, all_heights):
            res = f"{w}x{h}"
            stats['resolutions'][res] = stats['resolutions'].get(res, 0) + 1
    else:
        stats = {'total_clips': 0, 'errors': len(errors)}
    
    # Write summary
    print(f"Writing summary to: {summary_file}")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VIDEOMAE MANIFEST SUMMARY\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total clips processed: {stats['total_clips']}\n")
        f.write(f"Errors: {stats.get('errors', 0)}\n\n")
        
        if 'frames' in stats:
            f.write("Frame Statistics:\n")
            f.write(f"  Min frames: {stats['frames']['min']}\n")
            f.write(f"  Max frames: {stats['frames']['max']}\n")
            f.write(f"  Mean frames: {stats['frames']['mean']:.1f}\n\n")
            
            f.write("FPS Statistics:\n")
            f.write(f"  Min FPS: {stats['fps']['min']:.1f}\n")
            f.write(f"  Max FPS: {stats['fps']['max']:.1f}\n")
            f.write(f"  Mean FPS: {stats['fps']['mean']:.1f}\n\n")
            
            f.write("Resolution Distribution:\n")
            for res, count in sorted(stats['resolutions'].items(), key=lambda x: -x[1]):
                f.write(f"  {res}: {count} clips\n")
        
        f.write("\n" + "-" * 60 + "\n")
        f.write("Path Mapping:\n")
        f.write(f"  Local root: {local_root}\n")
        f.write(f"  Colab clips root: {colab_root}\n")
        f.write(f"  Colab tensors root: {colab_tensors_root}\n")
        
        if errors:
            f.write("\n" + "-" * 60 + "\n")
            f.write("Failed Clips:\n")
            for err in errors:
                f.write(f"  {err}\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MANIFEST BUILD COMPLETE")
    print("=" * 60)
    print(f"Clips processed: {stats['total_clips']}")
    if 'frames' in stats:
        print(f"Frames: min={stats['frames']['min']}, max={stats['frames']['max']}, mean={stats['frames']['mean']:.1f}")
        print(f"Resolutions: {stats['resolutions']}")
    
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Build VideoMAE clip manifest")
    parser.add_argument(
        "--clips_list",
        type=str,
        default=None,
        help="Path to sta_clips_ok.txt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path"
    )
    parser.add_argument(
        "--local_root",
        type=str,
        default=str(LOCAL_CLIPS_ROOT),
        help="Local clips root directory"
    )
    parser.add_argument(
        "--colab_root",
        type=str,
        default=COLAB_CLIPS_ROOT,
        help="Colab clips mount point"
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    
    clips_list = Path(args.clips_list) if args.clips_list else script_dir / "local_lists" / "sta_clips_ok.txt"
    output_file = Path(args.output) if args.output else script_dir / "manifests" / "ego4d_sta_clips.jsonl"
    summary_file = script_dir / "logs" / "manifest_summary.txt"
    
    # Print header
    print("=" * 60)
    print("VIDEOMAE MANIFEST BUILDER")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    print(f"Clips list: {clips_list}")
    print(f"Output: {output_file}")
    print(f"Local root: {args.local_root}")
    print(f"Colab root: {args.colab_root}")
    print()
    
    # Build manifest
    count = build_manifest(
        clips_list_file=clips_list,
        output_file=output_file,
        summary_file=summary_file,
        local_root=Path(args.local_root),
        colab_root=args.colab_root,
        colab_tensors_root=COLAB_TENSORS_ROOT,
    )
    
    return 0 if count > 0 else 1


if __name__ == "__main__":
    exit(main())
