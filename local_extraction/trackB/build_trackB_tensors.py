#!/usr/bin/env python3
"""
Build TrackB Annotation-Aligned Frame Tensors

Pre-extracts frame tensors for TrackB training, matching EXACTLY what
the training loop extracts on-the-fly. This speeds up training by ~10-30x
by avoiding repeated frame loading and transformation.

Key Design:
-----------
1. Uses the SAME frame selection logic as TrackB training:
   - sample_window_ending_at(): 16 frames ENDING at annotation frame, stride=2
   
2. Uses the SAME transformation pipeline:
   - Resize to 224x224
   - ToTensor (scales to [0,1])
   - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   
3. Output format:
   - Shape: (C=3, T=16, H=224, W=224)
   - Dtype: float32
   - Values: ImageNet-normalized
   - One .pt file per annotation (keyed by uid_frameIdx)

Usage:
------
    python build_trackB_tensors.py --manifest path/to/manifest.json --out_root path/to/output

Inputs:
-------
    - Manifest file (JSON or JSONL) with 'image' field pointing to annotation frames
    - Extracted frames in extracted_frames/<uid>/<frame>.jpg

Outputs:
--------
    - tensors/<uid>_<frame_idx>.pt - One tensor per annotation
    - logs/build_summary.txt - Build statistics
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("ERROR: torchvision is required. Install with: pip install torchvision")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x


# =============================================================================
# Configuration (matches trackB_tokenizer.py)
# =============================================================================

DEFAULT_TIME_LEN = 16       # Number of frames in temporal window
DEFAULT_TIME_STRIDE = 2     # Stride between frames
DEFAULT_IMG_SIZE = 224      # Spatial size (H, W)

# ImageNet normalization (same as trackB_tokenizer.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# Frame Utilities (copied from trackB_tokenizer.py for consistency)
# =============================================================================

def list_uid_frames(uid_dir: Path) -> List[Path]:
    """List all JPG frames in a UID directory, sorted by name."""
    if not uid_dir.exists():
        return []
    return sorted([p for p in uid_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])


def sample_window_ending_at(frame_path: Path, frames_root: Path, time_len: int, stride: int) -> List[Path]:
    """
    Return a list of frame paths of length <= time_len ending at frame_path.
    Steps backward by stride.
    
    This is the EXACT same logic as trackB_tokenizer.py to ensure consistency.
    
    Args:
        frame_path: Path to the annotation frame (last frame in window)
        frames_root: Root directory containing extracted frames
        time_len: Number of frames to sample
        stride: Step size when going backward
        
    Returns:
        List of frame paths, ordered chronologically (earliest to latest)
    """
    uid_dir = frame_path.parent
    all_frames = list_uid_frames(uid_dir)
    
    if not all_frames:
        return []
    
    # Find index of the annotation frame
    try:
        idx = all_frames.index(frame_path)
    except ValueError:
        # Fall back by matching stem
        stems = [p.stem for p in all_frames]
        if frame_path.stem in stems:
            idx = stems.index(frame_path.stem)
        else:
            return []
    
    # Sample frames going backward with stride
    selected = []
    i = idx
    while i >= 0 and len(selected) < time_len:
        selected.append(all_frames[i])
        i -= stride
    
    # Reverse to get chronological order (earliest to latest)
    selected.reverse()
    
    return selected


# =============================================================================
# Transform Pipeline
# =============================================================================

def build_transform(img_size: int = DEFAULT_IMG_SIZE) -> T.Compose:
    """
    Build the transformation pipeline matching trackB_tokenizer.py.
    
    Returns:
        Composed transform: Resize -> ToTensor -> Normalize
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# Tensor Building
# =============================================================================

def extract_window_tensor(
    window_paths: List[Path],
    transform: T.Compose,
    time_len: int = DEFAULT_TIME_LEN,
) -> Optional[torch.Tensor]:
    """
    Load frames and build tensor for a temporal window.
    
    Args:
        window_paths: List of frame paths (may be < time_len)
        transform: Image transformation pipeline
        time_len: Expected number of frames (will pad if needed)
        
    Returns:
        Tensor of shape (C=3, T=time_len, H=224, W=224), or None if failed
    """
    if not window_paths:
        return None
    
    frames = []
    last_frame = None
    
    for path in window_paths:
        try:
            with Image.open(str(path)) as im:
                im = im.convert("RGB")
                frame_tensor = transform(im)  # (C, H, W)
                last_frame = frame_tensor
                frames.append(frame_tensor)
        except Exception as e:
            # Use last valid frame if available
            if last_frame is not None:
                frames.append(last_frame.clone())
            else:
                return None
    
    if not frames:
        return None
    
    # Pad if needed (repeat last frame)
    while len(frames) < time_len:
        frames.append(frames[-1].clone())
    
    # Truncate if too many (take last time_len)
    if len(frames) > time_len:
        frames = frames[-time_len:]
    
    # Stack: (T, C, H, W) -> permute to (C, T, H, W)
    video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
    
    return video_tensor


def parse_annotation_key(image_path: str) -> Tuple[str, str]:
    """
    Extract UID and frame index from annotation image path.
    
    Args:
        image_path: Path like ".../extracted_frames/<uid>/<frame_idx>.jpg"
        
    Returns:
        (uid, frame_idx) tuple
    """
    path = Path(image_path)
    uid = path.parent.name
    frame_idx = path.stem
    return uid, frame_idx


# =============================================================================
# Manifest Loading
# =============================================================================

def load_manifest(path: Path) -> List[Dict]:
    """Load manifest from JSON or JSONL file."""
    if not path.exists():
        return []
    
    # Try JSON array first
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    
    # JSONL fallback
    records = []
    try:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    
    return records


# =============================================================================
# Main Builder
# =============================================================================

def build_tensors(
    manifest_path: Path,
    output_root: Path,
    frames_root: Optional[Path] = None,
    time_len: int = DEFAULT_TIME_LEN,
    time_stride: int = DEFAULT_TIME_STRIDE,
    img_size: int = DEFAULT_IMG_SIZE,
    skip_existing: bool = True,
) -> Dict:
    """
    Build annotation-aligned frame tensors.
    
    Args:
        manifest_path: Path to manifest file
        output_root: Output directory for tensors
        frames_root: Root directory for frames (inferred from manifest if None)
        time_len: Number of frames per tensor
        time_stride: Stride between frames
        img_size: Spatial size (H, W)
        skip_existing: Skip if tensor already exists
        
    Returns:
        Statistics dictionary
    """
    print(f"=" * 60)
    print("TrackB Tensor Builder")
    print(f"=" * 60)
    print(f"Manifest: {manifest_path}")
    print(f"Output:   {output_root}")
    print(f"Config:   time_len={time_len}, stride={time_stride}, img_size={img_size}")
    print()
    
    # Load manifest
    records = load_manifest(manifest_path)
    if not records:
        print("ERROR: No records found in manifest")
        return {'error': 'No records'}
    
    print(f"Loaded {len(records)} annotations from manifest")
    
    # Create output directories
    tensors_dir = output_root / 'tensors'
    logs_dir = output_root / 'logs'
    tensors_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Build transform
    transform = build_transform(img_size)
    
    # Process annotations
    stats = {
        'total': len(records),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'missing_frames': 0,
        'by_uid': defaultdict(int),
    }
    
    failed_records = []
    
    iterator = tqdm(records, desc="Building tensors") if HAS_TQDM else records
    
    # Detect repo root for relative path resolution
    repo_root = None
    manifest_abs = manifest_path.resolve()  # Make absolute for path detection
    
    for record in iterator:
        # Get image path from record
        image_path = record.get('image') or record.get('image_path') or record.get('frame_path')
        if not image_path:
            stats['failed'] += 1
            failed_records.append({'record': record, 'error': 'No image field'})
            continue
        
        # Parse UID and frame index
        uid, frame_idx = parse_annotation_key(image_path)
        tensor_key = f"{uid}_{frame_idx}"
        tensor_path = tensors_dir / f"{tensor_key}.pt"
        
        # Skip if exists
        if skip_existing and tensor_path.exists():
            stats['skipped'] += 1
            continue
        
        # Resolve frame path - handle multiple path formats
        frame_path = Path(image_path)
        
        if not frame_path.exists():
            # Try 1: Relative path from repo root (e.g., "local_extraction/v2/extracted_frames/...")
            if 'local_extraction' in str(image_path) or 'extracted_frames' in str(image_path):
                # Find repo root by going up from manifest until we find 'local_extraction'
                if repo_root is None:
                    current = manifest_abs.parent
                    while current.parent != current:
                        if current.name == 'local_extraction':
                            repo_root = current.parent
                            break
                        current = current.parent
                
                # Try resolving from repo root
                if repo_root is not None:
                    frame_path = repo_root / image_path
            
            # Try 2: Just uid/frame relative to frames_root
            if not frame_path.exists() and frames_root is not None:
                frame_path = frames_root / uid / f"{frame_idx}.jpg"
            
            if not frame_path.exists():
                stats['missing_frames'] += 1
                failed_records.append({'record': record, 'error': f'Frame not found: {frame_path}'})
                continue
        
        # Get frames root from first valid path
        if frames_root is None:
            frames_root = frame_path.parent.parent
        
        # Sample window ending at annotation frame
        window = sample_window_ending_at(frame_path, frames_root, time_len, time_stride)
        
        if not window:
            stats['failed'] += 1
            failed_records.append({'record': record, 'error': 'Could not sample window'})
            continue
        
        # Build tensor
        tensor = extract_window_tensor(window, transform, time_len)
        
        if tensor is None:
            stats['failed'] += 1
            failed_records.append({'record': record, 'error': 'Could not build tensor'})
            continue
        
        # Save tensor
        try:
            torch.save(tensor, tensor_path)
            stats['processed'] += 1
            stats['by_uid'][uid] += 1
        except Exception as e:
            stats['failed'] += 1
            failed_records.append({'record': record, 'error': str(e)})
    
    # Write summary
    summary_path = logs_dir / 'build_summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        f.write(f"TrackB Tensor Builder Summary\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Manifest: {manifest_path}\n")
        f.write(f"Output: {output_root}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  time_len: {time_len}\n")
        f.write(f"  time_stride: {time_stride}\n")
        f.write(f"  img_size: {img_size}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Total annotations: {stats['total']}\n")
        f.write(f"  Processed: {stats['processed']}\n")
        f.write(f"  Skipped (existing): {stats['skipped']}\n")
        f.write(f"  Failed: {stats['failed']}\n")
        f.write(f"  Missing frames: {stats['missing_frames']}\n")
        f.write(f"  Unique UIDs: {len(stats['by_uid'])}\n")
    
    # Write failed records
    if failed_records:
        failed_path = logs_dir / 'failed_records.jsonl'
        with failed_path.open('w', encoding='utf-8') as f:
            for item in failed_records:
                f.write(json.dumps(item) + '\n')
    
    # Print summary
    print()
    print(f"=" * 60)
    print("Summary")
    print(f"=" * 60)
    print(f"Total annotations: {stats['total']}")
    print(f"Processed:         {stats['processed']}")
    print(f"Skipped (exist):   {stats['skipped']}")
    print(f"Failed:            {stats['failed']}")
    print(f"Missing frames:    {stats['missing_frames']}")
    print(f"Unique UIDs:       {len(stats['by_uid'])}")
    print()
    print(f"Output saved to: {tensors_dir}")
    print(f"Summary saved to: {summary_path}")
    
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build TrackB annotation-aligned frame tensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build tensors for training manifest
  python build_trackB_tensors.py \\
      --manifest v2/manifests/head_train_clip.json \\
      --out_root videomae_trackB_tensors
  
  # Build for validation manifest
  python build_trackB_tensors.py \\
      --manifest v2/manifests/head_val_clip.json \\
      --out_root videomae_trackB_tensors
  
  # Custom frame window settings
  python build_trackB_tensors.py \\
      --manifest v2/manifests/head_train_clip.json \\
      --out_root videomae_trackB_tensors \\
      --time_len 16 --stride 2
"""
    )
    
    parser.add_argument(
        '--manifest', '-m',
        type=Path,
        required=True,
        help='Path to manifest file (JSON or JSONL)'
    )
    
    parser.add_argument(
        '--out_root', '-o',
        type=Path,
        default=Path('videomae_trackB_tensors'),
        help='Output directory for tensors (default: videomae_trackB_tensors)'
    )
    
    parser.add_argument(
        '--frames_root', '-f',
        type=Path,
        default=None,
        help='Root directory for extracted frames (inferred from manifest if not provided)'
    )
    
    parser.add_argument(
        '--time_len', '-t',
        type=int,
        default=DEFAULT_TIME_LEN,
        help=f'Number of frames in temporal window (default: {DEFAULT_TIME_LEN})'
    )
    
    parser.add_argument(
        '--stride', '-s',
        type=int,
        default=DEFAULT_TIME_STRIDE,
        help=f'Stride between frames (default: {DEFAULT_TIME_STRIDE})'
    )
    
    parser.add_argument(
        '--img_size', '-i',
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f'Image size (default: {DEFAULT_IMG_SIZE})'
    )
    
    parser.add_argument(
        '--no_skip',
        action='store_true',
        help='Rebuild tensors even if they already exist'
    )
    
    args = parser.parse_args()
    
    # Validate manifest exists
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}")
        sys.exit(1)
    
    # Build tensors
    stats = build_tensors(
        manifest_path=args.manifest,
        output_root=args.out_root,
        frames_root=args.frames_root,
        time_len=args.time_len,
        time_stride=args.stride,
        img_size=args.img_size,
        skip_existing=not args.no_skip,
    )
    
    if stats.get('error'):
        sys.exit(1)


if __name__ == '__main__':
    main()
