#!/usr/bin/env python3
"""
Pre-extract ResNet18 tokens for Track B training.

This script extracts ResNet18 feature tokens from all frames and saves them as .pt files.
This enables ~120× faster training by avoiding on-the-fly feature extraction.

Usage:
    python -m trackB.extract_resnet_tokens [--output_dir OUTPUT] [--device DEVICE]
    # or from local_extraction directory:
    python trackB/extract_resnet_tokens.py

Output structure:
    v2/resnet18_tokens/
    ├── <uid1>/
    │   ├── 0000001.pt  → (49, 512) tensor
    │   ├── 0000002.pt  → (49, 512) tensor
    │   └── ...
    ├── <uid2>/
    │   └── ...
    └── manifest.json   → metadata about extraction
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directories to path for imports
_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_EXTRACTION = _THIS_DIR.parent
for p in [str(_LOCAL_EXTRACTION), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from tqdm import tqdm

from trackB_tokenizer import (
    TokenizerConfig,
    build_backbone,
    build_transform,
    image_grid_tokens,
    list_uid_frames,
)


def extract_uid_tokens(
    uid_dir: Path,
    output_dir: Path,
    backbone: torch.nn.Module,
    transform,
    cfg: TokenizerConfig,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Extract tokens for all frames in a UID directory.
    
    Args:
        uid_dir: Path to UID frame directory
        output_dir: Output directory for this UID's tokens
        backbone: ResNet18 backbone
        transform: Image transform
        cfg: Tokenizer config
        skip_existing: Skip frames that already have .pt files
    
    Returns:
        Dict with extraction stats
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames = list_uid_frames(uid_dir)
    stats = {
        'uid': uid_dir.name,
        'total_frames': len(frames),
        'extracted': 0,
        'skipped': 0,
        'errors': 0,
        'time_seconds': 0.0,
    }
    
    start_time = time.time()
    
    for frame_path in frames:
        # Output path: same name but .pt extension
        out_path = output_dir / f"{frame_path.stem}.pt"
        
        if skip_existing and out_path.exists():
            stats['skipped'] += 1
            continue
        
        try:
            # Extract tokens
            tokens, hw = image_grid_tokens(frame_path, backbone, transform, cfg)
            
            # Save as .pt file
            torch.save({
                'tokens': tokens,  # (N, 512) where N = Hf * Wf = 49
                'hw': hw,          # (7, 7) grid size
                'source_frame': frame_path.name,
            }, out_path)
            
            stats['extracted'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            print(f"  [ERROR] {frame_path.name}: {e}")
    
    stats['time_seconds'] = time.time() - start_time
    return stats


def main():
    parser = argparse.ArgumentParser(description='Pre-extract ResNet18 tokens for Track B')
    parser.add_argument('--frames_dir', type=str, default=None,
                        help='Input frames directory (default: v2/extracted_frames)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output tokens directory (default: v2/resnet18_tokens)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip frames with existing .pt files')
    parser.add_argument('--no_skip', action='store_true',
                        help='Force re-extraction of all frames')
    parser.add_argument('--uid', type=str, default=None,
                        help='Extract only specific UID (for testing)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Extract only UIDs from this manifest (for efficiency)')
    parser.add_argument('--manifest_only', action='store_true',
                        help='Only extract UIDs that appear in training manifest (recommended)')
    args = parser.parse_args()
    
    # Resolve paths
    cwd = Path.cwd()
    if cwd.name == 'local_extraction':
        local_extraction = cwd
    else:
        local_extraction = cwd / 'local_extraction'
    
    frames_dir = Path(args.frames_dir) if args.frames_dir else local_extraction / 'v2' / 'extracted_frames'
    output_dir = Path(args.output_dir) if args.output_dir else local_extraction / 'v2' / 'resnet18_tokens'
    
    if not frames_dir.exists():
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        return
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    skip_existing = not args.no_skip
    
    print("=" * 60)
    print("ResNet18 Token Pre-Extraction for Track B")
    print("=" * 60)
    print(f"Frames directory: {frames_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Skip existing: {skip_existing}")
    
    # Build backbone and transform
    print("\n[1/3] Loading ResNet18 backbone...")
    cfg = TokenizerConfig()
    cfg.device = device
    backbone = build_backbone(cfg)
    transform = build_transform(cfg)
    print(f"      Backbone loaded on {device}")
    
    # Find UIDs to process
    if args.uid:
        # Single UID mode (for testing)
        uid_dirs = [frames_dir / args.uid]
        if not uid_dirs[0].exists():
            print(f"[ERROR] UID not found: {args.uid}")
            return
    elif args.manifest or args.manifest_only:
        # Extract only UIDs from manifest (MUCH faster)
        manifest_path = None
        if args.manifest:
            manifest_path = Path(args.manifest)
        else:
            # Auto-discover manifest
            trackA_runs = local_extraction / 'runs' / 'Track_A'
            for run_dir in sorted(trackA_runs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if run_dir.is_dir() and run_dir.name.startswith('trackA_stageB_'):
                    for name in ['head_train.jsonl', 'head_train.json']:
                        candidate = run_dir / name
                        if candidate.exists():
                            manifest_path = candidate
                            break
                    if manifest_path:
                        break
        
        if manifest_path is None or not manifest_path.exists():
            print(f"[ERROR] Manifest not found. Use --manifest to specify path.")
            return
        
        print(f"      Loading manifest: {manifest_path}")
        
        # Parse manifest to get unique UIDs
        uids_needed = set()
        with open(manifest_path, 'r') as f:
            if manifest_path.suffix == '.jsonl':
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        # Extract UID from image_path or uid field
                        if 'image_path' in rec:
                            # Format: local_extraction/v2/extracted_frames/<uid>/<frame>.jpg
                            parts = Path(rec['image_path']).parts
                            for i, p in enumerate(parts):
                                if p == 'extracted_frames' and i + 1 < len(parts):
                                    uids_needed.add(parts[i + 1])
                                    break
                        elif 'uid' in rec:
                            uids_needed.add(rec['uid'])
                        elif 'video_uid' in rec:
                            uids_needed.add(rec['video_uid'])
                    except:
                        continue
            else:
                data = json.load(f)
                for rec in data:
                    if 'image_path' in rec:
                        parts = Path(rec['image_path']).parts
                        for i, p in enumerate(parts):
                            if p == 'extracted_frames' and i + 1 < len(parts):
                                uids_needed.add(parts[i + 1])
                                break
                    elif 'uid' in rec:
                        uids_needed.add(rec['uid'])
                    elif 'video_uid' in rec:
                        uids_needed.add(rec['video_uid'])
        
        print(f"      Found {len(uids_needed)} unique UIDs in manifest")
        uid_dirs = [frames_dir / uid for uid in sorted(uids_needed) if (frames_dir / uid).exists()]
        print(f"      {len(uid_dirs)} UIDs have frames on disk")
    else:
        # All UIDs (slow!)
        uid_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
        print(f"\n      WARNING: Extracting ALL {len(uid_dirs)} UIDs. This will take ~18 hours!")
        print(f"      Use --manifest_only to extract only training UIDs (~30 min)")
    
    print(f"\n[2/3] Extracting tokens for {len(uid_dirs)} UIDs...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    total_start = time.time()
    
    for uid_dir in tqdm(uid_dirs, desc="UIDs"):
        uid_output = output_dir / uid_dir.name
        stats = extract_uid_tokens(
            uid_dir, uid_output, backbone, transform, cfg, skip_existing
        )
        all_stats.append(stats)
    
    total_time = time.time() - total_start
    
    # Summary
    total_frames = sum(s['total_frames'] for s in all_stats)
    total_extracted = sum(s['extracted'] for s in all_stats)
    total_skipped = sum(s['skipped'] for s in all_stats)
    total_errors = sum(s['errors'] for s in all_stats)
    
    print(f"\n[3/3] Extraction complete!")
    print(f"      Total frames: {total_frames}")
    print(f"      Extracted: {total_extracted}")
    print(f"      Skipped: {total_skipped}")
    print(f"      Errors: {total_errors}")
    print(f"      Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if total_extracted > 0:
        print(f"      Avg per frame: {total_time/total_extracted*1000:.1f}ms")
    
    # Save manifest
    manifest = {
        'extraction_time': datetime.now().isoformat(),
        'frames_dir': str(frames_dir),
        'output_dir': str(output_dir),
        'device': device,
        'backbone': 'resnet18',
        'token_dim': 512,
        'grid_size': [7, 7],
        'total_frames': total_frames,
        'total_extracted': total_extracted,
        'total_skipped': total_skipped,
        'total_errors': total_errors,
        'total_time_seconds': total_time,
        'uids': all_stats,
    }
    
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n      Manifest saved: {manifest_path}")
    
    # Estimate storage
    if total_extracted > 0:
        sample_pt = next(output_dir.rglob('*.pt'), None)
        if sample_pt:
            pt_size = sample_pt.stat().st_size
            total_size_mb = (total_extracted * pt_size) / (1024 * 1024)
            print(f"      Estimated storage: {total_size_mb:.1f} MB")


if __name__ == '__main__':
    main()
