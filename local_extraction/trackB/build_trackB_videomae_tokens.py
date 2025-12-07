#!/usr/bin/env python3
"""
Build TrackB Pre-Encoded VideoMAE Tokens

Pre-extracts VideoMAE ENCODER OUTPUT (tokens) for TrackB training.
This is different from build_trackB_tensors.py which only saves frame pixels.

Why This Is Faster:
-------------------
- build_trackB_tensors.py saves: frame pixels (C, T, H, W)
  → Still requires VideoMAE encoder forward pass at training time (~0.5s/sample)
  
- THIS script saves: VideoMAE encoder OUTPUT (img_tokens, vid_tokens)
  → Skips encoder entirely at training time (~0.01s/sample)
  → ~50x speedup!

Output Format:
--------------
Each .pt file contains a dict:
{
    'img_tokens': Tensor (N, D) - image tokens from middle frame (N=196, D=768)
    'vid_tokens': Tensor (T', N, D) - video tokens (T'=8, N=196, D=768)
    'grid_hw': (H', W') - spatial grid dimensions (14, 14)
}

Usage:
------
    python build_trackB_videomae_tokens.py \
        --manifest path/to/head_train.jsonl \
        --out_root ../videomae_trackB_tokens \
        --device cuda

This script MUST be run on GPU (CUDA) for reasonable speed.
Colab A100 recommended.

Dependencies:
-------------
- torch
- transformers (for VideoMAE)
- PIL
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install pillow")
    sys.exit(1)

try:
    import torchvision.transforms as T
except ImportError:
    print("ERROR: torchvision is required. Install with: pip install torchvision")
    sys.exit(1)


# =============================================================================
# Constants
# =============================================================================
DEFAULT_TIME_LEN = 16
DEFAULT_TIME_STRIDE = 2
DEFAULT_IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# VideoMAE Encoder Loading
# =============================================================================

_CACHED_ENCODER = None
_CACHED_ENCODER_DEVICE = None
_CACHED_ENCODER_MODEL = None

# Available VideoMAE models from HuggingFace
VIDEOMAE_MODELS = {
    "base": "MCG-NJU/videomae-base",
    "large": "MCG-NJU/videomae-large",
    "huge": "MCG-NJU/videomae-huge",
    "base-kinetics": "MCG-NJU/videomae-base-finetuned-kinetics",
    "large-kinetics": "MCG-NJU/videomae-large-finetuned-kinetics",
}


def load_videomae_encoder_from_scratch(weights_path: str, device: str = "cuda"):
    """Load VideoMAE encoder from custom scratch-trained checkpoint.
    
    Args:
        weights_path: Path to .pt checkpoint
        device: "cuda" or "cpu"
    
    Returns:
        VideoMAE encoder model
    """
    global _CACHED_ENCODER, _CACHED_ENCODER_DEVICE, _CACHED_ENCODER_MODEL
    
    if (_CACHED_ENCODER is not None and 
        _CACHED_ENCODER_DEVICE == device and 
        _CACHED_ENCODER_MODEL == weights_path):
        return _CACHED_ENCODER
    
    print(f"[VideoMAE] Loading custom encoder from: {weights_path}")
    
    ckpt = torch.load(weights_path, map_location=device)
    
    # Detect format
    is_huggingface = (
        'model' in ckpt 
        and 'config' in ckpt 
        and isinstance(ckpt.get('config'), dict)
        and 'hidden_size' in ckpt.get('config', {})
        and any(k.startswith('videomae.') for k in ckpt.get('model', {}).keys())
    )
    
    if is_huggingface:
        try:
            from transformers import VideoMAEModel, VideoMAEConfig as HFVideoMAEConfig
            
            hf_config = HFVideoMAEConfig(**ckpt['config'])
            encoder = VideoMAEModel(hf_config)
            
            # Load encoder weights
            encoder_weights = {k[len('videomae.'):]: v for k, v in ckpt['model'].items() if k.startswith('videomae.')}
            encoder.load_state_dict(encoder_weights, strict=False)
            
            print(f"[VideoMAE] Loaded HuggingFace format checkpoint")
            print(f"  Hidden size: {hf_config.hidden_size}, Layers: {hf_config.num_hidden_layers}")
        except ImportError:
            print("ERROR: transformers required for HuggingFace format checkpoint")
            sys.exit(1)
    else:
        # Custom format - use our videomae module
        try:
            from videomae.videomae_model import load_videomae_encoder as _custom_loader, VideoMAEConfig
        except ImportError:
            from trackB.videomae.videomae_model import load_videomae_encoder as _custom_loader, VideoMAEConfig
        
        # Try to infer config from checkpoint
        vmae_cfg = VideoMAEConfig(
            img_size=224,
            patch_size=16,
            tubelet_size=2,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )
        
        encoder = _custom_loader(weights_path, cfg=vmae_cfg, device=device)
        print(f"[VideoMAE] Loaded custom format checkpoint")
    
    encoder = encoder.to(device)
    encoder.eval()
    
    for param in encoder.parameters():
        param.requires_grad = False
    
    _CACHED_ENCODER = encoder
    _CACHED_ENCODER_DEVICE = device
    _CACHED_ENCODER_MODEL = weights_path
    
    return encoder


def load_videomae_encoder(device: str = "cuda", model_name: str = "base"):
    """Load VideoMAE encoder (HuggingFace implementation).
    
    Args:
        device: "cuda" or "cpu"
        model_name: One of "base", "large", "huge", "base-kinetics", "large-kinetics"
                   Or full HuggingFace model path like "MCG-NJU/videomae-base"
    """
    global _CACHED_ENCODER, _CACHED_ENCODER_DEVICE, _CACHED_ENCODER_MODEL
    
    # Resolve model name to full path
    if model_name in VIDEOMAE_MODELS:
        hf_model = VIDEOMAE_MODELS[model_name]
    else:
        hf_model = model_name  # Assume it's already a full path
    
    if (_CACHED_ENCODER is not None and 
        _CACHED_ENCODER_DEVICE == device and 
        _CACHED_ENCODER_MODEL == hf_model):
        return _CACHED_ENCODER
    
    try:
        from transformers import VideoMAEModel
        print(f"[VideoMAE] Loading encoder: {hf_model} on {device}...")
        encoder = VideoMAEModel.from_pretrained(hf_model)
        encoder = encoder.to(device)
        encoder.eval()
        
        # Freeze
        for param in encoder.parameters():
            param.requires_grad = False
        
        _CACHED_ENCODER = encoder
        _CACHED_ENCODER_DEVICE = device
        _CACHED_ENCODER_MODEL = hf_model
        print(f"[VideoMAE] Encoder loaded successfully (embed_dim={encoder.config.hidden_size})")
        return encoder
        
    except ImportError:
        print("ERROR: transformers is required. Install with: pip install transformers")
        sys.exit(1)


# =============================================================================
# Frame Utilities
# =============================================================================

def list_uid_frames(uid_dir: Path) -> List[Path]:
    """List all JPG frames for a UID, sorted numerically."""
    if not uid_dir.exists():
        return []
    return sorted([p for p in uid_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])


def sample_window_ending_at(frame_path: Path, frames_root: Path, time_len: int = 16, stride: int = 2) -> List[Path]:
    """
    Return a list of frame paths ending at frame_path.
    Steps backward by stride.
    """
    uid_dir = frame_path.parent
    all_frames = list_uid_frames(uid_dir)
    
    if not all_frames:
        return []
    
    # Find index of the annotation frame
    try:
        idx = all_frames.index(frame_path)
    except ValueError:
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
    
    selected.reverse()
    return selected


# =============================================================================
# Token Extraction
# =============================================================================

def build_transform(img_size: int = DEFAULT_IMG_SIZE) -> T.Compose:
    """Build the transformation pipeline."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def extract_frame_tensor(window_paths: List[Path], transform: T.Compose, time_len: int = 16) -> Optional[torch.Tensor]:
    """Load frames and build tensor."""
    if not window_paths:
        return None
    
    frames = []
    last_frame = None
    
    for path in window_paths:
        try:
            with Image.open(str(path)) as im:
                im = im.convert("RGB")
                frame_tensor = transform(im)
                last_frame = frame_tensor
                frames.append(frame_tensor)
        except Exception:
            if last_frame is not None:
                frames.append(last_frame.clone())
            else:
                return None
    
    if not frames:
        return None
    
    # Pad if needed
    while len(frames) < time_len:
        frames.append(frames[-1].clone())
    
    if len(frames) > time_len:
        frames = frames[-time_len:]
    
    video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
    
    return video_tensor


def encode_with_videomae(
    frames_tensor: torch.Tensor,
    encoder: torch.nn.Module,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Run VideoMAE encoder on frame tensor.
    
    Args:
        frames_tensor: (C, T, H, W) - ImageNet normalized
        encoder: VideoMAE encoder
        device: cuda/cpu
        
    Returns:
        img_tokens: (N, D) - image tokens from middle frame
        vid_tokens: (T', N, D) - video tokens  
        grid_hw: (H', W') - spatial grid (14, 14 for 224px)
    """
    # Prepare input: (1, C, T, H, W)
    video = frames_tensor.unsqueeze(0).to(device)
    
    # HuggingFace expects (B, T, C, H, W)
    video_hf = video.permute(0, 2, 1, 3, 4)  # (1, T, C, H, W)
    
    with torch.no_grad():
        outputs = encoder(pixel_values=video_hf, return_dict=True)
        encoded = outputs.last_hidden_state  # (1, num_patches+1, hidden_dim)
    
    # Get grid dimensions
    patch_size = encoder.config.patch_size  # 16
    tubelet_size = encoder.config.tubelet_size  # 2
    num_frames = encoder.config.num_frames  # 16
    image_size = encoder.config.image_size  # 224
    
    T_out = num_frames // tubelet_size  # 8
    H_out = image_size // patch_size  # 14
    W_out = image_size // patch_size  # 14
    
    expected_patches = T_out * H_out * W_out  # 1568
    
    # Remove CLS token if present
    if encoded.shape[1] == expected_patches + 1:
        tokens = encoded[:, 1:, :]  # (1, 1568, 768)
    else:
        tokens = encoded
    
    # Reshape to (T', H'*W', D)
    N_spatial = H_out * W_out  # 196
    vid_tokens = tokens.reshape(1, T_out, N_spatial, -1).squeeze(0)  # (8, 196, 768)
    vid_tokens = vid_tokens.float().cpu()
    
    # For image tokens, take the middle temporal token
    img_tokens = vid_tokens[T_out // 2]  # (196, 768)
    
    grid_hw = (H_out, W_out)
    
    return img_tokens, vid_tokens, grid_hw


# =============================================================================
# Manifest Loading
# =============================================================================

def load_manifest(path: Path) -> List[Dict]:
    """Load manifest from JSON or JSONL file."""
    if not path.exists():
        return []
    
    suffix = path.suffix.lower()
    
    if suffix == '.jsonl':
        records = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    else:
        with open(path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]


def parse_annotation_key(image_path: str) -> Tuple[str, str]:
    """Extract UID and frame index from annotation image path."""
    path = Path(image_path)
    uid = path.parent.name
    frame_idx = path.stem
    return uid, frame_idx


# =============================================================================
# Main Processing
# =============================================================================

def process_manifest(
    manifest_path: Path,
    frames_root: Path,
    out_root: Path,
    device: str = "cuda",
    skip_existing: bool = True,
    time_len: int = DEFAULT_TIME_LEN,
    time_stride: int = DEFAULT_TIME_STRIDE,
    model_name: str = "base",
    scratch_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process manifest and extract VideoMAE tokens.
    
    Args:
        model_name: VideoMAE model variant ("base", "large", "huge", "base-kinetics", "large-kinetics")
        scratch_path: Path to custom scratch-trained checkpoint (overrides model_name)
    """
    # Setup
    tensors_dir = out_root / "tokens"
    logs_dir = out_root / "logs"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    records = load_manifest(manifest_path)
    print(f"[Builder] Loaded {len(records)} annotations from {manifest_path.name}")
    
    if not records:
        return {"error": "No records found"}
    
    # Load encoder - use scratch checkpoint if provided, else HuggingFace model
    if scratch_path:
        encoder = load_videomae_encoder_from_scratch(scratch_path, device)
    else:
        encoder = load_videomae_encoder(device, model_name)
    transform = build_transform()
    
    # Group by unique (uid, frame_idx)
    unique_keys = {}
    for rec in records:
        img_path = rec.get('image') or rec.get('image_path')
        if not img_path:
            continue
        uid, frame_idx = parse_annotation_key(img_path)
        key = f"{uid}_{frame_idx}"
        if key not in unique_keys:
            unique_keys[key] = (uid, frame_idx, img_path)
    
    print(f"[Builder] Found {len(unique_keys)} unique (uid, frame) pairs")
    
    # Process
    stats = {
        "total": len(unique_keys),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }
    
    start_time = time.time()
    
    from tqdm import tqdm
    for key, (uid, frame_idx, img_path) in tqdm(unique_keys.items(), desc="Extracting tokens"):
        out_path = tensors_dir / f"{key}.pt"
        
        # Skip if exists
        if skip_existing and out_path.exists():
            stats["skipped"] += 1
            continue
        
        # Resolve frame path
        img_path_obj = Path(img_path)
        if img_path_obj.is_absolute() and img_path_obj.exists():
            frame_path = img_path_obj
        else:
            # Try relative to frames_root
            frame_path = frames_root / uid / f"{frame_idx}.jpg"
        
        if not frame_path.exists():
            stats["failed"] += 1
            continue
        
        # Get window
        window = sample_window_ending_at(frame_path, frames_root, time_len, time_stride)
        if not window:
            window = [frame_path]
        
        # Extract frame tensor
        frames_tensor = extract_frame_tensor(window, transform, time_len)
        if frames_tensor is None:
            stats["failed"] += 1
            continue
        
        # Encode with VideoMAE
        try:
            img_tokens, vid_tokens, grid_hw = encode_with_videomae(frames_tensor, encoder, device)
            
            # Save
            torch.save({
                'img_tokens': img_tokens,  # (N, D) = (196, 768)
                'vid_tokens': vid_tokens,  # (T', N, D) = (8, 196, 768)
                'grid_hw': grid_hw,  # (14, 14)
            }, out_path)
            
            stats["processed"] += 1
            
        except Exception as e:
            print(f"[Error] {key}: {e}")
            stats["failed"] += 1
    
    elapsed = time.time() - start_time
    stats["elapsed_sec"] = elapsed
    stats["elapsed_min"] = elapsed / 60
    
    # Save summary
    summary_path = logs_dir / "build_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"VideoMAE Token Extraction Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Manifest: {manifest_path}\n")
        f.write(f"Output: {tensors_dir}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Total unique frames: {stats['total']}\n")
        f.write(f"  Processed: {stats['processed']}\n")
        f.write(f"  Skipped (existing): {stats['skipped']}\n")
        f.write(f"  Failed: {stats['failed']}\n")
        f.write(f"\nTime: {stats['elapsed_min']:.1f} minutes\n")
    
    print(f"\n[Done] Processed {stats['processed']}, Skipped {stats['skipped']}, Failed {stats['failed']}")
    print(f"[Done] Time: {stats['elapsed_min']:.1f} minutes")
    print(f"[Done] Output: {tensors_dir}")
    
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build pre-encoded VideoMAE tokens for TrackB")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
    parser.add_argument("--frames_root", type=str, default=None, help="Root directory for extracted frames")
    parser.add_argument("--out_root", type=str, default="../videomae_trackB_tokens", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--model", type=str, default="base", 
                        choices=["base", "large", "huge", "base-kinetics", "large-kinetics"],
                        help="VideoMAE model variant from HuggingFace (default: base)")
    parser.add_argument("--scratch", type=str, default=None,
                        help="Path to custom scratch-trained .pt checkpoint (overrides --model)")
    parser.add_argument("--no_skip", action="store_true", help="Don't skip existing files")
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest).resolve()
    
    # Auto-detect frames_root
    if args.frames_root:
        frames_root = Path(args.frames_root).resolve()
    else:
        # Try to find from manifest location
        script_dir = Path(__file__).resolve().parent
        local_extraction = script_dir.parent
        frames_root = local_extraction / "v2" / "extracted_frames"
    
    out_root = Path(args.out_root).resolve()
    
    print(f"[Builder] Manifest: {manifest_path}")
    print(f"[Builder] Frames root: {frames_root}")
    print(f"[Builder] Output: {out_root}")
    print(f"[Builder] Device: {args.device}")
    
    if args.scratch:
        scratch_path = Path(args.scratch).resolve()
        if not scratch_path.exists():
            print(f"ERROR: Scratch checkpoint not found: {scratch_path}")
            sys.exit(1)
        print(f"[Builder] Using SCRATCH checkpoint: {scratch_path}")
    else:
        scratch_path = None
        print(f"[Builder] Model: {args.model}")
    
    if not frames_root.exists():
        print(f"ERROR: Frames root not found: {frames_root}")
        sys.exit(1)
    
    process_manifest(
        manifest_path=manifest_path,
        frames_root=frames_root,
        out_root=out_root,
        device=args.device,
        skip_existing=not args.no_skip,
        model_name=args.model,
        scratch_path=str(scratch_path) if scratch_path else None,
    )


if __name__ == "__main__":
    main()
