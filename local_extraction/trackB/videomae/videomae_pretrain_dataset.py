#!/usr/bin/env python3
"""
VideoMAE Pretraining Dataset

Consumes STA v2 train clips (egocentric video) for self-supervised VideoMAE pretraining.
Each sample is a T-frame window (e.g., 16 or 32 frames) from an egocentric clip.

Data sources (in priority order):
1. Extracted frames: local_extraction/v2/extracted_frames/<uid>/<frame:07d>.jpg
2. Raw clips: H:\\My Drive\\ego4d_data\\v2\\clips_540\\clips (optional, ~95GB)

This is SELF-SUPERVISED: no labels needed, only raw frames.
"""
from __future__ import annotations

import os
import sys
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

# Add parent paths for imports
_THIS_DIR = Path(__file__).resolve().parent
_TRACKB_DIR = _THIS_DIR.parent
_LOCAL_EXTRACTION = _TRACKB_DIR.parent
for p in [str(_LOCAL_EXTRACTION), str(_LOCAL_EXTRACTION.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from core import load_config, get_paths
except ImportError:
    load_config = None
    get_paths = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VideoMAEDatasetConfig:
    """Configuration for VideoMAE pretraining dataset."""
    
    # Temporal settings
    num_frames: int = 16          # T frames per sample
    frame_stride: int = 2         # stride between sampled frames
    
    # Spatial settings  
    img_size: int = 224           # resize to this size
    crop_size: int = 224          # random crop size (after resize)
    
    # Augmentation
    use_augmentation: bool = True
    random_flip: bool = True
    color_jitter: bool = True
    color_jitter_strength: float = 0.4
    
    # Normalization (ImageNet stats)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data source
    frames_root: Optional[Path] = None
    clips_root: Optional[Path] = None  # Optional: raw video clips
    train_manifest: Optional[Path] = None  # STA v2 train manifest for UIDs
    
    # Sampling
    samples_per_uid: int = 4      # how many T-frame windows per clip
    min_frames_required: int = 32  # minimum frames in clip to sample from
    
    def __post_init__(self):
        if self.frames_root is not None:
            self.frames_root = Path(self.frames_root)
        if self.clips_root is not None:
            self.clips_root = Path(self.clips_root)
        if self.train_manifest is not None:
            self.train_manifest = Path(self.train_manifest)


# =============================================================================
# Dataset
# =============================================================================

class VideoMAEPretrainDataset(Dataset):
    """
    Dataset for VideoMAE self-supervised pretraining on egocentric clips.
    
    Each __getitem__ returns:
        video: Tensor of shape (C, T, H, W) or (T, C, H, W) depending on format
        metadata: dict with uid, frame indices, etc.
    """
    
    def __init__(
        self,
        cfg: VideoMAEDatasetConfig,
        split: str = "train",
        output_format: str = "CTHW",  # "CTHW" or "TCHW"
    ):
        self.cfg = cfg
        self.split = split
        self.output_format = output_format
        
        # Build transforms
        self.transform = self._build_transform()
        
        # Collect samples
        self.samples = self._collect_samples()
        
        print(f"[VideoMAE Dataset] Created {len(self.samples)} samples from {split} split")
    
    def _build_transform(self) -> T.Compose:
        """Build data augmentation transforms."""
        transforms = []
        
        # Resize (slightly larger for random crop)
        resize_size = int(self.cfg.img_size * 1.14) if self.cfg.use_augmentation else self.cfg.img_size
        transforms.append(T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC))
        
        # Random crop or center crop
        if self.cfg.use_augmentation:
            transforms.append(T.RandomCrop(self.cfg.crop_size))
        else:
            transforms.append(T.CenterCrop(self.cfg.crop_size))
        
        # Random horizontal flip
        if self.cfg.use_augmentation and self.cfg.random_flip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        
        # Color jitter
        if self.cfg.use_augmentation and self.cfg.color_jitter:
            s = self.cfg.color_jitter_strength
            transforms.append(T.ColorJitter(brightness=s, contrast=s, saturation=s, hue=s/4))
        
        # To tensor and normalize
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=self.cfg.mean, std=self.cfg.std))
        
        return T.Compose(transforms)
    
    def _collect_samples(self) -> List[Dict[str, Any]]:
        """Collect all samples (T-frame windows) from available clips."""
        samples = []
        
        # Get frames root
        frames_root = self.cfg.frames_root
        if frames_root is None and get_paths is not None:
            try:
                paths = get_paths()
                frames_root = paths.frames_root
            except Exception:
                pass
        
        if frames_root is None or not frames_root.exists():
            print(f"[VideoMAE Dataset] Warning: frames_root not found: {frames_root}")
            return samples
        
        # Get list of UIDs (from manifest or directory scan)
        uids = self._get_train_uids(frames_root)
        
        print(f"[VideoMAE Dataset] Found {len(uids)} UIDs in {frames_root}")
        
        for uid in uids:
            uid_dir = frames_root / uid
            if not uid_dir.is_dir():
                continue
            
            # Get all frames for this clip
            frame_files = sorted([
                f for f in uid_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            
            # Skip if not enough frames
            if len(frame_files) < self.cfg.min_frames_required:
                continue
            
            # Sample multiple windows per clip
            num_samples = min(self.cfg.samples_per_uid, len(frame_files) // self.cfg.num_frames)
            
            for _ in range(num_samples):
                # Random start index
                window_length = self.cfg.num_frames * self.cfg.frame_stride
                max_start = len(frame_files) - window_length
                
                if max_start <= 0:
                    start_idx = 0
                else:
                    start_idx = random.randint(0, max_start)
                
                # Select frame indices with stride
                frame_indices = list(range(
                    start_idx, 
                    start_idx + window_length, 
                    self.cfg.frame_stride
                ))[:self.cfg.num_frames]
                
                samples.append({
                    'uid': uid,
                    'uid_dir': uid_dir,
                    'frame_files': frame_files,
                    'frame_indices': frame_indices,
                })
        
        return samples
    
    def _get_train_uids(self, frames_root: Path) -> List[str]:
        """Get list of training UIDs from manifest or directory scan."""
        uids = []
        
        # Try to load from train manifest
        if self.cfg.train_manifest and self.cfg.train_manifest.exists():
            try:
                with open(self.cfg.train_manifest) as f:
                    manifest = json.load(f)
                
                # Extract unique UIDs from manifest
                if isinstance(manifest, dict):
                    # Try different manifest structures
                    if 'clips' in manifest:
                        uids = list({c.get('video_uid', c.get('uid', '')) for c in manifest['clips']})
                    elif 'samples' in manifest:
                        uids = list({s.get('uid', '') for s in manifest['samples']})
                elif isinstance(manifest, list):
                    uids = list({item.get('uid', item.get('video_uid', '')) for item in manifest})
                
                uids = [u for u in uids if u]  # Remove empty strings
                
                if uids:
                    print(f"[VideoMAE Dataset] Loaded {len(uids)} UIDs from manifest")
                    return uids
            except Exception as e:
                print(f"[VideoMAE Dataset] Warning: Could not load manifest: {e}")
        
        # Fall back to directory scan
        uids = [d.name for d in frames_root.iterdir() if d.is_dir()]
        return uids
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            video: Tensor of shape (C, T, H, W) or (T, C, H, W)
            metadata: dict with sample info
        """
        sample = self.samples[idx]
        
        # Load frames
        frames = []
        for frame_idx in sample['frame_indices']:
            frame_path = sample['frame_files'][frame_idx]
            
            with Image.open(str(frame_path)) as img:
                img = img.convert('RGB')
                
                # Apply consistent random seed for all frames in same clip
                # This ensures same spatial transform across temporal dimension
                if idx == 0:
                    random.seed(idx)
                    torch.manual_seed(idx)
                
                frame_tensor = self.transform(img)
                frames.append(frame_tensor)
        
        # Stack frames: (T, C, H, W)
        video = torch.stack(frames, dim=0)
        
        # Convert to output format
        if self.output_format == "CTHW":
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        # else keep as (T, C, H, W)
        
        metadata = {
            'uid': sample['uid'],
            'frame_indices': sample['frame_indices'],
            'num_frames': len(frames),
        }
        
        return video, metadata
    
    def resample(self):
        """Re-collect samples with new random windows (call between epochs)."""
        self.samples = self._collect_samples()


# =============================================================================
# Demo / Test
# =============================================================================

def _demo():
    """Test the dataset."""
    print("=" * 60)
    print("VideoMAE Pretrain Dataset Demo")
    print("=" * 60)
    
    # Try to get paths
    try:
        paths = get_paths()
        frames_root = paths.frames_root
    except Exception:
        frames_root = Path("local_extraction/v2/extracted_frames")
    
    print(f"Frames root: {frames_root}")
    
    if not frames_root.exists():
        print(f"[Demo] Frames root does not exist, skipping test")
        return
    
    # Create config
    cfg = VideoMAEDatasetConfig(
        frames_root=frames_root,
        num_frames=16,
        frame_stride=2,
        samples_per_uid=2,
    )
    
    # Create dataset
    dataset = VideoMAEPretrainDataset(cfg)
    
    if len(dataset) == 0:
        print("[Demo] No samples found")
        return
    
    # Load one sample
    video, meta = dataset[0]
    
    print(f"\n[Demo] Sample 0:")
    print(f"  Video shape: {video.shape}")
    print(f"  UID: {meta['uid']}")
    print(f"  Frame indices: {meta['frame_indices']}")
    print(f"  dtype: {video.dtype}")
    print(f"  value range: [{video.min():.3f}, {video.max():.3f}]")
    
    print("\n[Demo] Success!")


if __name__ == "__main__":
    _demo()
