#!/usr/bin/env python3
"""
VideoMAE Pretraining Dataset

Consumes STA v2 train clips (egocentric video) for self-supervised VideoMAE pretraining.
Each sample is a T-frame window (e.g., 16 or 32 frames) from an egocentric clip.

Data sources (in priority order):
1. Raw clips (.mp4): clips_root/<uid>.mp4 (preferred for Colab)
2. Extracted frames: frames_root/<uid>/<frame:07d>.jpg

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

# Try to import video reading library
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

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
        """Collect all samples (T-frame windows) from available clips or videos."""
        samples = []
        
        # Check clips_root first (video files)
        clips_root = self.cfg.clips_root
        if clips_root is not None and clips_root.exists():
            return self._collect_samples_from_clips(clips_root)
        
        # Fall back to frames_root (extracted frames)
        frames_root = self.cfg.frames_root
        if frames_root is None and get_paths is not None:
            try:
                paths = get_paths()
                frames_root = paths.frames_root
            except Exception:
                pass
        
        if frames_root is not None and frames_root.exists():
            return self._collect_samples_from_frames(frames_root)
        
        print(f"[VideoMAE Dataset] Warning: No data source found!")
        print(f"  clips_root: {clips_root}")
        print(f"  frames_root: {frames_root}")
        return samples
    
    def _collect_samples_from_clips(self, clips_root: Path) -> List[Dict[str, Any]]:
        """Collect samples from video clip files (.mp4)."""
        samples = []
        
        if not HAS_CV2:
            print("[VideoMAE Dataset] ERROR: cv2 not available for video reading!")
            print("  Install with: pip install opencv-python")
            return samples
        
        # Find all video files
        video_files = sorted([
            f for f in clips_root.iterdir()
            if f.is_file() and f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')
        ])
        
        print(f"[VideoMAE Dataset] Found {len(video_files)} video files in {clips_root}")
        
        for video_path in video_files:
            uid = video_path.stem  # filename without extension
            
            # Get video info
            try:
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception as e:
                print(f"[VideoMAE Dataset] Warning: Could not read {video_path}: {e}")
                continue
            
            # Skip if not enough frames
            if total_frames < self.cfg.min_frames_required:
                continue
            
            # Sample multiple windows per video
            window_length = self.cfg.num_frames * self.cfg.frame_stride
            num_samples = min(self.cfg.samples_per_uid, total_frames // window_length)
            
            for _ in range(max(1, num_samples)):
                max_start = total_frames - window_length
                if max_start <= 0:
                    start_idx = 0
                else:
                    start_idx = random.randint(0, max_start)
                
                frame_indices = list(range(
                    start_idx,
                    start_idx + window_length,
                    self.cfg.frame_stride
                ))[:self.cfg.num_frames]
                
                samples.append({
                    'uid': uid,
                    'video_path': video_path,
                    'frame_indices': frame_indices,
                    'source': 'video',
                })
        
        print(f"[VideoMAE Dataset] Created {len(samples)} samples from video clips")
        return samples
    
    def _collect_samples_from_frames(self, frames_root: Path) -> List[Dict[str, Any]]:
        """Collect samples from extracted frame directories."""
        samples = []
        
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
                    'source': 'frames',
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
        
        # Load frames based on source type
        if sample.get('source') == 'video':
            frames = self._load_frames_from_video(sample, idx)
        else:
            frames = self._load_frames_from_dir(sample, idx)
        
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
    
    def _load_frames_from_video(self, sample: Dict, idx: int) -> List[torch.Tensor]:
        """Load frames from a video file using cv2."""
        frames = []
        video_path = sample['video_path']
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Set consistent random seed for spatial transforms
        random.seed(idx)
        torch.manual_seed(idx)
        
        for frame_idx in sample['frame_indices']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame read fails, duplicate last frame or use zeros
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, self.cfg.img_size, self.cfg.img_size))
                continue
            
            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            frame_tensor = self.transform(img)
            frames.append(frame_tensor)
        
        cap.release()
        return frames
    
    def _load_frames_from_dir(self, sample: Dict, idx: int) -> List[torch.Tensor]:
        """Load frames from extracted frame files."""
        frames = []
        
        # Set consistent random seed for spatial transforms
        random.seed(idx)
        torch.manual_seed(idx)
        
        for frame_idx in sample['frame_indices']:
            frame_path = sample['frame_files'][frame_idx]
            
            with Image.open(str(frame_path)) as img:
                img = img.convert('RGB')
                frame_tensor = self.transform(img)
                frames.append(frame_tensor)
        
        return frames
    
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
