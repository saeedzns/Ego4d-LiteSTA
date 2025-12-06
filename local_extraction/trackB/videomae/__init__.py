"""
VideoMAE Module for Ego-only Self-Supervised Pretraining

This module provides:
1. VideoMAE pretraining dataset for egocentric clips
2. VideoMAE encoder/decoder model
3. Pretraining script with masked autoencoding

Usage:
    # Pretraining
    python -m trackB.videomae.videomae_pretrain --epochs 100
    
    # Load pretrained encoder for Track B
    from trackB.videomae import load_videomae_encoder
    encoder = load_videomae_encoder("runs/VideoMAE/videomae_ego_encoder.pt")
"""

from .videomae_model import VideoMAEEncoder, VideoMAE, VideoMAEConfig, load_videomae_encoder
from .videomae_pretrain_dataset import VideoMAEPretrainDataset

__all__ = [
    "VideoMAEEncoder",
    "VideoMAE",
    "VideoMAEConfig",
    "load_videomae_encoder",
    "VideoMAEPretrainDataset",
]
