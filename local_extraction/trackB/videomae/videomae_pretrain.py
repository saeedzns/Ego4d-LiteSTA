#!/usr/bin/env python3
"""
VideoMAE Pretraining Script

Self-supervised pretraining on egocentric video clips (STA v2 train).
Uses masked autoencoding to learn spatiotemporal representations.

Usage:
    # From local_extraction directory
    python -m trackB.videomae.videomae_pretrain --epochs 100 --batch_size 8
    
    # Demo mode (quick test)
    python -m trackB.videomae.videomae_pretrain --demo
    
    # Resume training
    python -m trackB.videomae.videomae_pretrain --resume runs/VideoMAE/checkpoint_latest.pt

Output:
    - Encoder checkpoint: runs/VideoMAE/videomae_ego_encoder.pt
    - Training logs: runs/VideoMAE/pretrain_log.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent paths for imports
_THIS_DIR = Path(__file__).resolve().parent
_TRACKB_DIR = _THIS_DIR.parent
_LOCAL_EXTRACTION = _TRACKB_DIR.parent
for p in [str(_LOCAL_EXTRACTION), str(_LOCAL_EXTRACTION.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from .videomae_model import VideoMAE, VideoMAEConfig, VideoMAEEncoder
from .videomae_pretrain_dataset import VideoMAEPretrainDataset, VideoMAEDatasetConfig

# Try to import core modules
try:
    from core import load_config, get_paths, RunLogger, create_run_logger
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    load_config = None
    get_paths = None
    RunLogger = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PretrainConfig:
    """Configuration for VideoMAE pretraining."""
    
    # Model
    model_variant: str = "base"      # "tiny", "small", "base", "large"
    img_size: int = 224
    patch_size: int = 16
    tubelet_size: int = 2
    num_frames: int = 16
    frame_stride: int = 2
    mask_ratio: float = 0.9          # VideoMAE uses 90% masking
    
    # Training
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1.5e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    weight_decay: float = 0.05
    
    # Optimizer
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.95)
    
    # Learning rate schedule
    lr_schedule: str = "cosine"      # "cosine" or "step"
    
    # Data
    num_workers: int = 4
    pin_memory: bool = True
    samples_per_uid: int = 4
    
    # Augmentation
    use_augmentation: bool = True
    random_flip: bool = True
    color_jitter: bool = True
    
    # Checkpointing
    save_freq: int = 10              # Save every N epochs
    save_best: bool = True
    
    # Demo mode
    demo: bool = False
    demo_steps: int = 10
    
    # Device
    device: str = "auto"
    use_amp: bool = True             # Automatic mixed precision
    
    # Paths
    output_dir: str = "runs/VideoMAE"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(self.output_dir)


# =============================================================================
# Training Utilities
# =============================================================================

def get_lr_scheduler(optimizer, cfg: PretrainConfig, steps_per_epoch: int):
    """Create learning rate scheduler."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return cfg.min_lr / cfg.lr + (1 - cfg.min_lr / cfg.lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


import math


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(
    model: VideoMAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    cfg: PretrainConfig,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
    
    for step, (video, meta) in enumerate(pbar):
        # Move to device
        video = video.to(cfg.device, non_blocking=True)
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=cfg.use_amp and cfg.device == "cuda"):
            output = model(video, mask_ratio=cfg.mask_ratio)
            loss = output['loss']
        
        # Backward pass
        optimizer.zero_grad()
        
        if cfg.use_amp and cfg.device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Update metrics
        loss_meter.update(loss.item(), video.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })
        
        # Demo mode: early exit
        if cfg.demo and step >= cfg.demo_steps:
            break
    
    return {
        'loss': loss_meter.avg,
        'lr': optimizer.param_groups[0]['lr'],
    }


def save_checkpoint(
    model: VideoMAE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    cfg: PretrainConfig,
    metrics: Dict[str, float],
    is_best: bool = False,
):
    """Save training checkpoint."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'config': asdict(cfg),
        'model_config': asdict(model.cfg),
        'state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    
    # Save latest
    latest_path = cfg.output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save periodic
    if (epoch + 1) % cfg.save_freq == 0:
        epoch_path = cfg.output_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
        torch.save(checkpoint, epoch_path)
    
    # Save best
    if is_best:
        best_path = cfg.output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
    
    # Save encoder-only checkpoint (for downstream use)
    encoder_checkpoint = {
        'config': asdict(model.cfg),
        'encoder_state_dict': model.encoder.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
    }
    encoder_path = cfg.output_dir / "videomae_ego_encoder.pt"
    torch.save(encoder_checkpoint, encoder_path)
    
    print(f"[Checkpoint] Saved to {cfg.output_dir}")


def load_checkpoint(
    model: VideoMAE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_path: str,
) -> int:
    """Load training checkpoint. Returns starting epoch."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    
    start_epoch = ckpt['epoch'] + 1
    print(f"[Checkpoint] Loaded from epoch {ckpt['epoch']}")
    
    return start_epoch


# =============================================================================
# Main Training Function
# =============================================================================

def train(cfg: PretrainConfig):
    """Main training function."""
    print("=" * 60)
    print("VideoMAE Pretraining")
    print("=" * 60)
    
    # Create run logger if available
    logger = None
    if HAS_CORE and RunLogger is not None:
        logger = create_run_logger(
            track='VideoMAE',
            stage='pretrain',
            run_dir=cfg.output_dir,
        )
        logger.log_config(asdict(cfg))
        logger.log_start()
    
    # Get data paths
    frames_root = None
    if HAS_CORE and get_paths is not None:
        try:
            paths = get_paths()
            frames_root = paths.frames_root
        except Exception:
            pass
    
    if frames_root is None:
        frames_root = Path("local_extraction/v2/extracted_frames")
    
    print(f"Frames root: {frames_root}")
    print(f"Device: {cfg.device}")
    print(f"Demo mode: {cfg.demo}")
    
    # Create dataset
    dataset_cfg = VideoMAEDatasetConfig(
        num_frames=cfg.num_frames,
        frame_stride=cfg.frame_stride,
        img_size=cfg.img_size,
        use_augmentation=cfg.use_augmentation,
        random_flip=cfg.random_flip,
        color_jitter=cfg.color_jitter,
        frames_root=frames_root,
        samples_per_uid=cfg.samples_per_uid,
    )
    
    dataset = VideoMAEPretrainDataset(dataset_cfg, split="train", output_format="CTHW")
    
    if len(dataset) == 0:
        print("[Error] No samples found. Check frames_root path.")
        return
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    
    # Create model
    model_configs = {
        "tiny": dict(embed_dim=192, depth=6, num_heads=3, decoder_embed_dim=96, decoder_depth=2),
        "small": dict(embed_dim=384, depth=8, num_heads=6, decoder_embed_dim=192, decoder_depth=3),
        "base": dict(embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=384, decoder_depth=4),
        "large": dict(embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8),
    }
    
    model_cfg_dict = model_configs.get(cfg.model_variant, model_configs["base"])
    model_cfg_dict.update({
        'img_size': cfg.img_size,
        'patch_size': cfg.patch_size,
        'tubelet_size': cfg.tubelet_size,
        'mask_ratio': cfg.mask_ratio,
    })
    
    model_cfg = VideoMAEConfig(**model_cfg_dict)
    model = VideoMAE(model_cfg).to(cfg.device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_params_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"Model: {cfg.model_variant}, Params: {n_params/1e6:.2f}M (Encoder: {n_params_enc/1e6:.2f}M)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    
    # Create scheduler
    steps_per_epoch = len(dataloader)
    scheduler = get_lr_scheduler(optimizer, cfg, steps_per_epoch)
    
    # Create AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and cfg.device == "cuda")
    
    # Resume if specified
    start_epoch = 0
    best_loss = float('inf')
    
    # Training loop
    print(f"\nStarting training for {cfg.epochs} epochs...")
    training_log = []
    
    try:
        for epoch in range(start_epoch, cfg.epochs):
            # Train one epoch
            metrics = train_one_epoch(
                model, dataloader, optimizer, scheduler, scaler, epoch, cfg
            )
            
            # Track best
            is_best = metrics['loss'] < best_loss
            if is_best:
                best_loss = metrics['loss']
            
            # Save checkpoint
            if (epoch + 1) % cfg.save_freq == 0 or is_best or epoch == cfg.epochs - 1:
                save_checkpoint(model, optimizer, scheduler, epoch, cfg, metrics, is_best)
            
            # Log
            metrics['epoch'] = epoch + 1
            training_log.append(metrics)
            
            print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {metrics['loss']:.4f} | LR: {metrics['lr']:.2e}")
            
            # Resample dataset between epochs (different windows)
            dataset.resample()
            
            # Demo mode: early exit
            if cfg.demo:
                break
        
        # Save final log
        log_path = cfg.output_dir / "pretrain_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                'config': asdict(cfg),
                'model_config': asdict(model_cfg),
                'training_log': training_log,
                'best_loss': best_loss,
            }, f, indent=2)
        
        print(f"\n[Done] Training complete. Best loss: {best_loss:.4f}")
        print(f"Encoder saved to: {cfg.output_dir / 'videomae_ego_encoder.pt'}")
        
        # Log completion
        if logger:
            logger.log_metrics({
                'best_loss': best_loss,
                'final_loss': training_log[-1]['loss'] if training_log else None,
                'epochs_trained': len(training_log),
            })
            logger.log_artifacts([
                str(cfg.output_dir / 'videomae_ego_encoder.pt'),
                str(cfg.output_dir / 'pretrain_log.json'),
            ])
            logger.log_end(success=True)
            logger.save()
        
    except Exception as e:
        if logger:
            logger.log_end(success=False, error=str(e))
            logger.save()
        raise


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VideoMAE Pretraining")
    
    # Model
    parser.add_argument("--model", type=str, default="base", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.9)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    
    # Data
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--samples_per_uid", type=int, default=4)
    
    # Checkpointing
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="runs/VideoMAE")
    parser.add_argument("--save_freq", type=int, default=10)
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    
    # Demo
    parser.add_argument("--demo", action="store_true", help="Quick demo mode")
    parser.add_argument("--demo_steps", type=int, default=10)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = PretrainConfig(
        model_variant=args.model,
        img_size=args.img_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        num_frames=args.num_frames,
        mask_ratio=args.mask_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        samples_per_uid=args.samples_per_uid,
        output_dir=args.output_dir,
        save_freq=args.save_freq,
        device=args.device,
        use_amp=not args.no_amp,
        demo=args.demo,
        demo_steps=args.demo_steps,
    )
    
    train(cfg)


if __name__ == "__main__":
    main()
