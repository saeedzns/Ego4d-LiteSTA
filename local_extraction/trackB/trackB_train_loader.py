#!/usr/bin/env python3
"""
Track B DataLoader Training Demo
- Uses TrackBDataset + collate
- Performs tokenization inside Dataset (__getitem__) and fusion in the training loop
- Applies label smoothing and TTC normalization (if available)
- Logs progress with tqdm

Configuration is loaded from configs/trackB.yaml (or --config argument)

Usage:
    # Default (uses trackB.yaml)
    python -m trackB.trackB_train_loader
    
    # With specific config preset
    python -m trackB.trackB_train_loader --config trackB_resnet18_baseline
    python -m trackB.trackB_train_loader --config trackB_videomae_ego

Note: To avoid multiprocessing issues with model objects in Dataset, set num_workers=0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
from PIL import Image

# ================== CLI ARGUMENT PARSING ==================
def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Track B Training')
    parser.add_argument('--config', type=str, default='trackB',
                        help='Config name to load (e.g., trackB, trackB_resnet18_baseline, trackB_videomae_ego)')
    parser.add_argument('--demo', action='store_true',
                        help='Enable demo mode (quick test)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    return parser.parse_args()

# Parse args before loading config (so we can use --config)
_cli_args = _parse_args()
# ==========================================================

# ================== YAML CONFIG LOADING ==================
_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_EXTRACTION = _THIS_DIR.parent
_REPO_ROOT = _LOCAL_EXTRACTION.parent

for p in [str(_LOCAL_EXTRACTION), str(_REPO_ROOT), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core import load_config

# Load configuration from YAML (using --config if specified)
_config_name = _cli_args.config
print(f"[TrackB] Loading config: {_config_name}")
_cfg = load_config(_config_name)

# Log which backbone is being used
_video_backbone = _cfg.get('model.tokenizer.video_backbone', 'resnet18')
print(f"[TrackB] Video backbone: {_video_backbone}")

# Set the tokenizer config to use the same config (before importing tokenizer)
import trackB_tokenizer
trackB_tokenizer.set_tokenizer_config(_cfg)
# =========================================================

from trackB_dataset import (
    TrackBDataset,
    trackB_collate,
    latest_stageB_run,
    resolve_stageB_manifest,
)
from trackB_metrics import (
    binary_accuracy, multiclass_accuracy, binary_average_precision,
    per_class_ap, ttc_mae, denorm_ttc,
)
from trackB_tokenizer import (
    TokenizerConfig,
    roi_pool_tokens_mean,
    build_backbone,
    build_transform,
    list_uid_frames,
    sample_window_ending_at,
    image_grid_tokens,
    video_grid_tokens,
)
from trackB_fusion import FusionConfig, TrackBFusion
from trackB_head import HeadConfig, TrackBHead


# --- Configuration Class (populated from YAML) ---


class TrainConfig:
    """Configuration for training, populated from YAML config.
    mode: 'demo' uses a tiny fixed-step loop; 'main' iterates full epochs.
    demo_variant selects between DataLoader-based or standalone sampling demo flows.
    """

    # Training mode
    mode: str = 'demo' if _cfg.get('demo.enabled', False) else 'main'
    
    # Demo settings
    demo_steps: int = _cfg.get('demo.steps', 10)
    demo_variant: str = _cfg.get('demo.variant', 'loader')
    demo_use_real_labels: bool = _cfg.get('demo.use_real_labels', True)
    demo_manifest: str | None = None
    
    # Training hyperparameters
    epochs: int = _cfg.get('training.epochs', 20)
    batch_size: int = _cfg.get('training.batch_size', 8)
    lr: float = _cfg.get('training.lr', 1e-3)
    min_lr: float = _cfg.get('training.min_lr', 1e-5)
    warmup_epochs: float = _cfg.get('training.warmup_epochs', 1.0)
    candidate_limit: int = _cfg.get('training.candidate_limit', 16)
    label_smoothing: float = _cfg.get('training.label_smoothing', 0.05)
    normalize_ttc: bool = _cfg.get('training.normalize_ttc', True)
    
    # Checkpointing
    save_epoch_checkpoints: bool = _cfg.get('training.save_epoch_checkpoints', True)
    save_best_checkpoint: bool = _cfg.get('training.save_best_checkpoint', True)
    
    # Mixed precision
    amp: bool = _cfg.get('training.amp', False)
    
    # Evaluation
    eval_every: int = _cfg.get('training.eval_every', 1)
    
    # Early stopping
    early_stopping_patience: int = _cfg.get('training.early_stopping.patience', 5)
    monitor_metric: str = _cfg.get('training.early_stopping.monitor', 'mAP')
    greater_is_better: bool = _cfg.get('training.early_stopping.mode', 'max') == 'max'
    
    # Data sources
    val_manifest: str | None = _cfg.get('data.val_manifest', None)
    train_manifest: str | None = _cfg.get('data.train_manifest', None)
    stageB_run: str | None = _cfg.get('data.stageB_run', None)
    
    # Pre-extracted tokens (for ~120x faster training)
    # Set to path like "v2/resnet18_tokens" to use pre-extracted ResNet18 tokens
    tokens_root: str | None = _cfg.get('data.tokens_root', None)
    
    # Multi-task settings
    use_multi_task_labels: bool = _cfg.get('multi_task.enabled', True)
    use_ttc_bins: bool = _cfg.get('multi_task.ttc_mode', 'reg') == 'bin'
    
    # Multi-task loss weights
    loss_w_next: float = _cfg.get('multi_task.loss_weights.next_active', 1.5)
    loss_w_noun: float = _cfg.get('multi_task.loss_weights.noun', 0.25)
    loss_w_verb: float = _cfg.get('multi_task.loss_weights.verb', 0.25)
    loss_w_ttc: float = _cfg.get('multi_task.loss_weights.ttc', 1.0)


def _config_to_dict(cfg: TrainConfig) -> Dict[str, Any]:
    """Serialize TrainConfig (class + instance attributes) to a plain dict."""
    out: Dict[str, Any] = {}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        try:
            val = getattr(cfg, name)
        except Exception:
            continue
        # Only keep simple JSON-serializable scalars
        if isinstance(val, (int, float, bool, str)) or val is None:
            out[name] = val
    return out


# Model architecture constants (from YAML)
TOKEN_DIM: int = _cfg.get('model.projector.out_dim', 256)
FUSION_LAYERS: int = _cfg.get('model.fusion.layers', 2)
HEAD_HIDDEN: int = _cfg.get('model.head.hidden', 256)
TTC_LOSS_WEIGHT: float = _cfg.get('multi_task.loss_weights.ttc', 1.0) * 0.1
DEMO_SYNTHETIC_BOX_COUNT: int = 4
DEMO_MANIFEST_PRIORITY: List[str] = [
    'head_train.jsonl',
    'head_train.json',
    'head_train_clip.json',
    'head_train_clip.jsonl',
    'head_train_video.json',
    'head_train_video.jsonl',
]
DEMO_SYNTHETIC_TTC_MAX: float = 2.0


def pad_and_stack(cand_list: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    cand_list: list of (Nc_i, C)
    returns:
      feats: (B, Nc_max, C)
      mask:  (B, Nc_max) 1=valid, 0=pad
    """
    B = len(cand_list)
    C = cand_list[0].shape[-1]
    Nc_max = max(x.shape[0] for x in cand_list)
    feats = cand_list[0].new_full((B, Nc_max, C), pad_value)
    mask = cand_list[0].new_zeros((B, Nc_max), dtype=torch.bool)
    for i, f in enumerate(cand_list):
        n = f.shape[0]
        feats[i, :n, :] = f
        mask[i, :n] = True
    return feats, mask


def _find_default_val_manifest(root: Path) -> Path | None:
    for name in [
        'head_val.jsonl', 'head_val.json',
        'head_val_clip.json', 'head_val_clip.jsonl',
        'head_val_video.json', 'head_val_video.jsonl',
        'val.json', 'val.jsonl'
    ]:
        p = root / name
        if p.exists():
            return p
    return None


def _print_config(cfg: TrainConfig):
    print("[TrackB] Config:", {
        'mode': cfg.mode,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'lr': cfg.lr,
        'min_lr': cfg.min_lr,
        'warmup_epochs': cfg.warmup_epochs,
        'candidate_limit': cfg.candidate_limit,
        'label_smoothing': cfg.label_smoothing,
        'normalize_ttc': cfg.normalize_ttc,
        'amp': cfg.amp,
        'demo_steps': cfg.demo_steps,
        'demo_variant': cfg.demo_variant,
        'demo_use_real_labels': cfg.demo_use_real_labels,
        'demo_manifest': cfg.demo_manifest,
        'eval_every': cfg.eval_every,
        'early_stopping_patience': cfg.early_stopping_patience,
        'monitor_metric': cfg.monitor_metric,
        'save_best_checkpoint': cfg.save_best_checkpoint,
        'val_manifest': cfg.val_manifest,
        'train_manifest': cfg.train_manifest,
        'stageB_run': cfg.stageB_run,
        'use_multi_task_labels': cfg.use_multi_task_labels,
        'use_ttc_bins': cfg.use_ttc_bins,
        'loss_w_next': cfg.loss_w_next,
        'loss_w_noun': cfg.loss_w_noun,
        'loss_w_verb': cfg.loss_w_verb,
        'loss_w_ttc': cfg.loss_w_ttc,
    })


def _run_demo(
    cfg: TrainConfig,
    loader: torch.utils.data.DataLoader,
    projector: nn.Module,
    fusion: TrackBFusion,
    head: TrackBHead,
    ce: nn.Module,
    l1: nn.Module,
    opt: optim.Optimizer,
    device: str,
    frames_root: Path,
    manif_root: Path,
    tcfg: TokenizerConfig,
):
        variant = cfg.demo_variant.lower().strip()
        if variant not in {'loader', 'standalone'}:
            raise ValueError(f"Unknown demo_variant '{cfg.demo_variant}'. Use 'loader' or 'standalone'.")

        if variant == 'loader':
            steps = cfg.demo_steps
            batch_iter = iter(loader)
            pbar = tqdm(range(steps), desc='[TrackB demo] steps (loader)')
            for _ in pbar:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(loader)
                    batch = next(batch_iter)
                if not batch.get('valid'):
                    continue
                samples: List[Dict[str, Any]] = batch['samples']
                img_cands: List[torch.Tensor] = []
                vid_cands: List[torch.Tensor] = []
                labels_list: List[torch.Tensor] = []
                ttc_list: List[torch.Tensor] = []

                for s in samples:
                    img_b = projector(s['img_tokens'].to(device)).unsqueeze(0)
                    vid_b = projector(s['vid_tokens'].to(device)).unsqueeze(0)
                    fused_img, fused_vid = fusion(img_b, vid_b)
                    W, H = s['image_size']; hw = s['hw']
                    pooled_img = []
                    pooled_vid = []
                    for box in s['bboxes']:
                        pooled_img.append(roi_pool_tokens_mean(hw, fused_img[0], box, (W, H)))
                        pooled_vid.append(roi_pool_tokens_mean(hw, fused_vid[0], box, (W, H)))
                    if not pooled_img:
                        continue
                    img_cands.append(torch.stack(pooled_img, dim=0))
                    vid_cands.append(torch.stack(pooled_vid, dim=0))
                    labels_list.append(s.get('is_positive', s['labels']).to(device))
                    ttc_list.append(s.get('ttc_norm', s['ttc']).to(device))

                if not img_cands:
                    continue

                img_pad, mask = pad_and_stack(img_cands)
                vid_pad, _ = pad_and_stack(vid_cands)
                labels_pad, _ = pad_and_stack([x.unsqueeze(-1).float() for x in labels_list])
                labels_pad = labels_pad.long().squeeze(-1)
                ttc_pad, _ = pad_and_stack([x.unsqueeze(-1) for x in ttc_list])

                out = head(img_pad.to(device), vid_pad.to(device))
                B, Nc, K = out['cls_logits'].shape
                ce_loss = ce(out['cls_logits'].reshape(B * Nc, K), labels_pad.reshape(B * Nc))
                ttc_loss_all = l1(out['ttc'], ttc_pad).squeeze(-1)
                ttc_loss = (ttc_loss_all * mask.to(device).float()).sum() / mask.to(device).float().sum().clamp_min(1.0)
                loss = ce_loss + TTC_LOSS_WEIGHT * ttc_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'cls': f"{ce_loss.item():.4f}",
                    'ttc': f"{ttc_loss.item():.4f}",
                })

            print('[trackB.demo] Training complete (loader variant).')
            return

        # Standalone variant mirrors trackB_train_demo.py logic
        manif_paths: List[Path] = []
        if cfg.demo_manifest:
            mp = Path(cfg.demo_manifest)
            if not mp.is_absolute():
                mp = manif_root / mp
            manif_paths.append(mp)
        else:
            for name in DEMO_MANIFEST_PRIORITY:
                manif_paths.append(manif_root / name)

        def _load_manifest(path: Path) -> List[Dict[str, Any]]:
            if not path.exists():
                return []
            try:
                with path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
            records: List[Dict[str, Any]] = []
            try:
                with path.open('r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(obj, dict):
                            records.append(obj)
            except Exception:
                pass
            return records

        manifest_records: List[Dict[str, Any]] = []
        manifest_used: Optional[Path] = None
        for candidate in manif_paths:
            manifest_records = _load_manifest(candidate)
            if manifest_records:
                manifest_used = candidate
                break

        use_real_labels = cfg.demo_use_real_labels and bool(manifest_records)
        if use_real_labels:
            print(f"[trackB.demo] Using manifest={manifest_used} records={len(manifest_records)}")
        else:
            if cfg.demo_use_real_labels:
                print('[trackB.demo] Manifest not found or empty; falling back to synthetic samples.')
            else:
                print('[trackB.demo] demo_use_real_labels disabled; using synthetic samples.')

        backbone_demo = build_backbone(tcfg)
        transform_demo = build_transform(tcfg)

        def _pick_sample(fr_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
            uids = [d for d in fr_root.iterdir() if d.is_dir()]
            if not uids:
                return None, None
            uid_dir = random.choice(uids)
            frames = list_uid_frames(uid_dir)
            if not frames:
                return None, None
            return uid_dir, random.choice(frames[max(0, len(frames) - 10):])

        def _resolve_from_record(rec: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path]]:
            uid = rec.get('uid') or rec.get('video_uid') or rec.get('video_id')
            if uid is None:
                return None, None
            uid_dir = frames_root / str(uid)
            frame_name = rec.get('frame_name') or rec.get('image')
            frame_idx = rec.get('frame') or rec.get('frame_idx') or rec.get('frame_index')
            if frame_name:
                cand = uid_dir / Path(frame_name).name
                if cand.exists():
                    return uid_dir, cand
            if isinstance(frame_idx, int):
                cand = uid_dir / f"{frame_idx:07d}.jpg"
                if cand.exists():
                    return uid_dir, cand
            frames = list_uid_frames(uid_dir) if uid_dir.exists() else []
            if frames:
                return uid_dir, frames[-1]
            return None, None

        def _extract_candidates(rec: Dict[str, Any]) -> Tuple[List[Tuple[float, float, float, float]], List[int], List[float]]:
            boxes: List[Tuple[float, float, float, float]] = []
            labels: List[int] = []
            ttcs: List[float] = []
            candidates = rec.get('candidates') or rec.get('boxes') or rec.get('objects') or []
            for c in candidates:
                if isinstance(c, dict):
                    x1 = c.get('x1') or c.get('xmin') or c.get('left')
                    y1 = c.get('y1') or c.get('ymin') or c.get('top')
                    x2 = c.get('x2') or c.get('xmax') or c.get('right')
                    y2 = c.get('y2') or c.get('ymax') or c.get('bottom')
                    if None in (x1, y1, x2, y2):
                        continue
                    boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    cls = c.get('cls')
                    if cls is None:
                        cls = c.get('label')
                        if isinstance(cls, bool):
                            cls = 1 if cls else 0
                        elif isinstance(cls, str):
                            cls = 1 if cls.lower() in ("1", "true", "active", "pos", "positive") else 0
                    if not isinstance(cls, int):
                        cls = 0
                    labels.append(int(cls))
                    ttc_val = c.get('ttc') or c.get('time_to_contact') or 0.0
                    ttcs.append(float(ttc_val) if isinstance(ttc_val, (int, float)) else 0.0)
                elif isinstance(c, (list, tuple)) and len(c) >= 4:
                    boxes.append((float(c[0]), float(c[1]), float(c[2]), float(c[3])))
                    labels.append(0)
                    ttcs.append(0.0)
            return boxes, labels, ttcs

        steps = cfg.demo_steps
        pbar = tqdm(range(steps), desc='[TrackB demo] steps (standalone)')
        for _ in pbar:
            if use_real_labels and manifest_records:
                rec = random.choice(manifest_records)
                uid_dir, frame = _resolve_from_record(rec)
            else:
                rec = None
                uid_dir, frame = _pick_sample(frames_root)

            if uid_dir is None or frame is None:
                print('[trackB.demo] WARNING: unable to resolve frame; skipping step.')
                continue

            img_tokens, hw = image_grid_tokens(frame, backbone_demo, transform_demo, tcfg)
            window_paths = sample_window_ending_at(frame, frames_root, tcfg.time_len, tcfg.time_stride)
            vid_tokens, _ = video_grid_tokens(window_paths, backbone_demo, transform_demo, tcfg)
            if vid_tokens.numel() == 0:
                print('[trackB.demo] WARNING: empty temporal window; skipping step.')
                continue

            img_b = projector(img_tokens.to(device)).unsqueeze(0)
            vid_b = projector(vid_tokens.to(device)).unsqueeze(0)
            fused_img, fused_vid = fusion(img_b, vid_b)

            with Image.open(str(frame)) as im:
                W, H = im.size

            if use_real_labels and rec is not None:
                boxes, labels_demo, ttcs_demo = _extract_candidates(rec)
            else:
                boxes = []
                labels_demo = []
                ttcs_demo = []
                for _ in range(DEMO_SYNTHETIC_BOX_COUNT):
                    x1 = random.uniform(0, max(1.0, W * 0.7))
                    y1 = random.uniform(0, max(1.0, H * 0.7))
                    x2 = x1 + random.uniform(W * 0.1, W * 0.3)
                    y2 = y1 + random.uniform(H * 0.1, H * 0.3)
                    boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                    labels_demo.append(random.randint(0, head.config.num_classes - 1))
                    ttcs_demo.append(random.random() * DEMO_SYNTHETIC_TTC_MAX)

            if cfg.candidate_limit and len(boxes) > cfg.candidate_limit:
                keep = list(range(len(boxes)))
                random.shuffle(keep)
                keep = keep[:cfg.candidate_limit]
                boxes = [boxes[i] for i in keep]
                labels_demo = [labels_demo[i] for i in keep]
                ttcs_demo = [ttcs_demo[i] for i in keep]

            pooled_img_feats: List[torch.Tensor] = []
            pooled_vid_feats: List[torch.Tensor] = []
            for box in boxes:
                pooled_img_feats.append(roi_pool_tokens_mean(hw, fused_img[0], box, (W, H)))
                pooled_vid_feats.append(roi_pool_tokens_mean(hw, fused_vid[0], box, (W, H)))

            if not pooled_img_feats:
                print('[trackB.demo] WARNING: zero candidates after pooling; skipping step.')
                continue

            img_cand = torch.stack(pooled_img_feats, dim=0).unsqueeze(0).to(device)
            vid_cand = torch.stack(pooled_vid_feats, dim=0).unsqueeze(0).to(device)
            out = head(img_cand, vid_cand)

            B, Nc, K = out['cls_logits'].shape
            if use_real_labels and labels_demo:
                cls_target = torch.tensor(labels_demo, dtype=torch.long, device=device).view(1, Nc)
                ttc_target = torch.tensor(ttcs_demo, dtype=torch.float32, device=device).view(1, Nc, 1)
            else:
                cls_target = torch.randint(0, K, (B, Nc), device=device)
                ttc_target = torch.rand(B, Nc, 1, device=device) * DEMO_SYNTHETIC_TTC_MAX

            loss_cls = ce(out['cls_logits'].reshape(B * Nc, K), cls_target.reshape(B * Nc))
            loss_ttc = l1(out['ttc'], ttc_target)
            loss = loss_cls + TTC_LOSS_WEIGHT * loss_ttc

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls': f"{loss_cls.item():.4f}",
                'ttc': f"{loss_ttc.item():.4f}",
                'cands': len(boxes),
            })

        print('[trackB.demo] Training complete (standalone variant).')

@torch.no_grad()
def _evaluate_loader(ds_val: TrackBDataset, loader, projector, fusion, head, device: str) -> Dict[str, float] | None:
    """Compute accuracy/mAP and TTC MAE on a dataset loader."""
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    pred_ttc_s_list: List[torch.Tensor] = []
    gt_ttc_s_list: List[torch.Tensor] = []
    total_samples = 0
    total_with_boxes = 0
    for batch in loader:
        if not batch.get('valid'):
            continue
        for s in batch['samples']:
            total_samples += 1
            img_b = projector(s['img_tokens'].to(device)).unsqueeze(0)
            vid_b = projector(s['vid_tokens'].to(device)).unsqueeze(0)
            fused_img, fused_vid = fusion(img_b, vid_b)
            W, H = s['image_size']; hw = s['hw']
            pooled_img = []; pooled_vid = []
            for box in s['bboxes']:
                pooled_img.append(roi_pool_tokens_mean(hw, fused_img[0], box, (W, H)))
                pooled_vid.append(roi_pool_tokens_mean(hw, fused_vid[0], box, (W, H)))
            if not pooled_img:
                continue
            total_with_boxes += 1
            img_c = torch.stack(pooled_img, dim=0).unsqueeze(0)
            vid_c = torch.stack(pooled_vid, dim=0).unsqueeze(0)
            out = head(img_c.to(device), vid_c.to(device))
            logits = out['cls_logits'][0].cpu()
            ttc_norm = out['ttc'][0, :, 0].cpu()
            mean = ds_val.stats.ttc_mean if ds_val.normalize_ttc else 0.0
            std = ds_val.stats.ttc_std if ds_val.normalize_ttc else 1.0
            ttc_s = denorm_ttc(ttc_norm, mean, std)
            labels = s.get('is_positive', s['labels']).cpu()
            gt_ttc = s['ttc'].cpu()

            logits_list.append(logits)
            labels_list.append(labels)
            pred_ttc_s_list.append(ttc_s)
            gt_ttc_s_list.append(gt_ttc)

    if not labels_list:
        print(f"[TrackB val] No usable candidates. samples={total_samples} with_boxes={total_with_boxes}. Check your val manifest and frames_root.")
        return None
    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    pred_ttc_all = torch.cat(pred_ttc_s_list, dim=0)
    gt_ttc_all = torch.cat(gt_ttc_s_list, dim=0)
    K = logits_all.shape[1]
    acc = multiclass_accuracy(logits_all, labels_all) if K > 2 else binary_accuracy(logits_all, labels_all)
    # Class distribution
    cls_counts = {int(c.item()): int((labels_all == c).sum().item()) for c in torch.unique(labels_all)}
    num_pos = cls_counts.get(1, 0) if K == 2 else None
    if K == 2:
        ap = binary_average_precision(logits_all, labels_all)
        if num_pos == 0:
            print(f"[TrackB val] WARNING: no positive examples in validation set (class 1). AP/mAP not informative.")
            map_score = float('nan')
        else:
            map_score = ap
    else:
        ap_dict = per_class_ap(logits_all, labels_all, K)
        map_score = float(sum(ap_dict.values()) / max(1, len(ap_dict)))
    mae = ttc_mae(pred_ttc_all, gt_ttc_all)
    return {'accuracy': acc, 'mAP': map_score, 'ttc_mae_seconds': mae, 'num_candidates': int(labels_all.numel()), 'class_counts': cls_counts}


def _lr_schedule(cfg: TrainConfig, epoch: int, step_in_epoch: int, steps_per_epoch: int, opt: optim.Optimizer):
    total_steps = cfg.epochs * steps_per_epoch
    global_step = epoch * steps_per_epoch + step_in_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    if global_step < warmup_steps:
        lr = cfg.lr * (global_step / max(1, warmup_steps))
    else:
        progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
        # cosine decay
        lr = cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))
    for g in opt.param_groups:
        g['lr'] = lr
    return lr


def _print_config(cfg: TrainConfig):
    print("[TrackB] Config:", {
        'config_name': _config_name,
        'video_backbone': _video_backbone,
        'projector_in_dim': _cfg.get('model.projector.in_dim', 512),
        'mode': cfg.mode,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'lr': cfg.lr,
        'min_lr': cfg.min_lr,
        'warmup_epochs': cfg.warmup_epochs,
        'candidate_limit': cfg.candidate_limit,
        'label_smoothing': cfg.label_smoothing,
        'normalize_ttc': cfg.normalize_ttc,
        'amp': cfg.amp,
        'demo_steps': cfg.demo_steps,
        'demo_variant': cfg.demo_variant,
        'demo_use_real_labels': cfg.demo_use_real_labels,
        'demo_manifest': cfg.demo_manifest,
        'eval_every': cfg.eval_every,
        'early_stopping_patience': cfg.early_stopping_patience,
        'monitor_metric': cfg.monitor_metric,
        'save_best_checkpoint': cfg.save_best_checkpoint,
        'val_manifest': cfg.val_manifest,
    })


def main():
    torch.manual_seed(0)
    cfg = TrainConfig()  # edit TrainConfig above to change behavior
    
    # Apply CLI argument overrides
    if _cli_args.demo:
        cfg.mode = 'demo'
    if _cli_args.epochs is not None:
        cfg.epochs = _cli_args.epochs
    if _cli_args.batch_size is not None:
        cfg.batch_size = _cli_args.batch_size
    if _cli_args.lr is not None:
        cfg.lr = _cli_args.lr
    
    # Initialize run logger
    from core import RunLogger
    run_logger = RunLogger(track='trackB')
    run_logger.log_config(_config_to_dict(cfg))
    run_logger.log_start()
    
    # Determine if we're running from local_extraction or repo root
    _cwd = Path.cwd()
    if _cwd.name == "local_extraction":
        _local_extraction = _cwd
    else:
        _local_extraction = _cwd / "local_extraction"
    
    frames_root = _local_extraction / "v2" / "extracted_frames"

    trackA_runs_root = _local_extraction / "runs" / "Track_A"
    stageB_run = Path(cfg.stageB_run) if cfg.stageB_run else latest_stageB_run(trackA_runs_root)
    stageB_train_manifest = resolve_stageB_manifest(stageB_run, 'head_train')
    stageB_val_manifest = resolve_stageB_manifest(stageB_run, 'head_val')

    if cfg.train_manifest is None and stageB_train_manifest is not None:
        cfg.train_manifest = str(stageB_train_manifest)
    if cfg.demo_manifest is None and stageB_train_manifest is not None:
        cfg.demo_manifest = str(stageB_train_manifest)
    if cfg.val_manifest is None and stageB_val_manifest is not None:
        cfg.val_manifest = str(stageB_val_manifest)

    manif_root = stageB_run if stageB_run is not None else _local_extraction / "v2" / "manifests"

    if stageB_run is not None:
        print(f"[TrackB] Using TrackA StageB run: {stageB_run}")
    else:
        print(f"[TrackB] StageB run not found; falling back to manifests under {manif_root}")

    _print_config(cfg)

    train_manifest_path = Path(cfg.train_manifest) if cfg.train_manifest else stageB_train_manifest

    tcfg = TokenizerConfig()
    
    # Resolve tokens_root for pre-extracted features
    tokens_root = None
    if cfg.tokens_root:
        tokens_root = Path(cfg.tokens_root)
        if not tokens_root.is_absolute():
            tokens_root = _local_extraction / cfg.tokens_root
        if tokens_root.exists():
            print(f"[TrackB] Using pre-extracted tokens: {tokens_root}")
        else:
            print(f"[TrackB] Warning: tokens_root not found, falling back to on-the-fly extraction: {tokens_root}")
            tokens_root = None

    # Dataset (tokenizes in __getitem__, or loads pre-extracted tokens)
    ds = TrackBDataset(
        frames_root=frames_root,
        manifests_root=manif_root,
        manifest_path=train_manifest_path,
        tokenizer_cfg=tcfg,
        candidate_limit=cfg.candidate_limit,
        normalize_ttc=cfg.normalize_ttc,
        tokens_root=tokens_root,
    )

    # Models
    C = TOKEN_DIM
    # Get projector input dim from config (512 for resnet18, 768 for videomae_ego)
    projector_in_dim = _cfg.get('model.projector.in_dim', 512)
    print(f"[TrackB] Projector: {projector_in_dim} -> {C}")
    projector = nn.Linear(projector_in_dim, C)
    fusion = TrackBFusion(FusionConfig(dim=C, layers=FUSION_LAYERS))
    # Infer class vocabularies from dataset labels.
    # Next-active: keep generic cls labels (usually 0/1).
    max_label = 0
    used_noun_ids: set[int] = set()
    used_verb_ids: set[int] = set()
    used_ttc_bins: set[int] = set()
    for p in ds.parsed:
        for c in p.get('candidates', []):
            try:
                max_label = max(max_label, int(c.get('cls', 0)))
            except Exception:
                pass
            try:
                nid = int(c.get('noun_id', -1))
                if nid >= 0:
                    used_noun_ids.add(nid)
            except Exception:
                pass
            try:
                vid = int(c.get('verb_id', -1))
                if vid >= 0:
                    used_verb_ids.add(vid)
            except Exception:
                pass
            try:
                tb = int(c.get('ttc_bin', -1))
                if tb >= 0:
                    used_ttc_bins.add(tb)
            except Exception:
                pass
    num_classes = max_label + 1 if max_label >= 1 else 2  # at least binary
    # Restrict noun/verb vocabularies to IDs actually observed in manifests.
    noun_id_list = sorted(used_noun_ids) if cfg.use_multi_task_labels else []
    verb_id_list = sorted(used_verb_ids) if cfg.use_multi_task_labels else []
    noun_global_to_local = {gid: idx for idx, gid in enumerate(noun_id_list)}
    verb_global_to_local = {gid: idx for idx, gid in enumerate(verb_id_list)}
    num_noun_classes = len(noun_id_list)
    num_verb_classes = len(verb_id_list)
    num_ttc_bins = (max(used_ttc_bins) + 1) if (cfg.use_ttc_bins and used_ttc_bins) else 0
    head_cfg = HeadConfig(
        dim=C,
        num_classes=num_classes,
        hidden=HEAD_HIDDEN,
        num_noun_classes=num_noun_classes,
        num_verb_classes=num_verb_classes,
        num_ttc_bins=num_ttc_bins,
        ttc_mode="bin" if cfg.use_ttc_bins else "reg",
    )
    head = TrackBHead(head_cfg)

    device = tcfg.device
    projector.to(device); fusion.to(device); head.to(device)

    # Optimizer
    opt = optim.Adam(list(projector.parameters()) + list(fusion.parameters()) + list(head.parameters()), lr=cfg.lr)

    # Losses
    try:
        ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    except TypeError:
        ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss(reduction='none')

    # DataLoader (keep workers=0 as dataset holds model references for tokenization)
    batch_size = cfg.batch_size if cfg.mode == 'main' else cfg.batch_size
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=trackB_collate)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # Prepare validation loader if in main mode
    val_loader = None
    ds_val = None
    if cfg.mode == 'main':
        val_manifest = Path(cfg.val_manifest) if cfg.val_manifest else _find_default_val_manifest(manif_root)
        try:
            ds_val = TrackBDataset(
                frames_root=frames_root,
                manifests_root=manif_root,
                # If no explicit val manifest, force synthetic by pointing to a non-existent file
                manifest_path=(val_manifest if val_manifest is not None else (manif_root / 'val.jsonl')),
                tokenizer_cfg=tcfg,
                candidate_limit=cfg.candidate_limit,
                normalize_ttc=cfg.normalize_ttc,
                synthetic_if_empty=True,
                tokens_root=tokens_root,  # Use same pre-extracted tokens for validation
            )
            val_loader = torch.utils.data.DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=trackB_collate)
        except Exception as e:
            print('[TrackB] WARNING: failed to build val loader:', e)

    if cfg.mode == 'demo':
        _run_demo(cfg, loader, projector, fusion, head, ce, l1, opt, device, frames_root, manif_root, tcfg)
    else:
        steps_per_epoch = math.ceil(len(ds) / batch_size) if len(ds) else 1
        best_val = None
        no_improve = 0
        metric_name = cfg.monitor_metric
        greater_is_better = cfg.greater_is_better if metric_name != 'ttc_mae' else False
        overall_total = cfg.epochs * steps_per_epoch
        overall_pbar = tqdm(total=overall_total, desc='[TrackB main] overall', position=0)
        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            epoch_next = 0.0
            epoch_ttc = 0.0
            epoch_noun = 0.0
            epoch_verb = 0.0
            n_steps = 0
            epoch_pbar = tqdm(total=steps_per_epoch, desc=f"[TrackB main] epoch {epoch+1}/{cfg.epochs}", position=1, leave=False)
            for step_in_epoch, batch in enumerate(loader):
                if step_in_epoch >= steps_per_epoch:
                    break
                epoch_pbar.update(1)
                overall_pbar.update(1)
                if not batch.get('valid'):
                    continue
                samples: List[Dict[str, Any]] = batch['samples']
                img_cands: List[torch.Tensor] = []
                vid_cands: List[torch.Tensor] = []
                labels_list: List[torch.Tensor] = []
                ttc_list: List[torch.Tensor] = []
                noun_list: List[torch.Tensor] = []
                verb_list: List[torch.Tensor] = []
                ttc_bin_list: List[torch.Tensor] = []
                for s in samples:
                    img_b = projector(s['img_tokens'].to(device)).unsqueeze(0)
                    vid_b = projector(s['vid_tokens'].to(device)).unsqueeze(0)
                    fused_img, fused_vid = fusion(img_b, vid_b)
                    W, H = s['image_size']; hw = s['hw']
                    pooled_img = []
                    pooled_vid = []
                    for box in s['bboxes']:
                        pooled_img.append(roi_pool_tokens_mean(hw, fused_img[0], box, (W, H)))
                        pooled_vid.append(roi_pool_tokens_mean(hw, fused_vid[0], box, (W, H)))
                    if not pooled_img:
                        continue
                    img_cands.append(torch.stack(pooled_img, dim=0))
                    vid_cands.append(torch.stack(pooled_vid, dim=0))
                    # Use explicit next-active flag when available; fall back to legacy labels.
                    pos_labels = s.get('is_positive', s['labels']).to(device)
                    labels_list.append(pos_labels)
                    ttc_list.append(s.get('ttc_norm', s['ttc']).to(device))
                    if cfg.use_multi_task_labels:
                        # Map global noun/verb IDs to local contiguous vocab indices
                        g_nouns = s['gt_noun_id'].tolist()
                        g_verbs = s['gt_verb_id'].tolist()
                        local_nouns = [noun_global_to_local.get(int(g), -1) for g in g_nouns]
                        local_verbs = [verb_global_to_local.get(int(g), -1) for g in g_verbs]
                        noun_list.append(torch.tensor(local_nouns, dtype=torch.long, device=device))
                        verb_list.append(torch.tensor(local_verbs, dtype=torch.long, device=device))
                    if cfg.use_ttc_bins:
                        ttc_bin_list.append(s['gt_ttc_bin'].to(device))
                if not img_cands:
                    continue
                img_pad, mask = pad_and_stack(img_cands)
                vid_pad, _ = pad_and_stack(vid_cands)
                labels_pad, _ = pad_and_stack([x.unsqueeze(-1).float() for x in labels_list])
                labels_pad = labels_pad.long().squeeze(-1)
                ttc_pad, _ = pad_and_stack([x.unsqueeze(-1) for x in ttc_list])
                noun_pad = verb_pad = ttc_bin_pad = None
                if cfg.use_multi_task_labels and noun_list and verb_list:
                    noun_pad, _ = pad_and_stack([x.unsqueeze(-1).float() for x in noun_list])
                    noun_pad = noun_pad.long().squeeze(-1)
                    verb_pad, _ = pad_and_stack([x.unsqueeze(-1).float() for x in verb_list])
                    verb_pad = verb_pad.long().squeeze(-1)
                if cfg.use_ttc_bins and ttc_bin_list:
                    ttc_bin_pad, _ = pad_and_stack([x.unsqueeze(-1).float() for x in ttc_bin_list])
                    ttc_bin_pad = ttc_bin_pad.long().squeeze(-1)

                with torch.cuda.amp.autocast(enabled=cfg.amp):
                    out = head(img_pad.to(device), vid_pad.to(device))
                    B, Nc, K = out['cls_logits'].shape
                    # Next-active (binary or multi-class) loss
                    logits_flat = out['cls_logits'].reshape(B * Nc, K)
                    labels_flat = labels_pad.reshape(B * Nc)
                    loss_next = ce(logits_flat, labels_flat)
                    # TTC loss: regression or bins
                    if cfg.use_ttc_bins and ttc_bin_pad is not None and 'ttc_bin_logits' in out:
                        tb_logits = out['ttc_bin_logits']  # (B,N,num_bins)
                        tb_B, tb_N, tb_K = tb_logits.shape
                        tb_flat = tb_logits.reshape(tb_B * tb_N, tb_K)
                        tb_targets = ttc_bin_pad.reshape(tb_B * tb_N)
                        loss_ttc = ce(tb_flat, tb_targets)
                    else:
                        # Continuous regression (legacy)
                        ttc_loss_all = l1(out['ttc'], ttc_pad).squeeze(-1)
                        loss_ttc = (ttc_loss_all * mask.to(device).float()).sum() / mask.to(device).float().sum().clamp_min(1.0)
                    # Optional noun / verb losses (positive candidates only)
                    loss_noun = 0.0
                    loss_verb = 0.0
                    if cfg.use_multi_task_labels and noun_pad is not None and 'noun_logits' in out:
                        noun_logits = out['noun_logits']  # (B,N,Cn)
                        Bn, Nn, Cn = noun_logits.shape
                        noun_flat = noun_logits.reshape(Bn * Nn, Cn)
                        noun_targets = noun_pad.reshape(Bn * Nn)
                        pos_mask = (labels_flat == 1)
                        if pos_mask.any():
                            loss_noun = ce(noun_flat[pos_mask], noun_targets[pos_mask])
                    if cfg.use_multi_task_labels and verb_pad is not None and 'verb_logits' in out:
                        verb_logits = out['verb_logits']  # (B,N,Cv)
                        Bv, Nv, Cv = verb_logits.shape
                        verb_flat = verb_logits.reshape(Bv * Nv, Cv)
                        verb_targets = verb_pad.reshape(Bv * Nv)
                        pos_mask = (labels_flat == 1)
                        if pos_mask.any():
                            loss_verb = ce(verb_flat[pos_mask], verb_targets[pos_mask])
                    # Total loss with weights
                    loss = (
                        cfg.loss_w_next * loss_next
                        + cfg.loss_w_ttc * loss_ttc
                        + cfg.loss_w_noun * loss_noun
                        + cfg.loss_w_verb * loss_verb
                    )
                # Scalar views for logging
                loss_next_val = float(loss_next) if isinstance(loss_next, float) else float(loss_next.item())
                loss_ttc_val = float(loss_ttc) if isinstance(loss_ttc, float) else float(loss_ttc.item())
                loss_noun_val = float(loss_noun) if isinstance(loss_noun, float) else (float(loss_noun.item()) if hasattr(loss_noun, "item") else 0.0)
                loss_verb_val = float(loss_verb) if isinstance(loss_verb, float) else (float(loss_verb.item()) if hasattr(loss_verb, "item") else 0.0)

                lr = _lr_schedule(cfg, epoch, step_in_epoch, steps_per_epoch, opt)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                # Note: accumulated as plain floats
                epoch_loss += float(loss.item())
                epoch_next += loss_next_val
                epoch_ttc += loss_ttc_val
                epoch_noun += loss_noun_val
                epoch_verb += loss_verb_val
                n_steps += 1
                epoch_pbar.set_postfix({
                    'lr': f"{lr:.2e}",
                    'loss': f"{loss.item():.4f}",
                    'L_next': f"{loss_next_val:.4f}",
                    'L_ttc': f"{loss_ttc_val:.4f}",
                })
            epoch_pbar.close()
            if n_steps:
                print(
                    f"[TrackB main] epoch {epoch+1} "
                    f"avg_loss={epoch_loss/n_steps:.4f} "
                    f"next={epoch_next/n_steps:.4f} "
                    f"ttc={epoch_ttc/n_steps:.4f} "
                    f"noun={epoch_noun/n_steps:.4f} "
                    f"verb={epoch_verb/n_steps:.4f}"
                )
            if cfg.save_epoch_checkpoints:
                checkpoints_dir = Path("local_extraction") / "runs" / "Track_B" / "checkpoints"
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                ckpt_path_epoch = checkpoints_dir / f"trackB_epoch{epoch+1}_{ts}.pt"
                cfg_dict = _config_to_dict(cfg)
                torch.save({
                    'projector': projector.state_dict(),
                    'fusion': fusion.state_dict(),
                    'head': head.state_dict(),
                    'noun_id_list': noun_id_list,
                    'verb_id_list': verb_id_list,
                    'train_config': cfg_dict,
                }, ckpt_path_epoch)
                print(f"[TrackB main] Saved epoch checkpoint: {ckpt_path_epoch}")
            if val_loader is not None and ((epoch + 1) % max(1, cfg.eval_every) == 0):
                metrics = _evaluate_loader(ds_val, val_loader, projector, fusion, head, device)
                if metrics is None:
                    print(f"[TrackB val] epoch {epoch+1} metrics: unavailable (no candidates/labels)")
                    continue
                print(f"[TrackB val] epoch {epoch+1} metrics: {metrics}")
                cur = metrics.get(metric_name, None)
                if isinstance(cur, float) and (cur != cur):
                    print(f"[TrackB val] Metric {metric_name} is NaN; skipping early stopping/best tracking this epoch.")
                    continue
                if cur is not None:
                    improved = (best_val is None) or ((cur > best_val) if greater_is_better else (cur < best_val))
                    if improved:
                        best_val = cur
                        no_improve = 0
                        if cfg.save_best_checkpoint:
                            checkpoints_dir = Path("local_extraction") / "runs" / "Track_B" / "checkpoints"
                            checkpoints_dir.mkdir(parents=True, exist_ok=True)
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            best_path = checkpoints_dir / f"trackB_best_{metric_name}_{cur:.4f}_{ts}.pt"
                            cfg_dict = _config_to_dict(cfg)
                            payload = {
                                'projector': projector.state_dict(),
                                'fusion': fusion.state_dict(),
                                'head': head.state_dict(),
                                'noun_id_list': noun_id_list,
                                'verb_id_list': verb_id_list,
                                'train_config': cfg_dict,
                            }
                            torch.save(payload, best_path)
                            torch.save(payload, checkpoints_dir / 'trackB_best.pt')
                            # Save a summary JSON alongside the best checkpoint for quick inspection
                            summary = {
                                'checkpoint': str(best_path),
                                'monitor_metric': metric_name,
                                'monitor_value': cur,
                                'timestamp': ts,
                                'metrics': metrics,
                                'train_config': cfg_dict,
                            }
                            summary_path = checkpoints_dir / f"trackB_best_{metric_name}_{cur:.4f}_{ts}_summary.json"
                            try:
                                with summary_path.open('w', encoding='utf-8') as f:
                                    json.dump(summary, f, indent=2)
                                # Also mirror the summary to match trackB_best.pt for stable pathing
                                with (checkpoints_dir / 'trackB_best_summary.json').open('w', encoding='utf-8') as f:
                                    json.dump(summary, f, indent=2)
                            except Exception as e:
                                print(f"[TrackB val] WARNING: failed to write best summary json: {e}")
                            print(f"[TrackB val] New best {metric_name}={cur:.4f}; saved: {best_path}")
                    else:
                        no_improve += 1
                        if no_improve >= max(1, cfg.early_stopping_patience):
                            print(f"[TrackB] Early stopping at epoch {epoch+1} (no improvement on {metric_name} for {no_improve} evals).")
                            break
        overall_pbar.close()
        print("[TrackB main] Training complete.")

    # Save checkpoint with timestamp under local_extraction/runs/Track_B/checkpoints
    # Final timestamped checkpoint (applies to both demo and main)
    checkpoints_dir = Path("local_extraction") / "runs" / "Track_B" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ts_final = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = checkpoints_dir / f"trackB_final_{ts_final}.pt"
    cfg_dict = _config_to_dict(cfg)
    torch.save({
        'projector': projector.state_dict(),
        'fusion': fusion.state_dict(),
        'head': head.state_dict(),
        'noun_id_list': noun_id_list,
        'verb_id_list': verb_id_list,
        'train_config': cfg_dict,
    }, ckpt_path)
    print(f"[TrackB] Saved final checkpoint: {ckpt_path}")
    
    # Log artifacts
    run_logger.log_artifacts([str(ckpt_path)])
    run_logger.log_output('final_checkpoint', str(ckpt_path))

    # Tiny eval replaced by integrated metrics summary on one batch for quick sanity
    with torch.no_grad():
        batch = next(iter(loader))
        if batch.get('valid'):
            sample = batch['samples'][0]
            img_b = projector(sample['img_tokens'].to(device)).unsqueeze(0)
            vid_b = projector(sample['vid_tokens'].to(device)).unsqueeze(0)
            fused_img, fused_vid = fusion(img_b, vid_b)
            W, H = sample['image_size']; hw = sample['hw']
            pooled_img = [] ; pooled_vid = []
            for box in sample['bboxes']:
                pooled_img.append(roi_pool_tokens_mean(hw, fused_img[0], box, (W, H)))
                pooled_vid.append(roi_pool_tokens_mean(hw, fused_vid[0], box, (W, H)))
            if pooled_img:
                img_c = torch.stack(pooled_img, dim=0).unsqueeze(0)
                vid_c = torch.stack(pooled_vid, dim=0).unsqueeze(0)
                out = head(img_c.to(device), vid_c.to(device))
                logits = out['cls_logits'][0].cpu()
                ttc_norm = out['ttc'][0, :, 0].cpu()
                mean = ds.stats.ttc_mean if ds.normalize_ttc else 0.0
                std = ds.stats.ttc_std if ds.normalize_ttc else 1.0
                ttc_s = denorm_ttc(ttc_norm, mean, std)
                labels = sample.get('is_positive', sample['labels']).cpu()
                acc_preview = multiclass_accuracy(logits, labels) if logits.shape[1] > 2 else binary_accuracy(logits, labels)
                print(f"[TrackB] preview batch accuracy={acc_preview:.4f}; first ttc_pred_s={float(ttc_s[0].item()):.2f}s")
                
                # Log final preview metrics
                run_logger.log_metric('preview_accuracy', acc_preview)
    
    # Finalize run log
    run_logger.log_end(success=True)
    run_logger.print_summary()


if __name__ == "__main__":
    main()
