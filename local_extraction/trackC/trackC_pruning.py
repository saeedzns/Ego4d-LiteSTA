#!/usr/bin/env python3
"""Track C — Training-Free RGTP-style Token Pruning.

This script evaluates a Track B checkpoint with an optional Run-time Guided Token Pruning (RGTP)
step that discards low-importance spatial tokens before running the head. Importance scores are
computed in a training-free manner from Frame-Guided Temporal Pooling attentions and temporal
motion energy, following the thesis Track C specification:

    rollout at t-1  → importance
    track to t      → motion-aware adjustment
    prune           → drop bottom K% tokens (configurable)

Configuration is loaded from configs/trackC.yaml (or --config argument)

Key outputs are written to ``local_extraction/runs/Track_C`` (metrics JSON + console summary).

Usage:
    # Default (uses trackC.yaml)
    python -m trackC.trackC_pruning
    
    # With specific config (for VideoMAE variant)
    python -m trackC.trackC_pruning --config trackC
    python -m trackC.trackC_pruning --video_backbone videomae_ego

Supports both ResNet18 (default) and VideoMAE backbones via --video_backbone flag.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

import torch
import torch.utils.benchmark as benchmark
from tqdm import tqdm


# ================== CLI ARGUMENT PARSING ==================
def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Track C RGTP Pruning Evaluation')
    parser.add_argument('--config', type=str, default='trackC',
                        help='Config name to load (e.g., trackC)')
    parser.add_argument('--video_backbone', type=str, default=None,
                        choices=['resnet18', 'videomae_ego'],
                        help='Override video backbone (resnet18 or videomae_ego)')
    parser.add_argument('--rgtp_rate', type=float, default=None,
                        help='Override RGTP pruning rate (0.0-0.95)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--no_pruning', action='store_true',
                        help='Disable pruning (baseline mode)')
    return parser.parse_args()

# Parse args before loading config
_cli_args = _parse_args()
# ==========================================================


# ================== YAML CONFIG LOADING ==================
_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_EXTRACTION = _THIS_DIR.parent
_REPO_ROOT = _LOCAL_EXTRACTION.parent

for p in [str(_LOCAL_EXTRACTION), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core import load_config

# Load configuration from YAML (using --config if specified)
_config_name = _cli_args.config
print(f"[TrackC] Loading config: {_config_name}")
_cfg = load_config(_config_name)

# Get video backbone (CLI override or config)
_video_backbone = _cli_args.video_backbone or _cfg.get('model.tokenizer.video_backbone', 'resnet18')
print(f"[TrackC] Video backbone: {_video_backbone}")
# =========================================================

# Ensure Track B modules are importable when running directly
TRACKB_DIR = _LOCAL_EXTRACTION / "trackB"
if str(TRACKB_DIR) not in sys.path:
    sys.path.append(str(TRACKB_DIR))

from trackB_dataset import (
    TrackBDataset,
    trackB_collate,
    latest_stageB_run,
    resolve_stageB_manifest,
    ttc_to_bin,
)
from trackB_tokenizer import TokenizerConfig, roi_pool_tokens_mean
from trackB_fusion import FusionConfig, TrackBFusion
from trackB_head import HeadConfig, TrackBHead
from trackB_metrics import (
    binary_accuracy,
    multiclass_accuracy,
    binary_average_precision,
    per_class_ap,
    ttc_mae,
    denorm_ttc,
)


# ------------------------------ Config (from YAML) ------------------------------

TOKEN_DIM = _cfg.get('model.projector.out_dim', 256)
FUSION_LAYERS = _cfg.get('model.fusion.layers', 2)
FUSION_HEADS = _cfg.get('model.fusion.heads', 8)
HEAD_HIDDEN = _cfg.get('model.head.hidden', 256)
LOGIT_FILL = _cfg.get('rgtp.logit_fill', -12.0)


@dataclass
class EvalConfig:
    frames_root: Path = Path(_cfg.get('paths.extracted_frames', 'local_extraction/v2/extracted_frames'))
    manifests_root: Path = Path(_cfg.get('paths.manifests', 'local_extraction/v2/manifests'))
    trackA_runs_root: Path = Path(_cfg.get('paths.runs', 'local_extraction/runs')) / "Track_A"
    checkpoint: Optional[Path] = Path(_cfg.get('evaluation.checkpoint')) if _cfg.get('evaluation.checkpoint') else None
    val_manifest: Optional[Path] = Path(_cfg.get('evaluation.val_manifest')) if _cfg.get('evaluation.val_manifest') else None
    stageB_run: Optional[Path] = Path(_cfg.get('evaluation.stageB_run')) if _cfg.get('evaluation.stageB_run') else None
    batch_size: int = _cfg.get('evaluation.batch_size', 1)
    candidate_limit: int = _cfg.get('training.candidate_limit', 16)
    normalize_ttc: bool = _cfg.get('training.normalize_ttc', True)
    num_workers: int = _cfg.get('runtime.num_workers', 0)


@dataclass
class RGTPConfig:
    enabled: bool = _cfg.get('rgtp.enabled', True)
    rate: float = _cfg.get('rgtp.rate', 0.1)
    min_keep: int = _cfg.get('rgtp.min_keep', 2)
    temporal_decay: float = _cfg.get('rgtp.temporal_decay', 0.6)

    def keep_count(self, num_tokens: int) -> int:
        if not self.enabled:
            return num_tokens
        rate = min(max(self.rate, 0.0), 0.95)
        keep = num_tokens - int(round(num_tokens * rate))
        return max(self.min_keep, min(num_tokens, keep))


class RuntimeConfig:
    """Runtime toggles populated from YAML config."""

    # Optional overrides; set to string paths or leave None for auto-discovery
    checkpoint: Optional[str] = _cfg.get('evaluation.checkpoint', None)
    stageB_run: Optional[str] = _cfg.get('evaluation.stageB_run', None)
    val_manifest: Optional[str] = _cfg.get('evaluation.val_manifest', None)

    # Pruning behaviour
    pruning_enabled: bool = _cfg.get('rgtp.enabled', True)
    rgtp_rate: float = _cfg.get('rgtp.rate', 0.1)
    min_keep: int = _cfg.get('rgtp.min_keep', 2)

    # Instrumentation toggles
    measure_latency: bool = _cfg.get('instrumentation.measure_latency', True)
    measure_vram: bool = _cfg.get('instrumentation.measure_vram', True)
    measure_flops: bool = _cfg.get('instrumentation.measure_flops', True)
    bench_warmup: int = _cfg.get('instrumentation.bench_warmup', 1)
    bench_iters: int = _cfg.get('instrumentation.bench_iters', 5)
    bench_samples: int = 1                # number of samples to micro-benchmark


@dataclass
class InstrumentationConfig:
    enabled: bool = True
    record_vram: bool = True
    record_flops: bool = True
    use_cuda_events: bool = True
    bench_warmup: int = 1
    bench_iters: int = 5
    bench_samples: int = 1


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def _serialize_cfg(cfg: Optional[Any]) -> Optional[Dict[str, Any]]:
    if cfg is None:
        return None
    try:
        obj = asdict(cfg)
    except Exception:
        try:
            obj = dict(cfg.__dict__)
        except Exception:
            return None
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


# ------------------------------ Helpers -----------------------------

def _find_latest_checkpoint() -> Optional[Path]:
    checkpoints_dir = Path('local_extraction') / 'runs' / 'Track_B' / 'checkpoints'
    if not checkpoints_dir.exists():
        return None
    cands = sorted(checkpoints_dir.glob('trackB_final_*.pt'))
    return cands[-1] if cands else None


def _load_models(checkpoint: Path, device: str) -> Tuple[torch.nn.Module, TrackBFusion, TrackBHead, int, Dict[str, Any]]:
    ckpt = torch.load(str(checkpoint), map_location=device)
    head_state = ckpt.get('head', {})
    cls_weight = head_state.get('cls_head.weight')
    if cls_weight is None:
        raise ValueError(f"Checkpoint {checkpoint} missing head.cls_head.weight")
    num_classes = cls_weight.shape[0]

    # Infer optional multi-task head sizes (noun/verb/TTC-bin)
    noun_w = head_state.get('noun_head.weight', None)
    verb_w = head_state.get('verb_head.weight', None)
    ttc_bin_w = head_state.get('ttc_bin_head.weight', None)
    num_noun_classes = noun_w.shape[0] if noun_w is not None else 0
    num_verb_classes = verb_w.shape[0] if verb_w is not None else 0
    num_ttc_bins = ttc_bin_w.shape[0] if ttc_bin_w is not None else 0

    # Get projector input dim based on video backbone (512 for resnet18, 768 for videomae_ego)
    projector_in_dim = _cfg.get('model.projector.in_dim', 512)
    if _video_backbone == 'videomae_ego':
        projector_in_dim = 768
    print(f"[TrackC] Projector: {projector_in_dim} -> {TOKEN_DIM}")
    
    projector = torch.nn.Linear(projector_in_dim, TOKEN_DIM).to(device)
    fusion = TrackBFusion(FusionConfig(dim=TOKEN_DIM, layers=FUSION_LAYERS, heads=FUSION_HEADS)).to(device)
    head_cfg = HeadConfig(
        dim=TOKEN_DIM,
        num_classes=num_classes,
        hidden=HEAD_HIDDEN,
        num_noun_classes=num_noun_classes,
        num_verb_classes=num_verb_classes,
        num_ttc_bins=num_ttc_bins,
        ttc_mode="bin" if num_ttc_bins > 0 else "reg",
    )
    head = TrackBHead(head_cfg).to(device)

    projector.load_state_dict(ckpt['projector'])
    fusion.load_state_dict(ckpt['fusion'])
    head.load_state_dict(head_state)

    projector.eval(); fusion.eval(); head.eval()
    extra = {
        "noun_id_list": ckpt.get("noun_id_list"),
        "verb_id_list": ckpt.get("verb_id_list"),
        "train_config": ckpt.get("train_config"),
    }
    return projector, fusion, head, num_classes, extra


def _grid_rollout(fusion: TrackBFusion, img_tokens: torch.Tensor, vid_tokens: torch.Tensor) -> torch.Tensor:
    """Return rollout attention scores for each spatial token from FGTP (shape: B x N)."""
    fgtp = fusion.fgtp
    q = fgtp.proj_q(img_tokens)
    k = fgtp.proj_k(vid_tokens)
    k = k.transpose(1, 2)  # (B, N, T, C)
    sim = torch.einsum('bnc,bntc->bnt', q, k) * fgtp.scale
    attn = sim.softmax(dim=-1)
    if attn.size(-1) >= 2:
        return attn[:, :, -2]
    return attn[:, :, -1]


def _motion_energy(vid_tokens: torch.Tensor) -> torch.Tensor:
    if vid_tokens.size(1) < 2:
        return torch.ones(vid_tokens.size(0), vid_tokens.size(2), device=vid_tokens.device)
    delta = vid_tokens[:, -1] - vid_tokens[:, -2]
    energy = delta.pow(2).mean(dim=-1)
    return energy


def _candidate_scores(
    fusion: TrackBFusion,
    img_tokens: torch.Tensor,
    vid_tokens: torch.Tensor,
    hw: Tuple[int, int],
    boxes: List[Tuple[float, float, float, float]],
    img_size: Tuple[int, int],
    rgtp_cfg: RGTPConfig,
) -> torch.Tensor:
    if not boxes:
        return torch.empty(0, device=img_tokens.device)

    rollout = _grid_rollout(fusion, img_tokens, vid_tokens)
    motion = _motion_energy(vid_tokens)

    rollout = rollout.squeeze(0)
    motion = motion.squeeze(0)

    if rollout.numel() == 0 or motion.numel() == 0:
        return torch.ones(len(boxes), device=img_tokens.device)

    rollout = rollout / rollout.max().clamp_min(1e-6)
    motion = motion / motion.max().clamp_min(1e-6)
    grid_score = rgtp_cfg.temporal_decay * rollout + (1.0 - rgtp_cfg.temporal_decay) * motion

    tokens = grid_score.unsqueeze(-1).detach().cpu()  # (N,1)
    scores: List[float] = []
    for box in boxes:
        pooled = roi_pool_tokens_mean(hw, tokens, box, img_size)
        scores.append(float(pooled.item()))
    scores_t = torch.tensor(scores, device=img_tokens.device)
    if scores_t.max() <= 0:
        return torch.ones_like(scores_t)
    return scores_t / scores_t.max().clamp_min(1e-6)


def _build_mask(scores: torch.Tensor, cfg: RGTPConfig) -> torch.Tensor:
    if scores.numel() == 0 or not cfg.enabled or cfg.rate <= 0:
        return torch.ones_like(scores, dtype=torch.bool)
    keep = cfg.keep_count(scores.numel())
    if keep >= scores.numel():
        return torch.ones_like(scores, dtype=torch.bool)
    keep = max(1, keep)
    topk = torch.topk(scores, keep, sorted=False)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[topk.indices] = True
    return mask


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def _metrics_from_logits(
    logits_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    pred_ttc_list: List[torch.Tensor],
    gt_ttc_list: List[torch.Tensor],
) -> Dict[str, Any]:
    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    pred_ttc_all = torch.cat(pred_ttc_list, dim=0)
    gt_ttc_all = torch.cat(gt_ttc_list, dim=0)
    num_classes = logits_all.shape[1]

    if num_classes > 2:
        acc = multiclass_accuracy(logits_all, labels_all)
        ap_dict = per_class_ap(logits_all, labels_all, num_classes)
        map_score = float(sum(ap_dict.values()) / max(1, len(ap_dict)))
    else:
        acc = binary_accuracy(logits_all, labels_all)
        map_score = binary_average_precision(logits_all, labels_all)
        ap_dict = {1: map_score}

    mae = ttc_mae(pred_ttc_all, gt_ttc_all)
    return {
        'accuracy': acc,
        'mAP': map_score,
        'ap_per_class': {str(k): float(v) for k, v in ap_dict.items()},
        'ttc_mae_seconds': mae,
        'num_candidates': int(labels_all.numel()),
    }


# ------------------------------ Main Eval ------------------------------

def evaluate(cfg: EvalConfig, rgtp_cfg: RGTPConfig, instr_cfg: Optional[InstrumentationConfig] = None) -> Dict[str, Any]:
    device = TokenizerConfig().device
    is_cuda = device.startswith("cuda")
    instr = instr_cfg or InstrumentationConfig()
    if instr.enabled and is_cuda and instr.record_vram:
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    lat_ms: List[float] = []
    bench_ms: List[float] = []
    bench_used = 0
    head_flops: Optional[float] = None

    stageB_run = cfg.stageB_run or latest_stageB_run(cfg.trackA_runs_root)
    manifests_root = stageB_run if stageB_run is not None else cfg.manifests_root
    if stageB_run is not None:
        print(f"[trackC] Using TrackA StageB run: {stageB_run}")
    else:
        print(f"[trackC] StageB run not found; using manifests under {manifests_root}")

    val_manifest = cfg.val_manifest
    if val_manifest is None:
        val_manifest = resolve_stageB_manifest(stageB_run, 'head_val') if stageB_run is not None else None
    if val_manifest is None:
        candidates = [
            'head_val.jsonl', 'head_val.json',
            'head_val_clip.json', 'head_val_clip.jsonl',
            'head_val_video.json', 'head_val_video.jsonl',
        ]
        for name in candidates:
            cand = manifests_root / name
            if cand.exists():
                val_manifest = cand
                break
    if val_manifest is None:
        raise FileNotFoundError("Validation manifest could not be resolved for Track C evaluation.")

    checkpoint = cfg.checkpoint or _find_latest_checkpoint()
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError("Track B checkpoint not found; train Track B first or pass --checkpoint.")

    projector, fusion, head, num_classes, extra = _load_models(checkpoint, device)
    noun_id_list = extra.get("noun_id_list") or []
    verb_id_list = extra.get("verb_id_list") or []
    train_config_from_ckpt = extra.get("train_config")
    tokenizer_cfg = TokenizerConfig()

    ds_val = TrackBDataset(
        frames_root=cfg.frames_root,
        manifests_root=manifests_root,
        manifest_path=val_manifest,
        tokenizer_cfg=tokenizer_cfg,
        candidate_limit=cfg.candidate_limit,
        normalize_ttc=cfg.normalize_ttc,
        synthetic_if_empty=False,
    )

    loader = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=trackB_collate,
    )

    print(f"[trackC] checkpoint={checkpoint}")
    print(f"[trackC] val_manifest={val_manifest} records={len(ds_val.records)} parsed={len(ds_val.parsed)}")
    print(f"[trackC] pruning enabled={rgtp_cfg.enabled} rate={rgtp_cfg.rate:.2f} min_keep={rgtp_cfg.min_keep}")

    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    pred_ttc_s_list: List[torch.Tensor] = []
    gt_ttc_s_list: List[torch.Tensor] = []

    prune_kept: List[int] = []
    prune_total: List[int] = []
    candidates_seen = 0

    total_samples = 0
    empty_candidates = 0

    # Frame-level N / N+V / N+δ / All (top-1) counters
    n_total_N = n_correct_N = 0
    n_total_NV = n_correct_NV = 0
    n_total_Nd = n_correct_Nd = 0
    n_total_All = n_correct_All = 0
    # Top-5 frame-level counters
    n_total_N_top5 = n_correct_N_top5 = 0
    n_total_NV_top5 = n_correct_NV_top5 = 0
    n_total_Nd_top5 = n_correct_Nd_top5 = 0
    n_total_All_top5 = n_correct_All_top5 = 0
    # Per-noun / per-verb breakdown (global IDs)
    noun_stats_total: Dict[int, int] = {}
    noun_stats_correct: Dict[int, int] = {}
    verb_stats_total: Dict[int, int] = {}
    verb_stats_correct: Dict[int, int] = {}
    # Top-5 candidate-level score/label lists for AP
    top5_scores_N: List[float] = []
    top5_labels_N: List[int] = []
    top5_scores_NV: List[float] = []
    top5_labels_NV: List[int] = []
    top5_scores_Nd: List[float] = []
    top5_labels_Nd: List[int] = []
    top5_scores_All: List[float] = []
    top5_labels_All: List[int] = []

    cuda_start = cuda_end = None
    if instr.enabled and instr.use_cuda_events and is_cuda:
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc='[trackC] evaluate'):
            if not batch.get('valid'):
                continue
            for sample in batch['samples']:
                total_samples += 1
                if not sample['bboxes']:
                    empty_candidates += 1
                    continue

                if instr.enabled:
                    start_t = time.perf_counter()
                    if cuda_start is not None and cuda_end is not None:
                        cuda_start.record()

                img_tok = sample['img_tokens'].to(device)
                vid_tok = sample['vid_tokens'].to(device)
                if vid_tok.numel() == 0:
                    vid_tok = img_tok.unsqueeze(0)
                img_proj = projector(img_tok).unsqueeze(0)
                vid_proj = projector(vid_tok).unsqueeze(0)

                fused_img, fused_vid = fusion(img_proj, vid_proj)

                pooled_img: List[torch.Tensor] = []
                pooled_vid: List[torch.Tensor] = []
                fused_img_cpu = fused_img[0].detach().cpu()
                fused_vid_cpu = fused_vid[0].detach().cpu()
                for box in sample['bboxes']:
                    pooled_img.append(roi_pool_tokens_mean(sample['hw'], fused_img_cpu, box, sample['image_size']).to(device))
                    pooled_vid.append(roi_pool_tokens_mean(sample['hw'], fused_vid_cpu, box, sample['image_size']).to(device))

                if not pooled_img:
                    empty_candidates += 1
                    continue

                img_c = torch.stack(pooled_img, dim=0).unsqueeze(0)
                vid_c = torch.stack(pooled_vid, dim=0).unsqueeze(0)
                candidates_seen += img_c.size(1)

                scores = _candidate_scores(
                    fusion,
                    img_proj,
                    vid_proj,
                    sample['hw'],
                    sample['bboxes'],
                    sample['image_size'],
                    rgtp_cfg,
                )
                mask = _build_mask(scores, rgtp_cfg)
                keep_idx = torch.nonzero(mask, as_tuple=True)[0]
                drop_idx = torch.nonzero(~mask, as_tuple=True)[0]

                prune_kept.append(int(keep_idx.numel()))
                prune_total.append(len(sample['bboxes']))

                if keep_idx.numel() == len(sample['bboxes']):
                    out = head(img_c.to(device), vid_c.to(device))
                else:
                    num_cand = len(sample['bboxes'])
                    partial = head(img_c[:, keep_idx].to(device), vid_c[:, keep_idx].to(device))

                    logits_full = torch.full(
                        (1, num_cand, num_classes),
                        LOGIT_FILL,
                        device=device,
                        dtype=partial['cls_logits'].dtype,
                    )
                    logits_full[:, keep_idx] = partial['cls_logits']

                    ttc_full = torch.zeros(
                        (1, num_cand, partial['ttc'].shape[-1]),
                        device=device,
                        dtype=partial['ttc'].dtype,
                    )
                    ttc_full[:, keep_idx] = partial['ttc']

                    out = {'cls_logits': logits_full, 'ttc': ttc_full}

                    if 'noun_logits' in partial:
                        noun_full = torch.full(
                            (1, num_cand, partial['noun_logits'].shape[-1]),
                            LOGIT_FILL,
                            device=device,
                            dtype=partial['noun_logits'].dtype,
                        )
                        noun_full[:, keep_idx] = partial['noun_logits']
                        out['noun_logits'] = noun_full
                    if 'verb_logits' in partial:
                        verb_full = torch.full(
                            (1, num_cand, partial['verb_logits'].shape[-1]),
                            LOGIT_FILL,
                            device=device,
                            dtype=partial['verb_logits'].dtype,
                        )
                        verb_full[:, keep_idx] = partial['verb_logits']
                        out['verb_logits'] = verb_full
                    if 'ttc_bin_logits' in partial:
                        ttc_bin_full = torch.full(
                            (1, num_cand, partial['ttc_bin_logits'].shape[-1]),
                            LOGIT_FILL,
                            device=device,
                            dtype=partial['ttc_bin_logits'].dtype,
                        )
                        ttc_bin_full[:, keep_idx] = partial['ttc_bin_logits']
                        out['ttc_bin_logits'] = ttc_bin_full

                logits = out['cls_logits'][0].cpu()
                ttc_norm = out['ttc'][0, :, 0].cpu()

                if instr.enabled:
                    if cuda_start is not None and cuda_end is not None:
                        cuda_end.record()
                        torch.cuda.synchronize()
                        elapsed_ms = float(cuda_start.elapsed_time(cuda_end))
                    else:
                        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                    lat_ms.append(elapsed_ms)
                    # Optional micro-benchmark on a limited number of samples (head forward only)
                    if bench_used < instr.bench_samples and instr.bench_iters > 0:
                        def _bench_fn():
                            _ = head(img_c.to(device), vid_c.to(device))
                            if is_cuda:
                                torch.cuda.synchronize()
                        try:
                            timer = benchmark.Timer(stmt="fn()", globals={"fn": _bench_fn})
                            for _ in range(max(0, instr.bench_warmup)):
                                timer.timeit(1)
                            res = timer.timeit(instr.bench_iters)
                            bench_ms.append(res.mean * 1000.0)
                            bench_used += 1
                        except Exception as e:
                            print(f"[trackC] benchmark failed: {e}")
                            bench_used = instr.bench_samples
                    # Optional FLOPs profile (once)
                    if head_flops is None and instr.record_flops:
                        try:
                            with torch.autograd.profiler.profile(use_cuda=is_cuda, with_flops=True) as prof:
                                _ = head(img_c.to(device), vid_c.to(device))
                                if is_cuda:
                                    torch.cuda.synchronize()
                            total_flops = 0.0
                            for evt in prof.key_averages():
                                fl = getattr(evt, "flops", None)
                                if fl:
                                    total_flops += float(fl)
                            head_flops = total_flops if total_flops > 0 else None
                        except Exception as e:
                            print(f"[trackC] FLOPs profiling failed: {e}")
                            head_flops = None

                mean = ds_val.stats.ttc_mean if ds_val.normalize_ttc else 0.0
                std = ds_val.stats.ttc_std if ds_val.normalize_ttc else 1.0
                ttc_s = denorm_ttc(ttc_norm, mean, std)

                logits_list.append(logits)
                labels = sample.get('is_positive', sample['labels']).cpu()
                labels_list.append(labels)
                pred_ttc_s_list.append(ttc_s)
                gt_ttc_s_list.append(sample['ttc'].cpu())

                # ---- Frame-level N / N+V / N+δ / All metrics ----
                has_noun = hasattr(head, "noun_head") and "gt_noun_id" in sample
                has_verb = hasattr(head, "verb_head") and "gt_verb_id" in sample
                gt_noun_ids = sample.get("gt_noun_id")
                gt_verb_ids = sample.get("gt_verb_id")
                gt_ttc_bins = sample.get("gt_ttc_bin")

                if has_noun or has_verb:
                    # Identify GT positive candidate (assume at most one per frame)
                    is_pos = sample.get("is_positive", labels).cpu()
                    pos_indices = torch.nonzero(is_pos == 1, as_tuple=True)[0]
                    if pos_indices.numel() > 0:
                        gt_idx = int(pos_indices[0].item())
                        gt_box = tuple(sample['bboxes'][gt_idx])
                        gt_noun = int(gt_noun_ids[gt_idx].item()) if gt_noun_ids is not None else -1
                        gt_verb = int(gt_verb_ids[gt_idx].item()) if gt_verb_ids is not None else -1
                        if gt_ttc_bins is not None:
                            gt_bin = int(gt_ttc_bins[gt_idx].item())
                        else:
                            gt_bin = ttc_to_bin(float(sample['ttc'][gt_idx].item()))

                        # Next-active probabilities per candidate
                        probs = torch.softmax(logits, dim=1)
                        base_scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                        score_tensor = base_scores
                        order_full = torch.argsort(score_tensor, descending=True)

                        # Top-1 index (for legacy N / N+V / N+δ / All)
                        pred_idx = int(order_full[0].item())
                        pred_box = tuple(sample['bboxes'][pred_idx])
                        iou = _box_iou(gt_box, pred_box)
                        if iou >= 0.5:
                            # Predicted noun / verb / TTC bin (if available)
                            pred_noun = pred_verb = None
                            pred_bin = None
                            if has_noun and hasattr(head, "noun_head"):
                                noun_logits = head.noun_head(logits.new_zeros(1, logits.size(0), HEAD_HIDDEN))  # placeholder, not used directly
                            # We use stored logits if available
                            if has_noun and hasattr(head, "noun_head") and "noun_logits" in out:
                                noun_logits_full = out["noun_logits"][0].cpu()
                                pred_noun = int(torch.argmax(noun_logits_full[pred_idx]).item())
                            if has_verb and hasattr(head, "verb_head") and "verb_logits" in out:
                                verb_logits_full = out["verb_logits"][0].cpu()
                                pred_verb = int(torch.argmax(verb_logits_full[pred_idx]).item())
                            if hasattr(head, "ttc_bin_head") and "ttc_bin_logits" in out:
                                tb_logits = out["ttc_bin_logits"][0].cpu()
                                pred_bin = int(torch.argmax(tb_logits[pred_idx]).item())
                            else:
                                pred_bin = ttc_to_bin(float(ttc_s[pred_idx].item()))

                            pred_noun_global = pred_verb_global = None
                            if has_noun and noun_id_list and pred_noun is not None and 0 <= pred_noun < len(noun_id_list):
                                pred_noun_global = int(noun_id_list[pred_noun])
                            if has_verb and verb_id_list and pred_verb is not None and 0 <= pred_verb < len(verb_id_list):
                                pred_verb_global = int(verb_id_list[pred_verb])

                            # N
                            if has_noun and gt_noun >= 0 and pred_noun_global is not None:
                                n_total_N += 1
                                if pred_noun_global == gt_noun:
                                    n_correct_N += 1
                                noun_stats_total[gt_noun] = noun_stats_total.get(gt_noun, 0) + 1
                                if pred_noun_global == gt_noun:
                                    noun_stats_correct[gt_noun] = noun_stats_correct.get(gt_noun, 0) + 1
                            # N+V
                            if has_noun and has_verb and gt_noun >= 0 and gt_verb >= 0 and pred_noun_global is not None and pred_verb_global is not None:
                                n_total_NV += 1
                                if pred_noun_global == gt_noun and pred_verb_global == gt_verb:
                                    n_correct_NV += 1
                                verb_stats_total[gt_verb] = verb_stats_total.get(gt_verb, 0) + 1
                                if pred_verb_global == gt_verb:
                                    verb_stats_correct[gt_verb] = verb_stats_correct.get(gt_verb, 0) + 1
                            # N+δ
                            if pred_bin is not None and has_noun and gt_noun >= 0 and pred_noun_global is not None:
                                n_total_Nd += 1
                                if pred_noun_global == gt_noun and pred_bin == gt_bin:
                                    n_correct_Nd += 1
                            # All
                            if (
                                pred_bin is not None
                                and has_noun and has_verb
                                and gt_noun >= 0 and gt_verb >= 0
                                and pred_noun_global is not None and pred_verb_global is not None
                            ):
                                n_total_All += 1
                                if (
                                    pred_noun_global == gt_noun
                                    and pred_verb_global == gt_verb
                                    and pred_bin == gt_bin
                                ):
                                    n_correct_All += 1

                        # Top-5 metrics
                        top_inds = order_full[: min(5, order_full.numel())].tolist()
                        hit_N = hit_NV = hit_Nd = hit_All = False
                        for idx in top_inds:
                            box_i = tuple(sample['bboxes'][idx])
                            iou_i = _box_iou(gt_box, box_i)
                            score_i = float(score_tensor[idx].item())
                            pred_noun_i = pred_verb_i = None
                            pred_bin_i = None
                            pred_noun_global_i = pred_verb_global_i = None
                            if has_noun and hasattr(head, "noun_head") and "noun_logits" in out:
                                noun_logits_full = out["noun_logits"][0].cpu()
                                pred_noun_i = int(torch.argmax(noun_logits_full[idx]).item())
                                if noun_id_list and 0 <= pred_noun_i < len(noun_id_list):
                                    pred_noun_global_i = int(noun_id_list[pred_noun_i])
                            if has_verb and hasattr(head, "verb_head") and "verb_logits" in out:
                                verb_logits_full = out["verb_logits"][0].cpu()
                                pred_verb_i = int(torch.argmax(verb_logits_full[idx]).item())
                                if verb_id_list and 0 <= pred_verb_i < len(verb_id_list):
                                    pred_verb_global_i = int(verb_id_list[pred_verb_i])
                            if hasattr(head, "ttc_bin_head") and "ttc_bin_logits" in out:
                                tb_logits = out["ttc_bin_logits"][0].cpu()
                                pred_bin_i = int(torch.argmax(tb_logits[idx]).item())
                            else:
                                pred_bin_i = ttc_to_bin(float(ttc_s[idx].item()))

                            cond_N = has_noun and gt_noun >= 0 and pred_noun_global_i is not None and iou_i >= 0.5 and pred_noun_global_i == gt_noun
                            cond_NV = (
                                cond_N
                                and has_verb
                                and gt_verb >= 0
                                and pred_verb_global_i is not None
                                and pred_verb_global_i == gt_verb
                            )
                            cond_Nd = cond_N and pred_bin_i is not None and pred_bin_i == gt_bin
                            cond_All = cond_NV and pred_bin_i is not None and pred_bin_i == gt_bin

                            if has_noun and gt_noun >= 0:
                                top5_scores_N.append(score_i)
                                top5_labels_N.append(1 if cond_N else 0)
                            if has_noun and has_verb and gt_noun >= 0 and gt_verb >= 0:
                                top5_scores_NV.append(score_i)
                                top5_labels_NV.append(1 if cond_NV else 0)
                            if has_noun and gt_noun >= 0:
                                top5_scores_Nd.append(score_i)
                                top5_labels_Nd.append(1 if cond_Nd else 0)
                            if has_noun and has_verb and gt_noun >= 0 and gt_verb >= 0:
                                top5_scores_All.append(score_i)
                                top5_labels_All.append(1 if cond_All else 0)

                            hit_N = hit_N or cond_N
                            hit_NV = hit_NV or cond_NV
                            hit_Nd = hit_Nd or cond_Nd
                            hit_All = hit_All or cond_All

                        if has_noun and gt_noun >= 0:
                            n_total_N_top5 += 1
                            if hit_N:
                                n_correct_N_top5 += 1
                        if has_noun and has_verb and gt_noun >= 0 and gt_verb >= 0:
                            n_total_NV_top5 += 1
                            if hit_NV:
                                n_correct_NV_top5 += 1
                        if has_noun and gt_noun >= 0:
                            n_total_Nd_top5 += 1
                            if hit_Nd:
                                n_correct_Nd_top5 += 1
                        if has_noun and has_verb and gt_noun >= 0 and gt_verb >= 0:
                            n_total_All_top5 += 1
                            if hit_All:
                                n_correct_All_top5 += 1

    if not logits_list:
        raise RuntimeError("No valid samples were processed during Track C evaluation.")

    metrics = _metrics_from_logits(logits_list, labels_list, pred_ttc_s_list, gt_ttc_s_list)

    kept = torch.tensor(prune_kept, dtype=torch.float32)
    total = torch.tensor(prune_total, dtype=torch.float32)
    prune_rate = 1.0 - (kept / total.clamp_min(1)).mean().item()
    metrics.update({
        'rgtp_enabled': rgtp_cfg.enabled,
        'rgtp_rate_request': rgtp_cfg.rate,
        'rgtp_mean_fraction_pruned': prune_rate,
        'samples_total': total_samples,
        'samples_without_candidates': empty_candidates,
        'checkpoint': str(checkpoint),
        'val_manifest': str(val_manifest),
    })
    if lat_ms:
        total_time_s = sum(lat_ms) / 1000.0 if sum(lat_ms) > 0 else 0.0
        metrics.update({
            'latency_ms_mean': statistics.mean(lat_ms),
            'latency_ms_median': statistics.median(lat_ms),
            'latency_ms_p90': _percentile(lat_ms, 90),
            'latency_ms_p95': _percentile(lat_ms, 95),
            'throughput_samples_per_s': len(lat_ms) / total_time_s if total_time_s > 0 else 0.0,
            'throughput_candidates_per_s': candidates_seen / total_time_s if total_time_s > 0 else 0.0,
        })
    if bench_ms:
        metrics['head_benchmark_ms_mean'] = statistics.mean(bench_ms)
    if head_flops is not None:
        metrics['head_flops'] = head_flops
    if instr.enabled and instr.record_vram and is_cuda:
        try:
            metrics['peak_vram_bytes'] = torch.cuda.max_memory_allocated(device)
            metrics['peak_vram_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        except Exception:
            pass
    # Frame-level N / N+V / N+δ / All (top-1)
    if n_total_N > 0:
        metrics['N_mAP'] = n_correct_N / n_total_N
    if n_total_NV > 0:
        metrics['Nv_mAP'] = n_correct_NV / n_total_NV
    if n_total_Nd > 0:
        metrics['N_delta_mAP'] = n_correct_Nd / n_total_Nd
    if n_total_All > 0:
        metrics['All_mAP'] = n_correct_All / n_total_All
    # Top-5 hit-rate
    if n_total_N_top5 > 0:
        metrics['N_top5_acc'] = n_correct_N_top5 / n_total_N_top5
    if n_total_NV_top5 > 0:
        metrics['Nv_top5_acc'] = n_correct_NV_top5 / n_total_NV_top5
    if n_total_Nd_top5 > 0:
        metrics['N_delta_top5_acc'] = n_correct_Nd_top5 / n_total_Nd_top5
    if n_total_All_top5 > 0:
        metrics['All_top5_acc'] = n_correct_All_top5 / n_total_All_top5
    # Top-5 candidate-level AP
    if top5_scores_N:
        scores = torch.tensor(top5_scores_N, dtype=torch.float32)
        labels_top = torch.tensor(top5_labels_N, dtype=torch.long)
        metrics['N_top5_mAP'] = binary_average_precision(scores, labels_top)
    if top5_scores_NV:
        scores = torch.tensor(top5_scores_NV, dtype=torch.float32)
        labels_top = torch.tensor(top5_labels_NV, dtype=torch.long)
        metrics['Nv_top5_mAP'] = binary_average_precision(scores, labels_top)
    if top5_scores_Nd:
        scores = torch.tensor(top5_scores_Nd, dtype=torch.float32)
        labels_top = torch.tensor(top5_labels_Nd, dtype=torch.long)
        metrics['N_delta_top5_mAP'] = binary_average_precision(scores, labels_top)
    if top5_scores_All:
        scores = torch.tensor(top5_scores_All, dtype=torch.float32)
        labels_top = torch.tensor(top5_labels_All, dtype=torch.long)
        metrics['All_top5_mAP'] = binary_average_precision(scores, labels_top)
    # Per-noun / per-verb breakdowns
    if noun_stats_total:
        per_noun = {}
        for nid, tot in noun_stats_total.items():
            corr = noun_stats_correct.get(nid, 0)
            per_noun[str(nid)] = {
                "total": tot,
                "correct": corr,
                "accuracy": float(corr) / float(tot) if tot > 0 else 0.0,
            }
        metrics['per_noun_stats'] = per_noun
    if verb_stats_total:
        per_verb = {}
        for vid, tot in verb_stats_total.items():
            corr = verb_stats_correct.get(vid, 0)
            per_verb[str(vid)] = {
                "total": tot,
                "correct": corr,
                "accuracy": float(corr) / float(tot) if tot > 0 else 0.0,
            }
        metrics['per_verb_stats'] = per_verb
    # Propagate train_config from Track B checkpoint if present
    if train_config_from_ckpt:
        metrics['train_config'] = train_config_from_ckpt
    return metrics


def _save_metrics(metrics: Dict[str, Any], eval_cfg: EvalConfig, rgtp_cfg: RGTPConfig, instr_cfg: Optional[InstrumentationConfig]) -> Path:
    out_dir = Path('local_extraction') / 'runs' / 'Track_C' / 'metrics'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rate = metrics.get('rgtp_rate_request', 0.0)
    out_path = out_dir / f"trackC_val_rate{int(rate * 100):02d}_{ts}.json"
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    summary = {
        "metrics": metrics,
        "eval_config": _serialize_cfg(eval_cfg),
        "rgtp_config": _serialize_cfg(rgtp_cfg),
        "instrumentation": _serialize_cfg(instr_cfg),
    }
    summary_path = out_dir / f"trackC_val_rate{int(rate * 100):02d}_{ts}_summary.json"
    try:
        with summary_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"[trackC] WARNING: failed to write summary json: {e}")
    return out_path


def main() -> None:
    runtime_cfg = RuntimeConfig()
    
    # Apply CLI argument overrides
    if _cli_args.checkpoint:
        runtime_cfg.checkpoint = _cli_args.checkpoint
    if _cli_args.rgtp_rate is not None:
        runtime_cfg.rgtp_rate = _cli_args.rgtp_rate
    if _cli_args.no_pruning:
        runtime_cfg.pruning_enabled = False
        runtime_cfg.rgtp_rate = 0.0
    
    print("[trackC] Runtime toggles:", {
        'config_name': _config_name,
        'video_backbone': _video_backbone,
        'checkpoint': runtime_cfg.checkpoint,
        'stageB_run': runtime_cfg.stageB_run,
        'val_manifest': runtime_cfg.val_manifest,
        'pruning_enabled': runtime_cfg.pruning_enabled,
        'rgtp_rate': runtime_cfg.rgtp_rate,
        'min_keep': runtime_cfg.min_keep,
        'measure_latency': runtime_cfg.measure_latency,
        'measure_vram': runtime_cfg.measure_vram,
        'measure_flops': runtime_cfg.measure_flops,
    })

    eval_cfg = EvalConfig()
    if runtime_cfg.checkpoint:
        eval_cfg.checkpoint = Path(runtime_cfg.checkpoint)
    if runtime_cfg.val_manifest:
        eval_cfg.val_manifest = Path(runtime_cfg.val_manifest)
    if runtime_cfg.stageB_run:
        eval_cfg.stageB_run = Path(runtime_cfg.stageB_run)

    rgtp_cfg = RGTPConfig(
        enabled=runtime_cfg.pruning_enabled and runtime_cfg.rgtp_rate > 0,
        rate=max(0.0, min(0.95, runtime_cfg.rgtp_rate)),
        min_keep=max(1, runtime_cfg.min_keep),
    )
    instr_cfg = InstrumentationConfig(
        enabled=runtime_cfg.measure_latency or runtime_cfg.measure_vram or runtime_cfg.measure_flops,
        record_vram=runtime_cfg.measure_vram,
        record_flops=runtime_cfg.measure_flops,
        bench_warmup=max(0, runtime_cfg.bench_warmup),
        bench_iters=max(0, runtime_cfg.bench_iters),
        bench_samples=max(0, runtime_cfg.bench_samples),
    )

    torch.set_grad_enabled(False)
    metrics = evaluate(eval_cfg, rgtp_cfg, instr_cfg)
    out_path = _save_metrics(metrics, eval_cfg, rgtp_cfg, instr_cfg)

    print("\n[trackC] Metrics:")
    for k in ['accuracy', 'mAP', 'ttc_mae_seconds', 'num_candidates', 'rgtp_mean_fraction_pruned']:
        if k in metrics:
            print(f"  {k}: {metrics[k]}")
    print(f"[trackC] metrics saved to {out_path}")
    
    # Log run using RunLogger
    try:
        from core import RunLogger
        run_dir = Path("local_extraction") / "runs" / "Track_C"
        run_logger = RunLogger(track='trackC', run_dir=run_dir)
        run_logger.log_config({
            'eval_config': asdict(eval_cfg),
            'rgtp_config': asdict(rgtp_cfg),
            'instrumentation_config': asdict(instr_cfg),
            'runtime_config': {
                'checkpoint': runtime_cfg.checkpoint,
                'pruning_enabled': runtime_cfg.pruning_enabled,
                'rgtp_rate': runtime_cfg.rgtp_rate,
            }
        })
        run_logger.log_metrics(metrics)
        run_logger.log_artifacts([str(out_path)])
        run_logger.log_end(success=True)
        run_logger.print_summary()
    except Exception as e:
        print(f"[trackC] WARNING: failed to write run log: {e}")


if __name__ == '__main__':
    main()
