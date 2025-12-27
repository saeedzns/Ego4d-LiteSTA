#!/usr/bin/env python3
"""
Track B Evaluation

Loads a saved checkpoint, runs the validation manifest, computes metrics (accuracy, mAP, TTC MAE),
and writes:
- metrics JSON under local_extraction/runs/Track_B/metrics
- predictions CSV and JSONL under local_extraction/runs/Track_B/predictions
- qualitative overlays (top candidates) under local_extraction/runs/Track_B/overlays/val

Configuration is loaded from configs/trackB.yaml
"""
from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import csv
import json

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ================== CLI ARGUMENT PARSING ==================
def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Track B multi-task evaluation")
    parser.add_argument(
        '--config',
        type=str,
        default='trackB',
        help='Config name to load (e.g., trackB, trackB_resnet18_baseline, trackB_videomae_ego)',
    )
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path (defaults to best/final under runs/Track_B/checkpoints)')
    parser.add_argument('--val_manifest', type=str, default=None, help='Explicit validation manifest path')
    parser.add_argument('--stageB_run', type=str, default=None, help='TrackA StageB run directory')
    parser.add_argument('--ttc_mode', type=str, default='reg', choices=['reg', 'binned'], help='TTC mode for N+δ (reg or binned)')
    parser.add_argument('--hotspot', type=str, default=None, choices=['on', 'off'], help='Enable/disable hotspot priors (overrides config)')
    parser.add_argument('--clip', type=str, default=None, choices=['on', 'off'], help='Enable/disable CLIP re-ranking (overrides config)')
    args, _unknown = parser.parse_known_args()
    return args


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
print(f"[trackB.eval] Loading config: {_config_name}")
_cfg = load_config(_config_name)

# Ensure tokenizer reads the same config
import trackB_tokenizer
trackB_tokenizer.set_tokenizer_config(_cfg)
# =========================================================

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
    binary_accuracy, multiclass_accuracy, binary_average_precision,
    per_class_ap, ttc_mae, denorm_ttc,
)


@dataclass
class EvalConfig:
    # Paths (from YAML)
    frames_root: Path = Path(_cfg.get('paths.extracted_frames', 'local_extraction/v2/extracted_frames'))
    manifests_root: Path = Path(_cfg.get('paths.manifests', 'local_extraction/v2/manifests'))
    trackA_runs_root: Path = Path(_cfg.get('paths.runs', 'local_extraction/runs')) / "Track_A"
    val_manifest: Optional[Path] = Path(_cfg.get('data.val_manifest')) if _cfg.get('data.val_manifest') else None
    checkpoint_path: Optional[Path] = Path(_cfg.get('evaluation.checkpoint')) if _cfg.get('evaluation.checkpoint') else None
    stageB_run: Optional[str] = _cfg.get('data.stageB_run', None)

    # Model/tokenizer (from YAML)
    token_dim: int = _cfg.get('model.projector.out_dim', 256)
    num_classes: int = _cfg.get('model.head.num_classes', 2)
    fusion_layers: int = _cfg.get('model.fusion.layers', 2)

    # Fusion/head wiring (from YAML; checkpoint may override for safety)
    fusion_heads: int = _cfg.get('model.fusion.heads', 8)
    fusion_dropout: float = _cfg.get('model.fusion.dropout', 0.1)
    fgtp_stride_t: int = _cfg.get('model.fusion.fgtp_stride_t', 2)
    fusion_ff_mult: int = _cfg.get('model.fusion.ff_mult', 4)
    head_dropout: float = _cfg.get('model.head.dropout', 0.1)

    # Data (from YAML)
    batch_size: int = _cfg.get('evaluation.batch_size', 8)
    candidate_limit: int = _cfg.get('training.candidate_limit', 16)
    normalize_ttc: bool = _cfg.get('training.normalize_ttc', True)

    # Outputs (from YAML)
    topk_overlay: int = _cfg.get('evaluation.topk_overlay', 3)
    save_overlays: bool = _cfg.get('evaluation.save_overlays', True)

    # Metric toggles (from YAML)
    compute_map: bool = _cfg.get('evaluation.compute_map', True)
    compute_ttc_mae: bool = _cfg.get('evaluation.compute_ttc_mae', True)
    
    # Multi-task / evaluation options
    ttc_mode: str = _cfg.get('multi_task.ttc_mode', 'reg')
    iou_thresh: float = _cfg.get('stage_b.iou_thresh', 0.5) if _cfg.get('stage_b.iou_thresh') else 0.5
    
    # Hotspot priors (PEAR-style) - from YAML
    use_hotspot_priors: bool = _cfg.get('evaluation.hotspot_priors.enabled', True)
    hotspot_prior_path: Optional[Path] = Path(_cfg.get('evaluation.hotspot_priors.path')) if _cfg.get('evaluation.hotspot_priors.path') else None
    hotspot_alpha: float = _cfg.get('evaluation.hotspot_priors.alpha', 0.3)
    
    # CLIP re-ranking - from YAML
    use_clip_rerank: bool = _cfg.get('evaluation.clip_rerank.enabled', True)
    clip_weight: float = _cfg.get('evaluation.clip_rerank.weight', 0.3)
    clip_model: str = _cfg.get('evaluation.clip_rerank.model', 'ViT-B/32')
    noun_label_path: Optional[Path] = Path(_cfg.get('paths.org_annotations', 'local_extraction/v2/org_annotations')) / "fho_sta_val_height-540.json"
    
    # Official Ego4D TTC threshold (seconds) for N+δ and All metrics
    # Per official benchmark: |pred_ttc - gt_ttc| <= 0.25s
    ttc_threshold: float = _cfg.get('evaluation.ttc_threshold', 0.25)


def _find_default_val_manifest(root: Path) -> Optional[Path]:
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


def _find_latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    if not checkpoints_dir.exists():
        return None
    cands = sorted(checkpoints_dir.glob('trackB_final_*.pt'))
    return cands[-1] if cands else None


def _load_models(cfg: EvalConfig, device: str, checkpoint: Path):
    ckpt = torch.load(str(checkpoint), map_location=device)
    h_state = ckpt.get('head', {})
    # Infer class counts from checkpoint head weights
    h_w = h_state.get('cls_head.weight', None)
    if h_w is None:
        num_classes = cfg.num_classes
    else:
        num_classes = h_w.shape[0]
    noun_w = h_state.get('noun_head.weight', None)
    verb_w = h_state.get('verb_head.weight', None)
    ttc_bin_w = h_state.get('ttc_bin_head.weight', None)
    num_noun_classes = noun_w.shape[0] if noun_w is not None else 0
    num_verb_classes = verb_w.shape[0] if verb_w is not None else 0
    num_ttc_bins = ttc_bin_w.shape[0] if ttc_bin_w is not None else 0
    
    # Infer projector input dimension from checkpoint weights
    proj_state = ckpt.get('projector', {})
    proj_weight = proj_state.get('weight', None)
    if proj_weight is not None:
        projector_in_dim = proj_weight.shape[1]  # (out_dim, in_dim)
    else:
        projector_in_dim = 512  # default for ResNet18
    
    # Prefer architecture params stored in checkpoint (prevents silent mismatch at eval)
    train_cfg = ckpt.get('train_config') or {}
    fusion_heads = int(train_cfg.get('fusion_heads', cfg.fusion_heads))
    fusion_dropout = float(train_cfg.get('fusion_dropout', cfg.fusion_dropout))
    fgtp_stride_t = int(train_cfg.get('fgtp_stride_t', cfg.fgtp_stride_t))
    fusion_ff_mult = int(train_cfg.get('fusion_ff_mult', cfg.fusion_ff_mult))
    head_dropout = float(train_cfg.get('head_dropout', cfg.head_dropout))

    projector = torch.nn.Linear(projector_in_dim, cfg.token_dim)
    fusion = TrackBFusion(
        FusionConfig(
            dim=cfg.token_dim,
            heads=fusion_heads,
            layers=cfg.fusion_layers,
            dropout=fusion_dropout,
            fgtp_stride_t=fgtp_stride_t,
            ff_mult=fusion_ff_mult,
        )
    )

    # Infer hidden dim from checkpoint weights when available (prevents load_state_dict mismatch)
    hidden_dim = 256
    if h_w is not None and hasattr(h_w, 'shape') and len(h_w.shape) == 2:
        hidden_dim = int(h_w.shape[1])
    head_cfg = HeadConfig(
        dim=cfg.token_dim,
        num_classes=num_classes,
        hidden=hidden_dim,
        dropout=head_dropout,
        num_noun_classes=num_noun_classes,
        num_verb_classes=num_verb_classes,
        num_ttc_bins=num_ttc_bins,
        ttc_mode="bin" if num_ttc_bins > 0 else "reg",
    )
    head = TrackBHead(head_cfg)
    projector.load_state_dict(ckpt['projector'])
    fusion.load_state_dict(ckpt['fusion'])
    head.load_state_dict(ckpt['head'])
    projector.to(device).eval(); fusion.to(device).eval(); head.to(device).eval()
    id_maps = {
        'noun_id_list': ckpt.get('noun_id_list'),
        'verb_id_list': ckpt.get('verb_id_list'),
        'train_config': ckpt.get('train_config'),
    }
    return projector, fusion, head, id_maps


def _ensure_dirs() -> Dict[str, Path]:
    base = Path('local_extraction') / 'runs' / 'Track_B'
    out = {
        'metrics': base / 'metrics',
        'pred_csv': base / 'predictions',
        'pred_json': base / 'predictions',
        'overlays': base / 'overlays' / 'val',
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def _serialize_eval_config(cfg: "EvalConfig") -> Dict[str, Any]:
    def _norm(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        return v
    return {k: _norm(v) for k, v in asdict(cfg).items()}


def _draw_overlay(img_path: Path, boxes: List[Tuple[float,float,float,float]], labels: List[int], probs: List[float], ttc_pred_s: List[float], ttc_gt_s: List[Optional[float]], save_path: Path):
    im = Image.open(str(img_path)).convert('RGB')
    draw = ImageDraw.Draw(im)
    W, H = im.size
    colors = [(46,204,113), (231,76,60), (41,128,185), (241,196,15)]
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = box
        color = colors[labels[i] % len(colors)] if labels else (46,204,113)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        txt = f"p={probs[i]:.2f} ttc={ttc_pred_s[i]:.2f}s"
        if ttc_gt_s[i] is not None:
            txt += f" | gt={ttc_gt_s[i]:.2f}s"
        draw.text((max(0,x1), max(0,y1-12)), txt, fill=color)
    im.save(str(save_path))


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


def _load_hotspot_priors(path: Optional[Path]) -> Tuple[Dict[Tuple[int, int], float], float]:
    """
    Load hotspot priors from JSON.

    Expected format:
      {
        "pairs": {
          "noun_id,verb_id": score,
          ...
        },
        "default": 0.0
      }
    """
    if path is None:
        return {}, 0.0
    p = Path(path)
    if not p.exists():
        print(f"[trackB.eval] hotspot prior file not found: {p}")
        return {}, 0.0
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[trackB.eval] failed to load hotspot priors from {p}: {e}")
        return {}, 0.0
    pairs_raw = obj.get("pairs", {})
    default = float(obj.get("default", 0.0))
    table: Dict[Tuple[int, int], float] = {}
    for k, v in pairs_raw.items():
        try:
            ks = str(k).split(",")
            if len(ks) != 2:
                continue
            nid = int(ks[0].strip())
            vid = int(ks[1].strip())
            table[(nid, vid)] = float(v)
        except Exception:
            continue
    return table, default


def _load_noun_labels(path: Optional[Path]) -> Optional[list[str]]:
    """
    Load noun labels for CLIP prompts from STA height-540 JSONs.

    Expects noun_categories: [{id, name}, ...] in the JSON.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[trackB.eval] noun label file not found: {p}")
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("noun_categories"), list):
            labels = []
            for item in obj.get("noun_categories") or []:
                if "id" not in item:
                    continue
                idx = int(item["id"])
                while len(labels) <= idx:
                    labels.append("")
                labels[idx] = str(item.get("name", ""))
            return labels
        # fallback: if a list is provided
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception as e:
        print(f"[trackB.eval] failed to load noun labels from {p}: {e}")
        return None


def _maybe_load_clip(cfg: EvalConfig, device: str, noun_id_list: list[int]) -> Tuple[Optional[Any], Optional[Any], Dict[int, torch.Tensor]]:
    """
    Optionally load CLIP model and precompute noun text embeddings for observed noun IDs.

    Returns:
      (clip_model, preprocess_fn, {global_noun_id: text_embedding})
    """
    if not cfg.use_clip_rerank:
        return None, None, {}
    try:
        import clip  # type: ignore
    except Exception as e:
        print(f"[trackB.eval] CLIP not available ({e}); disabling clip re-ranking.")
        return None, None, {}
    model, preprocess = clip.load(cfg.clip_model, device=device)
    model.eval()
    noun_labels = _load_noun_labels(cfg.noun_label_path)
    text_embeds: Dict[int, torch.Tensor] = {}
    if not noun_id_list:
        return model, preprocess, text_embeds
    with torch.no_grad():
        for gid in noun_id_list:
            if noun_labels and 0 <= gid < len(noun_labels):
                label = noun_labels[gid]
            else:
                label = f"noun {gid}"
            prompt = f"a photo of {label}"
            tokens = clip.tokenize([prompt]).to(device)
            txt = model.encode_text(tokens)[0]
            txt = txt / txt.norm(p=2)
            text_embeds[gid] = txt
    return model, preprocess, text_embeds


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    device = TokenizerConfig().device
    out_dirs = _ensure_dirs()

    stageB_run = Path(cfg.stageB_run) if cfg.stageB_run else latest_stageB_run(cfg.trackA_runs_root)
    manifests_root = stageB_run if stageB_run is not None else cfg.manifests_root

    if stageB_run is not None:
        print(f"[trackB.eval] Using TrackA StageB run: {stageB_run}")
    else:
        print(f"[trackB.eval] StageB run not found; using manifests under {manifests_root}")

    val_manifest = cfg.val_manifest
    if val_manifest is None:
        val_manifest = resolve_stageB_manifest(stageB_run, 'head_val') if stageB_run is not None else None
    if val_manifest is None:
        val_manifest = _find_default_val_manifest(manifests_root)
    if isinstance(val_manifest, str):
        val_manifest = Path(val_manifest)
    if val_manifest is None or not val_manifest.exists():
        print("[trackB.eval] WARNING: No validation manifest found; falling back to synthetic dataset.")
        val_manifest = manifests_root / 'val.jsonl'

    checkpoints_dir = Path('local_extraction') / 'runs' / 'Track_B' / 'checkpoints'
    ckpt_path = cfg.checkpoint_path or _find_latest_checkpoint(checkpoints_dir)
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found under runs/Track_B/checkpoints. Train first.")

    # Load checkpoint to get train_config (for tokens_root, video_backbone, etc.)
    ckpt_for_config = torch.load(str(ckpt_path), map_location='cpu')
    train_config_from_ckpt = ckpt_for_config.get('train_config', {})
    tokens_root_from_ckpt = train_config_from_ckpt.get('tokens_root')
    video_backbone_from_ckpt = train_config_from_ckpt.get('video_backbone', 'resnet18')
    
    # Build tokenizer config - use checkpoint's tokens_root if available
    tok_cfg = TokenizerConfig()
    
    # Resolve tokens_root: training scripts save relative to local_extraction,
    # but eval may run from the project root, so prepend local_extraction if needed
    tokens_root_resolved = None
    if tokens_root_from_ckpt:
        tokens_root_resolved = Path('local_extraction') / tokens_root_from_ckpt
        if not tokens_root_resolved.exists():
            # Try as-is (may already be full path or run from local_extraction)
            tokens_root_resolved = Path(tokens_root_from_ckpt)
        tok_cfg.tokens_root = str(tokens_root_resolved)
        print(f"[trackB.eval] Using tokens_root from checkpoint: {tokens_root_from_ckpt} -> resolved: {tokens_root_resolved}")
    if video_backbone_from_ckpt:
        tok_cfg.video_backbone = video_backbone_from_ckpt
        print(f"[trackB.eval] Using video_backbone from checkpoint: {video_backbone_from_ckpt}")

    # Build dataset/loader
    ds_val = TrackBDataset(
        frames_root=cfg.frames_root,
        manifests_root=manifests_root,
        manifest_path=val_manifest,
        tokenizer_cfg=tok_cfg,
        candidate_limit=cfg.candidate_limit,
        normalize_ttc=cfg.normalize_ttc,
        synthetic_if_empty=True,
        tokens_root=tokens_root_resolved,
    )
    print(f"[trackB.eval] val_manifest={val_manifest} records={len(ds_val.records)} parsed={len(ds_val.parsed)} frames_root={cfg.frames_root}")
    # Quick debug: count parsed candidates and missing frame paths
    missing_paths = 0
    empty_cand = 0
    for p in ds_val.parsed[:50]:  # limit to first 50 for speed
        if not p['candidates']:
            empty_cand += 1
        # Attempt resolve to detect path issues
        uid = p['uid']
        rec = p
        # mimic internal logic
        frame_path = ds_val._resolve_frame_path(uid, rec.get('frame_idx'), rec.get('frame_name'), rec.get('frame_abs'))
        if frame_path is None:
            missing_paths += 1
    if empty_cand or missing_paths:
        print(f"[trackB.eval][debug] first50 empty_cand={empty_cand} missing_paths={missing_paths}")
        # UID coverage check
        uids = [p['uid'] for p in ds_val.parsed[:500]]  # sample
        uids_unique = sorted(set(uids))
        missing_uid_dirs = [u for u in uids_unique if not (cfg.frames_root / u).exists()]
        if missing_uid_dirs:
            print(f"[trackB.eval][debug] frames_root missing UID dirs: {len(missing_uid_dirs)} (showing up to 10): {missing_uid_dirs[:10]}")
            cache_dir = Path('local_extraction') / 'runs' / 'Track_B' / 'cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            missing_file = cache_dir / 'missing_val_uids.txt'
            with missing_file.open('w', encoding='utf-8') as f:
                for u in missing_uid_dirs:
                    f.write(u + '\n')
            print(f"[trackB.eval][debug] wrote missing UID list: {missing_file}")
    loader = torch.utils.data.DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=trackB_collate)

    # Load models
    projector, fusion, head, id_maps = _load_models(cfg, device, ckpt_path)
    noun_id_list = id_maps.get('noun_id_list') or []
    verb_id_list = id_maps.get('verb_id_list') or []
    train_config_from_ckpt = id_maps.get('train_config')
    # Load hotspot priors and CLIP (optional)
    # hotspot_default is read from the hotspot JSON ("default" field).
    # Resolve hotspot path - use absolute path to avoid CWD issues
    hotspot_path = None
    if cfg.use_hotspot_priors and cfg.hotspot_prior_path:
        hotspot_path = Path(cfg.hotspot_prior_path)
        if not hotspot_path.exists():
            # Try as absolute path under project root
            _project_root = Path(__file__).resolve().parent.parent.parent  # Ego4d-LiteSTA
            abs_path = _project_root / cfg.hotspot_prior_path
            if abs_path.exists():
                hotspot_path = abs_path
                print(f"[trackB.eval] Resolved hotspot path: {hotspot_path}")
            else:
                # Try under local_extraction
                alt_path = _project_root / 'local_extraction' / 'v2' / hotspot_path.name
                if alt_path.exists():
                    hotspot_path = alt_path
                    print(f"[trackB.eval] Resolved hotspot path: {hotspot_path}")
    hotspot_table, hotspot_default = _load_hotspot_priors(hotspot_path)
    clip_model, clip_preprocess, clip_noun_text = _maybe_load_clip(cfg, device, noun_id_list)

    # Accumulators (candidate-level)
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_pred_ttc_s: List[torch.Tensor] = []
    all_gt_ttc_s: List[torch.Tensor] = []
    pred_rows: List[Dict[str, Any]] = []
    # Accumulators (frame-level N / N+V / N+δ / All)
    n_total_N = n_correct_N = 0
    n_total_NV = n_correct_NV = 0
    n_total_Nd = n_correct_Nd = 0
    n_total_All = n_correct_All = 0
    # Top-5 frame-level hit-rate counters
    n_total_N_top5 = n_correct_N_top5 = 0
    n_total_NV_top5 = n_correct_NV_top5 = 0
    n_total_Nd_top5 = n_correct_Nd_top5 = 0
    n_total_All_top5 = n_correct_All_top5 = 0
    n_total_All = n_correct_All = 0
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

    # Iterate
    total_samples = 0
    total_with_boxes = 0
    overall_pbar = tqdm(total=len(ds_val.parsed), desc='[TrackB eval] samples', position=0)
    with torch.no_grad():
        for batch in loader:
            if not batch.get('valid'):
                continue
            batch_samples = batch['samples']
            batch_pbar = tqdm(total=len(batch_samples), desc='[TrackB eval] batch', position=1, leave=False)
            for s in batch_samples:
                batch_pbar.update(1)
                overall_pbar.update(1)
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
                logits = out['cls_logits'][0].cpu()  # (Nc,K)
                ttc_norm = out['ttc'][0, :, 0].cpu()  # (Nc,)
                mean = ds_val.stats.ttc_mean if ds_val.normalize_ttc else 0.0
                std = ds_val.stats.ttc_std if ds_val.normalize_ttc else 1.0
                ttc_s = denorm_ttc(ttc_norm, mean, std)
                labels = s.get('is_positive', s['labels']).cpu()
                gt_ttc = s['ttc'].cpu()
                has_noun = 'noun_logits' in out
                has_verb = 'verb_logits' in out
                has_ttc_bin_head = 'ttc_bin_logits' in out
                noun_logits_full = out['noun_logits'][0].cpu() if has_noun else None
                verb_logits_full = out['verb_logits'][0].cpu() if has_verb else None
                ttc_bin_logits_full = out['ttc_bin_logits'][0].cpu() if has_ttc_bin_head else None
                gt_noun_ids = s.get('gt_noun_id')
                gt_verb_ids = s.get('gt_verb_id')
                gt_ttc_bins = s.get('gt_ttc_bin')

                all_logits.append(logits)
                all_labels.append(labels)
                all_pred_ttc_s.append(ttc_s)
                all_gt_ttc_s.append(gt_ttc)

                # Predictions rows
                probs = F.softmax(logits, dim=1)  # (Nc,K)
                topk = min(cfg.topk_overlay, logits.shape[0])
                pred_label = torch.argmax(probs, dim=1)
                # Base next-active scores per candidate
                base_scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                
                # Find GT positive box for this frame (for error analysis)
                is_pos = s.get('is_positive', labels).cpu()
                pos_indices = torch.nonzero(is_pos == 1, as_tuple=True)[0]
                gt_box_coords = None
                if pos_indices.numel() > 0:
                    gt_idx = int(pos_indices[0].item())
                    gt_box_coords = s['bboxes'][gt_idx]
                
                final_scores: List[float] = []
                for i, box in enumerate(s['bboxes']):
                    # Candidate-level predictions for error analysis
                    pred_noun_local = None
                    pred_noun_global = None
                    pred_verb_local = None
                    pred_verb_global = None
                    pred_ttc_bin = None
                    gt_noun = int(gt_noun_ids[i].item()) if gt_noun_ids is not None and i < len(gt_noun_ids) else None
                    gt_verb = int(gt_verb_ids[i].item()) if gt_verb_ids is not None and i < len(gt_verb_ids) else None
                    if has_noun and noun_logits_full is not None:
                        pred_noun_local = int(torch.argmax(noun_logits_full[i]).item())
                        if noun_id_list and 0 <= pred_noun_local < len(noun_id_list):
                            pred_noun_global = int(noun_id_list[pred_noun_local])
                    if has_verb and verb_logits_full is not None:
                        pred_verb_local = int(torch.argmax(verb_logits_full[i]).item())
                        if verb_id_list and 0 <= pred_verb_local < len(verb_id_list):
                            pred_verb_global = int(verb_id_list[pred_verb_local])
                    # Predicted TTC bin for CSV (independent of evaluation mode)
                    if has_ttc_bin_head and ttc_bin_logits_full is not None:
                        pred_ttc_bin = int(torch.argmax(ttc_bin_logits_full[i]).item())
                    else:
                        pred_ttc_bin = ttc_to_bin(float(ttc_s[i].item()))
                    gt_ttc_bin = None
                    if gt_ttc_bins is not None and i < len(gt_ttc_bins):
                        gt_ttc_bin = int(gt_ttc_bins[i].item())
                    else:
                        gt_ttc_bin = ttc_to_bin(float(gt_ttc[i].item())) if i < len(gt_ttc) else None
                    ttc_pred_val = float(ttc_s[i].item())
                    ttc_gt_val = float(gt_ttc[i].item()) if i < len(gt_ttc) else None
                    ttc_error = None
                    if ttc_gt_val is not None:
                        ttc_error = ttc_pred_val - ttc_gt_val
                    # Combine scores: base → hotspot → CLIP
                    base_pos = float(base_scores[i].item())
                    score = base_pos
                    # Hotspot prior based on predicted noun/verb (global IDs)
                    score_hotspot = None
                    if cfg.use_hotspot_priors and pred_noun_global is not None and pred_verb_global is not None and hotspot_table:
                        # If (noun, verb) is unseen in the hotspot table, fall back to JSON-level "default"
                        score_hotspot = hotspot_table.get((pred_noun_global, pred_verb_global), hotspot_default)
                        score = (1.0 - cfg.hotspot_alpha) * score + cfg.hotspot_alpha * float(score_hotspot)
                    # CLIP noun re-ranking
                    score_clip = None
                    if cfg.use_clip_rerank and clip_model is not None and clip_preprocess is not None and clip_noun_text and pred_noun_global is not None and pred_noun_global in clip_noun_text:
                        try:
                            img_path = Path(s['frame_path'])
                            img = Image.open(str(img_path)).convert("RGB")
                            crop = img.crop((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                            clip_in = clip_preprocess(crop).unsqueeze(0).to(device)
                            with torch.no_grad():
                                img_feat = clip_model.encode_image(clip_in)[0]
                                img_feat = img_feat / img_feat.norm(p=2)
                                txt_feat = clip_noun_text[pred_noun_global]
                                sim = float((img_feat @ txt_feat).item())  # cosine similarity
                            # Map similarity [-1,1] → [0,1]
                            score_clip = 0.5 * (sim + 1.0)
                            score = (1.0 - cfg.clip_weight) * score + cfg.clip_weight * score_clip
                        except Exception as e:
                            print(f"[trackB.eval] CLIP scoring failed for {img_path}: {e}")
                    final_scores.append(score)

                    row = {
                        'uid': s['uid'],
                        'frame_path': str(s['frame_path']),
                        'cand_idx': i,
                        'x1': float(box[0]), 'y1': float(box[1]), 'x2': float(box[2]), 'y2': float(box[3]),
                        'gt_x1': float(gt_box_coords[0]) if gt_box_coords else None,
                        'gt_y1': float(gt_box_coords[1]) if gt_box_coords else None,
                        'gt_x2': float(gt_box_coords[2]) if gt_box_coords else None,
                        'gt_y2': float(gt_box_coords[3]) if gt_box_coords else None,
                        'label': int(labels[i].item()) if i < len(labels) else None,
                        'pred_label': int(pred_label[i].item()),
                        'prob_pos': base_pos,
                        'score_final': score,
                        'score_hotspot': score_hotspot,
                        'score_clip': score_clip,
                        'ttc_pred_s': ttc_pred_val,
                        'ttc_gt_s': ttc_gt_val,
                        'ttc_error_s': ttc_error,
                        'gt_noun_id': gt_noun,
                        'pred_noun_id': pred_noun_global,
                        'gt_verb_id': gt_verb,
                        'pred_verb_id': pred_verb_global,
                        'gt_ttc_bin': gt_ttc_bin,
                        'pred_ttc_bin': pred_ttc_bin,
                    }
                    pred_rows.append(row)

                # Overlays
                if cfg.save_overlays:
                    order = torch.argsort(probs[:, 1] if probs.shape[1] > 1 else probs[:, 0], descending=True)
                    sel = order[:topk]
                    boxes = [s['bboxes'][i] for i in sel.tolist()]
                    labels_sel = [int(pred_label[i].item()) for i in sel.tolist()]
                    probs_sel = [float((probs[i, 1] if probs.shape[1] > 1 else probs[i, 0]).item()) for i in sel.tolist()]
                    ttc_pred_sel = [float(ttc_s[i].item()) for i in sel.tolist()]
                    ttc_gt_sel = [float(gt_ttc[i].item()) if i < len(gt_ttc) else None for i in sel.tolist()]
                    fname = Path(s['frame_path']).stem
                    save_p = out_dirs['overlays'] / f"{s['uid']}_{fname}.jpg"
                    try:
                        _draw_overlay(Path(s['frame_path']), boxes, labels_sel, probs_sel, ttc_pred_sel, ttc_gt_sel, save_p)
                    except Exception as e:
                        print("[trackB.eval] overlay failed:", e)

                # Frame-level + top-5 N / N+V / N+δ / All metrics (if multi-task heads present)
                has_noun = 'noun_logits' in out
                has_verb = 'verb_logits' in out
                # Need at least one positive candidate with valid noun/verb
                if has_noun or has_verb:
                    # Identify GT positive candidate (assume at most one per frame)
                    is_pos = s.get('is_positive', labels).cpu()
                    pos_indices = torch.nonzero(is_pos == 1, as_tuple=True)[0]
                    if pos_indices.numel() > 0:
                        gt_idx = int(pos_indices[0].item())
                        gt_box = tuple(s['bboxes'][gt_idx])
                        gt_noun_ids = s.get('gt_noun_id')
                        gt_verb_ids = s.get('gt_verb_id')
                        gt_ttc_bins = s.get('gt_ttc_bin')
                        gt_noun = int(gt_noun_ids[gt_idx].item()) if gt_noun_ids is not None else -1
                        gt_verb = int(gt_verb_ids[gt_idx].item()) if gt_verb_ids is not None else -1
                        if gt_ttc_bins is not None:
                            gt_bin = int(gt_ttc_bins[gt_idx].item())
                        else:
                            gt_bin = ttc_to_bin(float(gt_ttc[gt_idx].item()))

                        # Determine sorted candidate indices by final score
                        if final_scores:
                            score_tensor = torch.tensor(final_scores)
                        else:
                            # fall back to base next-active scores
                            base_scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                            score_tensor = base_scores
                        order_full = torch.argsort(score_tensor, descending=True)

                        # Top-1 index (for legacy N / Nv / N+δ / All)
                        pred_idx = int(order_full[0].item())
                        pred_box = tuple(s['bboxes'][pred_idx])
                        iou = _box_iou(gt_box, pred_box)
                        if iou >= cfg.iou_thresh:
                            # Predicted noun / verb / TTC bin (if available)
                            pred_noun = None
                            pred_verb = None
                            pred_bin = None
                            if has_noun:
                                noun_logits = out['noun_logits'][0].cpu()
                                pred_noun = int(torch.argmax(noun_logits[pred_idx]).item())
                            if has_verb:
                                verb_logits = out['verb_logits'][0].cpu()
                                pred_verb = int(torch.argmax(verb_logits[pred_idx]).item())
                            if 'ttc_bin_logits' in out and cfg.ttc_mode == 'binned':
                                tb_logits = out['ttc_bin_logits'][0].cpu()
                                pred_bin = int(torch.argmax(tb_logits[pred_idx]).item())
                            elif cfg.ttc_mode == 'reg':
                                pred_bin = ttc_to_bin(float(ttc_s[pred_idx].item()))
                            # Map local predicted IDs back to global taxonomy IDs when lists are available
                            pred_noun_global = None
                            pred_verb_global = None
                            if has_noun and noun_id_list and pred_noun is not None and 0 <= pred_noun < len(noun_id_list):
                                pred_noun_global = int(noun_id_list[pred_noun])
                            if has_verb and verb_id_list and pred_verb is not None and 0 <= pred_verb < len(verb_id_list):
                                pred_verb_global = int(verb_id_list[pred_verb])

                            # N mAP (box IoU good & noun correct)
                            if has_noun and gt_noun >= 0 and pred_noun_global is not None:
                                n_total_N += 1
                                if pred_noun_global == gt_noun:
                                    n_correct_N += 1
                                # Per-noun breakdown
                                noun_stats_total[gt_noun] = noun_stats_total.get(gt_noun, 0) + 1
                                if pred_noun_global == gt_noun:
                                    noun_stats_correct[gt_noun] = noun_stats_correct.get(gt_noun, 0) + 1
                            # N+V mAP (noun + verb)
                            if has_noun and has_verb and gt_noun >= 0 and gt_verb >= 0 and pred_noun_global is not None and pred_verb_global is not None:
                                n_total_NV += 1
                                if (pred_noun_global == gt_noun) and (pred_verb_global == gt_verb):
                                    n_correct_NV += 1
                                # Per-verb breakdown
                                verb_stats_total[gt_verb] = verb_stats_total.get(gt_verb, 0) + 1
                                if pred_verb_global == gt_verb:
                                    verb_stats_correct[gt_verb] = verb_stats_correct.get(gt_verb, 0) + 1
                            # N+δ mAP (noun + TTC within threshold) - Official Ego4D: |error| <= 0.25s
                            pred_ttc_val = float(ttc_s[pred_idx].item())
                            gt_ttc_val_for_match = float(gt_ttc[pred_idx].item()) if pred_idx < len(gt_ttc) else 0.0
                            ttc_match = abs(pred_ttc_val - gt_ttc_val_for_match) <= cfg.ttc_threshold
                            if has_noun and gt_noun >= 0 and pred_noun_global is not None:
                                n_total_Nd += 1
                                if (pred_noun_global == gt_noun) and ttc_match:
                                    n_correct_Nd += 1
                            # All: noun + verb + TTC (N+V+δ) - Official Ego4D threshold
                            if (
                                has_noun and has_verb
                                and gt_noun >= 0 and gt_verb >= 0
                                and pred_noun_global is not None and pred_verb_global is not None
                            ):
                                n_total_All += 1
                                if (
                                    pred_noun_global == gt_noun
                                    and pred_verb_global == gt_verb
                                    and ttc_match
                                ):
                                    n_correct_All += 1

                        # Top-5 metrics (hit-rate + AP) using top-5 candidates by score
                        top_inds = order_full[: min(5, order_full.numel())].tolist()
                        hit_N = hit_NV = hit_Nd = hit_All = False
                        for idx in top_inds:
                            box_i = tuple(s['bboxes'][idx])
                            iou_i = _box_iou(gt_box, box_i)
                            score_i = float(score_tensor[idx].item())
                            # Predict semantics for this candidate
                            pred_noun_i = pred_verb_i = None
                            pred_bin_i = None
                            pred_noun_global_i = pred_verb_global_i = None
                            if has_noun:
                                noun_logits = out['noun_logits'][0].cpu()
                                pred_noun_i = int(torch.argmax(noun_logits[idx]).item())
                                if noun_id_list and 0 <= pred_noun_i < len(noun_id_list):
                                    pred_noun_global_i = int(noun_id_list[pred_noun_i])
                            if has_verb:
                                verb_logits = out['verb_logits'][0].cpu()
                                pred_verb_i = int(torch.argmax(verb_logits[idx]).item())
                                if verb_id_list and 0 <= pred_verb_i < len(verb_id_list):
                                    pred_verb_global_i = int(verb_id_list[pred_verb_i])
                            if 'ttc_bin_logits' in out and cfg.ttc_mode == 'binned':
                                tb_logits = out['ttc_bin_logits'][0].cpu()
                                pred_bin_i = int(torch.argmax(tb_logits[idx]).item())
                            else:
                                pred_bin_i = ttc_to_bin(float(ttc_s[idx].item()))

                            # conditions
                            cond_N = has_noun and gt_noun >= 0 and pred_noun_global_i is not None and iou_i >= cfg.iou_thresh and pred_noun_global_i == gt_noun
                            cond_NV = (
                                cond_N
                                and has_verb
                                and gt_verb >= 0
                                and pred_verb_global_i is not None
                                and pred_verb_global_i == gt_verb
                            )
                            # Official Ego4D TTC matching: |pred - gt| <= 0.25s
                            pred_ttc_i = float(ttc_s[idx].item())
                            gt_ttc_i = float(gt_ttc[gt_idx].item()) if gt_idx < len(gt_ttc) else 0.0
                            ttc_match_i = abs(pred_ttc_i - gt_ttc_i) <= cfg.ttc_threshold
                            cond_Nd = cond_N and ttc_match_i
                            cond_All = cond_NV and ttc_match_i

                            # Append AP lists (top-5 only)
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

                            # Track per-frame hits
                            hit_N = hit_N or cond_N
                            hit_NV = hit_NV or cond_NV
                            hit_Nd = hit_Nd or cond_Nd
                            hit_All = hit_All or cond_All

                        # Frame-level top-5 hit-rate counters
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
            batch_pbar.close()
    overall_pbar.close()

    # Stack
    if not all_labels:
        print(f"[trackB.eval] No usable candidates. samples={total_samples} with_boxes={total_with_boxes}. Check your val manifest and frames_root.")
        return {}
    logits_all = torch.cat(all_logits, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    pred_ttc_all = torch.cat(all_pred_ttc_s, dim=0)
    gt_ttc_all = torch.cat(all_gt_ttc_s, dim=0)

    # Metrics
    K = logits_all.shape[1]
    acc = multiclass_accuracy(logits_all, labels_all) if K > 2 else binary_accuracy(logits_all, labels_all)
    map_score = None
    ap_per_class = None
    if cfg.compute_map:
        if K == 2:
            ap = binary_average_precision(logits_all, labels_all)
            ap_per_class = {1: ap}  # positive class AP
            map_score = ap
        else:
            ap_per_class = per_class_ap(logits_all, labels_all, K)
            map_score = float(sum(ap_per_class.values()) / max(1, len(ap_per_class)))

    mae_ttc = None
    if cfg.compute_ttc_mae:
        mae_ttc = ttc_mae(pred_ttc_all, gt_ttc_all)

    metrics: Dict[str, Any] = {
        'accuracy': acc,
        'mAP': map_score,
        'ap_per_class': ap_per_class,
        'ttc_mae_seconds': mae_ttc,
        'num_candidates': int(labels_all.numel()),
        'checkpoint': str(ckpt_path),
        'val_manifest': str(val_manifest) if val_manifest else None,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    if train_config_from_ckpt:
        metrics['train_config'] = train_config_from_ckpt
    # Frame-level N / N+V / N+δ metrics (if available)
    if cfg.compute_map:
        if n_total_N > 0:
            metrics['N_mAP'] = n_correct_N / n_total_N
        if n_total_NV > 0:
            metrics['Nv_mAP'] = n_correct_NV / n_total_NV
        if n_total_Nd > 0:
            metrics['N_delta_mAP'] = n_correct_Nd / n_total_Nd
        if n_total_All > 0:
            metrics['All_mAP'] = n_correct_All / n_total_All
    # Top-5 hit-rate metrics
    if n_total_N_top5 > 0:
        metrics['N_top5_acc'] = n_correct_N_top5 / n_total_N_top5
    if n_total_NV_top5 > 0:
        metrics['Nv_top5_acc'] = n_correct_NV_top5 / n_total_NV_top5
    if n_total_Nd_top5 > 0:
        metrics['N_delta_top5_acc'] = n_correct_Nd_top5 / n_total_Nd_top5
    if n_total_All_top5 > 0:
        metrics['All_top5_acc'] = n_correct_All_top5 / n_total_All_top5
    # Top-5 AP metrics (global over candidates)
    if cfg.compute_map:
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
    # Per-noun / per-verb accuracy breakdowns
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

    # Write metrics + summary (metrics + full eval config)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = out_dirs['metrics'] / f'metrics_val_{ts}.json'
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "metrics": metrics,
        "eval_config": _serialize_eval_config(cfg),
    }
    summary_path = out_dirs['metrics'] / f'metrics_val_{ts}_summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Write predictions
    csv_path = out_dirs['pred_csv'] / f'predictions_val_{ts}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader()
        for r in pred_rows:
            w.writerow(r)
    jsonl_path = out_dirs['pred_json'] / f'predictions_val_{ts}.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as f:
        for r in pred_rows:
            f.write(json.dumps(r) + '\n')

    print('[trackB.eval] metrics:', metrics)
    print('[trackB.eval] wrote:', metrics_path, summary_path, csv_path, jsonl_path)
    
    # Log run using RunLogger
    try:
        from core import RunLogger
        run_logger = RunLogger(track='trackB')
        # Persist full resolved YAML config alongside the run log (for provenance).
        resolved_cfg_path = run_logger.run_dir / "resolved_config.json"
        try:
            resolved_cfg_path.write_text(
                json.dumps(
                    {
                        'config_name': _config_name,
                        'config_resolved': _cfg.to_dict(),
                        'config_flat': _cfg.flat(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding='utf-8',
            )
        except Exception:
            pass

        run_logger.log_config({
            'config_name': _config_name,
            'eval_config': _serialize_eval_config(cfg),
            'config_resolved_path': str(resolved_cfg_path),
        })
        run_logger.log_metrics(metrics)
        run_logger.log_artifacts([str(metrics_path), str(summary_path), str(csv_path), str(jsonl_path), str(resolved_cfg_path)])
        run_logger.log_end(success=True)
        run_logger.print_summary()
    except Exception as e:
        print(f"[trackB.eval] WARNING: failed to write run log: {e}")
    
    return metrics


if __name__ == '__main__':
    cfg = EvalConfig()
    if _cli_args.checkpoint:
        cfg.checkpoint_path = Path(_cli_args.checkpoint)
    if _cli_args.val_manifest:
        cfg.val_manifest = Path(_cli_args.val_manifest)
    if _cli_args.stageB_run:
        cfg.stageB_run = _cli_args.stageB_run
    cfg.ttc_mode = _cli_args.ttc_mode
    
    # Override hotspot/clip from CLI
    if _cli_args.hotspot is not None:
        cfg.use_hotspot_priors = (_cli_args.hotspot == 'on')
    if _cli_args.clip is not None:
        cfg.use_clip_rerank = (_cli_args.clip == 'on')

    evaluate(cfg)
