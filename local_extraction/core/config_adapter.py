#!/usr/bin/env python3
"""
Config Adapter for Track Scripts

This module provides a bridge between the new YAML-based config system
and the existing toggle-based scripts. It can:

1. Load YAML configs and apply them to existing module-level toggles
2. Generate override dicts for class-based configs (TrainConfig, EvalConfig, etc.)
3. Support CLI overrides and environment variables

Usage:
    from core.config_adapter import apply_config_to_module, get_track_config

    # Apply to module-level toggles
    import trackA_stageA
    apply_config_to_module(trackA_stageA, 'trackA')

    # Get config dict for class instantiation
    train_cfg = get_track_config('trackB', section='training')
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

# Ensure parent is importable
_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_EXTRACTION = _THIS_DIR.parent
if str(_LOCAL_EXTRACTION) not in sys.path:
    sys.path.insert(0, str(_LOCAL_EXTRACTION))

from core.config_loader import load_config, Config


# =============================================================================
# Mapping: YAML keys -> Module attributes
# =============================================================================

# Track A Stage A mappings
TRACKA_STAGEA_MAPPINGS = {
    # paths
    "paths.extracted_frames": "FRAMES_ROOT",
    "paths.yolo_labels": "LABELS_ROOT",
    "version": "VERSION",

    # stage_a settings (matching trackA.yaml structure)
    "stage_a.mode": "DETECTION_MODE",
    "stage_a.k": "K",
    "stage_a.yolo.conf_thresh": "YOLO_CONF",
    "stage_a.yolo.nms_iou": "YOLO_IOU",
    "stage_a.yolo.weights": "YOLO_WEIGHTS",
    "stage_a.yolo.imgsz": "YOLO_IMGSZ",
    "stage_a.last_frame_only": "LAST_FRAME_ONLY",
    "stage_a.max_images": "MAX_IMAGES",
    "stage_a.oracle.label_space": "SPACE",
    "stage_a.output.save_per_image_csv": "SAVE_PER_IMAGE_CSV",

    # demo
    "demo.enabled": "DEMO_MODE",
    "demo.max_samples": "DEMO_N",

    # runtime
    "runtime.print_progress": "PRINT_PROGRESS",
}

# Track A Stage B mappings
TRACKA_STAGEB_MAPPINGS = {
    "version": "VERSION",
    "paths.extracted_frames": "FRAMES_ROOT",
    "paths.yolo_labels": "LABELS_ROOT",

    # stage_b settings (matching trackA.yaml structure)
    "stage_b.iou_thresh": "IOU_THRESH",
    "stage_b.keep_top_n": "KEEP_TOP_N",
    "stage_b.crop_size": "CROP_SIZE",  # list [W, H]
    "stage_b.eval_with_labels": "EVAL_WITH_LABELS",
    "stage_b.manifests.use_clip_manifest": "USE_CLIP_MANIFEST",
    "stage_b.manifests.train": "HEAD_TRAIN_MANIFEST",
    "stage_b.manifests.val": "HEAD_VAL_MANIFEST",
    "stage_b.output.write_head_train_val": "WRITE_HEAD_TRAIN_VAL",
    "stage_b.output.write_semantics_in_manifest": "WRITE_SEMANTICS_IN_MANIFEST",
    "stage_b.output.write_back_to_stagea_summary": "WRITE_BACK_TO_STAGEA_SUMMARY",

    # demo
    "demo.enabled": "DEMO_MODE",
    "demo.max_samples": "DEMO_N",
    "runtime.max_images": "MAX_IMAGES",
    "runtime.print_progress": "PRINT_PROGRESS",
}

# Track B training mappings (for TrainConfig class)
TRACKB_TRAIN_MAPPINGS = {
    "training.epochs": "epochs",
    "training.batch_size": "batch_size",
    "training.lr": "lr",
    "training.min_lr": "min_lr",
    "training.warmup_epochs": "warmup_epochs",
    "training.label_smoothing": "label_smoothing",
    "training.amp": "amp",
    "training.save_epoch_checkpoints": "save_epoch_checkpoints",
    "training.save_best_checkpoint": "save_best_checkpoint",
    "training.early_stopping.patience": "early_stopping_patience",
    "training.eval_every": "eval_every",
    "training.candidate_limit": "candidate_limit",
    "training.normalize_ttc": "normalize_ttc",

    # multi-task
    "multi_task.enabled": "use_multi_task_labels",
    "multi_task.loss_weights.next_active": "loss_w_next",
    "multi_task.loss_weights.noun": "loss_w_noun",
    "multi_task.loss_weights.verb": "loss_w_verb",
    "multi_task.loss_weights.ttc": "loss_w_ttc",

    # demo
    "demo.enabled": ("mode", lambda v: "demo" if v else "main"),
    "demo.steps": "demo_steps",
    "demo.variant": "demo_variant",
    "demo.use_real_labels": "demo_use_real_labels",

    # data
    "data.train_manifest": "train_manifest",
    "data.val_manifest": "val_manifest",
    "data.stageB_run": "stageB_run",
}

# Track B tokenizer mappings (for TokenizerConfig)
TRACKB_TOKENIZER_MAPPINGS = {
    "model.tokenizer.backbone": "backbone",
    "model.tokenizer.pretrained": "pretrained",
    "model.tokenizer.img_size": "img_size",
    "model.tokenizer.freeze_backbone": "freeze_backbone",
    "model.tokenizer.time_len": "time_len",
    "model.tokenizer.time_stride": "time_stride",
    "runtime.device": "device",
    "runtime.use_half": "use_half",
}

# Track B fusion mappings (for FusionConfig)
TRACKB_FUSION_MAPPINGS = {
    "model.fusion.dim": "dim",
    "model.fusion.layers": "layers",
    "model.fusion.heads": "heads",
    "model.fusion.dropout": "dropout",
}

# Track B head mappings (for HeadConfig)
TRACKB_HEAD_MAPPINGS = {
    "model.head.dim": "dim",
    "model.head.hidden": "hidden",
    "model.head.num_classes": "num_classes",
    "model.head.dropout": "dropout",
}

# Track B eval mappings (for EvalConfig)
TRACKB_EVAL_MAPPINGS = {
    "paths.extracted_frames": "frames_root",
    "paths.manifests": "manifests_root",
    "evaluation.batch_size": "batch_size",
    "training.candidate_limit": "candidate_limit",
    "training.normalize_ttc": "normalize_ttc",
    "model.projector.out_dim": "token_dim",
    "model.fusion.layers": "fusion_layers",
}

# Track C pruning mappings
TRACKC_MAPPINGS = {
    "rgtp.enabled": "pruning_enabled",
    "rgtp.rate": "rgtp_rate",
    "rgtp.min_keep": "min_keep",

    # instrumentation
    "instrumentation.measure_latency": "measure_latency",
    "instrumentation.measure_vram": "measure_vram",
    "instrumentation.measure_flops": "measure_flops",
}


# =============================================================================
# Core Functions
# =============================================================================

def apply_config_to_module(
    module,
    track: str,
    stage: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Apply YAML config values to module-level attributes.

    Args:
        module: The module to update (e.g., imported trackA_stageA)
        track: Track name ('trackA', 'trackB', 'trackC')
        stage: Optional stage ('stageA', 'stageB') for Track A
        overrides: Additional config overrides

    Returns:
        The loaded Config object
    """
    # Load config
    cfg = load_config(track, overrides=overrides)

    # Select mapping based on track/stage
    if track == "trackA":
        if stage == "stageB":
            mappings = TRACKA_STAGEB_MAPPINGS
        else:
            mappings = TRACKA_STAGEA_MAPPINGS
    elif track == "trackB":
        # TrackB uses class-based config, skip module-level
        return cfg
    elif track == "trackC":
        mappings = TRACKC_MAPPINGS
    else:
        mappings = {}

    # Apply mappings
    for yaml_key, module_attr in mappings.items():
        value = cfg.get(yaml_key)
        if value is None:
            continue

        if isinstance(module_attr, tuple):
            if callable(module_attr[1]):
                # Transform function
                attr_name, transform = module_attr
                value = transform(value)
                if hasattr(module, attr_name):
                    setattr(module, attr_name, value)
            else:
                # Tuple index (e.g., CROP_SIZE)
                attr_name, idx = module_attr
                if hasattr(module, attr_name):
                    current = getattr(module, attr_name)
                    if isinstance(current, (list, tuple)):
                        current = list(current)
                        current[idx] = value
                        setattr(module, attr_name, tuple(current))
        else:
            if hasattr(module, module_attr):
                current_val = getattr(module, module_attr)
                # Handle Path conversion
                if isinstance(current_val, Path):
                    value = Path(value) if isinstance(value, str) else value
                # Handle tuple conversion (e.g., CROP_SIZE = (256, 256))
                elif isinstance(current_val, tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(module, module_attr, value)

    return cfg


def get_track_config(
    track: str,
    section: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get config dict for class-based configs.

    Args:
        track: Track name
        section: Config section to return (e.g., 'training', 'model')
        overrides: Additional overrides

    Returns:
        Dict suitable for unpacking into config classes
    """
    cfg = load_config(track, overrides=overrides)

    if section:
        return cfg.get(section, {})
    return cfg.to_dict()


def create_train_config(
    base_cfg: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Create a TrainConfig instance from YAML config.

    Args:
        base_cfg: Optional pre-loaded config dict
        overrides: Additional overrides

    Returns:
        TrainConfig instance with YAML values applied
    """
    # Import TrainConfig from trackB
    try:
        from trackB_train_loader import TrainConfig
    except ImportError:
        # Fallback: return dict instead
        return get_track_config('trackB', section='training', overrides=overrides)

    if base_cfg is None:
        cfg = load_config('trackB', overrides=overrides)
        base_cfg = cfg.to_dict()

    # Create config with defaults
    train_cfg = TrainConfig()

    # Apply mappings
    for yaml_key, attr in TRACKB_TRAIN_MAPPINGS.items():
        # Navigate nested keys
        parts = yaml_key.split(".")
        value = base_cfg
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break

        if value is None:
            continue

        if isinstance(attr, tuple):
            attr_name, transform = attr
            value = transform(value)
            setattr(train_cfg, attr_name, value)
        else:
            setattr(train_cfg, attr, value)

    return train_cfg


def create_rgtp_config(
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Create an RGTPConfig instance from YAML config.

    Returns:
        RGTPConfig-compatible dict (or dataclass if importable)
    """
    cfg = load_config('trackC', overrides=overrides)

    return {
        'enabled': cfg.get('rgtp.enabled', True),
        'rate': cfg.get('rgtp.rate', 0.1),
        'min_keep': cfg.get('rgtp.min_keep', 2),
        'temporal_decay': cfg.get('rgtp.temporal_decay', 0.6),
    }


def create_eval_config(
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Create an EvalConfig instance from YAML config.

    Returns:
        EvalConfig-compatible dict
    """
    cfg = load_config('trackC', overrides=overrides)

    return {
        'frames_root': Path(cfg.get('paths.extracted_frames', 'local_extraction/v2/extracted_frames')),
        'manifests_root': Path(cfg.get('paths.manifests', 'local_extraction/v2/manifests')),
        'trackA_runs_root': Path('local_extraction/runs/Track_A'),
        'checkpoint': cfg.get('evaluation.checkpoint'),
        'val_manifest': cfg.get('evaluation.val_manifest'),
        'stageB_run': cfg.get('evaluation.stageB_run'),
        'batch_size': cfg.get('evaluation.batch_size', 1),
        'candidate_limit': cfg.get('training.candidate_limit', 16),
        'normalize_ttc': cfg.get('training.normalize_ttc', True),
        'num_workers': cfg.get('runtime.num_workers', 0),
    }


def create_fusion_config(
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Create a FusionConfig instance from YAML config.

    Returns:
        FusionConfig instance or dict
    """
    cfg = load_config('trackB', overrides=overrides)

    try:
        from trackB_fusion import FusionConfig
        return FusionConfig(
            dim=cfg.get('model.fusion.dim', 256),
            layers=cfg.get('model.fusion.layers', 2),
            heads=cfg.get('model.fusion.heads', 8),
            dropout=cfg.get('model.fusion.dropout', 0.1),
        )
    except ImportError:
        return {
            'dim': cfg.get('model.fusion.dim', 256),
            'layers': cfg.get('model.fusion.layers', 2),
            'heads': cfg.get('model.fusion.heads', 8),
            'dropout': cfg.get('model.fusion.dropout', 0.1),
        }


def create_head_config(
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Create a HeadConfig instance from YAML config.

    Returns:
        HeadConfig instance or dict
    """
    cfg = load_config('trackB', overrides=overrides)

    try:
        from trackB_head import HeadConfig
        return HeadConfig(
            dim=cfg.get('model.head.dim', 256),
            hidden=cfg.get('model.head.hidden', 256),
            num_classes=cfg.get('model.head.num_classes', 2),
            dropout=cfg.get('model.head.dropout', 0.1),
        )
    except ImportError:
        return {
            'dim': cfg.get('model.head.dim', 256),
            'hidden': cfg.get('model.head.hidden', 256),
            'num_classes': cfg.get('model.head.num_classes', 2),
            'dropout': cfg.get('model.head.dropout', 0.1),
        }


def create_tokenizer_config(
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Create a TokenizerConfig instance from YAML config.

    Returns:
        TokenizerConfig instance or dict
    """
    cfg = load_config('trackB', overrides=overrides)

    try:
        from trackB_tokenizer import TokenizerConfig
        return TokenizerConfig(
            img_size=cfg.get('model.tokenizer.img_size', 224),
            time_len=cfg.get('model.tokenizer.time_len', 8),
            time_stride=cfg.get('model.tokenizer.time_stride', 2),
        )
    except ImportError:
        return {
            'img_size': cfg.get('model.tokenizer.img_size', 224),
            'time_len': cfg.get('model.tokenizer.time_len', 8),
            'time_stride': cfg.get('model.tokenizer.time_stride', 2),
        }


def create_trackb_eval_config(
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Create an EvalConfig for Track B evaluation from YAML.

    Returns:
        EvalConfig-compatible dict for trackB_eval.py
    """
    cfg = load_config('trackB', overrides=overrides)

    return {
        'frames_root': Path(cfg.get('paths.extracted_frames', 'local_extraction/v2/extracted_frames')),
        'manifests_root': Path(cfg.get('paths.manifests', 'local_extraction/v2/manifests')),
        'batch_size': cfg.get('evaluation.batch_size', 8),
        'candidate_limit': cfg.get('training.candidate_limit', 16),
        'normalize_ttc': cfg.get('training.normalize_ttc', True),
        'token_dim': cfg.get('model.projector.out_dim', 256),
        'fusion_layers': cfg.get('model.fusion.layers', 2),
        'hotspot_priors_enabled': cfg.get('evaluation.hotspot_priors.enabled', True),
        'hotspot_priors_path': cfg.get('evaluation.hotspot_priors.path'),
        'hotspot_alpha': cfg.get('evaluation.hotspot_priors.alpha', 0.3),
        'clip_rerank_enabled': cfg.get('evaluation.clip_rerank.enabled', True),
        'clip_model': cfg.get('evaluation.clip_rerank.model', 'ViT-B/32'),
        'clip_weight': cfg.get('evaluation.clip_rerank.weight', 0.3),
    }


# =============================================================================
# CLI Integration
# =============================================================================

def main_with_config(
    track: str,
    module_main_fn,
    stage: Optional[str] = None,
):
    """
    Run a track's main() function with YAML config applied.

    Args:
        track: Track name
        module_main_fn: The module's main function to call
        stage: Optional stage for Track A
    """
    import argparse

    parser = argparse.ArgumentParser(description=f"Run {track} with YAML config")
    parser.add_argument("--config", type=str, help="Path to custom YAML config")
    parser.add_argument("--demo", action="store_true", help="Enable demo mode")
    parser.add_argument("--device", type=str, help="Device override (cpu/cuda)")
    parser.add_argument("-o", "--override", action="append", nargs=2, metavar=("KEY", "VALUE"),
                       help="Config override (e.g., -o training.lr 0.001)")

    args, remaining = parser.parse_known_args()

    # Build overrides
    overrides = {}
    if args.demo:
        overrides['demo'] = {'enabled': True}
    if args.device:
        overrides['runtime'] = {'device': args.device}
    if args.override:
        for key, val in args.override:
            # Try to parse as number/bool
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    if val.lower() in ('true', 'false'):
                        val = val.lower() == 'true'

            # Set nested key
            parts = key.split(".")
            d = overrides
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = val

    # Load and apply config
    cfg = load_config(track, overrides=overrides)

    # Run main
    return module_main_fn(config=cfg)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Demo: show how configs map to module attributes
    print("Config Adapter for Track Scripts")
    print("=" * 50)

    cfg = load_config('trackA')
    print("\nTrack A Config (stage_a section):")
    print(f"  detector: {cfg.get('stage_a.detector')}")
    print(f"  k: {cfg.get('stage_a.k')}")
    print(f"  demo.enabled: {cfg.get('demo.enabled')}")

    cfg = load_config('trackB')
    print("\nTrack B Config (training section):")
    print(f"  epochs: {cfg.get('training.epochs')}")
    print(f"  batch_size: {cfg.get('training.batch_size')}")
    print(f"  lr: {cfg.get('training.lr')}")

    cfg = load_config('trackC')
    print("\nTrack C Config (rgtp section):")
    print(f"  enabled: {cfg.get('rgtp.enabled')}")
    print(f"  rate: {cfg.get('rgtp.rate')}")
    print(f"  min_keep: {cfg.get('rgtp.min_keep')}")
