#!/usr/bin/env python3
"""
Run Categorization and Visualization Script
============================================
Analyzes all Track B and Track C runs, categorizes them, and generates
visualizations explaining which runs to use for thesis and why.

Usage:
    python local_extraction/final_scripts/run_categorization_report.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np

# Configuration
WORKSPACE = Path(__file__).resolve().parents[2]
TRACK_B_METRICS = WORKSPACE / "local_extraction" / "runs" / "Track_B" / "metrics"
TRACK_B_CHECKPOINTS = WORKSPACE / "local_extraction" / "runs" / "Track_B" / "checkpoints"
TRACK_C_METRICS = WORKSPACE / "local_extraction" / "runs" / "Track_C" / "metrics"
OUTPUT_DIR = WORKSPACE / "local_extraction" / "final_scripts" / "comparison_results"

# Key checkpoints
WEIGHTED_CHECKPOINT = "0.3708"
UNWEIGHTED_CHECKPOINT = "0.3904"

# Composite score keys
SCORE_KEYS = ["mAP", "N_top5_mAP", "Nv_top5_mAP", "N_delta_top5_mAP", "All_top5_mAP"]

# Backbone/Pretraining categories
BACKBONE_VIDEOMAE = "videomae_ego"
BACKBONE_RESNET18 = "resnet18"
PRETRAINING_EXO = "exo"  # COCO/ImageNet pretrained
PRETRAINING_EGO = "ego"  # Ego4D pretrained


def _detect_backbone(run_name: str, checkpoint: str, run_dir: Optional[str] = None,
                     json_data: Optional[Dict[str, Any]] = None) -> str:
    """Detect backbone type from JSON config, run name, checkpoint, or run directory.

    Priority order:
    1. Explicit video_backbone field in JSON data (handles both structures:
       - metrics.train_config.video_backbone for run metrics
       - train_config.video_backbone for checkpoint summaries)
    2. tokens_root path containing 'videomae'
    3. Run name or checkpoint containing 'videomae'
    4. Default: resnet18
    """
    # Priority 1: Check JSON data for explicit video_backbone field
    if json_data:
        # Try metrics.train_config.video_backbone (run metrics format)
        train_config = json_data.get("metrics", {}).get("train_config", {})

        # Also try top-level train_config (checkpoint summary format)
        if not train_config:
            train_config = json_data.get("train_config", {})

        video_backbone = train_config.get("video_backbone", "")
        if video_backbone:
            if "videomae" in video_backbone.lower():
                return BACKBONE_VIDEOMAE
            if "resnet" in video_backbone.lower():
                return BACKBONE_RESNET18

        # Priority 2: Check tokens_root path
        tokens_root = train_config.get("tokens_root", "")
        if tokens_root and "videomae" in tokens_root.lower():
            return BACKBONE_VIDEOMAE

    # Priority 3: Check run name and checkpoint
    name_lower = (run_name + checkpoint).lower()
    if "videomae" in name_lower:
        return BACKBONE_VIDEOMAE

    # Default is ResNet18 (exo-transfer baseline)
    return BACKBONE_RESNET18


def _detect_pretraining(backbone: str) -> str:
    """Determine pretraining source based on backbone."""
    if backbone == BACKBONE_VIDEOMAE:
        return PRETRAINING_EGO
    # ResNet18 uses ImageNet, YOLO uses COCO - both exo
    return PRETRAINING_EXO


def _basename(path_str: Optional[str]) -> str:
    """Get basename from path string."""
    if not path_str:
        return ""
    return Path(path_str).name


def _extract_map_from_ckpt(name: str) -> Optional[float]:
    """Extract mAP value from checkpoint name like trackB_best_mAP_0.3708_..."""
    if "mAP_" in name:
        try:
            parts = name.split("mAP_")[1].split("_")[0]
            return float(parts)
        except:
            pass
    return None


def _score_row(row: Dict[str, Any]) -> float:
    """Calculate composite score."""
    total = 0.0
    for k in SCORE_KEYS:
        v = row.get(k, 0.0) or 0.0
        try:
            total += float(v)
        except (ValueError, TypeError):
            pass
    return total


def load_track_b_metrics() -> List[Dict[str, Any]]:
    """Load all Track B metrics."""
    rows = []
    if not TRACK_B_METRICS.exists():
        return rows

    for f in sorted(TRACK_B_METRICS.glob("*_summary.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            train = metrics.get("train_config", {})

            ckpt = _basename(metrics.get("checkpoint", ""))
            ckpt_map = _extract_map_from_ckpt(ckpt)

            # Determine if weighted
            use_class_weights = train.get("use_class_weights", False)
            class_weight_alpha = train.get("class_weight_alpha")

            # Extract date from filename
            # Filename: metrics_val_20251226_231700_summary.json
            parts = f.stem.split("_")
            date_str = parts[2] if len(parts) > 3 else ""

            row = {
                "run": f.name,
                "timestamp": metrics.get("timestamp", ""),
                "date": date_str,
                "checkpoint": ckpt,
                "checkpoint_map": ckpt_map,
                "mAP": metrics.get("mAP"),
                "accuracy": metrics.get("accuracy"),
                "N_top5_mAP": metrics.get("N_top5_mAP"),
                "Nv_top5_mAP": metrics.get("Nv_top5_mAP"),
                "N_delta_top5_mAP": metrics.get("N_delta_top5_mAP"),
                "All_top5_mAP": metrics.get("All_top5_mAP"),
                "ttc_mae_seconds": metrics.get("ttc_mae_seconds"),
                "use_class_weights": use_class_weights,
                "class_weight_alpha": class_weight_alpha,
                "is_weighted": bool(use_class_weights),
                "source": "trackB",
            }

            # Detect backbone and pretraining from JSON config
            backbone = _detect_backbone(f.name, ckpt, json_data=data)
            row["backbone"] = backbone
            row["pretraining"] = _detect_pretraining(backbone)

            row["score"] = _score_row(row)

            # Categorize
            if WEIGHTED_CHECKPOINT in ckpt:
                row["category"] = "weighted_thesis"
                row["use_for_thesis"] = True
                row["skip_reason"] = None
            elif UNWEIGHTED_CHECKPOINT in ckpt:
                row["category"] = "unweighted_legacy"
                row["use_for_thesis"] = False
                row["skip_reason"] = "Unweighted checkpoint - lower composite score"
            elif use_class_weights:
                row["category"] = "weighted_other"
                row["use_for_thesis"] = False
                row["skip_reason"] = "Not best weighted checkpoint"
            else:
                row["category"] = "unweighted_development"
                row["use_for_thesis"] = False
                row["skip_reason"] = "Development/early training - no class weights"

            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to parse {f.name}: {e}")

    return rows


def load_track_c_metrics() -> List[Dict[str, Any]]:
    """Load all Track C metrics."""
    rows = []
    if not TRACK_C_METRICS.exists():
        return rows

    for f in sorted(TRACK_C_METRICS.glob("trackC_val_rate*_summary.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})

            ckpt = _basename(metrics.get("checkpoint", ""))
            ckpt_map = _extract_map_from_ckpt(ckpt)

            # Extract date and rate from filename
            # Filename: trackC_val_rate00_20251226_141723_summary.json
            parts = f.stem.split("_")
            rate_str = parts[2] if len(parts) > 2 else "00"
            # Date is at index 3 (20251226), timestamp is at index 4 (141723)
            date_str = parts[3] if len(parts) > 4 else ""

            # Determine efficiency config
            eff_parts = []
            rgtp_enabled = metrics.get("rgtp_enabled", False)
            rgtp_rate = metrics.get("rgtp_rate_request", 0)
            frame_sub = metrics.get("efficiency_frame_subsample_enabled", False)
            frame_keep = metrics.get("efficiency_frame_subsample_keep_frames")
            frame_strat = metrics.get("efficiency_frame_subsample_strategy", "uniform")
            token_prune = metrics.get("efficiency_token_prune_enabled", False)
            token_rate = metrics.get("efficiency_token_prune_rate")

            if rgtp_enabled and rgtp_rate > 0:
                eff_parts.append(f"RGTP={rgtp_rate:.1f}")
            if frame_sub and frame_keep:
                eff_parts.append(f"Fr={frame_keep}({frame_strat[:3]})")
            if token_prune and token_rate:
                eff_parts.append(f"Tok={token_rate:.1f}")

            eff_config = " + ".join(eff_parts) if eff_parts else "Baseline"

            row = {
                "run": f.name,
                "timestamp": f.stem.split("_")[-1] if "_" in f.stem else "",
                "date": date_str,
                "rgtp_rate": rate_str,
                "checkpoint": ckpt,
                "checkpoint_map": ckpt_map,
                "efficiency_config": eff_config,
                "mAP": metrics.get("mAP"),
                "accuracy": metrics.get("accuracy"),
                "N_top5_mAP": metrics.get("N_top5_mAP"),
                "Nv_top5_mAP": metrics.get("Nv_top5_mAP"),
                "N_delta_top5_mAP": metrics.get("N_delta_top5_mAP"),
                "All_top5_mAP": metrics.get("All_top5_mAP"),
                "ttc_mae_seconds": metrics.get("ttc_mae_seconds"),
                "latency_ms_mean": metrics.get("latency_ms_mean"),
                "rgtp_enabled": rgtp_enabled,
                "rgtp_rate_request": rgtp_rate,
                "frame_subsample_enabled": frame_sub,
                "frame_subsample_keep_frames": frame_keep,
                "token_prune_enabled": token_prune,
                "token_prune_rate": token_rate,
                "source": "trackC",
            }

            # Detect backbone and pretraining from JSON config
            backbone = _detect_backbone(f.name, ckpt, json_data=data)
            row["backbone"] = backbone
            row["pretraining"] = _detect_pretraining(backbone)

            row["score"] = _score_row(row)

            # Categorize
            if WEIGHTED_CHECKPOINT in ckpt:
                if date_str >= "20251230":
                    row["category"] = "weighted_efficiency_v5"
                    row["use_for_thesis"] = True
                    row["skip_reason"] = None
                elif date_str >= "20251226":
                    row["category"] = "weighted_baseline_v4"
                    row["use_for_thesis"] = True
                    row["skip_reason"] = None
                else:
                    row["category"] = "weighted_early"
                    row["use_for_thesis"] = False
                    row["skip_reason"] = "Early weighted runs before final baseline"
            elif UNWEIGHTED_CHECKPOINT in ckpt:
                row["category"] = "unweighted_legacy"
                row["use_for_thesis"] = False
                row["skip_reason"] = "Uses unweighted 0.3904 checkpoint"
            else:
                row["category"] = "development"
                row["use_for_thesis"] = False
                row["skip_reason"] = "Development/early checkpoint"

            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to parse {f.name}: {e}")

    return rows


def load_checkpoints() -> List[Dict[str, Any]]:
    """Load checkpoint info."""
    rows = []
    if not TRACK_B_CHECKPOINTS.exists():
        return rows

    for f in sorted(TRACK_B_CHECKPOINTS.glob("trackB_best_mAP_*.pt")):
        name = f.name
        ckpt_map = _extract_map_from_ckpt(name)
        date_str = name.split("_")[-2] if "_" in name else ""

        # Determine if weighted by checking summary json
        summary_file = f.with_suffix(".json").parent / (f.stem + "_summary.json")
        is_weighted = False
        alpha = None
        summary_data = None
        if summary_file.exists():
            try:
                summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                train = summary_data.get("train_config", {})
                is_weighted = bool(train.get("use_class_weights", False))
                alpha = train.get("class_weight_alpha")
            except:
                pass

        row = {
            "checkpoint": name,
            "mAP": ckpt_map,
            "date": date_str,
            "is_weighted": is_weighted,
            "class_weight_alpha": alpha,
        }

        # Detect backbone and pretraining from summary JSON
        # summary_data has train_config at top level, pass directly
        backbone = _detect_backbone(name, name, json_data=summary_data)
        row["backbone"] = backbone
        row["pretraining"] = _detect_pretraining(backbone)

        # Categorize
        if str(ckpt_map) == WEIGHTED_CHECKPOINT.lstrip("0."):
            row["category"] = "thesis_primary"
            row["use_for_thesis"] = True
            row["skip_reason"] = None
        elif str(ckpt_map) == UNWEIGHTED_CHECKPOINT.lstrip("0."):
            row["category"] = "legacy_reference"
            row["use_for_thesis"] = False
            row["skip_reason"] = "Unweighted - lower composite score"
        elif is_weighted:
            row["category"] = "weighted_other"
            row["use_for_thesis"] = False
            row["skip_reason"] = "Not the best weighted checkpoint"
        else:
            row["category"] = "development"
            row["use_for_thesis"] = False
            row["skip_reason"] = "Development checkpoint"

        rows.append(row)

    return sorted(rows, key=lambda r: r.get("mAP", 0) or 0, reverse=True)


def save_plot_data(data: Dict[str, Any], filename: str, output_dir: Path):
    """Save plot data to JSON file."""
    json_path = output_dir / f"{filename}_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"      [DATA] Saved {json_path.name}")


def create_plots(track_b: List[Dict[str, Any]], track_c: List[Dict[str, Any]],
                checkpoints: List[Dict[str, Any]], output_dir: Path):
    """Create comprehensive visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Checkpoint Timeline with Use/Skip Categories (by date)
    fig, ax = plt.subplots(figsize=(14, 6))

    color_map = {
        "thesis_primary": "#4CAF50",      # Green - use
        "legacy_reference": "#FF9800",     # Orange - reference only
        "weighted_other": "#2196F3",       # Blue - weighted but not best
        "development": "#9E9E9E",          # Gray - skip
    }

    ckpt_dates = []
    ckpt_maps = []
    ckpt_colors = []
    ckpt_labels = []

    valid_ckpts = [(i, ckpt) for i, ckpt in enumerate(checkpoints)
                   if ckpt.get("mAP") and ckpt.get("date") and len(ckpt.get("date", "")) == 8]

    for i, (_, ckpt) in enumerate(valid_ckpts):
        date_str = ckpt.get("date", "")
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            ckpt_dates.append(date_obj)
        except:
            continue
        ckpt_maps.append(ckpt["mAP"])
        ckpt_colors.append(color_map.get(ckpt.get("category", "development"), "#9E9E9E"))
        ckpt_labels.append(f"{ckpt['mAP']:.4f}")

    if ckpt_dates:
        scatter = ax.scatter(ckpt_dates, ckpt_maps, c=ckpt_colors, s=100, edgecolors='black', zorder=5)

        # Highlight key checkpoints
        for date, m, c, l in zip(ckpt_dates, ckpt_maps, ckpt_colors, ckpt_labels):
            if "3708" in l or "3904" in l:
                ax.annotate(l, (date, m), textcoords="offset points", xytext=(5, 10),
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax.axhline(y=0.3708, color='#4CAF50', linestyle='--', alpha=0.5, label='0.3708 (Thesis)')
        ax.axhline(y=0.3904, color='#FF9800', linestyle='--', alpha=0.5, label='0.3904 (Legacy)')

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Legend
        legend_patches = [
            mpatches.Patch(color='#4CAF50', label='[USE] Thesis Primary (0.3708)'),
            mpatches.Patch(color='#FF9800', label='[REF] Legacy Reference (0.3904)'),
            mpatches.Patch(color='#2196F3', label='[SKIP] Other Weighted'),
            mpatches.Patch(color='#9E9E9E', label='[SKIP] Development'),
        ]
        ax.legend(handles=legend_patches, loc='lower right')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_title('Track B Checkpoint Evolution by Date', fontsize=14)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'checkpoint_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save plot data
    save_plot_data({
        "plot_type": "checkpoint_timeline",
        "checkpoints": [{"date": str(d), "mAP": float(m), "color": c, "label": l}
                       for d, m, c, l in zip(ckpt_dates, ckpt_maps, ckpt_colors, ckpt_labels)],
        "color_map": color_map
    }, "checkpoint_timeline", output_dir)

    # Plot 1b: Track B Checkpoints - All Metrics Comparison (like efficiency_configs)
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get top checkpoints by mAP for comparison
    top_ckpts = sorted([c for c in checkpoints if c.get("mAP")],
                       key=lambda x: x.get("mAP", 0), reverse=True)[:10]

    if top_ckpts:
        ckpt_names = []
        ckpt_mAP = []
        ckpt_acc = []
        ckpt_all_top5 = []
        ckpt_n_top5 = []
        ckpt_nv_top5 = []
        ckpt_nd_top5 = []
        ckpt_scores = []

        for ckpt in top_ckpts:
            # Short name from checkpoint
            name = ckpt.get("checkpoint", "")
            # Extract mAP value from name (e.g., 0.3708)
            if "mAP_" in name:
                short = name.split("mAP_")[1][:6]
            elif "best_" in name:
                short = name.split("best_")[1][:10]
            else:
                short = name[-15:]
            ckpt_names.append(short)
            ckpt_mAP.append((ckpt.get("mAP") or 0) * 100)
            ckpt_acc.append((ckpt.get("accuracy") or 0) * 100)
            ckpt_all_top5.append((ckpt.get("All_top5_mAP") or 0) * 100)
            ckpt_n_top5.append((ckpt.get("N_top5_mAP") or 0) * 100)
            ckpt_nv_top5.append((ckpt.get("Nv_top5_mAP") or 0) * 100)
            ckpt_nd_top5.append((ckpt.get("N_delta_top5_mAP") or 0) * 100)
            ckpt_scores.append(ckpt.get("score") or 0)

        x = np.arange(len(ckpt_names))
        width = 0.12

        ax2 = ax.twinx()

        # Plot all metrics as bars
        bars1 = ax.bar(x - 2.5*width, ckpt_mAP, width, label='mAP (%)', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x - 1.5*width, ckpt_all_top5, width, label='All_top5', color='#4CAF50', alpha=0.8)
        bars3 = ax.bar(x - 0.5*width, ckpt_n_top5, width, label='N_top5', color='#8BC34A', alpha=0.8)
        bars4 = ax.bar(x + 0.5*width, ckpt_nv_top5, width, label='Nv_top5', color='#CDDC39', alpha=0.8)
        bars5 = ax.bar(x + 1.5*width, ckpt_nd_top5, width, label='N_delta', color='#FFC107', alpha=0.8)
        bars6 = ax.bar(x + 2.5*width, ckpt_acc, width, label='Accuracy (%)', color='#9C27B0', alpha=0.8)

        # Composite score as line
        line = ax2.plot(x, ckpt_scores, 'ro-', label='Composite Score', markersize=10, linewidth=2)

        ax.set_xlabel('Checkpoint (by mAP value)')
        ax.set_ylabel('Metric Value (%)')
        ax2.set_ylabel('Composite Score', color='red')
        ax.set_title('Track B Top 10 Checkpoints: All Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(ckpt_names, rotation=30, ha='right')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        # Highlight the thesis checkpoint (0.3708)
        for i, name in enumerate(ckpt_names):
            if "3708" in name:
                ax.axvline(x=i, color='#4CAF50', linestyle='--', alpha=0.5, linewidth=2)
                ax.annotate('THESIS', (i, max(ckpt_mAP) + 2), ha='center', fontsize=9,
                           fontweight='bold', color='#4CAF50')

    plt.tight_layout()
    plt.savefig(output_dir / 'trackB_checkpoints_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save plot data
    if top_ckpts:
        save_plot_data({
            "plot_type": "trackB_checkpoints_metrics",
            "checkpoints": [{"name": n, "mAP": float(m), "accuracy": float(a), "All_top5_mAP": float(at),
                            "N_top5_mAP": float(nt), "Nv_top5_mAP": float(nv), "N_delta_top5_mAP": float(nd),
                            "score": float(s)}
                           for n, m, a, at, nt, nv, nd, s in zip(ckpt_names, ckpt_mAP, ckpt_acc,
                                                                   ckpt_all_top5, ckpt_n_top5,
                                                                   ckpt_nv_top5, ckpt_nd_top5, ckpt_scores)]
        }, "trackB_checkpoints_metrics", output_dir)

    # Plot 1c: Track B Runs - All Metrics Comparison (like efficiency_configs for Track C)
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get runs sorted by score for comparison
    top_runs = sorted([r for r in track_b if r.get("score")],
                      key=lambda x: x.get("score", 0), reverse=True)[:12]

    if top_runs:
        run_names = []
        run_mAP = []
        run_acc = []
        run_all_top5 = []
        run_n_top5 = []
        run_nv_top5 = []
        run_nd_top5 = []
        run_scores = []
        run_colors = []

        for r in top_runs:
            # Short name from run - extract checkpoint mAP and date
            ckpt = r.get("checkpoint", "")
            date = r.get("date", "")[-4:] if r.get("date") else ""  # MMDD
            if "mAP_" in ckpt:
                mAP_val = ckpt.split("mAP_")[1][:6]
                short = f"{mAP_val}\n({date})"
            else:
                short = date
            run_names.append(short)
            run_mAP.append((r.get("mAP") or 0) * 100)
            run_acc.append((r.get("accuracy") or 0) * 100)
            run_all_top5.append((r.get("All_top5_mAP") or 0) * 100)
            run_n_top5.append((r.get("N_top5_mAP") or 0) * 100)
            run_nv_top5.append((r.get("Nv_top5_mAP") or 0) * 100)
            run_nd_top5.append((r.get("N_delta_top5_mAP") or 0) * 100)
            run_scores.append(r.get("score") or 0)

            # Color by use status
            if r.get("use_for_thesis"):
                run_colors.append("#4CAF50")
            elif UNWEIGHTED_CHECKPOINT in ckpt:
                run_colors.append("#FF9800")
            else:
                run_colors.append("#9E9E9E")

        x = np.arange(len(run_names))
        width = 0.12

        ax2 = ax.twinx()

        # Plot all metrics as bars
        bars1 = ax.bar(x - 2.5*width, run_mAP, width, label='mAP (%)', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x - 1.5*width, run_all_top5, width, label='All_top5', color='#4CAF50', alpha=0.8)
        bars3 = ax.bar(x - 0.5*width, run_n_top5, width, label='N_top5', color='#8BC34A', alpha=0.8)
        bars4 = ax.bar(x + 0.5*width, run_nv_top5, width, label='Nv_top5', color='#CDDC39', alpha=0.8)
        bars5 = ax.bar(x + 1.5*width, run_nd_top5, width, label='N_delta', color='#FFC107', alpha=0.8)
        bars6 = ax.bar(x + 2.5*width, run_acc, width, label='Accuracy (%)', color='#9C27B0', alpha=0.8)

        # Composite score as line
        line = ax2.plot(x, run_scores, 'ro-', label='Composite Score', markersize=10, linewidth=2)

        # Add background colors for USE vs SKIP
        for i, color in enumerate(run_colors):
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color=color, zorder=0)

        ax.set_xlabel('Checkpoint (mAP) and Date')
        ax.set_ylabel('Metric Value (%)')
        ax2.set_ylabel('Composite Score', color='red')
        ax.set_title('Track B Top 12 Runs: All Metrics Comparison\n(Green=USE, Orange=0.3904, Gray=Other)')
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=0, ha='center', fontsize=8)
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'trackB_runs_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save plot data
    if top_runs:
        save_plot_data({
            "plot_type": "trackB_runs_metrics",
            "runs": [{"name": n, "mAP": float(m), "accuracy": float(a), "All_top5_mAP": float(at),
                     "N_top5_mAP": float(nt), "Nv_top5_mAP": float(nv), "N_delta_top5_mAP": float(nd),
                     "score": float(s), "color": c}
                    for n, m, a, at, nt, nv, nd, s, c in zip(run_names, run_mAP, run_acc,
                                                               run_all_top5, run_n_top5,
                                                               run_nv_top5, run_nd_top5,
                                                               run_scores, run_colors)]
        }, "trackB_runs_metrics", output_dir)

    # Plot 2: Weighted vs Unweighted Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar chart of key metrics
    ax1 = axes[0]
    metrics = ['mAP', 'accuracy', 'All_top5_mAP', 'N_top5_mAP', 'Nv_top5_mAP', 'N_delta_top5_mAP']
    labels = ['mAP', 'accuracy', 'All_top5', 'N_top5', 'Nv_top5', 'N_delta']

    # Find best weighted and unweighted runs
    weighted_runs = [r for r in track_b if WEIGHTED_CHECKPOINT in r.get("checkpoint", "")]
    unweighted_runs = [r for r in track_b if UNWEIGHTED_CHECKPOINT in r.get("checkpoint", "")]

    if weighted_runs and unweighted_runs:
        best_weighted = max(weighted_runs, key=lambda r: r.get("score", 0))
        best_unweighted = max(unweighted_runs, key=lambda r: r.get("score", 0))

        x = np.arange(len(metrics))
        width = 0.35

        w_vals = [(best_weighted.get(m) or 0) * 100 for m in metrics]
        u_vals = [(best_unweighted.get(m) or 0) * 100 for m in metrics]

        bars1 = ax1.bar(x - width/2, w_vals, width, label=f'Weighted (0.3708)', color='#4CAF50', alpha=0.8)
        bars2 = ax1.bar(x + width/2, u_vals, width, label=f'Unweighted (0.3904)', color='#FF9800', alpha=0.8)

        ax1.set_ylabel('Score (%)')
        ax1.set_title('Weighted vs Unweighted Checkpoint Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                            ha='center', va='bottom', fontsize=7)

    # Right: Composite score comparison
    ax2 = axes[1]
    if weighted_runs and unweighted_runs:
        categories = ['Composite\nScore', 'mAP\n(×100)', 'All_top5\n(×100)']
        w_score = best_weighted.get("score", 0)
        u_score = best_unweighted.get("score", 0)

        scores = [
            [w_score, u_score],
            [(best_weighted.get("mAP") or 0) * 100, (best_unweighted.get("mAP") or 0) * 100],
            [(best_weighted.get("All_top5_mAP") or 0) * 100, (best_unweighted.get("All_top5_mAP") or 0) * 100],
        ]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax2.bar(x - width/2, [s[0] for s in scores], width, label='Weighted (0.3708) [USE]', color='#4CAF50')
        bars2 = ax2.bar(x + width/2, [s[1] for s in scores], width, label='Unweighted (0.3904) [SKIP]', color='#FF9800')

        ax2.set_ylabel('Value')
        ax2.set_title('Why Use Weighted Checkpoint?')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Highlight winner
        for i, (w, u) in enumerate(scores):
            winner = "W" if w > u else "U"
            y_pos = max(w, u) + 1
            ax2.annotate(f'Winner: {"Weighted" if w > u else "Unweighted"}',
                        xy=(i, y_pos), ha='center', fontsize=8,
                        color='#4CAF50' if w > u else '#FF9800', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'weighted_vs_unweighted.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save plot data
    if weighted_runs and unweighted_runs:
        save_plot_data({
            "plot_type": "weighted_vs_unweighted",
            "weighted": {
                "checkpoint": "0.3708",
                "metrics": dict(zip(metrics, w_vals)),
                "score": float(w_score)
            },
            "unweighted": {
                "checkpoint": "0.3904",
                "metrics": dict(zip(metrics, u_vals)),
                "score": float(u_score)
            },
            "scores_comparison": [[w_score, u_score],
                                 [(best_weighted.get("mAP") or 0) * 100, (best_unweighted.get("mAP") or 0) * 100],
                                 [(best_weighted.get("All_top5_mAP") or 0) * 100, (best_unweighted.get("All_top5_mAP") or 0) * 100]]
        }, "weighted_vs_unweighted", output_dir)

    # Plot 3: Track C Run Categories
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Pie chart of run categories
    ax1 = axes[0]
    category_counts = {}
    for r in track_c:
        cat = r.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    cat_colors = {
        "weighted_efficiency_v5": "#4CAF50",
        "weighted_baseline_v4": "#8BC34A",
        "weighted_early": "#CDDC39",
        "unweighted_legacy": "#FF9800",
        "development": "#9E9E9E",
    }
    cat_labels = {
        "weighted_efficiency_v5": "[USE] Dec 30 Efficiency",
        "weighted_baseline_v4": "[USE] Dec 26 Baseline",
        "weighted_early": "[SKIP] Early Weighted",
        "unweighted_legacy": "[SKIP] Unweighted 0.3904",
        "development": "[SKIP] Development",
    }

    labels = [cat_labels.get(c, c) for c in category_counts.keys()]
    sizes = list(category_counts.values())
    colors = [cat_colors.get(c, "#9E9E9E") for c in category_counts.keys()]

    explode = [0.1 if "weighted_efficiency" in c or "weighted_baseline" in c else 0
               for c in category_counts.keys()]

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%',
            shadow=True, startangle=90)
    ax1.set_title('Track C Run Categories')

    # Right: Score distribution by category
    ax2 = axes[1]
    use_runs = [r for r in track_c if r.get("use_for_thesis")]
    skip_runs = [r for r in track_c if not r.get("use_for_thesis")]

    use_scores = [r.get("score", 0) for r in use_runs if r.get("score")]
    skip_scores = [r.get("score", 0) for r in skip_runs if r.get("score")]

    if use_scores or skip_scores:
        bp = ax2.boxplot([use_scores, skip_scores] if use_scores and skip_scores else
                        [use_scores] if use_scores else [skip_scores],
                        tick_labels=['[USE] For Thesis', '[SKIP]'] if use_scores and skip_scores else
                        ['[USE] For Thesis'] if use_scores else ['[SKIP]'],
                        patch_artist=True)

        colors_bp = ['#4CAF50', '#FF9800'] if use_scores and skip_scores else ['#4CAF50'] if use_scores else ['#FF9800']
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel('Composite Score')
        ax2.set_title('Score Distribution: Use vs Skip Runs')
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'trackC_categories.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save plot data
    save_plot_data({
        "plot_type": "trackC_categories",
        "category_counts": category_counts,
        "use_scores": use_scores if use_scores else [],
        "skip_scores": skip_scores if skip_scores else []
    }, "trackC_categories", output_dir)

    # Plot 4: Efficiency Configurations Comparison (Dec 30 runs only)
    fig, ax = plt.subplots(figsize=(12, 7))

    dec30_runs = [r for r in track_c if r.get("date", "") >= "20251230" and r.get("use_for_thesis")]

    if dec30_runs:
        # Group by efficiency config
        configs = {}
        for r in dec30_runs:
            cfg = r.get("efficiency_config", "Unknown")
            if cfg not in configs:
                configs[cfg] = []
            configs[cfg].append(r)

        # Get best score for each config
        config_names = []
        config_scores = []
        config_latencies = []
        config_mAP = []
        config_all_top5 = []
        config_n_top5 = []
        config_nv_top5 = []
        config_nd_top5 = []

        for cfg, runs in sorted(configs.items(), key=lambda x: max(r.get("score", 0) for r in x[1]), reverse=True):
            best = max(runs, key=lambda r: r.get("score", 0))
            config_names.append(cfg)
            config_scores.append(best.get("score", 0))
            config_latencies.append(best.get("latency_ms_mean", 0))
            config_mAP.append((best.get("mAP") or 0) * 100)
            config_all_top5.append((best.get("All_top5_mAP") or 0) * 100)
            config_n_top5.append((best.get("N_top5_mAP") or 0) * 100)
            config_nv_top5.append((best.get("Nv_top5_mAP") or 0) * 100)
            config_nd_top5.append((best.get("N_delta_top5_mAP") or 0) * 100)

        x = np.arange(len(config_names))
        width = 0.12

        ax2 = ax.twinx()

        # Plot all top5 metrics plus mAP
        bars1 = ax.bar(x - 2*width, config_mAP, width, label='mAP (%)', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x - width, config_all_top5, width, label='All_top5', color='#4CAF50', alpha=0.8)
        bars3 = ax.bar(x, config_n_top5, width, label='N_top5', color='#8BC34A', alpha=0.8)
        bars4 = ax.bar(x + width, config_nv_top5, width, label='Nv_top5', color='#CDDC39', alpha=0.8)
        bars5 = ax.bar(x + 2*width, config_nd_top5, width, label='N_delta', color='#FFC107', alpha=0.8)
        line = ax2.plot(x, config_latencies, 'ro-', label='Latency (ms)', markersize=10, linewidth=2)

        ax.set_xlabel('Efficiency Configuration')
        ax.set_ylabel('Metric Value (%)')
        ax2.set_ylabel('Latency (ms)', color='red')
        ax.set_title('Dec 30 Efficiency Runs: All Metrics vs Latency Trade-off')
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=30, ha='right')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_configs.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save plot data
    if dec30_runs:
        save_plot_data({
            "plot_type": "efficiency_configs",
            "configs": [{"name": n, "mAP": float(m), "All_top5_mAP": float(at),
                        "N_top5_mAP": float(nt), "Nv_top5_mAP": float(nv), "N_delta_top5_mAP": float(nd),
                        "score": float(s), "latency_ms": float(l)}
                       for n, m, at, nt, nv, nd, s, l in zip(config_names, config_mAP,
                                                              config_all_top5, config_n_top5,
                                                              config_nv_top5, config_nd_top5,
                                                              config_scores, config_latencies)]
        }, "efficiency_configs", output_dir)

    # Plot 5: Complete Run Timeline (by date)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Track B timeline
    ax1 = axes[0]
    b_dates = []
    b_scores = []
    b_colors = []

    valid_b = [(i, r) for i, r in enumerate(sorted(track_b, key=lambda x: x.get("date", "")))
               if r.get("date") and len(r.get("date", "")) == 8]

    for i, (_, r) in enumerate(valid_b):
        # Parse date string YYYYMMDD to datetime
        date_str = r.get("date", "")
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            b_dates.append(date_obj)
        except:
            continue
        b_scores.append(r.get("score", 0))
        if r.get("use_for_thesis"):
            b_colors.append("#4CAF50")
        elif UNWEIGHTED_CHECKPOINT in r.get("checkpoint", ""):
            b_colors.append("#FF9800")
        else:
            b_colors.append("#9E9E9E")

    if b_dates:
        ax1.scatter(b_dates, b_scores, c=b_colors, s=80, edgecolors='black', alpha=0.7)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Composite Score')
        ax1.set_title('Track B Runs Timeline by Date')
        ax1.grid(alpha=0.3)

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        legend_patches = [
            mpatches.Patch(color='#4CAF50', label='Use (0.3708)'),
            mpatches.Patch(color='#FF9800', label='Skip (0.3904)'),
            mpatches.Patch(color='#9E9E9E', label='Skip (Other)'),
        ]
        ax1.legend(handles=legend_patches, loc='lower right')

    # Track C timeline
    ax2 = axes[1]
    c_dates = []
    c_scores = []
    c_colors = []

    valid_c = [(i, r) for i, r in enumerate(sorted(track_c, key=lambda x: x.get("date", "")))
               if r.get("date") and len(r.get("date", "")) == 8]

    for i, (_, r) in enumerate(valid_c):
        date_str = r.get("date", "")
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            c_dates.append(date_obj)
        except:
            continue
        c_scores.append(r.get("score", 0))
        if r.get("use_for_thesis"):
            c_colors.append("#4CAF50")
        elif UNWEIGHTED_CHECKPOINT in r.get("checkpoint", ""):
            c_colors.append("#FF9800")
        else:
            c_colors.append("#9E9E9E")

    if c_dates:
        ax2.scatter(c_dates, c_scores, c=c_colors, s=80, edgecolors='black', alpha=0.7)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Composite Score')
        ax2.set_title('Track C Runs Timeline by Date')
        ax2.grid(alpha=0.3)

        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add vertical lines for key dates
        try:
            dec26 = datetime(2025, 12, 26)
            dec30 = datetime(2025, 12, 30)
            ax2.axvline(x=dec26, color='#4CAF50', linestyle='--', alpha=0.7, label='Dec 26: Baseline')
            ax2.axvline(x=dec30, color='#2196F3', linestyle='--', alpha=0.7, label='Dec 30: Efficiency')
        except:
            pass

        legend_patches = [
            mpatches.Patch(color='#4CAF50', label='Use (Weighted 0.3708)'),
            mpatches.Patch(color='#FF9800', label='Skip (0.3904)'),
            mpatches.Patch(color='#9E9E9E', label='Skip (Other)'),
        ]
        ax2.legend(handles=legend_patches, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'run_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save plot data
    save_plot_data({
        "plot_type": "run_timeline",
        "track_b": [{"date": str(d), "score": float(s), "color": c}
                   for d, s, c in zip(b_dates, b_scores, b_colors)],
        "track_c": [{"date": str(d), "score": float(s), "color": c}
                   for d, s, c in zip(c_dates, c_scores, c_colors)]
    }, "run_timeline", output_dir)

    # Plot 6: Backbone Comparison (ResNet18 vs VideoMAE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Backbone distribution
    ax1 = axes[0]
    all_runs = track_b + track_c
    backbone_counts = {}
    for r in all_runs:
        bb = r.get("backbone", BACKBONE_RESNET18)
        backbone_counts[bb] = backbone_counts.get(bb, 0) + 1

    bb_colors = {
        BACKBONE_RESNET18: "#2196F3",  # Blue for exo-transfer
        BACKBONE_VIDEOMAE: "#4CAF50",  # Green for ego-pretrained
    }
    bb_labels = {
        BACKBONE_RESNET18: "ResNet18 (ImageNet)",
        BACKBONE_VIDEOMAE: "VideoMAE (Ego4D)",
    }

    if backbone_counts:
        labels = [bb_labels.get(k, k) for k in backbone_counts.keys()]
        sizes = list(backbone_counts.values())
        colors = [bb_colors.get(k, "#9E9E9E") for k in backbone_counts.keys()]

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                shadow=True, startangle=90)
        ax1.set_title('Backbone Distribution Across All Runs')

    # Right: Pretraining source comparison
    ax2 = axes[1]
    pretraining_counts = {}
    for r in all_runs:
        pt = r.get("pretraining", PRETRAINING_EXO)
        pretraining_counts[pt] = pretraining_counts.get(pt, 0) + 1

    pt_colors = {
        PRETRAINING_EXO: "#FF9800",   # Orange for exo (third-person pretrained)
        PRETRAINING_EGO: "#4CAF50",   # Green for ego (first-person pretrained)
    }
    pt_labels = {
        PRETRAINING_EXO: "Exo-Transfer\n(COCO/ImageNet)",
        PRETRAINING_EGO: "Ego-Pretrained\n(Ego4D)",
    }

    if pretraining_counts:
        labels = [pt_labels.get(k, k) for k in pretraining_counts.keys()]
        sizes = list(pretraining_counts.values())
        colors = [pt_colors.get(k, "#9E9E9E") for k in pretraining_counts.keys()]

        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                shadow=True, startangle=90)
        ax2.set_title('Pretraining Source Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / 'backbone_pretraining.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save plot data
    save_plot_data({
        "plot_type": "backbone_pretraining",
        "backbone_counts": backbone_counts,
        "pretraining_counts": pretraining_counts
    }, "backbone_pretraining", output_dir)

    # Plot 7: Backbone Metrics Comparison (if we have VideoMAE runs)
    videomae_runs = [r for r in all_runs if r.get("backbone") == BACKBONE_VIDEOMAE]
    resnet_runs = [r for r in all_runs if r.get("backbone") == BACKBONE_RESNET18]

    if videomae_runs and resnet_runs:
        fig, ax = plt.subplots(figsize=(12, 7))

        # Compare best runs from each backbone
        best_videomae = max(videomae_runs, key=lambda r: r.get("score", 0))
        best_resnet = max(resnet_runs, key=lambda r: r.get("score", 0))

        metrics = ['mAP', 'accuracy', 'All_top5_mAP', 'N_top5_mAP', 'Nv_top5_mAP', 'N_delta_top5_mAP']
        labels = ['mAP', 'Accuracy', 'All_top5', 'N_top5', 'Nv_top5', 'N_delta']

        x = np.arange(len(metrics))
        width = 0.35

        videomae_vals = [(best_videomae.get(m) or 0) * 100 for m in metrics]
        resnet_vals = [(best_resnet.get(m) or 0) * 100 for m in metrics]

        bars1 = ax.bar(x - width/2, resnet_vals, width, label='ResNet18 (Exo)', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, videomae_vals, width, label='VideoMAE (Ego)', color='#4CAF50', alpha=0.8)

        ax.set_ylabel('Score (%)')
        ax.set_title('ResNet18 (Exo-Transfer) vs VideoMAE (Ego-Pretrained) Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                            ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'backbone_metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save plot data
        save_plot_data({
            "plot_type": "backbone_metrics_comparison",
            "resnet18": {"metrics": dict(zip(metrics, resnet_vals)), "label": labels},
            "videomae": {"metrics": dict(zip(metrics, videomae_vals)), "label": labels}
        }, "backbone_metrics_comparison", output_dir)

    # Plot 8: Track B vs Track C - USE Runs Comparison
    # Compare all suitable runs from both tracks in the same plot
    b_use_runs = [r for r in track_b if r.get("use_for_thesis")]
    c_use_runs = [r for r in track_c if r.get("use_for_thesis")]

    if b_use_runs or c_use_runs:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Grouped bar chart comparing metrics
        ax1 = axes[0]

        # Collect metrics for both tracks
        metrics_to_compare = ['mAP', 'accuracy', 'All_top5_mAP', 'N_top5_mAP', 'Nv_top5_mAP', 'N_delta_top5_mAP']
        metric_labels = ['mAP', 'Accuracy', 'All_top5', 'N_top5', 'Nv_top5', 'N_delta']

        # Get best runs from each track
        if b_use_runs:
            best_b = max(b_use_runs, key=lambda r: r.get("score", 0))
            b_vals = [(best_b.get(m) or 0) * 100 for m in metrics_to_compare]
        else:
            b_vals = [0] * len(metrics_to_compare)

        if c_use_runs:
            best_c = max(c_use_runs, key=lambda r: r.get("score", 0))
            c_vals = [(best_c.get(m) or 0) * 100 for m in metrics_to_compare]
        else:
            c_vals = [0] * len(metrics_to_compare)

        x = np.arange(len(metrics_to_compare))
        width = 0.35

        bars1 = ax1.bar(x - width/2, b_vals, width, label=f'Track B Best ({len(b_use_runs)} runs)',
                       color='#2196F3', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, c_vals, width, label=f'Track C Best ({len(c_use_runs)} runs)',
                       color='#4CAF50', alpha=0.8, edgecolor='black')

        ax1.set_ylabel('Score (%)')
        ax1.set_title('Track B (Baseline) vs Track C (Efficiency)\nBest Suitable Runs')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels, rotation=15)
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels and delta
        for i, (b_val, c_val) in enumerate(zip(b_vals, c_vals)):
            # Track B value
            ax1.annotate(f'{b_val:.1f}', xy=(i - width/2, b_val + 0.5),
                        ha='center', va='bottom', fontsize=7, color='#2196F3', fontweight='bold')
            # Track C value
            ax1.annotate(f'{c_val:.1f}', xy=(i + width/2, c_val + 0.5),
                        ha='center', va='bottom', fontsize=7, color='#4CAF50', fontweight='bold')
            # Delta
            delta = c_val - b_val
            delta_color = '#4CAF50' if delta >= 0 else '#F44336'
            ax1.annotate(f'Δ{delta:+.1f}', xy=(i, max(b_val, c_val) + 3),
                        ha='center', va='bottom', fontsize=7, color=delta_color)

        # Right: Scatter plot showing all USE runs from both tracks
        ax2 = axes[1]

        # Plot Track B runs
        if b_use_runs:
            b_maps = [(r.get("mAP") or 0) * 100 for r in b_use_runs]
            b_accs = [(r.get("accuracy") or 0) * 100 for r in b_use_runs]
            ax2.scatter(b_maps, b_accs, c='#2196F3', s=150, label=f'Track B ({len(b_use_runs)})',
                       alpha=0.8, edgecolors='black', marker='o')

            # Annotate best
            for r in b_use_runs:
                if r == best_b:
                    ax2.annotate('Best B',
                               ((r.get("mAP") or 0) * 100, (r.get("accuracy") or 0) * 100),
                               textcoords="offset points", xytext=(10, 5), fontsize=8,
                               fontweight='bold', color='#2196F3')

        # Plot Track C runs
        if c_use_runs:
            c_maps = [(r.get("mAP") or 0) * 100 for r in c_use_runs]
            c_accs = [(r.get("accuracy") or 0) * 100 for r in c_use_runs]
            ax2.scatter(c_maps, c_accs, c='#4CAF50', s=150, label=f'Track C ({len(c_use_runs)})',
                       alpha=0.8, edgecolors='black', marker='s')

            # Annotate best
            for r in c_use_runs:
                if r == best_c:
                    ax2.annotate('Best C',
                               ((r.get("mAP") or 0) * 100, (r.get("accuracy") or 0) * 100),
                               textcoords="offset points", xytext=(10, 5), fontsize=8,
                               fontweight='bold', color='#4CAF50')

        ax2.set_xlabel('mAP (%)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('All Suitable Runs: mAP vs Accuracy\n(Track B = circles, Track C = squares)')
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'trackB_vs_trackC_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save plot data
        save_plot_data({
            "plot_type": "trackB_vs_trackC_comparison",
            "track_b": {
                "best_metrics": dict(zip(metrics_to_compare, b_vals)),
                "all_runs": [{"mAP": (r.get("mAP") or 0) * 100,
                             "accuracy": (r.get("accuracy") or 0) * 100} for r in b_use_runs]
            },
            "track_c": {
                "best_metrics": dict(zip(metrics_to_compare, c_vals)),
                "all_runs": [{"mAP": (r.get("mAP") or 0) * 100,
                             "accuracy": (r.get("accuracy") or 0) * 100} for r in c_use_runs]
            },
            "metric_labels": metric_labels
        }, "trackB_vs_trackC_comparison", output_dir)

    print(f"[INFO] Plots saved to {output_dir}")
    print(f"[INFO] Plot data files saved (*.json)")



def generate_report(track_b: List[Dict[str, Any]], track_c: List[Dict[str, Any]],
                   checkpoints: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Generate comprehensive markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Track B and Track C Run Categorization Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")
    lines.append("This report categorizes all Track B and Track C runs, explaining which to use for thesis and which to skip, with detailed reasoning.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("### Key Decisions")
    lines.append("")

    # Find best weighted and unweighted runs for summary
    weighted_runs = [r for r in track_b if WEIGHTED_CHECKPOINT in r.get("checkpoint", "")]
    unweighted_runs = [r for r in track_b if UNWEIGHTED_CHECKPOINT in r.get("checkpoint", "")]

    # Get best run metrics and checkpoint names directly from runs
    if weighted_runs:
        best_weighted = max(weighted_runs, key=lambda r: r.get("score", 0))
        w_mAP = (best_weighted.get("mAP") or 0) * 100
        w_N_top5 = (best_weighted.get("N_top5_mAP") or 0) * 100
        w_Nv_top5 = (best_weighted.get("Nv_top5_mAP") or 0) * 100
        w_ttc = best_weighted.get("ttc_mae_seconds") or 0
        w_run = best_weighted.get("run", "metrics_val_20251226_231700")
        weighted_ckpt_name = best_weighted.get("checkpoint", "trackB_best_mAP_0.3708_20251225_224220.pt")
        weighted_ckpt_name = best_weighted.get("checkpoint", "trackB_best_mAP_0.3708_20251225_224220.pt")
    else:
        w_mAP = w_N_top5 = w_Nv_top5 = w_ttc = 0
        w_run = "N/A"
        weighted_ckpt_name = "trackB_best_mAP_0.3708_*.pt"

    if unweighted_runs:
        best_unweighted = max(unweighted_runs, key=lambda r: r.get("score", 0))
        u_mAP = (best_unweighted.get("mAP") or 0) * 100
        u_N_top5 = (best_unweighted.get("N_top5_mAP") or 0) * 100
        u_Nv_top5 = (best_unweighted.get("Nv_top5_mAP") or 0) * 100
        u_ttc = best_unweighted.get("ttc_mae_seconds") or 0
        unweighted_ckpt_name = best_unweighted.get("checkpoint", "trackB_best_mAP_0.3904_20251220_024940.pt")
    else:
        u_mAP = u_N_top5 = u_Nv_top5 = u_ttc = 0
        unweighted_ckpt_name = "trackB_best_mAP_0.3904_*.pt"

    lines.append("| Decision | Checkpoint File | mAP | N_top5 | Nv_top5 | TTC_MAE | Best Run | Reason |")
    lines.append("|----------|----------------|-----|--------|---------|---------|----------|--------|")
    lines.append(f"| ✅ **USE** | `{weighted_ckpt_name}` | **{w_mAP:.2f}%** | **{w_N_top5:.2f}%** | **{w_Nv_top5:.2f}%** | **{w_ttc:.4f}s** | `{w_run[:40]}` | Class-weighted training, used in all Dec 26-30 experiments |")
    lines.append(f"| ❌ **SKIP** | `{unweighted_ckpt_name}` | {u_mAP:.2f}% | {u_N_top5:.2f}% | {u_Nv_top5:.2f}% | {u_ttc:.4f}s | - | Different training methodology, not used in efficiency experiments |")
    lines.append("| ❌ **SKIP** | Other checkpoints | Various | Various | Various | Various | - | Development/early training, not final quality |")
    lines.append("")

    # Add quick reference for error analysis
    lines.append("### Quick Reference for Error Analysis")
    lines.append("")
    lines.append("**Primary Checkpoint for Thesis:**")
    lines.append(f"- **File:** `{weighted_ckpt_name}`")
    lines.append(f"- **Location:** `local_extraction/runs/Track_B/checkpoints/{weighted_ckpt_name}`")
    lines.append(f"- **Best Evaluation Run:** `{w_run}`")
    lines.append(f"- **Performance:** mAP={w_mAP:.2f}%, N_top5={w_N_top5:.2f}%, Nv_top5={w_Nv_top5:.2f}%, TTC_MAE={w_ttc:.4f}s")
    lines.append("")
    lines.append("**For Error Analysis Command:**")
    lines.append("```bash")
    lines.append("python -m local_extraction.trackB.error_analysis \\")
    lines.append(f"  --metrics \"local_extraction/runs/Track_B/metrics/{w_run.replace('_summary', '')}.json\" \\")
    lines.append(f"  --predictions \"local_extraction/runs/Track_B/predictions/{w_run.replace('_summary.json', '').replace('_summary', '')}.csv\" \\")
    lines.append("  --output_dir \"local_extraction/runs/Track_B/error_analysis\"")
    lines.append("```")
    lines.append("")

    lines.append("### Run Count Summary")
    lines.append("")

    b_use = len([r for r in track_b if r.get("use_for_thesis")])
    b_skip = len([r for r in track_b if not r.get("use_for_thesis")])
    c_use = len([r for r in track_c if r.get("use_for_thesis")])
    c_skip = len([r for r in track_c if not r.get("use_for_thesis")])

    lines.append("| Track | Total Runs | ✅ Use | ❌ Skip |")
    lines.append("|-------|------------|--------|---------|")
    lines.append(f"| Track B | {len(track_b)} | {b_use} | {b_skip} |")
    lines.append(f"| Track C | {len(track_c)} | {c_use} | {c_skip} |")
    lines.append(f"| **Total** | {len(track_b) + len(track_c)} | {b_use + c_use} | {b_skip + c_skip} |")
    lines.append("")

    # Backbone and Pretraining Summary
    lines.append("### Backbone and Pretraining Summary")
    lines.append("")
    all_runs = track_b + track_c
    resnet_count = len([r for r in all_runs if r.get("backbone") == BACKBONE_RESNET18])
    videomae_count = len([r for r in all_runs if r.get("backbone") == BACKBONE_VIDEOMAE])
    exo_count = len([r for r in all_runs if r.get("pretraining") == PRETRAINING_EXO])
    ego_count = len([r for r in all_runs if r.get("pretraining") == PRETRAINING_EGO])

    lines.append("| Aspect | Category | Count | Description |")
    lines.append("|--------|----------|-------|-------------|")
    lines.append(f"| **Backbone** | ResNet18 | {resnet_count} | ImageNet-pretrained CNN (frozen feature extractor) |")
    lines.append(f"| **Backbone** | VideoMAE | {videomae_count} | Ego4D-pretrained video transformer |")
    lines.append(f"| **Pretraining** | Exo-Transfer | {exo_count} | Third-person pretrained (COCO/ImageNet) |")
    lines.append(f"| **Pretraining** | Ego-Only | {ego_count} | First-person pretrained (Ego4D) |")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Checkpoint Analysis
    lines.append("## Checkpoint Analysis")
    lines.append("")
    lines.append("### All Track B Checkpoints (Best mAP checkpoints)")
    lines.append("")
    lines.append("| Checkpoint | mAP | Date | Weighted | Category | Use? | Skip Reason |")
    lines.append("|------------|-----|------|----------|----------|------|-------------|")

    for ckpt in checkpoints[:15]:  # Top 15
        name = ckpt.get("checkpoint", "")[:50]
        mAP = ckpt.get("mAP", 0)
        date = ckpt.get("date", "")
        weighted = "✅" if ckpt.get("is_weighted") else "❌"
        cat = ckpt.get("category", "")
        use = "✅ **USE**" if ckpt.get("use_for_thesis") else "❌ Skip"
        reason = ckpt.get("skip_reason", "-") or "-"
        lines.append(f"| `{name}` | {mAP:.4f} | {date} | {weighted} | {cat} | {use} | {reason} |")

    lines.append("")

    # Key Checkpoint Comparison
    lines.append("### Key Checkpoint Comparison: 0.3708 vs 0.3904")
    lines.append("")
    lines.append("![Weighted vs Unweighted](weighted_vs_unweighted.png)")
    lines.append("")

    weighted_runs = [r for r in track_b if WEIGHTED_CHECKPOINT in r.get("checkpoint", "")]
    unweighted_runs = [r for r in track_b if UNWEIGHTED_CHECKPOINT in r.get("checkpoint", "")]

    if weighted_runs and unweighted_runs:
        best_w = max(weighted_runs, key=lambda r: r.get("score", 0))
        best_u = max(unweighted_runs, key=lambda r: r.get("score", 0))

        lines.append("| Metric | 0.3708 (Weighted) | 0.3904 (Unweighted) | Winner | Δ |")
        lines.append("|--------|-------------------|---------------------|--------|---|")

        metrics = [
            ("Composite Score", "score", False),
            ("mAP", "mAP", True),
            ("accuracy", "accuracy", True),
            ("TTC MAE (seconds)", "ttc_mae_seconds", False),
            ("", "", False),  # separator
            ("All_top5_mAP", "All_top5_mAP", True),
            ("N_top5_mAP", "N_top5_mAP", True),
            ("Nv_top5_mAP", "Nv_top5_mAP", True),
            ("N_delta_top5_mAP", "N_delta_top5_mAP", True),
            ("", "", False),  # separator
            ("All_top5_acc", "All_top5_acc", True),
            ("N_top5_acc", "N_top5_acc", True),
            ("Nv_top5_acc", "Nv_top5_acc", True),
            ("N_delta_top5_acc", "N_delta_top5_acc", True),
        ]

        for label, key, is_pct in metrics:
            if not label:  # separator row
                lines.append("|--------|-------------------|---------------------|--------|---|")
                continue
            w_val = best_w.get(key, 0) or 0
            u_val = best_u.get(key, 0) or 0

            # For TTC MAE, lower is better
            is_lower_better = "mae" in key.lower()

            if is_pct:
                w_str = f"{w_val*100:.2f}%"
                u_str = f"{u_val*100:.2f}%"
                delta = f"{(w_val - u_val)*100:+.2f}%"
            else:
                w_str = f"{w_val:.4f}"
                u_str = f"{u_val:.4f}"
                delta = f"{w_val - u_val:+.4f}"

            if is_lower_better:
                winner = "**0.3708**" if w_val < u_val else "0.3904"
            else:
                winner = "**0.3708**" if w_val > u_val else "0.3904"
            lines.append(f"| {label} | {w_str} | {u_str} | {winner} | {delta} |")

        lines.append("")

        # Analyze actual winner
        w_score = best_w.get("score", 0)
        u_score = best_u.get("score", 0)

        if w_score > u_score:
            lines.append("**Conclusion:** **0.3708 (Weighted) wins on composite score** because:")
            lines.append("- Better balanced noun/verb predictions through class weighting")
            lines.append("- Class weighting helps with imbalanced Ego4D distribution")
            lines.append("- More robust generalization")
        else:
            lines.append("**Note:** While 0.3904 shows higher raw scores in this evaluation, we use **0.3708 (Weighted)** for thesis because:")
            lines.append("- **Consistent checkpoint:** Dec 26-30 experiments all used 0.3708")
            lines.append("- **Class weighting:** Training with class weights provides better handling of imbalanced classes")
            lines.append("- **Fair comparison:** Comparing efficiency variants requires same baseline checkpoint")
            lines.append("- **Reproducibility:** 0.3708 represents our final training methodology with weighted loss")
        lines.append("")

    lines.append("### Checkpoint Timeline")
    lines.append("")
    lines.append("![Checkpoint Timeline](checkpoint_timeline.png)")
    lines.append("")

    lines.append("### Top Checkpoints: All Metrics Comparison")
    lines.append("")
    lines.append("![Track B Checkpoints Metrics](trackB_checkpoints_metrics.png)")
    lines.append("")
    lines.append("This plot shows all top-5 metrics (mAP, All_top5, N_top5, Nv_top5, N_delta) and accuracy as grouped bars,")
    lines.append("with the composite score as a red line. The thesis checkpoint (0.3708) is highlighted.")
    lines.append("")

    lines.append("### Top Runs: All Metrics Comparison")
    lines.append("")
    lines.append("![Track B Runs Metrics](trackB_runs_metrics.png)")
    lines.append("")
    lines.append("This plot shows top 12 Track B evaluation runs with all metrics as grouped bars.")
    lines.append("Background colors indicate: **Green=USE for thesis**, **Orange=0.3904 (skip)**, **Gray=Other (skip)**")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Track B Runs Detail
    lines.append("## Track B Runs Detail")
    lines.append("")
    lines.append("### Runs to USE for Thesis")
    lines.append("")

    use_b = [r for r in track_b if r.get("use_for_thesis")]
    if use_b:
        lines.append("| Run | Score | mAP | Acc | All_top5 | N_top5 | Nv_top5 | N_delta | TTC_MAE |")
        lines.append("|-----|-------|-----|-----|----------|--------|---------|---------|---------|")
        for r in sorted(use_b, key=lambda x: x.get("score", 0), reverse=True):
            run = r.get("run", "")[:35]
            score = r.get("score", 0)
            mAP = (r.get("mAP") or 0) * 100
            acc = (r.get("accuracy") or 0) * 100
            all5 = (r.get("All_top5_mAP") or 0) * 100
            n5 = (r.get("N_top5_mAP") or 0) * 100
            nv5 = (r.get("Nv_top5_mAP") or 0) * 100
            nd5 = (r.get("N_delta_top5_mAP") or 0) * 100
            ttc = r.get("ttc_mae_seconds") or 0
            lines.append(f"| `{run}` | {score:.4f} | {mAP:.1f}% | {acc:.1f}% | {all5:.2f}% | {n5:.2f}% | {nv5:.2f}% | {nd5:.2f}% | {ttc:.4f} |")
    else:
        lines.append("No runs to use.")
    lines.append("")

    lines.append("### Runs to SKIP")
    lines.append("")

    skip_b = [r for r in track_b if not r.get("use_for_thesis")]
    if skip_b:
        lines.append("| Run | Date | Checkpoint | Score | Category | Skip Reason |")
        lines.append("|-----|------|------------|-------|----------|-------------|")
        for r in sorted(skip_b, key=lambda x: x.get("date", ""), reverse=True)[:15]:
            run = r.get("run", "")[:40]
            date = r.get("date", "")
            ckpt = r.get("checkpoint", "")[:25]
            score = r.get("score", 0)
            cat = r.get("category", "")
            reason = r.get("skip_reason", "")[:40]
            lines.append(f"| `{run}` | {date} | `{ckpt}` | {score:.4f} | {cat} | {reason} |")
        if len(skip_b) > 15:
            lines.append(f"| ... | ... | ... | ... | ... | ({len(skip_b) - 15} more) |")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Track C Runs Detail
    lines.append("## Track C Runs Detail")
    lines.append("")
    lines.append("### Category Distribution")
    lines.append("")
    lines.append("![Track C Categories](trackC_categories.png)")
    lines.append("")

    # Count by category
    cat_counts = {}
    for r in track_c:
        cat = r.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    lines.append("| Category | Count | Use? | Description |")
    lines.append("|----------|-------|------|-------------|")

    cat_descs = {
        "weighted_efficiency_v5": ("✅", "Dec 30 efficiency experiments with frame/token pruning"),
        "weighted_baseline_v4": ("✅", "Dec 26 baseline runs with correct weighted checkpoint"),
        "weighted_early": ("❌", "Early weighted runs before final baseline established"),
        "unweighted_legacy": ("❌", "Runs using unweighted 0.3904 checkpoint"),
        "development": ("❌", "Development/early checkpoint runs"),
    }

    for cat, count in sorted(cat_counts.items()):
        use, desc = cat_descs.get(cat, ("❌", "Unknown category"))
        lines.append(f"| {cat} | {count} | {use} | {desc} |")

    lines.append("")

    lines.append("### Runs to USE for Thesis")
    lines.append("")

    use_c = [r for r in track_c if r.get("use_for_thesis")]
    if use_c:
        lines.append("| Config | Score | mAP | Acc | All_top5 | N_top5 | Nv_top5 | N_delta | Latency | TTC_MAE |")
        lines.append("|--------|-------|-----|-----|----------|--------|---------|---------|---------|---------|")
        for r in sorted(use_c, key=lambda x: x.get("score", 0), reverse=True):
            cfg = r.get("efficiency_config", "")[:25]
            score = r.get("score", 0)
            mAP = (r.get("mAP") or 0) * 100
            acc = (r.get("accuracy") or 0) * 100
            all5 = (r.get("All_top5_mAP") or 0) * 100
            n5 = (r.get("N_top5_mAP") or 0) * 100
            nv5 = (r.get("Nv_top5_mAP") or 0) * 100
            nd5 = (r.get("N_delta_top5_mAP") or 0) * 100
            lat = r.get("latency_ms_mean") or 0
            ttc = r.get("ttc_mae_seconds") or 0
            lines.append(f"| `{cfg}` | {score:.4f} | {mAP:.1f}% | {acc:.1f}% | {all5:.2f}% | {n5:.2f}% | {nv5:.2f}% | {nd5:.2f}% | {lat:.1f}ms | {ttc:.4f} |")
    else:
        lines.append("No runs to use.")
    lines.append("")

    lines.append("### Efficiency Configuration Comparison (Dec 30)")
    lines.append("")
    lines.append("![Efficiency Configs](efficiency_configs.png)")
    lines.append("")

    lines.append("### Runs to SKIP")
    lines.append("")

    skip_c = [r for r in track_c if not r.get("use_for_thesis")]
    if skip_c:
        lines.append("| Timestamp | Checkpoint | Score | Category | Skip Reason |")
        lines.append("|-----------|------------|-------|----------|-------------|")
        for r in sorted(skip_c, key=lambda x: x.get("date", ""), reverse=True)[:20]:
            ts = r.get("timestamp", "")
            ckpt = r.get("checkpoint", "")[:25]
            score = r.get("score", 0)
            cat = r.get("category", "")
            reason = r.get("skip_reason", "")[:45]
            lines.append(f"| {ts} | `{ckpt}` | {score:.4f} | {cat} | {reason} |")
        if len(skip_c) > 20:
            lines.append(f"| ... | ... | ... | ... | ({len(skip_c) - 20} more) |")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Timeline
    lines.append("## Run Timeline")
    lines.append("")
    lines.append("![Run Timeline](run_timeline.png)")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Backbone and Pretraining Analysis
    lines.append("## Backbone and Pretraining Analysis")
    lines.append("")
    lines.append("### Exo-Transfer vs Ego-Only Pretraining")
    lines.append("")
    lines.append("This thesis uses an **exo-transfer baseline** approach:")
    lines.append("")
    lines.append("| Component | Model | Pretrained On | Type |")
    lines.append("|-----------|-------|---------------|------|")
    lines.append("| **Track A Detector** | YOLOv8-s | COCO (80 classes) | Exo-Transfer |")
    lines.append("| **Track B Tokenizer** | ResNet18 | ImageNet (1000 classes) | Exo-Transfer |")
    lines.append("| **Track B VideoMAE** | VideoMAE-ViT | Ego4D (if used) | Ego-Only |")
    lines.append("")
    lines.append("### Distribution")
    lines.append("")
    lines.append("![Backbone and Pretraining](backbone_pretraining.png)")
    lines.append("")

    # Check if we have VideoMAE runs to compare
    all_runs = track_b + track_c
    videomae_runs = [r for r in all_runs if r.get("backbone") == BACKBONE_VIDEOMAE]
    resnet_runs = [r for r in all_runs if r.get("backbone") == BACKBONE_RESNET18]

    if videomae_runs and resnet_runs:
        lines.append("### Backbone Performance Comparison")
        lines.append("")
        lines.append("![Backbone Metrics Comparison](backbone_metrics_comparison.png)")
        lines.append("")

        best_videomae = max(videomae_runs, key=lambda r: r.get("score", 0))
        best_resnet = max(resnet_runs, key=lambda r: r.get("score", 0))

        lines.append("| Metric | ResNet18 (Exo) | VideoMAE (Ego) | Winner | Δ |")
        lines.append("|--------|----------------|----------------|--------|---|")

        metrics = [
            ("mAP", "mAP", True),
            ("Accuracy", "accuracy", True),
            ("All_top5_mAP", "All_top5_mAP", True),
            ("N_top5_mAP", "N_top5_mAP", True),
            ("Nv_top5_mAP", "Nv_top5_mAP", True),
            ("N_delta_top5_mAP", "N_delta_top5_mAP", True),
        ]

        for label, key, is_pct in metrics:
            r_val = best_resnet.get(key, 0) or 0
            v_val = best_videomae.get(key, 0) or 0

            if is_pct:
                r_str = f"{r_val*100:.2f}%"
                v_str = f"{v_val*100:.2f}%"
                delta = f"{(v_val - r_val)*100:+.2f}%"
            else:
                r_str = f"{r_val:.4f}"
                v_str = f"{v_val:.4f}"
                delta = f"{v_val - r_val:+.4f}"

            winner = "**VideoMAE**" if v_val > r_val else "ResNet18"
            lines.append(f"| {label} | {r_str} | {v_str} | {winner} | {delta} |")

        lines.append("")
    else:
        lines.append("### Note on Backbone Comparison")
        lines.append("")
        lines.append(f"Current runs primarily use **ResNet18 (Exo-Transfer)** backbone ({len(resnet_runs)} runs).")
        lines.append(f"VideoMAE (Ego-Only) runs: {len(videomae_runs)}")
        lines.append("")
        lines.append("The exo-transfer baseline demonstrates what can be achieved with off-the-shelf")
        lines.append("third-person pretrained models (COCO for detection, ImageNet for features).")
        lines.append("")
        lines.append("**Future work:** Compare with in-domain egocentric pretraining (VideoMAE on Ego4D).")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Track B vs Track C Comparison
    lines.append("## Track B vs Track C Comparison")
    lines.append("")
    lines.append("Track B provides the **baseline STA head** training, while Track C applies **efficiency optimizations** (token pruning, frame subsampling) on top of the best Track B checkpoint.")
    lines.append("")

    b_use_runs = [r for r in track_b if r.get("use_for_thesis")]
    c_use_runs = [r for r in track_c if r.get("use_for_thesis")]

    lines.append("### Suitable Runs Summary")
    lines.append("")
    lines.append(f"| Track | Suitable Runs | Purpose |")
    lines.append("|-------|---------------|---------|")
    lines.append(f"| **Track B** | {len(b_use_runs)} | Baseline STA head (full inference, no efficiency) |")
    lines.append(f"| **Track C** | {len(c_use_runs)} | Efficiency optimized (pruning, subsampling) |")
    lines.append("")

    if b_use_runs and c_use_runs:
        lines.append("### Best Run Comparison")
        lines.append("")
        lines.append("![Track B vs Track C Comparison](trackB_vs_trackC_comparison.png)")
        lines.append("")

        best_b = max(b_use_runs, key=lambda r: r.get("score", 0))
        best_c = max(c_use_runs, key=lambda r: r.get("score", 0))

        lines.append("| Metric | Track B (Baseline) | Track C (Efficiency) | Δ | Winner |")
        lines.append("|--------|-------------------|---------------------|---|--------|")

        metrics = [
            ("mAP", "mAP"),
            ("Accuracy", "accuracy"),
            ("All_top5_mAP", "All_top5_mAP"),
            ("N_top5_mAP", "N_top5_mAP"),
            ("Nv_top5_mAP", "Nv_top5_mAP"),
            ("N_delta_top5_mAP", "N_delta_top5_mAP"),
            ("TTC MAE (s)", "ttc_mae_seconds"),
        ]

        for label, key in metrics:
            b_val = best_b.get(key, 0) or 0
            c_val = best_c.get(key, 0) or 0

            if key == "ttc_mae_seconds":
                # Lower is better for MAE
                b_str = f"{b_val:.4f}"
                c_str = f"{c_val:.4f}"
                delta = f"{(c_val - b_val):+.4f}"
                winner = "**Track C**" if c_val < b_val else "Track B" if b_val < c_val else "Tie"
            else:
                b_str = f"{b_val*100:.2f}%"
                c_str = f"{c_val*100:.2f}%"
                delta = f"{(c_val - b_val)*100:+.2f}%"
                winner = "**Track C**" if c_val > b_val else "Track B" if b_val > c_val else "Tie"

            lines.append(f"| {label} | {b_str} | {c_str} | {delta} | {winner} |")

        lines.append("")

        # Add latency comparison if Track C has latency data
        c_latency = best_c.get("latency_ms_mean")
        if c_latency:
            lines.append("### Efficiency Trade-off")
            lines.append("")
            lines.append(f"Track C achieves efficiency with average latency of **{c_latency:.1f}ms** per sample.")
            lines.append("")

            # Calculate retention percentages
            b_mAP = (best_b.get("mAP") or 0)
            c_mAP = (best_c.get("mAP") or 0)
            if b_mAP > 0:
                retention = (c_mAP / b_mAP) * 100
                lines.append(f"- **mAP Retention:** {retention:.1f}% of baseline")
            lines.append("")
    else:
        lines.append("*Comparison requires suitable runs from both tracks.*")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Skip Reasons Summary
    lines.append("## Skip Reasons Summary")
    lines.append("")

    skip_reasons = {}
    for r in track_b + track_c:
        if not r.get("use_for_thesis"):
            reason = r.get("skip_reason", "Unknown")
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    lines.append("| Skip Reason | Count |")
    lines.append("|-------------|-------|")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {reason} | {count} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Recommendations
    lines.append("## Thesis Recommendations")
    lines.append("")
    lines.append("### For Track B Comparison:")
    lines.append("- Use **3 runs** with checkpoint `0.3708` (weighted)")
    lines.append("- Best run: `metrics_val_20251226_231700` (score: 0.7507)")
    lines.append("")
    lines.append("### For Track C Efficiency Analysis:")
    lines.append(f"- Use **{c_use} runs** from Dec 26-30 with weighted checkpoint")
    lines.append("- Focus on Dec 30 efficiency configurations for main thesis results")
    lines.append("- Key configurations: Baseline, Fr=8(uni), Fr=4+Tok=0.3")
    lines.append("")
    lines.append("### Why Use 0.3708 (Weighted) instead of 0.3904?")
    lines.append("1. **Consistency:** All Dec 26-30 experiments used 0.3708 checkpoint")
    lines.append("2. **Class weighting:** 0.3708 was trained WITH class weights (use_class_weights=True)")
    lines.append("3. **Fair comparison:** Efficiency comparisons require consistent baseline")
    lines.append("4. **Methodology:** 0.3708 represents our final training approach for thesis")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `run_categorization_report.py` on {timestamp}*")

    # Write report
    report_path = output_dir / "run_categorization_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Report saved to {report_path}")

    return report_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("Run Categorization and Visualization")
    print("=" * 60)

    print("\n[1/5] Loading Track B metrics...")
    track_b = load_track_b_metrics()
    print(f"      Found {len(track_b)} runs")

    print("\n[2/5] Loading Track C metrics...")
    track_c = load_track_c_metrics()
    print(f"      Found {len(track_c)} runs")

    print("\n[3/5] Loading checkpoints...")
    checkpoints = load_checkpoints()
    print(f"      Found {len(checkpoints)} checkpoints")

    print("\n[4/5] Creating plots...")
    create_plots(track_b, track_c, checkpoints, OUTPUT_DIR)

    print("\n[5/5] Generating report...")
    report_path = generate_report(track_b, track_c, checkpoints, OUTPUT_DIR)

    # Print summary
    b_use = len([r for r in track_b if r.get("use_for_thesis")])
    c_use = len([r for r in track_c if r.get("use_for_thesis")])

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nTrack B: {b_use} USE / {len(track_b) - b_use} SKIP")
    print(f"Track C: {c_use} USE / {len(track_c) - c_use} SKIP")
    print(f"\nOutput: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
