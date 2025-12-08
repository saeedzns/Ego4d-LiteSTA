#!/usr/bin/env python3
"""
Track B Ablation Comparison Tool

Compare evaluation results across different configurations (hotspot on/off, CLIP on/off).
Parses recent metrics_val_*_summary.json files and generates comparison tables and plots.

Usage:
  python local_extraction/trackB/trackB_ablation_compare.py
  python local_extraction/trackB/trackB_ablation_compare.py --last 8
  python local_extraction/trackB/trackB_ablation_compare.py --out_dir local_extraction/runs/Track_B/ablation_plots
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class EvalResult:
    """Single evaluation result."""
    timestamp: str
    checkpoint: str
    backbone: str  # resnet18 or videomae_ego
    hotspot: bool
    clip: bool
    mAP: float
    accuracy: float
    ttc_mae: float
    N_mAP: float
    Nv_mAP: float
    N_delta_mAP: float
    All_mAP: float
    N_top5_mAP: float
    Nv_top5_mAP: float
    N_delta_top5_mAP: float
    All_top5_mAP: float


def infer_backbone(summary: Dict[str, Any]) -> str:
    """Infer backbone type from summary JSON."""
    train_config = summary.get("metrics", {}).get("train_config", {})
    if not train_config:
        train_config = summary.get("train_config", {})
    
    video_backbone = train_config.get("video_backbone", "")
    tokens_root = train_config.get("tokens_root", "")
    
    if video_backbone == "videomae_ego" or (tokens_root and "videomae" in tokens_root.lower()):
        return "videomae_ego"
    return "resnet18"


def infer_hotspot_clip(summary: Dict[str, Any]) -> tuple[bool, bool]:
    """Infer hotspot and CLIP settings from summary JSON."""
    eval_config = summary.get("eval_config", {})
    hotspot = eval_config.get("use_hotspot_priors", False)
    clip = eval_config.get("use_clip_rerank", False)
    return hotspot, clip


def load_summaries(metrics_dir: Path, last_n: Optional[int] = None) -> List[EvalResult]:
    """Load summary JSON files and extract results."""
    files = sorted(metrics_dir.glob("metrics_val_*_summary.json"), key=lambda p: p.stat().st_mtime)
    
    if last_n is not None:
        files = files[-last_n:]
    
    results: List[EvalResult] = []
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ablation] Failed to load {path}: {e}")
            continue
        
        metrics = obj.get("metrics", obj)  # Handle both nested and flat formats
        eval_config = obj.get("eval_config", {})
        
        backbone = infer_backbone(obj)
        hotspot = eval_config.get("use_hotspot_priors", False)
        clip = eval_config.get("use_clip_rerank", False)
        
        result = EvalResult(
            timestamp=metrics.get("timestamp", path.stem),
            checkpoint=Path(metrics.get("checkpoint", "")).name,
            backbone=backbone,
            hotspot=hotspot,
            clip=clip,
            mAP=metrics.get("mAP", 0.0),
            accuracy=metrics.get("accuracy", 0.0),
            ttc_mae=metrics.get("ttc_mae_seconds", 0.0),
            N_mAP=metrics.get("N_mAP", 0.0),
            Nv_mAP=metrics.get("Nv_mAP", 0.0),
            N_delta_mAP=metrics.get("N_delta_mAP", 0.0),
            All_mAP=metrics.get("All_mAP", 0.0),
            N_top5_mAP=metrics.get("N_top5_mAP", 0.0),
            Nv_top5_mAP=metrics.get("Nv_top5_mAP", 0.0),
            N_delta_top5_mAP=metrics.get("N_delta_top5_mAP", 0.0),
            All_top5_mAP=metrics.get("All_top5_mAP", 0.0),
        )
        results.append(result)
    
    return results


def print_comparison_table(results: List[EvalResult]) -> None:
    """Print comparison table to console."""
    print("\n" + "=" * 160)
    print("ABLATION COMPARISON TABLE - Core Metrics (%)")
    print("=" * 160)
    
    header = f"{'Backbone':<12} {'Hotspot':<8} {'CLIP':<6} {'mAP':>8} {'Acc':>8} {'TTC MAE':>10} {'N_mAP':>8} {'Nv_mAP':>8} {'N+δ':>8} {'All':>8} {'Checkpoint':<45}"
    print(header)
    print("-" * 160)
    
    for r in results:
        hotspot_str = "✓" if r.hotspot else "✗"
        clip_str = "✓" if r.clip else "✗"
        # Extract short checkpoint name
        ckpt_short = Path(r.checkpoint).stem if r.checkpoint else "N/A"
        if len(ckpt_short) > 42:
            ckpt_short = ckpt_short[:42] + "..."
        row = f"{r.backbone:<12} {hotspot_str:<8} {clip_str:<6} {r.mAP*100:>8.2f} {r.accuracy*100:>8.2f} {r.ttc_mae:>10.4f} {r.N_mAP*100:>8.2f} {r.Nv_mAP*100:>8.2f} {r.N_delta_mAP*100:>8.2f} {r.All_mAP*100:>8.2f} {ckpt_short:<45}"
        print(row)
    
    print("=" * 160)
    
    # Print Top-5 metrics table
    print("\n" + "=" * 160)
    print("ABLATION COMPARISON TABLE - Top-5 Metrics (%)")
    print("=" * 160)
    
    header_top5 = f"{'Backbone':<12} {'Hotspot':<8} {'CLIP':<6} {'N_top5':>10} {'Nv_top5':>10} {'N+δ_top5':>10} {'All_top5':>10} {'Checkpoint':<45}"
    print(header_top5)
    print("-" * 160)
    
    for r in results:
        hotspot_str = "✓" if r.hotspot else "✗"
        clip_str = "✓" if r.clip else "✗"
        ckpt_short = Path(r.checkpoint).stem if r.checkpoint else "N/A"
        if len(ckpt_short) > 42:
            ckpt_short = ckpt_short[:42] + "..."
        row = f"{r.backbone:<12} {hotspot_str:<8} {clip_str:<6} {r.N_top5_mAP*100:>10.2f} {r.Nv_top5_mAP*100:>10.2f} {r.N_delta_top5_mAP*100:>10.2f} {r.All_top5_mAP*100:>10.2f} {ckpt_short:<45}"
        print(row)
    
    print("=" * 160)
    
    # Print checkpoint mapping table
    print("\n" + "=" * 120)
    print("CHECKPOINT MAPPING (Config → Checkpoint File)")
    print("=" * 120)
    print(f"{'Backbone':<14} {'Hotspot':<8} {'CLIP':<6} {'Timestamp':<18} {'Checkpoint File'}")
    print("-" * 120)
    for r in results:
        hotspot_str = "✓" if r.hotspot else "✗"
        clip_str = "✓" if r.clip else "✗"
        ckpt_name = Path(r.checkpoint).name if r.checkpoint else "N/A"
        print(f"{r.backbone:<14} {hotspot_str:<8} {clip_str:<6} {r.timestamp:<18} {ckpt_name}")
    print("=" * 120)


def group_by_config(results: List[EvalResult]) -> Dict[tuple, EvalResult]:
    """Group results by (backbone, hotspot, clip) config, keeping the latest."""
    grouped: Dict[tuple, EvalResult] = {}
    for r in results:
        key = (r.backbone, r.hotspot, r.clip)
        # Keep the latest result for each config
        if key not in grouped or r.timestamp > grouped[key].timestamp:
            grouped[key] = r
    return grouped


def plot_ablation(results: List[EvalResult], out_dir: Path, group_configs: bool = True) -> None:
    """Generate ablation comparison plots.
    
    Args:
        results: List of evaluation results
        out_dir: Output directory for plots
        group_configs: If True, group by config (default). If False, show all checkpoints individually.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[ablation] matplotlib/numpy not available; skipping plots.")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if group_configs:
        # Group by config (original behavior)
        grouped = group_by_config(results)
        _plot_grouped(grouped, results, out_dir, plt, np)
    else:
        # Show all checkpoints individually
        _plot_all_checkpoints(results, out_dir, plt, np)


def _plot_all_checkpoints(results: List[EvalResult], out_dir: Path, plt, np) -> None:
    """Plot all checkpoints individually (--all mode)."""
    
    # Sort by timestamp
    results_sorted = sorted(results, key=lambda r: r.timestamp)
    
    # Create labels from checkpoint names (shortened)
    labels = []
    for r in results_sorted:
        ckpt = Path(r.checkpoint).stem if r.checkpoint else "N/A"
        # Shorten: keep backbone indicator and mAP value if present
        if "mAP" in ckpt:
            # e.g., trackB_best_mAP_0.3580_20251206_083053 -> 0.3580 (R) or (V)
            parts = ckpt.split("_")
            mAP_idx = parts.index("mAP") if "mAP" in parts else -1
            if mAP_idx >= 0 and mAP_idx + 1 < len(parts):
                mAP_val = parts[mAP_idx + 1]
                backbone_tag = "R" if r.backbone == "resnet18" else "V"
                hotspot_tag = "H" if r.hotspot else ""
                clip_tag = "C" if r.clip else ""
                labels.append(f"{mAP_val}({backbone_tag}{hotspot_tag}{clip_tag})")
            else:
                labels.append(ckpt[:20])
        else:
            labels.append(ckpt[:20])
    
    # Plot 1: All metrics line plot over checkpoints
    metrics_to_plot = [
        ('mAP', 'mAP (%)'),
        ('N_mAP', 'N_mAP (%)'),
        ('Nv_mAP', 'N+V_mAP (%)'),
        ('N_delta_mAP', 'N+δ_mAP (%)'),
        ('All_mAP', 'All_mAP (%)'),
        ('N_top5_mAP', 'N_top5 (%)'),
        ('Nv_top5_mAP', 'N+V_top5 (%)'),
        ('N_delta_top5_mAP', 'N+δ_top5 (%)'),
        ('All_top5_mAP', 'All_top5 (%)'),
    ]
    
    n_metrics = len(metrics_to_plot)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes_flat = axes.flatten()
    
    x = np.arange(len(results_sorted))
    
    # Color by backbone
    colors = ['steelblue' if r.backbone == 'resnet18' else 'coral' for r in results_sorted]
    
    for ax, (metric_key, ylabel) in zip(axes_flat, metrics_to_plot):
        vals = [getattr(r, metric_key) * 100 for r in results_sorted]
        
        # Plot bars with backbone-specific colors
        bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=6)
    
    # Hide unused axes
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='ResNet18'),
                       Patch(facecolor='coral', label='VideoMAE-Ego')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.suptitle('All Checkpoints - Metrics Comparison\n(R=ResNet18, V=VideoMAE, H=Hotspot, C=CLIP)', fontsize=12)
    plt.tight_layout()
    out_path = out_dir / "all_checkpoints_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ablation] Saved all checkpoints plot → {out_path}")
    
    # Plot 2: Timeline view (mAP over time)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for backbone, color, marker in [("resnet18", "steelblue", "o"), ("videomae_ego", "coral", "s")]:
        backbone_results = [r for r in results_sorted if r.backbone == backbone]
        if backbone_results:
            x_vals = [results_sorted.index(r) for r in backbone_results]
            mAP_vals = [r.mAP * 100 for r in backbone_results]
            ax.scatter(x_vals, mAP_vals, c=color, marker=marker, s=100, label=backbone, zorder=3)
            ax.plot(x_vals, mAP_vals, c=color, alpha=0.5, linestyle='--', linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('mAP (%)')
    ax.set_xlabel('Checkpoint (chronological)')
    ax.set_title('mAP Timeline - All Checkpoints')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / "all_checkpoints_timeline.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ablation] Saved timeline plot → {out_path}")


def _plot_grouped(grouped: Dict[tuple, EvalResult], results: List[EvalResult], out_dir: Path, plt, np) -> None:
    """Plot grouped by config (original behavior)."""
    
    # Separate by backbone
    resnet_results = {k: v for k, v in grouped.items() if k[0] == "resnet18"}
    videomae_results = {k: v for k, v in grouped.items() if k[0] == "videomae_ego"}
    
    # Plot 1: Bar chart comparing mAP across configs
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    configs = [
        (False, False, "Base"),
        (True, False, "+Hotspot"),
        (False, True, "+CLIP"),
        (True, True, "+Both"),
    ]
    
    for ax, (backbone_results, title) in zip(axes, [(resnet_results, "ResNet18"), (videomae_results, "VideoMAE-Ego")]):
        if not backbone_results:
            ax.set_visible(False)
            continue
        
        x_labels = []
        mAP_values = []
        N_mAP_values = []
        Nv_mAP_values = []
        
        backbone_name = title.lower().replace("-", "_")
        for hotspot, clip, label in configs:
            key = (backbone_name if backbone_name == "resnet18" else "videomae_ego", hotspot, clip)
            if key in backbone_results:
                r = backbone_results[key]
                x_labels.append(label)
                mAP_values.append(r.mAP * 100)
                N_mAP_values.append(r.N_mAP * 100)
                Nv_mAP_values.append(r.Nv_mAP * 100)
        
        if not x_labels:
            ax.text(0.5, 0.5, f"No data for {title}", ha='center', va='center', transform=ax.transAxes)
            continue
        
        x = np.arange(len(x_labels))
        width = 0.25
        
        bars1 = ax.bar(x - width, mAP_values, width, label='mAP', color='steelblue')
        bars2 = ax.bar(x, N_mAP_values, width, label='N_mAP', color='forestgreen')
        bars3 = ax.bar(x + width, Nv_mAP_values, width, label='N+V_mAP', color='coral')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('mAP (%)')
        ax.set_title(f'{title} Ablation')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    out_path = out_dir / "ablation_mAP_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ablation] Saved mAP comparison → {out_path}")
    
    # Plot 2: Top-5 metrics comparison (all 4 top-5 metrics)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, (backbone_results, title) in zip(axes, [(resnet_results, "ResNet18"), (videomae_results, "VideoMAE-Ego")]):
        if not backbone_results:
            ax.set_visible(False)
            continue
        
        x_labels = []
        N_top5_values = []
        Nv_top5_values = []
        Nd_top5_values = []  # N+δ top5
        All_top5_values = []
        
        backbone_name = title.lower().replace("-", "_")
        for hotspot, clip, label in configs:
            key = (backbone_name if backbone_name == "resnet18" else "videomae_ego", hotspot, clip)
            if key in backbone_results:
                r = backbone_results[key]
                x_labels.append(label)
                N_top5_values.append(r.N_top5_mAP * 100)
                Nv_top5_values.append(r.Nv_top5_mAP * 100)
                Nd_top5_values.append(r.N_delta_top5_mAP * 100)
                All_top5_values.append(r.All_top5_mAP * 100)
        
        if not x_labels:
            ax.text(0.5, 0.5, f"No data for {title}", ha='center', va='center', transform=ax.transAxes)
            continue
        
        x = np.arange(len(x_labels))
        width = 0.2
        
        bars1 = ax.bar(x - 1.5*width, N_top5_values, width, label='N_top5', color='steelblue')
        bars2 = ax.bar(x - 0.5*width, Nv_top5_values, width, label='N+V_top5', color='forestgreen')
        bars3 = ax.bar(x + 0.5*width, Nd_top5_values, width, label='N+δ_top5', color='orange')
        bars4 = ax.bar(x + 1.5*width, All_top5_values, width, label='All_top5', color='coral')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Top-5 mAP (%)')
        ax.set_title(f'{title} Top-5 Ablation')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    out_path = out_dir / "ablation_top5_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ablation] Saved Top-5 comparison → {out_path}")
    
    # Plot 3: Heatmap of all metrics
    fig, ax = plt.subplots(figsize=(14, 10))
    
    all_configs = []
    all_metrics = []
    metric_names = ['mAP', 'Accuracy', 'N_mAP', 'N+V_mAP', 'N+δ_mAP', 'All_mAP', 'N_top5', 'Nv_top5', 'N+δ_top5', 'All_top5']
    
    for (backbone, hotspot, clip), r in sorted(grouped.items()):
        hotspot_str = "H+" if hotspot else "H-"
        clip_str = "C+" if clip else "C-"
        config_label = f"{backbone[:6]}_{hotspot_str}{clip_str}"
        all_configs.append(config_label)
        all_metrics.append([
            r.mAP * 100,
            r.accuracy * 100,
            r.N_mAP * 100,
            r.Nv_mAP * 100,
            r.N_delta_mAP * 100,
            r.All_mAP * 100,
            r.N_top5_mAP * 100,
            r.Nv_top5_mAP * 100,
            r.N_delta_top5_mAP * 100,
            r.All_top5_mAP * 100,
        ])
    
    if all_metrics:
        data = np.array(all_metrics)
        im = ax.imshow(data.T, aspect='auto', cmap='RdYlGn')
        
        ax.set_xticks(range(len(all_configs)))
        ax.set_xticklabels(all_configs, rotation=45, ha='right')
        ax.set_yticks(range(len(metric_names)))
        ax.set_yticklabels(metric_names)
        
        # Add text annotations
        for i in range(len(metric_names)):
            for j in range(len(all_configs)):
                text = ax.text(j, i, f'{data[j, i]:.1f}',
                              ha='center', va='center', color='black', fontsize=9)
        
        ax.set_title('Ablation Results Heatmap (% values)')
        fig.colorbar(im, ax=ax, label='Metric Value (%)')
    
    plt.tight_layout()
    out_path = out_dir / "ablation_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ablation] Saved heatmap → {out_path}")
    
    # Plot 4: Line plot showing metric trends across configs (3x3 grid for all key metrics)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    config_labels = ["Base", "+Hotspot", "+CLIP", "+Both"]
    config_keys = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    
    # Define all metrics to plot
    metrics_to_plot = [
        (0, 0, 'mAP', 'mAP (%)'),
        (0, 1, 'N_mAP', 'N_mAP (%)'),
        (0, 2, 'Nv_mAP', 'N+V_mAP (%)'),
        (1, 0, 'N_delta_mAP', 'N+δ_mAP (%)'),
        (1, 1, 'All_mAP', 'All_mAP (%)'),
        (1, 2, 'N_top5_mAP', 'N_top5 (%)'),
        (2, 0, 'Nv_top5_mAP', 'N+V_top5 (%)'),
        (2, 1, 'N_delta_top5_mAP', 'N+δ_top5 (%)'),
        (2, 2, 'All_top5_mAP', 'All_top5 (%)'),
    ]
    
    for row, col, metric_key, ylabel in metrics_to_plot:
        ax = axes[row, col]
        
        for backbone_name, color, marker in [("resnet18", "steelblue", "o"), ("videomae_ego", "coral", "s")]:
            x_vals = []
            vals = []
            for i, (hotspot, clip) in enumerate(config_keys):
                key = (backbone_name, hotspot, clip)
                if key in grouped:
                    x_vals.append(i)
                    vals.append(getattr(grouped[key], metric_key) * 100)
            if x_vals:
                ax.plot(x_vals, vals, marker=marker, color=color, linewidth=2, markersize=8, label=backbone_name)
        
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} Across Configs')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / "ablation_line_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ablation] Saved line plot → {out_path}")


def save_comparison_csv(results: List[EvalResult], out_dir: Path) -> None:
    """Save comparison table as CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_comparison.csv"
    
    with open(out_path, "w", encoding="utf-8") as f:
        header = "timestamp,checkpoint,backbone,hotspot,clip,mAP,accuracy,ttc_mae,N_mAP,Nv_mAP,N_delta_mAP,All_mAP,N_top5_mAP,Nv_top5_mAP,N_delta_top5_mAP,All_top5_mAP"
        f.write(header + "\n")
        for r in results:
            row = f"{r.timestamp},{r.checkpoint},{r.backbone},{r.hotspot},{r.clip},{r.mAP:.6f},{r.accuracy:.6f},{r.ttc_mae:.6f},{r.N_mAP:.6f},{r.Nv_mAP:.6f},{r.N_delta_mAP:.6f},{r.All_mAP:.6f},{r.N_top5_mAP:.6f},{r.Nv_top5_mAP:.6f},{r.N_delta_top5_mAP:.6f},{r.All_top5_mAP:.6f}"
            f.write(row + "\n")
    
    print(f"[ablation] Saved CSV → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Track B Ablation Comparison Tool")
    parser.add_argument("--metrics_dir", type=str, default="local_extraction/runs/Track_B/metrics",
                       help="Directory containing metrics_val_*_summary.json files")
    parser.add_argument("--out_dir", type=str, default="local_extraction/runs/Track_B/ablation_plots",
                       help="Output directory for plots and CSV")
    parser.add_argument("--last", type=int, default=None,
                       help="Only consider the last N evaluation runs")
    parser.add_argument("--all", action="store_true",
                       help="Include ALL checkpoints (don't group by config, show each eval separately)")
    args = parser.parse_args()
    
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    
    if not metrics_dir.exists():
        print(f"[ablation] Metrics directory not found: {metrics_dir}")
        return
    
    results = load_summaries(metrics_dir, args.last)
    
    if not results:
        print(f"[ablation] No summary files found in {metrics_dir}")
        return
    
    print(f"[ablation] Loaded {len(results)} evaluation results")
    if args.all:
        print("[ablation] --all mode: showing ALL checkpoints (not grouped by config)")
    
    # Print table
    print_comparison_table(results)
    
    # Save CSV
    save_comparison_csv(results, out_dir)
    
    # Generate plots (pass --all flag)
    plot_ablation(results, out_dir, group_configs=not args.all)
    
    # Print best configs
    print("\n" + "=" * 60)
    print("BEST CONFIGURATIONS BY METRIC")
    print("=" * 60)
    
    grouped = group_by_config(results)
    if grouped:
        best_mAP = max(grouped.values(), key=lambda r: r.mAP)
        best_N_mAP = max(grouped.values(), key=lambda r: r.N_mAP)
        best_Nv_mAP = max(grouped.values(), key=lambda r: r.Nv_mAP)
        best_N_top5 = max(grouped.values(), key=lambda r: r.N_top5_mAP)
        best_Nv_top5 = max(grouped.values(), key=lambda r: r.Nv_top5_mAP)
        best_All_top5 = max(grouped.values(), key=lambda r: r.All_top5_mAP)
        
        print(f"Best mAP:       {best_mAP.backbone} | Hotspot={best_mAP.hotspot} | CLIP={best_mAP.clip} → {best_mAP.mAP*100:.2f}%")
        print(f"Best N_mAP:     {best_N_mAP.backbone} | Hotspot={best_N_mAP.hotspot} | CLIP={best_N_mAP.clip} → {best_N_mAP.N_mAP*100:.2f}%")
        print(f"Best N+V:       {best_Nv_mAP.backbone} | Hotspot={best_Nv_mAP.hotspot} | CLIP={best_Nv_mAP.clip} → {best_Nv_mAP.Nv_mAP*100:.2f}%")
        print(f"Best N_top5:    {best_N_top5.backbone} | Hotspot={best_N_top5.hotspot} | CLIP={best_N_top5.clip} → {best_N_top5.N_top5_mAP*100:.2f}%")
        print(f"Best Nv_top5:   {best_Nv_top5.backbone} | Hotspot={best_Nv_top5.hotspot} | CLIP={best_Nv_top5.clip} → {best_Nv_top5.Nv_top5_mAP*100:.2f}%")
        print(f"Best All_top5:  {best_All_top5.backbone} | Hotspot={best_All_top5.hotspot} | CLIP={best_All_top5.clip} → {best_All_top5.All_top5_mAP*100:.2f}%")


if __name__ == "__main__":
    main()
