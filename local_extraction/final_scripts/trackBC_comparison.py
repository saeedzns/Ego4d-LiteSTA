#!/usr/bin/env python3
"""
Track B vs Track C Comparison Script
=====================================
Compares results using only the weighted-class checkpoint (0.3708).
Generates trade-off analysis, comparison plots, and detailed MD report.

Table format follows summarize_best_runs.py style.
Uses composite score for ranking: mAP + N_top5 + Nv_top5 + N_delta_top5 + All_top5

Usage:
    python local_extraction/final_scripts/trackBC_comparison.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Iterable
import matplotlib.pyplot as plt
import numpy as np

# Configuration
WORKSPACE = Path(r"D:\Thesis\Ego4d-LiteSTA")
TRACK_B_METRICS = WORKSPACE / "local_extraction" / "runs" / "Track_B" / "metrics"
TRACK_C_METRICS = WORKSPACE / "local_extraction" / "runs" / "Track_C" / "metrics"
OUTPUT_DIR = WORKSPACE / "local_extraction" / "final_scripts" / "comparison_results"
WEIGHTED_CHECKPOINT = "0.3708"  # Only use weighted-class checkpoint

# Scoring weights (same as summarize_best_runs.py)
SCORE_KEYS = ["mAP", "N_top5_mAP", "Nv_top5_mAP", "N_delta_top5_mAP", "All_top5_mAP"]
SCORE_WEIGHTS = {k: 1.0 for k in SCORE_KEYS}


def _fmt(v: Any, ndigits: int = 4) -> str:
    """Format value for table display."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def _pct(v: Any, ndigits: int = 2) -> str:
    """Format as percentage."""
    if v is None:
        return ""
    try:
        return f"{float(v) * 100:.{ndigits}f}%"
    except (ValueError, TypeError):
        return str(v)


def _basename(path_str: Optional[str]) -> str:
    """Get basename from path string."""
    if not path_str:
        return ""
    return Path(path_str).name


def _score_row(row: Dict[str, Any], keys: Iterable[str], weights: Dict[str, float]) -> float:
    """Calculate composite score."""
    total = 0.0
    for k in keys:
        w = weights.get(k, 1.0)
        v = row.get(k, 0.0) or 0.0
        try:
            total += w * float(v)
        except (ValueError, TypeError):
            pass
    return total


def _write_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]], 
                 pct_cols: Optional[List[str]] = None) -> List[str]:
    """Write markdown table."""
    if pct_cols is None:
        pct_cols = []
    lines = []
    header = "| " + " | ".join(c[0] for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = []
        for label, key in columns:
            val = r.get(key)
            if key in pct_cols:
                cells.append(_pct(val))
            else:
                cells.append(_fmt(val))
        line = "| " + " | ".join(cells) + " |"
        lines.append(line)
    return lines


def load_track_b_metrics() -> List[Dict[str, Any]]:
    """Load all Track B metrics with weighted checkpoint."""
    rows = []
    if not TRACK_B_METRICS.exists():
        print(f"[WARN] Track B metrics dir not found: {TRACK_B_METRICS}")
        return rows
    
    for f in sorted(TRACK_B_METRICS.glob("metrics_val_*_summary.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            train = metrics.get("train_config", {})
            
            # Only include weighted checkpoint runs
            ckpt = _basename(metrics.get("checkpoint", ""))
            if WEIGHTED_CHECKPOINT not in ckpt:
                continue
            
            # Must have use_class_weights
            if not train.get("use_class_weights"):
                continue
            
            row = {
                "run": f.name,
                "timestamp": metrics.get("timestamp", ""),
                "checkpoint": ckpt,
                "mAP": metrics.get("mAP"),
                "accuracy": metrics.get("accuracy"),
                "N_mAP": metrics.get("N_mAP"),
                "Nv_mAP": metrics.get("Nv_mAP"),
                "N_delta_mAP": metrics.get("N_delta_mAP"),
                "All_mAP": metrics.get("All_mAP"),
                "N_top5_mAP": metrics.get("N_top5_mAP"),
                "Nv_top5_mAP": metrics.get("Nv_top5_mAP"),
                "N_delta_top5_mAP": metrics.get("N_delta_top5_mAP"),
                "All_top5_mAP": metrics.get("All_top5_mAP"),
                "N_top5_acc": metrics.get("N_top5_acc"),
                "Nv_top5_acc": metrics.get("Nv_top5_acc"),
                "N_delta_top5_acc": metrics.get("N_delta_top5_acc"),
                "All_top5_acc": metrics.get("All_top5_acc"),
                "ttc_mae_seconds": metrics.get("ttc_mae_seconds"),
                "video_backbone": train.get("video_backbone"),
                "tokens_root": train.get("tokens_root"),
                "loss_w_noun": train.get("loss_w_noun"),
                "loss_w_verb": train.get("loss_w_verb"),
                "class_weight_alpha": train.get("class_weight_alpha"),
                "use_class_weights": train.get("use_class_weights"),
                "source": "trackB",
            }
            row["score"] = _score_row(row, SCORE_KEYS, SCORE_WEIGHTS)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to parse {f.name}: {e}")
    
    return rows


def load_track_c_metrics() -> List[Dict[str, Any]]:
    """Load all Track C metrics with weighted checkpoint."""
    rows = []
    if not TRACK_C_METRICS.exists():
        print(f"[WARN] Track C metrics dir not found: {TRACK_C_METRICS}")
        return rows
    
    for f in sorted(TRACK_C_METRICS.glob("trackC_val_rate*.json")):
        if "_summary" in f.name:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            
            # Only include weighted checkpoint runs
            ckpt = _basename(data.get("checkpoint", ""))
            if WEIGHTED_CHECKPOINT not in ckpt:
                continue
            
            # Build efficiency config string
            eff_parts = []
            if data.get("rgtp_enabled") and data.get("rgtp_rate_request", 0) > 0:
                eff_parts.append(f"RGTP={data.get('rgtp_rate_request'):.1f}")
            if data.get("efficiency_frame_subsample_enabled"):
                frames = data.get("efficiency_frame_subsample_keep_frames")
                strat = data.get("efficiency_frame_subsample_strategy", "uniform")[:3]
                if frames:
                    eff_parts.append(f"Fr={frames}({strat})")
            if data.get("efficiency_token_prune_enabled"):
                rate = data.get("efficiency_token_prune_rate")
                if rate:
                    eff_parts.append(f"Tok={rate:.1f}")
            eff_config = " + ".join(eff_parts) if eff_parts else "Baseline"
            
            row = {
                "run": f.name,
                "timestamp": f.stem.split("_")[-1] if "_" in f.stem else "",
                "checkpoint": ckpt,
                "efficiency_config": eff_config,
                "mAP": data.get("mAP"),
                "accuracy": data.get("accuracy"),
                "N_mAP": data.get("N_mAP"),
                "Nv_mAP": data.get("Nv_mAP"),
                "N_delta_mAP": data.get("N_delta_mAP"),
                "All_mAP": data.get("All_mAP"),
                "N_top5_mAP": data.get("N_top5_mAP"),
                "Nv_top5_mAP": data.get("Nv_top5_mAP"),
                "N_delta_top5_mAP": data.get("N_delta_top5_mAP"),
                "All_top5_mAP": data.get("All_top5_mAP"),
                "N_top5_acc": data.get("N_top5_acc"),
                "Nv_top5_acc": data.get("Nv_top5_acc"),
                "N_delta_top5_acc": data.get("N_delta_top5_acc"),
                "All_top5_acc": data.get("All_top5_acc"),
                "ttc_mae_seconds": data.get("ttc_mae_seconds"),
                "latency_ms_mean": data.get("latency_ms_mean"),
                "latency_ms_median": data.get("latency_ms_median"),
                "latency_ms_p95": data.get("latency_ms_p95"),
                "throughput_samples_per_s": data.get("throughput_samples_per_s"),
                "rgtp_rate_request": data.get("rgtp_rate_request"),
                "rgtp_enabled": data.get("rgtp_enabled"),
                "rgtp_mean_fraction_pruned": data.get("rgtp_mean_fraction_pruned"),
                "frame_subsample_enabled": data.get("efficiency_frame_subsample_enabled"),
                "frame_subsample_keep_frames": data.get("efficiency_frame_subsample_keep_frames"),
                "frame_subsample_strategy": data.get("efficiency_frame_subsample_strategy"),
                "token_prune_enabled": data.get("efficiency_token_prune_enabled"),
                "token_prune_rate": data.get("efficiency_token_prune_rate"),
                "source": "trackC",
            }
            row["score"] = _score_row(row, SCORE_KEYS, SCORE_WEIGHTS)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to parse {f.name}: {e}")
    
    return rows


def find_best_by_metric(rows: List[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    """Find best row by a specific metric."""
    valid = [r for r in rows if r.get(metric) is not None]
    if not valid:
        return None
    return max(valid, key=lambda r: r.get(metric, 0))


def find_pareto_optimal(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find Pareto-optimal runs (score vs latency trade-off)."""
    valid = [r for r in rows if r.get("latency_ms_mean") is not None and r.get("score") is not None]
    if not valid:
        return []
    
    pareto = []
    for r in valid:
        dominated = False
        for other in valid:
            if other is r:
                continue
            # Other dominates r if it has better score AND better latency
            if (other.get("score", 0) >= r.get("score", 0) and 
                other.get("latency_ms_mean", float('inf')) <= r.get("latency_ms_mean", float('inf')) and
                (other.get("score", 0) > r.get("score", 0) or 
                 other.get("latency_ms_mean", float('inf')) < r.get("latency_ms_mean", float('inf')))):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    
    return sorted(pareto, key=lambda r: r.get("score", 0), reverse=True)


def create_comparison_plots(track_b: List[Dict[str, Any]], track_c: List[Dict[str, Any]], 
                           output_dir: Path):
    """Create comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_b = find_best_by_metric(track_b, "score")
    best_c = find_best_by_metric(track_c, "score")
    
    # Plot 1: Bar chart comparing all metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: All top5 metrics comparison
    ax1 = axes[0]
    metrics = ['All_top5_mAP', 'N_top5_mAP', 'Nv_top5_mAP', 'N_delta_top5_mAP', 'mAP', 'accuracy']
    labels = ['All_top5', 'N_top5', 'Nv_top5', 'N_delta', 'mAP', 'accuracy']
    x = np.arange(len(metrics))
    width = 0.35
    
    if best_b and best_c:
        b_vals = [(best_b.get(m) or 0) * 100 for m in metrics]
        c_vals = [(best_c.get(m) or 0) * 100 for m in metrics]
        
        bars1 = ax1.bar(x - width/2, b_vals, width, label=f'Track B (score={best_b.get("score", 0):.4f})', 
                       color='#2196F3', alpha=0.8)
        bars2 = ax1.bar(x + width/2, c_vals, width, label=f'Track C (score={best_c.get("score", 0):.4f})', 
                       color='#4CAF50', alpha=0.8)
        
        ax1.set_ylabel('Score (%)')
        ax1.set_title('Best Track B vs Best Track C\n(Weighted Checkpoint, Composite Score)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15)
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        ha='center', va='bottom', fontsize=8)
    
    # Right: Score vs Latency scatter (Track C only)
    ax2 = axes[1]
    c_with_lat = [r for r in track_c if r.get("latency_ms_mean") is not None]
    if c_with_lat:
        # Group by efficiency config
        configs = list(set(r.get("efficiency_config", "Unknown") for r in c_with_lat))
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
        config_colors = {c: colors[i] for i, c in enumerate(configs)}
        
        for cfg in configs:
            cfg_runs = [r for r in c_with_lat if r.get("efficiency_config") == cfg]
            lats = [r.get("latency_ms_mean") for r in cfg_runs]
            scores = [r.get("score", 0) for r in cfg_runs]
            ax2.scatter(lats, scores, label=cfg, s=80, alpha=0.7, c=[config_colors[cfg]])
        
        # Add Track B reference line
        if best_b:
            ax2.axhline(y=best_b.get("score", 0), color='#2196F3', linestyle='--', 
                       linewidth=2, label=f"Track B Best (score={best_b.get('score', 0):.4f})")
        
        # Highlight Pareto optimal
        pareto = find_pareto_optimal(track_c)
        if pareto:
            p_lats = [r.get("latency_ms_mean") for r in pareto]
            p_scores = [r.get("score", 0) for r in pareto]
            ax2.scatter(p_lats, p_scores, c='red', s=150, marker='*', 
                       label='Pareto Optimal', zorder=10, edgecolors='black')
        
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Composite Score')
        ax2.set_title('Track C: Score vs Latency Trade-off')
        ax2.legend(loc='upper right', fontsize=7)
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trackBC_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Detailed Pareto frontier with all top5 metrics
    pareto_runs = find_pareto_optimal(track_c)
    if pareto_runs and len(pareto_runs) > 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # All runs in gray
        all_lats = [r.get("latency_ms_mean") for r in c_with_lat]
        all_scores = [r.get("score", 0) for r in c_with_lat]
        ax.scatter(all_lats, all_scores, c='lightgray', alpha=0.5, s=50, label='All Track C runs')
        
        # Pareto runs
        pareto_lats = [r.get("latency_ms_mean") for r in pareto_runs]
        pareto_scores = [r.get("score", 0) for r in pareto_runs]
        ax.scatter(pareto_lats, pareto_scores, c='#4CAF50', s=120, marker='*', 
                  label='Pareto Optimal', zorder=5, edgecolors='black')
        
        # Connect Pareto points
        sorted_pareto = sorted(zip(pareto_lats, pareto_scores), key=lambda x: x[0])
        ax.plot([p[0] for p in sorted_pareto], [p[1] for p in sorted_pareto], 
               'g--', alpha=0.6, linewidth=2)
        
        # Track B reference
        if best_b:
            ax.axhline(y=best_b.get("score", 0), color='#2196F3', linestyle='--', 
                      linewidth=2, label=f"Track B Best")
        
        # Annotate Pareto points
        for r in pareto_runs:
            ax.annotate(f"{r.get('efficiency_config', '')}\nAll={r.get('All_top5_mAP', 0)*100:.1f}%", 
                       (r.get("latency_ms_mean"), r.get("score", 0)),
                       textcoords="offset points", xytext=(5, 5), fontsize=7,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Composite Score (mAP + N_top5 + Nv_top5 + N_delta + All_top5)', fontsize=12)
        ax.set_title('Track C Pareto Frontier: Composite Score vs Latency', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trackC_pareto_frontier.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Trade-off heatmap for each metric
    if c_with_lat:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics_to_plot = ['All_top5_mAP', 'N_top5_mAP', 'Nv_top5_mAP', 'N_delta_top5_mAP', 'mAP', 'accuracy']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            lats = [r.get("latency_ms_mean") for r in c_with_lat]
            vals = [(r.get(metric) or 0) * 100 for r in c_with_lat]
            
            scatter = ax.scatter(lats, vals, c=vals, cmap='RdYlGn', s=60, alpha=0.7)
            
            if best_b and best_b.get(metric):
                ax.axhline(y=best_b.get(metric) * 100, color='#2196F3', linestyle='--', 
                          linewidth=2, label='Track B Best')
            
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel(f'{metric} (%)')
            ax.set_title(f'{metric} vs Latency')
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=7)
            plt.colorbar(scatter, ax=ax, label='%')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trackC_metrics_tradeoff.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[INFO] Plots saved to {output_dir}")


def generate_markdown_report(track_b: List[Dict[str, Any]], track_c: List[Dict[str, Any]],
                            output_dir: Path) -> Path:
    """Generate comprehensive markdown report following summarize_best_runs.py style."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    best_b_score = find_best_by_metric(track_b, "score")
    best_c_score = find_best_by_metric(track_c, "score")
    best_c_all = find_best_by_metric(track_c, "All_top5_mAP")
    best_c_n = find_best_by_metric(track_c, "N_top5_mAP")
    best_c_nv = find_best_by_metric(track_c, "Nv_top5_mAP")
    best_c_map = find_best_by_metric(track_c, "mAP")
    
    # Calculate baseline latency
    baseline_runs = [r for r in track_c if r.get("efficiency_config") == "Baseline" and r.get("latency_ms_mean")]
    baseline_lat = np.mean([r.get("latency_ms_mean") for r in baseline_runs]) if baseline_runs else 40.0
    
    # Percentage columns
    pct_cols = ["mAP", "accuracy", "N_mAP", "Nv_mAP", "N_delta_mAP", "All_mAP",
                "N_top5_mAP", "Nv_top5_mAP", "N_delta_top5_mAP", "All_top5_mAP",
                "N_top5_acc", "Nv_top5_acc", "N_delta_top5_acc", "All_top5_acc"]
    
    lines = []
    lines.append("# Track B vs Track C Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")
    lines.append(f"**Checkpoint:** Weighted-class only (`{WEIGHTED_CHECKPOINT}`)")
    lines.append("")
    lines.append(f"**Scoring:** Composite score = mAP + N_top5_mAP + Nv_top5_mAP + N_delta_top5_mAP + All_top5_mAP")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    if best_b_score and best_c_score:
        lines.append("### Best Runs Comparison (by Composite Score)")
        lines.append("")
        lines.append("| Metric | Track B Best | Track C Best | Winner | Δ |")
        lines.append("|--------|--------------|--------------|--------|---|")
        
        compare_metrics = [
            ("Composite Score", "score", False),
            ("mAP", "mAP", True),
            ("accuracy", "accuracy", True),
            ("All_top5_mAP", "All_top5_mAP", True),
            ("N_top5_mAP", "N_top5_mAP", True),
            ("Nv_top5_mAP", "Nv_top5_mAP", True),
            ("N_delta_top5_mAP", "N_delta_top5_mAP", True),
            ("All_top5_acc", "All_top5_acc", True),
            ("N_top5_acc", "N_top5_acc", True),
            ("ttc_mae_seconds", "ttc_mae_seconds", False),
        ]
        
        for label, key, is_pct in compare_metrics:
            b_val = best_b_score.get(key)
            c_val = best_c_score.get(key)
            
            if b_val is None and c_val is None:
                continue
            
            b_str = _pct(b_val) if is_pct else _fmt(b_val)
            c_str = _pct(c_val) if is_pct else _fmt(c_val)
            
            # Determine winner (higher is better for most, lower for ttc_mae)
            if b_val is not None and c_val is not None:
                if key == "ttc_mae_seconds":
                    winner = "Track B" if b_val < c_val else "Track C"
                    delta = c_val - b_val
                else:
                    winner = "Track B" if b_val > c_val else "Track C"
                    delta = c_val - b_val
                delta_str = f"{delta*100:+.2f}%" if is_pct else f"{delta:+.4f}"
            else:
                winner = ""
                delta_str = ""
            
            lines.append(f"| {label} | {b_str} | {c_str} | {winner} | {delta_str} |")
        
        lines.append("")
        if best_c_score.get("latency_ms_mean"):
            speedup = ((baseline_lat - best_c_score.get("latency_ms_mean")) / baseline_lat) * 100
            lines.append(f"**Track C Best Latency:** {best_c_score.get('latency_ms_mean'):.1f}ms "
                        f"({speedup:.0f}% faster than baseline)")
            lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Best Track B Run
    lines.append("## Best Track B Run")
    lines.append("")
    if best_b_score:
        lines.extend(_write_table([best_b_score], [
            ("run", "run"),
            ("checkpoint", "checkpoint"),
            ("score", "score"),
            ("mAP", "mAP"),
            ("accuracy", "accuracy"),
            ("N_top5_mAP", "N_top5_mAP"),
            ("Nv_top5_mAP", "Nv_top5_mAP"),
            ("N_delta_top5_mAP", "N_delta_top5_mAP"),
            ("All_top5_mAP", "All_top5_mAP"),
            ("ttc_mae_seconds", "ttc_mae_seconds"),
            ("video_backbone", "video_backbone"),
            ("class_weight_alpha", "class_weight_alpha"),
        ], pct_cols))
    else:
        lines.append("No Track B runs found with weighted checkpoint.")
    lines.append("")
    
    # Best Track C Runs by Category
    lines.append("## Best Track C Runs by Category")
    lines.append("")
    
    best_c_categories = [
        ("Best Composite Score", best_c_score),
        ("Best All_top5_mAP", best_c_all),
        ("Best N_top5_mAP", best_c_n),
        ("Best Nv_top5_mAP", best_c_nv),
        ("Best mAP", best_c_map),
    ]
    
    cat_rows = []
    for cat_name, run in best_c_categories:
        if run:
            row = dict(run)
            row["category"] = cat_name
            if run.get("latency_ms_mean"):
                speedup = ((baseline_lat - run.get("latency_ms_mean")) / baseline_lat) * 100
                row["speedup"] = f"{speedup:.0f}%"
            else:
                row["speedup"] = ""
            cat_rows.append(row)
    
    if cat_rows:
        lines.extend(_write_table(cat_rows, [
            ("Category", "category"),
            ("Config", "efficiency_config"),
            ("score", "score"),
            ("mAP", "mAP"),
            ("All_top5", "All_top5_mAP"),
            ("N_top5", "N_top5_mAP"),
            ("Nv_top5", "Nv_top5_mAP"),
            ("N_delta", "N_delta_top5_mAP"),
            ("Latency", "latency_ms_mean"),
            ("Speedup", "speedup"),
        ], pct_cols))
    lines.append("")
    
    # Pareto Optimal Configurations
    lines.append("## Pareto-Optimal Configurations (Score vs Latency)")
    lines.append("")
    lines.append("These configurations are NOT dominated by any other run:")
    lines.append("")
    
    pareto_runs = find_pareto_optimal(track_c)
    if pareto_runs:
        pareto_rows = []
        for i, run in enumerate(pareto_runs, 1):
            row = dict(run)
            row["rank"] = i
            if run.get("latency_ms_mean"):
                speedup = ((baseline_lat - run.get("latency_ms_mean")) / baseline_lat) * 100
                row["speedup"] = f"{speedup:.0f}%"
            else:
                row["speedup"] = ""
            pareto_rows.append(row)
        
        lines.extend(_write_table(pareto_rows, [
            ("Rank", "rank"),
            ("Config", "efficiency_config"),
            ("score", "score"),
            ("All_top5", "All_top5_mAP"),
            ("N_top5", "N_top5_mAP"),
            ("Nv_top5", "Nv_top5_mAP"),
            ("mAP", "mAP"),
            ("Latency", "latency_ms_mean"),
            ("Speedup", "speedup"),
        ], pct_cols))
    else:
        lines.append("No Pareto-optimal runs found.")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # All Track B Runs
    lines.append("## All Track B Runs (Weighted Checkpoint)")
    lines.append("")
    if track_b:
        sorted_b = sorted(track_b, key=lambda r: r.get("score", 0), reverse=True)
        lines.extend(_write_table(sorted_b, [
            ("run", "run"),
            ("checkpoint", "checkpoint"),
            ("score", "score"),
            ("mAP", "mAP"),
            ("accuracy", "accuracy"),
            ("N_top5_mAP", "N_top5_mAP"),
            ("Nv_top5_mAP", "Nv_top5_mAP"),
            ("N_delta_top5_mAP", "N_delta_top5_mAP"),
            ("All_top5_mAP", "All_top5_mAP"),
            ("ttc_mae_seconds", "ttc_mae_seconds"),
        ], pct_cols))
    else:
        lines.append("No Track B runs found.")
    lines.append("")
    
    # All Track C Runs
    lines.append("## All Track C Runs (Weighted Checkpoint)")
    lines.append("")
    if track_c:
        sorted_c = sorted(track_c, key=lambda r: r.get("score", 0), reverse=True)
        lines.extend(_write_table(sorted_c, [
            ("timestamp", "timestamp"),
            ("Config", "efficiency_config"),
            ("score", "score"),
            ("mAP", "mAP"),
            ("accuracy", "accuracy"),
            ("All_top5", "All_top5_mAP"),
            ("N_top5", "N_top5_mAP"),
            ("Nv_top5", "Nv_top5_mAP"),
            ("N_delta", "N_delta_top5_mAP"),
            ("Latency", "latency_ms_mean"),
            ("TTC MAE", "ttc_mae_seconds"),
        ], pct_cols))
    else:
        lines.append("No Track C runs found.")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Trade-off Analysis
    lines.append("## Trade-off Analysis")
    lines.append("")
    
    # Find key insights
    lines.append("### Key Trade-off Insights")
    lines.append("")
    
    if best_b_score and best_c_score:
        # Best accuracy config
        if best_c_all:
            lines.append(f"1. **Best All_top5_mAP:** {_pct(best_c_all.get('All_top5_mAP'))} "
                        f"({best_c_all.get('efficiency_config')})")
            if best_b_score.get("All_top5_mAP"):
                delta = (best_c_all.get("All_top5_mAP", 0) - best_b_score.get("All_top5_mAP", 0)) * 100
                lines.append(f"   - vs Track B: {delta:+.2f}%")
            if best_c_all.get("latency_ms_mean"):
                speedup = ((baseline_lat - best_c_all.get("latency_ms_mean")) / baseline_lat) * 100
                lines.append(f"   - Latency: {best_c_all.get('latency_ms_mean'):.1f}ms ({speedup:.0f}% faster)")
            lines.append("")
        
        # Best N_top5 config
        if best_c_n:
            lines.append(f"2. **Best N_top5_mAP:** {_pct(best_c_n.get('N_top5_mAP'))} "
                        f"({best_c_n.get('efficiency_config')})")
            if best_b_score.get("N_top5_mAP"):
                delta = (best_c_n.get("N_top5_mAP", 0) - best_b_score.get("N_top5_mAP", 0)) * 100
                lines.append(f"   - vs Track B: {delta:+.2f}% {'🎉' if delta > 0 else ''}")
            lines.append("")
        
        # Maximum speedup config
        fastest_runs = [r for r in track_c if r.get("latency_ms_mean")]
        if fastest_runs:
            fastest = min(fastest_runs, key=lambda r: r.get("latency_ms_mean", float('inf')))
            speedup = ((baseline_lat - fastest.get("latency_ms_mean")) / baseline_lat) * 100
            lines.append(f"3. **Maximum Speedup:** {speedup:.0f}% ({fastest.get('efficiency_config')})")
            lines.append(f"   - Latency: {fastest.get('latency_ms_mean'):.1f}ms (vs {baseline_lat:.1f}ms baseline)")
            lines.append(f"   - All_top5: {_pct(fastest.get('All_top5_mAP'))}")
            lines.append(f"   - Score: {fastest.get('score', 0):.4f}")
            lines.append("")
    
    # Recommendations
    lines.append("### Recommendations")
    lines.append("")
    lines.append("| Use Case | Recommended Config | Score | All_top5 | Latency | Speedup |")
    lines.append("|----------|-------------------|-------|----------|---------|---------|")
    
    if pareto_runs:
        # Best overall (highest score)
        best_overall = pareto_runs[0]
        speedup = ((baseline_lat - best_overall.get("latency_ms_mean", baseline_lat)) / baseline_lat) * 100
        lines.append(f"| **Max Score** | {best_overall.get('efficiency_config')} | "
                    f"{best_overall.get('score', 0):.4f} | {_pct(best_overall.get('All_top5_mAP'))} | "
                    f"{best_overall.get('latency_ms_mean', 0):.1f}ms | {speedup:.0f}% |")
        
        # Best trade-off (>30% speedup, highest score)
        tradeoff_runs = [r for r in pareto_runs if r.get("latency_ms_mean", baseline_lat) < baseline_lat * 0.7]
        if tradeoff_runs:
            best_trade = max(tradeoff_runs, key=lambda r: r.get("score", 0))
            speedup = ((baseline_lat - best_trade.get("latency_ms_mean", baseline_lat)) / baseline_lat) * 100
            lines.append(f"| **Best Trade-off** | {best_trade.get('efficiency_config')} | "
                        f"{best_trade.get('score', 0):.4f} | {_pct(best_trade.get('All_top5_mAP'))} | "
                        f"{best_trade.get('latency_ms_mean', 0):.1f}ms | {speedup:.0f}% |")
        
        # Max efficiency (lowest latency)
        fastest = min(pareto_runs, key=lambda r: r.get("latency_ms_mean", float('inf')))
        speedup = ((baseline_lat - fastest.get("latency_ms_mean", baseline_lat)) / baseline_lat) * 100
        lines.append(f"| **Max Efficiency** | {fastest.get('efficiency_config')} | "
                    f"{fastest.get('score', 0):.4f} | {_pct(fastest.get('All_top5_mAP'))} | "
                    f"{fastest.get('latency_ms_mean', 0):.1f}ms | {speedup:.0f}% |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Plots
    lines.append("## Plots")
    lines.append("")
    lines.append("### Track B vs Track C Comparison")
    lines.append("![Track B vs C Comparison](trackBC_comparison.png)")
    lines.append("")
    lines.append("### Pareto Frontier (Score vs Latency)")
    lines.append("![Pareto Frontier](trackC_pareto_frontier.png)")
    lines.append("")
    lines.append("### Individual Metrics Trade-off")
    lines.append("![Metrics Trade-off](trackC_metrics_tradeoff.png)")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `trackBC_comparison.py` on {timestamp}*")
    
    # Write report
    report_path = output_dir / "trackBC_comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Report saved to {report_path}")
    
    return report_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("Track B vs Track C Comparison")
    print(f"Using only weighted checkpoint: {WEIGHTED_CHECKPOINT}")
    print(f"Scoring: {' + '.join(SCORE_KEYS)}")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading Track B metrics...")
    track_b = load_track_b_metrics()
    print(f"      Found {len(track_b)} runs with weighted checkpoint")
    
    print("\n[2/4] Loading Track C metrics...")
    track_c = load_track_c_metrics()
    print(f"      Found {len(track_c)} runs with weighted checkpoint")
    
    if not track_b and not track_c:
        print("[ERROR] No runs found with weighted checkpoint!")
        return 1
    
    # Find best runs
    print("\n[3/4] Analyzing runs...")
    best_b = find_best_by_metric(track_b, "score")
    best_c = find_best_by_metric(track_c, "score")
    pareto = find_pareto_optimal(track_c)
    
    if best_b:
        print(f"      Track B best score: {best_b.get('score', 0):.4f}")
    if best_c:
        print(f"      Track C best score: {best_c.get('score', 0):.4f}")
    print(f"      Track C Pareto optimal runs: {len(pareto)}")
    
    # Generate outputs
    print("\n[4/4] Generating outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plots(track_b, track_c, OUTPUT_DIR)
    report_path = generate_markdown_report(track_b, track_c, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Print quick summary
    print("\n### Quick Summary ###")
    if best_b:
        print(f"Track B Best Score: {best_b.get('score', 0):.4f}")
        print(f"  - All_top5: {best_b.get('All_top5_mAP', 0)*100:.2f}%")
        print(f"  - mAP: {best_b.get('mAP', 0)*100:.2f}%")
    if best_c:
        print(f"\nTrack C Best Score: {best_c.get('score', 0):.4f}")
        print(f"  - Config: {best_c.get('efficiency_config')}")
        print(f"  - All_top5: {best_c.get('All_top5_mAP', 0)*100:.2f}%")
        print(f"  - mAP: {best_c.get('mAP', 0)*100:.2f}%")
        if best_c.get('latency_ms_mean'):
            print(f"  - Latency: {best_c.get('latency_ms_mean'):.1f}ms")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
