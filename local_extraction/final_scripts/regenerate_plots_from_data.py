#!/usr/bin/env python3
"""
Regenerate Plots from Saved JSON Data
======================================
Reads the *_data.json files and regenerates all plots without re-running
the full analysis pipeline.

Usage:
    python local_extraction/final_scripts/regenerate_plots_from_data.py
    python local_extraction/final_scripts/regenerate_plots_from_data.py --output-dir custom_output
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np


def load_json_data(filepath: Path) -> dict:
    """Load JSON data file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_checkpoint_timeline(data: dict, output_dir: Path):
    """Regenerate checkpoint timeline plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    checkpoints = data['checkpoints']
    color_map = data['color_map']
    
    dates = [datetime.fromisoformat(c['date']) for c in checkpoints]
    maps = [c['mAP'] for c in checkpoints]
    colors = [c['color'] for c in checkpoints]
    labels = [c['label'] for c in checkpoints]
    
    scatter = ax.scatter(dates, maps, c=colors, s=100, edgecolors='black', zorder=5)
    
    # Highlight key checkpoints
    for date, m, c, l in zip(dates, maps, colors, labels):
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
    print("  ✓ checkpoint_timeline.png")


def plot_trackB_checkpoints_metrics(data: dict, output_dir: Path):
    """Regenerate Track B checkpoints metrics plot."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    checkpoints = data['checkpoints']
    names = [c['name'] for c in checkpoints]
    mAPs = [c['mAP'] for c in checkpoints]
    accs = [c['accuracy'] for c in checkpoints]
    all5s = [c['All_top5_mAP'] for c in checkpoints]
    n5s = [c['N_top5_mAP'] for c in checkpoints]
    nv5s = [c['Nv_top5_mAP'] for c in checkpoints]
    nd5s = [c['N_delta_top5_mAP'] for c in checkpoints]
    scores = [c['score'] for c in checkpoints]
    
    x = np.arange(len(names))
    width = 0.12
    
    ax2 = ax.twinx()
    
    # Plot all metrics as bars
    bars1 = ax.bar(x - 2.5*width, mAPs, width, label='mAP (%)', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x - 1.5*width, all5s, width, label='All_top5', color='#4CAF50', alpha=0.8)
    bars3 = ax.bar(x - 0.5*width, n5s, width, label='N_top5', color='#8BC34A', alpha=0.8)
    bars4 = ax.bar(x + 0.5*width, nv5s, width, label='Nv_top5', color='#CDDC39', alpha=0.8)
    bars5 = ax.bar(x + 1.5*width, nd5s, width, label='N_delta', color='#FFC107', alpha=0.8)
    bars6 = ax.bar(x + 2.5*width, accs, width, label='Accuracy (%)', color='#9C27B0', alpha=0.8)
    
    # Composite score as line
    line = ax2.plot(x, scores, 'ro-', label='Composite Score', markersize=10, linewidth=2)
    
    ax.set_xlabel('Checkpoint (by mAP value)')
    ax.set_ylabel('Metric Value (%)')
    ax2.set_ylabel('Composite Score', color='red')
    ax.set_title('Track B Top 10 Checkpoints: All Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight the thesis checkpoint (0.3708)
    for i, name in enumerate(names):
        if "3708" in name:
            ax.axvline(x=i, color='#4CAF50', linestyle='--', alpha=0.5, linewidth=2)
            ax.annotate('THESIS', (i, max(mAPs) + 2), ha='center', fontsize=9, 
                       fontweight='bold', color='#4CAF50')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trackB_checkpoints_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ trackB_checkpoints_metrics.png")


def plot_trackB_runs_metrics(data: dict, output_dir: Path):
    """Regenerate Track B runs metrics plot."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    runs = data['runs']
    names = [r['name'] for r in runs]
    mAPs = [r['mAP'] for r in runs]
    accs = [r['accuracy'] for r in runs]
    all5s = [r['All_top5_mAP'] for r in runs]
    n5s = [r['N_top5_mAP'] for r in runs]
    nv5s = [r['Nv_top5_mAP'] for r in runs]
    nd5s = [r['N_delta_top5_mAP'] for r in runs]
    scores = [r['score'] for r in runs]
    colors = [r['color'] for r in runs]
    
    x = np.arange(len(names))
    width = 0.12
    
    ax2 = ax.twinx()
    
    # Plot all metrics as bars
    bars1 = ax.bar(x - 2.5*width, mAPs, width, label='mAP (%)', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x - 1.5*width, all5s, width, label='All_top5', color='#4CAF50', alpha=0.8)
    bars3 = ax.bar(x - 0.5*width, n5s, width, label='N_top5', color='#8BC34A', alpha=0.8)
    bars4 = ax.bar(x + 0.5*width, nv5s, width, label='Nv_top5', color='#CDDC39', alpha=0.8)
    bars5 = ax.bar(x + 1.5*width, nd5s, width, label='N_delta', color='#FFC107', alpha=0.8)
    bars6 = ax.bar(x + 2.5*width, accs, width, label='Accuracy (%)', color='#9C27B0', alpha=0.8)
    
    # Composite score as line
    line = ax2.plot(x, scores, 'ro-', label='Composite Score', markersize=10, linewidth=2)
    
    # Add background colors for USE vs SKIP
    for i, color in enumerate(colors):
        ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color=color, zorder=0)
    
    ax.set_xlabel('Checkpoint (mAP) and Date')
    ax.set_ylabel('Metric Value (%)')
    ax2.set_ylabel('Composite Score', color='red')
    ax.set_title('Track B Top 12 Runs: All Metrics Comparison\n(Green=USE, Orange=0.3904, Gray=Other)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, ha='center', fontsize=8)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trackB_runs_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ trackB_runs_metrics.png")


def plot_weighted_vs_unweighted(data: dict, output_dir: Path):
    """Regenerate weighted vs unweighted comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    weighted = data['weighted']
    unweighted = data['unweighted']
    
    # Left: Bar chart of key metrics
    ax1 = axes[0]
    metrics_labels = ['mAP', 'accuracy', 'All_top5', 'N_top5', 'Nv_top5', 'N_delta']
    
    w_metrics = weighted['metrics']
    u_metrics = unweighted['metrics']
    
    w_vals = [w_metrics['mAP'], w_metrics['accuracy'], w_metrics['All_top5_mAP'], 
              w_metrics['N_top5_mAP'], w_metrics['Nv_top5_mAP'], w_metrics['N_delta_top5_mAP']]
    u_vals = [u_metrics['mAP'], u_metrics['accuracy'], u_metrics['All_top5_mAP'],
              u_metrics['N_top5_mAP'], u_metrics['Nv_top5_mAP'], u_metrics['N_delta_top5_mAP']]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, w_vals, width, label='Weighted (0.3708)', color='#4CAF50', alpha=0.8)
    bars2 = ax1.bar(x + width/2, u_vals, width, label='Unweighted (0.3904)', color='#FF9800', alpha=0.8)
    
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Weighted vs Unweighted Checkpoint Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_labels, rotation=15)
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
    categories = ['Composite\nScore', 'mAP\n(×100)', 'All_top5\n(×100)']
    scores_comp = data['scores_comparison']
    
    x = np.arange(len(categories))
    
    bars1 = ax2.bar(x - width/2, [s[0] for s in scores_comp], width, label='Weighted (0.3708) [USE]', color='#4CAF50')
    bars2 = ax2.bar(x + width/2, [s[1] for s in scores_comp], width, label='Unweighted (0.3904) [SKIP]', color='#FF9800')
    
    ax2.set_ylabel('Value')
    ax2.set_title('Why Use Weighted Checkpoint?')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Highlight winner
    for i, (w, u) in enumerate(scores_comp):
        winner = "W" if w > u else "U"
        y_pos = max(w, u) + 1
        ax2.annotate(f'Winner: {"Weighted" if w > u else "Unweighted"}', 
                    xy=(i, y_pos), ha='center', fontsize=8,
                    color='#4CAF50' if w > u else '#FF9800', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weighted_vs_unweighted.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ weighted_vs_unweighted.png")


def plot_trackC_categories(data: dict, output_dir: Path):
    """Regenerate Track C categories plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Pie chart of run categories
    ax1 = axes[0]
    category_counts = data['category_counts']
    
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
    use_scores = data.get('use_scores', [])
    skip_scores = data.get('skip_scores', [])
    
    if use_scores or skip_scores:
        bp_data = []
        bp_labels = []
        bp_colors = []
        
        if use_scores:
            bp_data.append(use_scores)
            bp_labels.append('[USE] For Thesis')
            bp_colors.append('#4CAF50')
        if skip_scores:
            bp_data.append(skip_scores)
            bp_labels.append('[SKIP]')
            bp_colors.append('#FF9800')
        
        bp = ax2.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Composite Score')
        ax2.set_title('Score Distribution: Use vs Skip Runs')
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trackC_categories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ trackC_categories.png")


def plot_efficiency_configs(data: dict, output_dir: Path):
    """Regenerate efficiency configurations plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    configs = data['configs']
    names = [c['name'] for c in configs]
    mAPs = [c['mAP'] for c in configs]
    all5s = [c['All_top5_mAP'] for c in configs]
    n5s = [c['N_top5_mAP'] for c in configs]
    nv5s = [c['Nv_top5_mAP'] for c in configs]
    nd5s = [c['N_delta_top5_mAP'] for c in configs]
    latencies = [c['latency_ms'] for c in configs]
    
    x = np.arange(len(names))
    width = 0.12
    
    ax2 = ax.twinx()
    
    # Plot all top5 metrics plus mAP
    bars1 = ax.bar(x - 2*width, mAPs, width, label='mAP (%)', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x - width, all5s, width, label='All_top5', color='#4CAF50', alpha=0.8)
    bars3 = ax.bar(x, n5s, width, label='N_top5', color='#8BC34A', alpha=0.8)
    bars4 = ax.bar(x + width, nv5s, width, label='Nv_top5', color='#CDDC39', alpha=0.8)
    bars5 = ax.bar(x + 2*width, nd5s, width, label='N_delta', color='#FFC107', alpha=0.8)
    line = ax2.plot(x, latencies, 'ro-', label='Latency (ms)', markersize=10, linewidth=2)
    
    ax.set_xlabel('Efficiency Configuration')
    ax.set_ylabel('Metric Value (%)')
    ax2.set_ylabel('Latency (ms)', color='red')
    ax.set_title('Dec 30 Efficiency Runs: All Metrics vs Latency Trade-off')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_configs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ efficiency_configs.png")


def plot_run_timeline(data: dict, output_dir: Path):
    """Regenerate run timeline plot."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Track B timeline
    ax1 = axes[0]
    b_data = data['track_b']
    b_dates = [datetime.fromisoformat(r['date']) for r in b_data]
    b_scores = [r['score'] for r in b_data]
    b_colors = [r['color'] for r in b_data]
    
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
    c_data = data['track_c']
    c_dates = [datetime.fromisoformat(r['date']) for r in c_data]
    c_scores = [r['score'] for r in c_data]
    c_colors = [r['color'] for r in c_data]
    
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
    print("  ✓ run_timeline.png")


def plot_backbone_pretraining(data: dict, output_dir: Path):
    """Regenerate backbone and pretraining plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Backbone distribution
    ax1 = axes[0]
    backbone_counts = data['backbone_counts']
    
    bb_colors = {
        "resnet18": "#2196F3",
        "videomae_ego": "#4CAF50",
    }
    bb_labels = {
        "resnet18": "ResNet18 (ImageNet)",
        "videomae_ego": "VideoMAE (Ego4D)",
    }
    
    labels = [bb_labels.get(k, k) for k in backbone_counts.keys()]
    sizes = list(backbone_counts.values())
    colors = [bb_colors.get(k, "#9E9E9E") for k in backbone_counts.keys()]
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            shadow=True, startangle=90)
    ax1.set_title('Backbone Distribution Across All Runs')
    
    # Right: Pretraining source comparison
    ax2 = axes[1]
    pretraining_counts = data['pretraining_counts']
    
    pt_colors = {
        "exo": "#FF9800",
        "ego": "#4CAF50",
    }
    pt_labels = {
        "exo": "Exo-Transfer\n(COCO/ImageNet)",
        "ego": "Ego-Pretrained\n(Ego4D)",
    }
    
    labels = [pt_labels.get(k, k) for k in pretraining_counts.keys()]
    sizes = list(pretraining_counts.values())
    colors = [pt_colors.get(k, "#9E9E9E") for k in pretraining_counts.keys()]
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            shadow=True, startangle=90)
    ax2.set_title('Pretraining Source Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'backbone_pretraining.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ backbone_pretraining.png")


def plot_backbone_metrics_comparison(data: dict, output_dir: Path):
    """Regenerate backbone metrics comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    resnet = data['resnet18']
    videomae = data['videomae']
    
    metrics = list(resnet['metrics'].keys())
    labels = resnet['label']
    
    resnet_vals = list(resnet['metrics'].values())
    videomae_vals = list(videomae['metrics'].values())
    
    x = np.arange(len(metrics))
    width = 0.35
    
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
    print("  ✓ backbone_metrics_comparison.png")


def plot_trackB_vs_trackC_comparison(data: dict, output_dir: Path):
    """Regenerate Track B vs Track C comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    track_b = data['track_b']
    track_c = data['track_c']
    metric_labels = data['metric_labels']
    
    # Left: Grouped bar chart comparing metrics
    ax1 = axes[0]
    
    b_vals = list(track_b['best_metrics'].values())
    c_vals = list(track_c['best_metrics'].values())
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, b_vals, width, label=f'Track B Best ({len(track_b["all_runs"])} runs)', 
                   color='#2196F3', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, c_vals, width, label=f'Track C Best ({len(track_c["all_runs"])} runs)', 
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
    b_runs = track_b['all_runs']
    if b_runs:
        b_maps = [r['mAP'] for r in b_runs]
        b_accs = [r['accuracy'] for r in b_runs]
        ax2.scatter(b_maps, b_accs, c='#2196F3', s=150, label=f'Track B ({len(b_runs)})', 
                   alpha=0.8, edgecolors='black', marker='o')
        
        # Annotate best
        best_b_idx = b_maps.index(max(b_maps))
        ax2.annotate('Best B', 
                   (b_maps[best_b_idx], b_accs[best_b_idx]),
                   textcoords="offset points", xytext=(10, 5), fontsize=8,
                   fontweight='bold', color='#2196F3')
    
    # Plot Track C runs
    c_runs = track_c['all_runs']
    if c_runs:
        c_maps = [r['mAP'] for r in c_runs]
        c_accs = [r['accuracy'] for r in c_runs]
        ax2.scatter(c_maps, c_accs, c='#4CAF50', s=150, label=f'Track C ({len(c_runs)})', 
                   alpha=0.8, edgecolors='black', marker='s')
        
        # Annotate best
        best_c_idx = c_maps.index(max(c_maps))
        ax2.annotate('Best C', 
                   (c_maps[best_c_idx], c_accs[best_c_idx]),
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
    print("  ✓ trackB_vs_trackC_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Regenerate plots from saved JSON data')
    parser.add_argument('--data-dir', type=Path, 
                       default=Path('local_extraction/final_scripts/comparison_results'),
                       help='Directory containing *_data.json files')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('local_extraction/final_scripts/comparison_results'),
                       help='Directory to save regenerated plots')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Regenerating Plots from Saved JSON Data")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Map data files to plot functions
    plot_functions = {
        'checkpoint_timeline_data.json': plot_checkpoint_timeline,
        'trackB_checkpoints_metrics_data.json': plot_trackB_checkpoints_metrics,
        'trackB_runs_metrics_data.json': plot_trackB_runs_metrics,
        'weighted_vs_unweighted_data.json': plot_weighted_vs_unweighted,
        'trackC_categories_data.json': plot_trackC_categories,
        'efficiency_configs_data.json': plot_efficiency_configs,
        'run_timeline_data.json': plot_run_timeline,
        'backbone_pretraining_data.json': plot_backbone_pretraining,
        'backbone_metrics_comparison_data.json': plot_backbone_metrics_comparison,
        'trackB_vs_trackC_comparison_data.json': plot_trackB_vs_trackC_comparison,
    }
    
    regenerated = 0
    skipped = 0
    
    for data_file, plot_func in plot_functions.items():
        data_path = data_dir / data_file
        if data_path.exists():
            print(f"Processing {data_file}...")
            try:
                data = load_json_data(data_path)
                plot_func(data, output_dir)
                regenerated += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                skipped += 1
        else:
            print(f"Skipping {data_file} (not found)")
            skipped += 1
    
    print()
    print("=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nRegenerated: {regenerated} plots")
    print(f"Skipped: {skipped} plots")
    print(f"\nOutput: {output_dir}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
