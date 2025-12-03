#!/usr/bin/env python3
"""
Summarize Ablation: VideoMAE vs ResNet18 Baseline

Reads metrics from Track B runs and produces a Markdown comparison table.

Usage:
    python -m trackB.summarize_ablation
    python -m trackB.summarize_ablation --output results/ablation_videomae_vs_clip.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def find_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Find all Track B runs with metrics."""
    runs = []
    
    if not runs_dir.exists():
        print(f"[Ablation] Runs directory not found: {runs_dir}")
        return runs
    
    # Look for run directories (format: YYYYMMDD_HHMMSS)
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Check for metrics.json
        metrics_file = run_dir / 'metrics.json'
        config_file = run_dir / 'config.json'
        
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                config = {}
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                
                runs.append({
                    'name': run_dir.name,
                    'path': run_dir,
                    'metrics': metrics,
                    'config': config,
                })
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Ablation] Error reading {metrics_file}: {e}")
                continue
    
    return runs


def categorize_run(run: Dict[str, Any]) -> str:
    """Categorize a run as 'resnet18' or 'videomae_ego' based on config."""
    config = run.get('config', {})
    
    # Check for video_backbone setting
    video_backbone = config.get('video_backbone')
    if video_backbone:
        return video_backbone
    
    # Check for config_name that might indicate backbone
    config_name = config.get('config_name', '')
    if 'videomae' in config_name.lower():
        return 'videomae_ego'
    if 'resnet' in config_name.lower() or 'baseline' in config_name.lower():
        return 'resnet18'
    
    # Check projector in_dim as a heuristic (768 = videomae, 512 = resnet)
    projector_in_dim = config.get('projector_in_dim')
    if projector_in_dim == 768:
        return 'videomae_ego'
    if projector_in_dim == 512:
        return 'resnet18'
    
    # Default to resnet18 (baseline)
    return 'resnet18'


def get_metric(metrics: Dict[str, Any], key: str, default: str = '-') -> str:
    """Get a metric value, formatting it nicely."""
    value = metrics.get(key)
    
    if value is None:
        return default
    
    if isinstance(value, float):
        if 'mAP' in key or 'accuracy' in key:
            return f"{value * 100:.2f}%"  # Convert to percentage
        elif 'mae' in key.lower() or 'ttc' in key.lower():
            return f"{value:.3f}"  # TTC in seconds
        else:
            return f"{value:.4f}"
    
    return str(value)


def generate_table(runs: List[Dict[str, Any]]) -> str:
    """Generate Markdown table comparing runs."""
    
    # Group runs by backbone type
    resnet_runs = [r for r in runs if categorize_run(r) == 'resnet18']
    videomae_runs = [r for r in runs if categorize_run(r) == 'videomae_ego']
    
    # Define metrics columns
    metric_columns = [
        ('accuracy', 'Accuracy'),
        ('mAP', 'mAP'),
        ('ttc_mae_seconds', 'TTC MAE'),
        ('N_mAP', 'N mAP'),
        ('Nv_mAP', 'Nv mAP'),
        ('N_delta_mAP', 'N+δ mAP'),
        ('All_mAP', 'All mAP'),
        ('top5_accuracy', 'Top5 Acc'),
    ]
    
    # Build table header
    header = "| Model Variant | Video Backbone | Run ID |"
    separator = "|---------------|----------------|--------|"
    for key, name in metric_columns:
        header += f" {name} |"
        separator += "--------|"
    
    lines = [
        "# Ablation: VideoMAE vs ResNet18 Baseline",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        header,
        separator,
    ]
    
    # Add rows for ResNet18 baseline runs
    for run in resnet_runs:
        metrics = run['metrics']
        row = f"| Exo-transfer (base) | resnet18 | {run['name'][:15]} |"
        for key, _ in metric_columns:
            row += f" {get_metric(metrics, key)} |"
        lines.append(row)
    
    # Add rows for VideoMAE runs
    for run in videomae_runs:
        metrics = run['metrics']
        row = f"| Ego-only VideoMAE | videomae_ego | {run['name'][:15]} |"
        for key, _ in metric_columns:
            row += f" {get_metric(metrics, key)} |"
        lines.append(row)
    
    # Add summary if we have both types
    if resnet_runs and videomae_runs:
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        
        # Calculate best metrics for each type
        best_resnet = max(resnet_runs, key=lambda r: r['metrics'].get('mAP', 0))
        best_videomae = max(videomae_runs, key=lambda r: r['metrics'].get('mAP', 0))
        
        resnet_map = best_resnet['metrics'].get('mAP', 0)
        videomae_map = best_videomae['metrics'].get('mAP', 0)
        
        if resnet_map > 0 and videomae_map > 0:
            delta = (videomae_map - resnet_map) / resnet_map * 100
            lines.append(f"- **ResNet18 Best mAP**: {resnet_map * 100:.2f}% (Run: {best_resnet['name']})")
            lines.append(f"- **VideoMAE Best mAP**: {videomae_map * 100:.2f}% (Run: {best_videomae['name']})")
            lines.append(f"- **Relative Change**: {delta:+.2f}%")
    
    # Add notes
    lines.extend([
        "",
        "## Notes",
        "",
        "- **Exo-transfer (base)**: Uses ResNet18 pretrained on ImageNet (exocentric data)",
        "- **Ego-only VideoMAE**: Uses VideoMAE self-supervised pretrained on egocentric video",
        "- mAP and accuracy shown as percentages",
        "- TTC MAE shown in seconds",
        "- '-' indicates metric not available",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarize ablation results')
    parser.add_argument('--runs_dir', type=str, 
                        default='local_extraction/runs/Track_B',
                        help='Path to Track B runs directory')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Output file path (default: print to stdout)')
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    
    print(f"[Ablation] Scanning runs in: {runs_dir}")
    runs = find_runs(runs_dir)
    
    if not runs:
        print("[Ablation] No runs found with metrics.")
        print("[Ablation] Make sure to run training first to generate metrics.")
        return
    
    print(f"[Ablation] Found {len(runs)} run(s)")
    
    # Categorize runs
    resnet_count = sum(1 for r in runs if categorize_run(r) == 'resnet18')
    videomae_count = sum(1 for r in runs if categorize_run(r) == 'videomae_ego')
    print(f"[Ablation] ResNet18 runs: {resnet_count}, VideoMAE runs: {videomae_count}")
    
    # Generate table
    table = generate_table(runs)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"[Ablation] Saved to: {output_path}")
    else:
        print()
        print(table)


if __name__ == '__main__':
    main()
