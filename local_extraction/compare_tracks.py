#!/usr/bin/env python3
"""
Compare Results Across Track A, B, and C

This script aggregates and compares metrics from all three tracks:
- Track A: Candidate generation (Stage A + Stage B)
- Track B: Temporal fusion and classification
- Track C: RGTP pruning for efficiency

Usage:
    python local_extraction/compare_tracks.py
    python local_extraction/compare_tracks.py --latest  # Only latest runs
    python local_extraction/compare_tracks.py --output comparison_report.md
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add local_extraction to path
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def find_track_a_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Find all Track A Stage A and Stage B runs with their summaries."""
    results = []
    
    # Stage A runs
    for stage_a_dir in sorted(runs_dir.glob("trackA_stageA_*")):
        summary_path = stage_a_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                results.append({
                    'track': 'A',
                    'stage': 'stageA',
                    'run_id': stage_a_dir.name,
                    'timestamp': summary.get('timestamp', ''),
                    'metrics': summary,
                    'path': str(stage_a_dir),
                })
            except Exception as e:
                print(f"Warning: Could not load {summary_path}: {e}")
    
    # Stage B runs
    for stage_b_dir in sorted(runs_dir.glob("trackA_stageB_*")):
        summary_path = stage_b_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                results.append({
                    'track': 'A',
                    'stage': 'stageB',
                    'run_id': stage_b_dir.name,
                    'timestamp': summary.get('timestamp', ''),
                    'metrics': summary,
                    'path': str(stage_b_dir),
                })
            except Exception as e:
                print(f"Warning: Could not load {summary_path}: {e}")
    
    return results


def find_track_b_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Find all Track B runs from metrics files."""
    results = []
    metrics_dir = runs_dir / "metrics"
    
    if not metrics_dir.exists():
        return results
    
    # Find all summary files
    for summary_path in sorted(metrics_dir.glob("metrics_val_*_summary.json")):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            
            metrics = summary.get('metrics', {})
            eval_config = summary.get('eval_config', {})
            
            results.append({
                'track': 'B',
                'run_id': summary_path.stem,
                'timestamp': metrics.get('timestamp', ''),
                'metrics': metrics,
                'config': eval_config,
                'path': str(summary_path),
            })
        except Exception as e:
            print(f"Warning: Could not load {summary_path}: {e}")
    
    return results


def find_track_c_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Find all Track C runs from metrics files."""
    results = []
    metrics_dir = runs_dir / "metrics"
    
    if not metrics_dir.exists():
        return results
    
    # Find all summary files
    for summary_path in sorted(metrics_dir.glob("trackC_val_*_summary.json")):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            
            metrics = summary.get('metrics', {})
            
            results.append({
                'track': 'C',
                'run_id': summary_path.stem,
                'timestamp': metrics.get('timestamp', ''),
                'metrics': metrics,
                'rgtp_rate': metrics.get('rgtp_rate_request', 0.0),
                'pruning_enabled': metrics.get('rgtp_enabled', False),
                'path': str(summary_path),
            })
        except Exception as e:
            print(f"Warning: Could not load {summary_path}: {e}")
    
    return results


def format_metrics_table(runs: List[Dict[str, Any]], track: str) -> str:
    """Format metrics as a markdown table."""
    if not runs:
        return f"No {track} runs found.\n"
    
    output = []
    output.append(f"\n### Track {track} Results\n")
    
    if track == 'A':
        # Track A: Split by stage
        stage_a = [r for r in runs if r['stage'] == 'stageA']
        stage_b = [r for r in runs if r['stage'] == 'stageB']
        
        if stage_a:
            output.append("#### Stage A (Candidate Generation)\n")
            output.append("| Run ID | Timestamp | Precision | Recall | F1 | Candidates/Frame |")
            output.append("|--------|-----------|-----------|--------|----|--------------------|")
            for run in stage_a:
                m = run['metrics']
                output.append(
                    f"| {run['run_id']} | {run['timestamp'][:16]} | "
                    f"{m.get('precision', 0):.3f} | {m.get('recall', 0):.3f} | "
                    f"{m.get('f1', 0):.3f} | {m.get('avg_candidates_per_frame', 0):.1f} |"
                )
            output.append("")
        
        if stage_b:
            output.append("#### Stage B (Head Manifest)\n")
            output.append("| Run ID | Timestamp | Train Samples | Val Samples | Positives |")
            output.append("|--------|-----------|---------------|-------------|-----------|")
            for run in stage_b:
                m = run['metrics']
                output.append(
                    f"| {run['run_id']} | {run['timestamp'][:16]} | "
                    f"{m.get('train_samples', 0)} | {m.get('val_samples', 0)} | "
                    f"{m.get('positives', 0)} |"
                )
            output.append("")
    
    elif track == 'B':
        output.append("| Run ID | Timestamp | Accuracy | mAP | TTC MAE (s) | Candidates |")
        output.append("|--------|-----------|----------|-----|-------------|------------|")
        for run in runs:
            m = run['metrics']
            output.append(
                f"| {run['run_id'][-20:]} | {run['timestamp'][:16]} | "
                f"{m.get('accuracy', 0):.3f} | {m.get('mAP', 0):.3f} | "
                f"{m.get('ttc_mae_seconds', 0):.3f} | {m.get('num_candidates', 0)} |"
            )
        output.append("")
        
        # Add best metrics
        if runs:
            best_map = max(runs, key=lambda r: r['metrics'].get('mAP', 0))
            best_acc = max(runs, key=lambda r: r['metrics'].get('accuracy', 0))
            output.append(f"**Best mAP:** {best_map['metrics'].get('mAP', 0):.3f} ({best_map['run_id'][-20:]})\n")
            output.append(f"**Best Accuracy:** {best_acc['metrics'].get('accuracy', 0):.3f} ({best_acc['run_id'][-20:]})\n")
    
    elif track == 'C':
        output.append("| Run ID | Timestamp | RGTP Rate | Accuracy | mAP | TTC MAE | Latency (ms) |")
        output.append("|--------|-----------|-----------|----------|-----|---------|--------------|")
        for run in runs:
            m = run['metrics']
            lat = m.get('latency_mean_ms', m.get('latency_ms', 0))
            output.append(
                f"| {run['run_id'][-25:]} | {run['timestamp'][:16]} | "
                f"{run['rgtp_rate']:.2f} | {m.get('accuracy', 0):.3f} | "
                f"{m.get('mAP', 0):.3f} | {m.get('ttc_mae_seconds', 0):.3f} | "
                f"{lat:.1f} |"
            )
        output.append("")
        
        # Group by pruning rate to show Pareto curve
        by_rate = {}
        for run in runs:
            rate = run['rgtp_rate']
            if rate not in by_rate:
                by_rate[rate] = []
            by_rate[rate].append(run)
        
        if len(by_rate) > 1:
            output.append("\n#### Accuracy vs Pruning Rate (Pareto Analysis)\n")
            output.append("| RGTP Rate | Best mAP | Best Accuracy | Avg Latency (ms) |")
            output.append("|-----------|----------|---------------|-------------------|")
            for rate in sorted(by_rate.keys()):
                rate_runs = by_rate[rate]
                best_map = max(rate_runs, key=lambda r: r['metrics'].get('mAP', 0))
                best_acc = max(rate_runs, key=lambda r: r['metrics'].get('accuracy', 0))
                avg_lat = sum(r['metrics'].get('latency_mean_ms', r['metrics'].get('latency_ms', 0)) 
                             for r in rate_runs) / len(rate_runs)
                output.append(
                    f"| {rate:.2f} | {best_map['metrics'].get('mAP', 0):.3f} | "
                    f"{best_acc['metrics'].get('accuracy', 0):.3f} | {avg_lat:.1f} |"
                )
            output.append("")
    
    return "\n".join(output)


def generate_comparison_report(track_a: List, track_b: List, track_c: List, output_file: Optional[str] = None):
    """Generate a comprehensive comparison report."""
    lines = []
    lines.append("# Ego4D-LiteSTA Results Comparison")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")
    
    # Overview
    lines.append("## Overview\n")
    lines.append(f"- **Track A Runs:** {len(track_a)} (Stage A: {len([r for r in track_a if r.get('stage')=='stageA'])}, Stage B: {len([r for r in track_a if r.get('stage')=='stageB'])}")
    lines.append(f"- **Track B Runs:** {len(track_b)}")
    lines.append(f"- **Track C Runs:** {len(track_c)}\n")
    
    # Track summaries
    lines.append(format_metrics_table(track_a, 'A'))
    lines.append(format_metrics_table(track_b, 'B'))
    lines.append(format_metrics_table(track_c, 'C'))
    
    # Pipeline summary
    if track_a and track_b and track_c:
        lines.append("\n## Full Pipeline Performance\n")
        lines.append("Combining best results from each track:\n")
        
        latest_a = max([r for r in track_a if r.get('stage')=='stageB'], 
                      key=lambda r: r['timestamp'], default=None)
        best_b = max(track_b, key=lambda r: r['metrics'].get('mAP', 0), default=None)
        baseline_c = [r for r in track_c if r['rgtp_rate'] == 0.0]
        best_c = max(baseline_c, key=lambda r: r['metrics'].get('mAP', 0), default=None) if baseline_c else None
        
        if latest_a:
            lines.append(f"**Track A (Latest):** {latest_a['metrics'].get('val_samples', 0)} validation samples")
        if best_b:
            lines.append(f"**Track B (Best mAP):** {best_b['metrics'].get('mAP', 0):.3f} mAP, "
                        f"{best_b['metrics'].get('accuracy', 0):.3f} accuracy")
        if best_c:
            lines.append(f"**Track C (Baseline):** {best_c['metrics'].get('mAP', 0):.3f} mAP, "
                        f"{best_c['metrics'].get('latency_mean_ms', 0):.1f}ms latency")
    
    # Recommendations
    lines.append("\n## Next Steps\n")
    lines.append("### For Detailed Analysis:\n")
    lines.append("- **Track A plots:** `local_extraction/runs/Track_A/plots/`")
    lines.append("- **Track B plots:** `python local_extraction/trackB/trackB_plots.py`")
    lines.append("- **Track C plots:** `python local_extraction/trackC/trackC_plots.py`")
    lines.append("\n### Key Metrics to Compare:\n")
    lines.append("- **Track A → B:** Does Stage B recall match Track B candidate coverage?")
    lines.append("- **Track B → C:** Does pruning maintain mAP while reducing latency?")
    lines.append("- **Overall:** Accuracy/mAP trend from candidate generation through final prediction\n")
    
    report = "\n".join(lines)
    
    if output_file:
        Path(output_file).write_text(report, encoding='utf-8')
        print(f"Report saved to: {output_file}")
    else:
        print(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Compare Track A, B, and C results')
    parser.add_argument('--latest', action='store_true', help='Show only the latest run from each track')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: print to console)')
    parser.add_argument('--runs-dir', type=str, default='local_extraction/runs', 
                       help='Base runs directory')
    args = parser.parse_args()
    
    runs_base = Path(args.runs_dir)
    
    print("Scanning for runs...")
    track_a = find_track_a_runs(runs_base / "Track_A")
    track_b = find_track_b_runs(runs_base / "Track_B")
    track_c = find_track_c_runs(runs_base / "Track_C")
    
    print(f"Found: {len(track_a)} Track A, {len(track_b)} Track B, {len(track_c)} Track C runs\n")
    
    if args.latest:
        if track_a:
            track_a = [max(track_a, key=lambda r: r['timestamp'])]
        if track_b:
            track_b = [max(track_b, key=lambda r: r['timestamp'])]
        if track_c:
            track_c = [max(track_c, key=lambda r: r['timestamp'])]
    
    generate_comparison_report(track_a, track_b, track_c, args.output)


if __name__ == "__main__":
    main()
