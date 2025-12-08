#!/usr/bin/env python3
"""
Compare Track B (unpruned) and Track C (pruned) metrics.

This script loads:
  - One or more Track B metrics JSON files under local_extraction/runs/Track_B/metrics
  - One or more Track C metrics JSON files under local_extraction/runs/Track_C/metrics

and prints a side-by-side comparison for key metrics:
  - accuracy, mAP, ttc_mae_seconds
  - N_mAP, Nv_mAP, N_delta_mAP, All_mAP
  - N_top5_acc/mAP, Nv_top5_acc/mAP, N_delta_top5_acc/mAP, All_top5_acc/mAP

Usage (from repo root):

  # Compare latest Track B vs Track C (single pair):
  python local_extraction/trackC/trackC_compare_metrics.py

  # Compare all runs from both tracks and save to TSV:
  python local_extraction/trackC/trackC_compare_metrics.py --all --out comparison.tsv

  # Generate a line plot comparing Track B vs Track C over time:
  python local_extraction/trackC/trackC_compare_metrics.py --all --plot

You can override specific files with:

  --b_metrics path/to/metrics_val_YYYY.json
  --c_metrics path/to/trackC_val_rateXX_YYYY.json

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import re


KEYS_ORDER = [
    "N_top5_mAP",
    "Nv_top5_mAP",
    "N_delta_top5_mAP",
    "All_top5_mAP",
    "accuracy",
    "mAP",
    "ttc_mae_seconds",
    "N_mAP",
    "Nv_mAP",
    "N_delta_mAP",
    "All_mAP",
    "N_top5_acc",
    "Nv_top5_acc",
    "N_delta_top5_acc",
    "All_top5_acc",
]


def _latest_json(dir_path: Path, prefix: str) -> Optional[Path]:
    if not dir_path.exists():
        return None
    files = sorted(dir_path.glob(prefix + "*.json"))
    return files[-1] if files else None


def _all_jsons(dir_path: Path, prefix: str) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted(dir_path.glob(prefix + "*.json"))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.6f}"
    if isinstance(x, int):
        return str(x)
    return "-"


def _build_row(track: str, path: Path, data: Dict[str, Any]) -> Dict[str, str]:
    row: Dict[str, str] = {
        "track": track,
        "file": path.name,
        "checkpoint": Path(data.get("checkpoint", "")).name or "-",
        "timestamp": data.get("timestamp") or "-",
    }
    for key in KEYS_ORDER:
        row[key] = _fmt(data.get(key))
    return row


def compare_single(b_path: Path, c_path: Path) -> None:
    """Print a single pair comparison with metrics as columns."""
    b = _load_json(b_path)
    c = _load_json(c_path)

    print(f"Track B: {b_path}")
    print(f"Track C: {c_path}\n")

    # Header row
    header = ["track"] + KEYS_ORDER + ["file", "checkpoint", "timestamp"]
    print("\t".join(header))
    print("-" * 120)

    b_row = _build_row("B", b_path, b)
    c_row = _build_row("C", c_path, c)

    print("\t".join(b_row.get(h, "-") for h in header))
    print("\t".join(c_row.get(h, "-") for h in header))


def compare_all(b_dir: Path, c_dir: Path, out_path: Optional[Path], plot: bool = False, plot_metric: str = "mAP") -> None:
    """Load all Track B and C metrics, print table, optionally save to TSV and plot."""
    b_files = _all_jsons(b_dir, "metrics_val_")
    c_files = _all_jsons(c_dir, "trackC_val_rate")

    if not b_files and not c_files:
        print("[compare] No metrics files found in Track B or Track C.")
        return

    rows: List[Dict[str, str]] = []
    b_data_list: List[Dict[str, Any]] = []
    c_data_list: List[Dict[str, Any]] = []

    for p in b_files:
        try:
            data = _load_json(p)
            data["_path"] = p
            rows.append(_build_row("B", p, data))
            b_data_list.append(data)
        except Exception as e:
            print(f"[compare] Skipping {p}: {e}")

    for p in c_files:
        try:
            data = _load_json(p)
            data["_path"] = p
            rows.append(_build_row("C", p, data))
            c_data_list.append(data)
        except Exception as e:
            print(f"[compare] Skipping {p}: {e}")

    if not rows:
        print("[compare] No valid metrics loaded.")
        return

    header = ["track"] + KEYS_ORDER + ["file", "checkpoint", "timestamp"]

    # Print to console
    print("\t".join(header))
    print("-" * 160)
    for row in rows:
        print("\t".join(row.get(h, "-") for h in header))

    # Save to TSV if requested
    if out_path:
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(header) + "\n")
            for row in rows:
                f.write("\t".join(row.get(h, "-") for h in header) + "\n")
        print(f"\n[compare] Saved comparison table -> {out_path}")

    # Generate plot if requested
    if plot:
        _generate_checkpoint_plot(b_data_list, c_data_list, plot_metric)


def _extract_timestamp(data: Dict[str, Any]) -> Optional[datetime]:
    """Extract timestamp from metrics data or filename."""
    # Try timestamp field first
    ts = data.get("timestamp")
    if ts:
        try:
            return datetime.fromisoformat(ts)
        except:
            pass
    
    # Try to extract from filename (e.g., metrics_val_20251207_123456.json)
    path = data.get("_path")
    if path:
        name = path.name if isinstance(path, Path) else Path(path).name
        # Match patterns like 20251207_123456 or 20251207_1234
        match = re.search(r'(\d{8})_(\d{4,6})', name)
        if match:
            date_str, time_str = match.groups()
            # Pad time to 6 digits if needed
            time_str = time_str.ljust(6, '0')
            try:
                return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            except:
                pass
    return None


def _extract_checkpoint_name(data: Dict[str, Any]) -> str:
    """Extract a short checkpoint name for display."""
    ckpt = data.get("checkpoint", "")
    if not ckpt or ckpt == "-":
        return "unknown"
    
    name = Path(ckpt).stem if ckpt else "unknown"
    
    # Shorten common patterns
    name = name.replace("trackB_best_mAP_", "mAP_")
    name = name.replace("trackB_best_", "best_")
    name = name.replace("trackB_final_", "final_")
    
    # Extract mAP value if present (e.g., mAP_0.3580_20251206_083053 -> mAP_0.358)
    match = re.search(r'mAP_(\d+\.\d+)', name)
    if match:
        mAP_val = float(match.group(1))
        # Also extract date
        date_match = re.search(r'(\d{8})_\d+$', name)
        if date_match:
            return f"mAP_{mAP_val:.3f}_{date_match.group(1)[-4:]}"  # Last 4 digits of date (MMDD)
        return f"mAP_{mAP_val:.3f}"
    
    return name[:20]  # Truncate if too long


def _generate_checkpoint_plot(
    b_data_list: List[Dict[str, Any]], 
    c_data_list: List[Dict[str, Any]], 
    metric: str = "mAP"
) -> None:
    """Generate a bar/line plot comparing Track B vs Track C grouped by checkpoint."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[compare] matplotlib not installed. Install with: pip install matplotlib")
        return

    # Group data by checkpoint
    def group_by_checkpoint(data_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for data in data_list:
            ckpt = data.get("checkpoint", "unknown")
            if ckpt and ckpt != "-":
                ckpt_key = Path(ckpt).stem
            else:
                ckpt_key = "unknown"
            if ckpt_key not in groups:
                groups[ckpt_key] = []
            groups[ckpt_key].append(data)
        return groups

    b_groups = group_by_checkpoint(b_data_list)
    c_groups = group_by_checkpoint(c_data_list)

    # Get all unique checkpoints, sorted by timestamp of first occurrence
    all_checkpoints = set(b_groups.keys()) | set(c_groups.keys())
    
    def get_first_timestamp(groups: Dict, ckpt: str) -> datetime:
        if ckpt in groups and groups[ckpt]:
            ts = _extract_timestamp(groups[ckpt][0])
            return ts if ts else datetime.min
        return datetime.min
    
    checkpoints_sorted = sorted(all_checkpoints, 
                                 key=lambda c: max(get_first_timestamp(b_groups, c), 
                                                   get_first_timestamp(c_groups, c)))
    
    # Filter out 'unknown' and empty
    checkpoints_sorted = [c for c in checkpoints_sorted if c and c != "unknown"]
    
    if not checkpoints_sorted:
        print("[compare] No valid checkpoints found for plotting.")
        return

    # Create short names for x-axis
    short_names = [_extract_checkpoint_name({"checkpoint": c}) for c in checkpoints_sorted]

    # Metrics to plot
    metrics_to_plot = ["mAP", "N_mAP", "All_top5_mAP", "accuracy"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    bar_width = 0.35
    x = np.arange(len(checkpoints_sorted))

    for ax, m in zip(axes, metrics_to_plot):
        # Get best value for each checkpoint (or average if multiple runs)
        b_vals = []
        c_vals = []
        
        for ckpt in checkpoints_sorted:
            # Track B values for this checkpoint
            if ckpt in b_groups:
                vals = [d.get(m) for d in b_groups[ckpt] if d.get(m) is not None]
                b_vals.append(max(vals) if vals else 0)
            else:
                b_vals.append(0)
            
            # Track C values for this checkpoint
            if ckpt in c_groups:
                vals = [d.get(m) for d in c_groups[ckpt] if d.get(m) is not None]
                c_vals.append(max(vals) if vals else 0)
            else:
                c_vals.append(0)

        # Create grouped bar chart
        bars_b = ax.bar(x - bar_width/2, b_vals, bar_width, label='Track B (unpruned)', color='steelblue', alpha=0.8)
        bars_c = ax.bar(x + bar_width/2, c_vals, bar_width, label='Track C (pruned)', color='coral', alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars_b, b_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
        for bar, val in zip(bars_c, c_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

        # Also add line plot to show progression
        if any(v > 0 for v in b_vals):
            b_valid = [(i, v) for i, v in enumerate(b_vals) if v > 0]
            if b_valid:
                ax.plot([p[0] for p in b_valid], [p[1] for p in b_valid], 
                       'b--o', linewidth=1.5, markersize=4, alpha=0.6)
        if any(v > 0 for v in c_vals):
            c_valid = [(i, v) for i, v in enumerate(c_vals) if v > 0]
            if c_valid:
                ax.plot([p[0] for p in c_valid], [p[1] for p in c_valid], 
                       'r--s', linewidth=1.5, markersize=4, alpha=0.6)

        ax.set_title(f'{m}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel(m)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

    plt.suptitle('Track B vs Track C: Progress by Checkpoint', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_dir = Path("local_extraction") / "runs" / "Track_C" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"trackBC_checkpoint_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[compare] Saved checkpoint progress plot -> {plot_path}")

    # Generate second plot for top5 metrics
    top5_metrics = ["N_top5_mAP", "Nv_top5_mAP", "N_delta_top5_mAP", "All_top5_mAP"]
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    axes2 = axes2.flatten()

    for ax, m in zip(axes2, top5_metrics):
        # Get best value for each checkpoint
        b_vals = []
        c_vals = []
        
        for ckpt in checkpoints_sorted:
            if ckpt in b_groups:
                vals = [d.get(m) for d in b_groups[ckpt] if d.get(m) is not None]
                b_vals.append(max(vals) if vals else 0)
            else:
                b_vals.append(0)
            
            if ckpt in c_groups:
                vals = [d.get(m) for d in c_groups[ckpt] if d.get(m) is not None]
                c_vals.append(max(vals) if vals else 0)
            else:
                c_vals.append(0)

        # Create grouped bar chart
        bars_b = ax.bar(x - bar_width/2, b_vals, bar_width, label='Track B (unpruned)', color='steelblue', alpha=0.8)
        bars_c = ax.bar(x + bar_width/2, c_vals, bar_width, label='Track C (pruned)', color='coral', alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars_b, b_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
        for bar, val in zip(bars_c, c_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

        # Add line plot for progression
        if any(v > 0 for v in b_vals):
            b_valid = [(i, v) for i, v in enumerate(b_vals) if v > 0]
            if b_valid:
                ax.plot([p[0] for p in b_valid], [p[1] for p in b_valid], 
                       'b--o', linewidth=1.5, markersize=4, alpha=0.6)
        if any(v > 0 for v in c_vals):
            c_valid = [(i, v) for i, v in enumerate(c_vals) if v > 0]
            if c_valid:
                ax.plot([p[0] for p in c_valid], [p[1] for p in c_valid], 
                       'r--s', linewidth=1.5, markersize=4, alpha=0.6)

        ax.set_title(f'{m}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel(m)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

    plt.suptitle('Track B vs Track C: Top-5 Metrics by Checkpoint', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save top5 plot
    top5_plot_path = plot_dir / f"trackBC_checkpoint_top5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(top5_plot_path, dpi=150, bbox_inches='tight')
    print(f"[compare] Saved top5 metrics plot -> {top5_plot_path}")

    # Also show plot
    plt.show()


def _generate_progress_plot(
    b_data_list: List[Dict[str, Any]], 
    c_data_list: List[Dict[str, Any]], 
    metric: str = "mAP"
) -> None:
    """Generate a line plot comparing Track B vs Track C progress over time."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[compare] matplotlib not installed. Install with: pip install matplotlib")
        return

    # Extract timestamps and metric values
    b_points = []
    for data in b_data_list:
        ts = _extract_timestamp(data)
        val = data.get(metric)
        if ts and val is not None:
            b_points.append((ts, val))
    
    c_points = []
    for data in c_data_list:
        ts = _extract_timestamp(data)
        val = data.get(metric)
        if ts and val is not None:
            c_points.append((ts, val))

    # Sort by timestamp
    b_points.sort(key=lambda x: x[0])
    c_points.sort(key=lambda x: x[0])

    if not b_points and not c_points:
        print(f"[compare] No valid data points for metric '{metric}'")
        return

    # Create figure with multiple subplots for key metrics
    metrics_to_plot = ["mAP", "N_mAP", "All_top5_mAP", "accuracy"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, m in zip(axes, metrics_to_plot):
        # Get data for this metric
        b_pts = [(ts, data.get(m)) for data in b_data_list 
                 if (ts := _extract_timestamp(data)) and data.get(m) is not None]
        c_pts = [(ts, data.get(m)) for data in c_data_list 
                 if (ts := _extract_timestamp(data)) and data.get(m) is not None]
        
        b_pts.sort(key=lambda x: x[0])
        c_pts.sort(key=lambda x: x[0])

        if b_pts:
            b_times, b_vals = zip(*b_pts)
            ax.plot(b_times, b_vals, 'b-o', label='Track B (unpruned)', linewidth=2, markersize=6)
        
        if c_pts:
            c_times, c_vals = zip(*c_pts)
            ax.plot(c_times, c_vals, 'r-s', label='Track C (pruned)', linewidth=2, markersize=6)

        ax.set_title(f'{m}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(m)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Track B vs Track C Progress Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_dir = Path("local_extraction") / "runs" / "Track_C" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"trackBC_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[compare] Saved progress plot -> {plot_path}")

    # Also show plot
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Track B vs Track C metrics.")
    ap.add_argument(
        "--b_metrics",
        type=str,
        default=None,
        help="Track B metrics JSON path; defaults to latest metrics_val_*.json under runs/Track_B/metrics.",
    )
    ap.add_argument(
        "--c_metrics",
        type=str,
        default=None,
        help="Track C metrics JSON path; defaults to latest trackC_val_rate*.json under runs/Track_C/metrics.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Compare all runs from both Track B and Track C together.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output TSV file path (used with --all).",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Generate a line plot showing Track B vs Track C progress over time (used with --all).",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="mAP",
        help="Primary metric to highlight in plot (default: mAP).",
    )
    args = ap.parse_args()

    b_dir = Path("local_extraction") / "runs" / "Track_B" / "metrics"
    c_dir = Path("local_extraction") / "runs" / "Track_C" / "metrics"

    if args.all:
        out_path = Path(args.out) if args.out else None
        compare_all(b_dir, c_dir, out_path, plot=args.plot, plot_metric=args.metric)
    else:
        b_path = Path(args.b_metrics) if args.b_metrics else _latest_json(b_dir, "metrics_val_")
        c_path = Path(args.c_metrics) if args.c_metrics else _latest_json(c_dir, "trackC_val_rate")

        if b_path is None or not b_path.exists():
            raise FileNotFoundError(f"Track B metrics not found (searched in {b_dir}); specify --b_metrics.")
        if c_path is None or not c_path.exists():
            raise FileNotFoundError(f"Track C metrics not found (searched in {c_dir}); specify --c_metrics.")

        compare_single(b_path, c_path)


if __name__ == "__main__":
    main()

