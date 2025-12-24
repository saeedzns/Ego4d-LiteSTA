#!/usr/bin/env python3
"""
Track C Plotting Utilities

Quick visualizations for Track C pruning runs under:
  local_extraction/runs/Track_C/

What it does:
- Scans metrics JSON files (metrics/trackC_val_rate*.json)
- Aggregates scalar metrics per evaluation (accuracy, mAP, TTC MAE, N/N+V/N+δ/All, top-5 variants, RGTP stats)
- Produces simple line plots showing how pruning runs performed over time

Usage (from repo root):

  python local_extraction/trackC/trackC_plots.py \
      --runs_dir local_extraction/runs/Track_C \
      --out_dir  local_extraction/runs/Track_C/plots
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MetricPoint:
    """Single evaluation snapshot for one pruning run."""

    idx: int
    checkpoint: str
    rate: Optional[float]
    timestamp: Optional[str]
    values: Dict[str, float] = field(default_factory=dict)


IMPORTANT_KEYS_ORDER = [
    "accuracy",
    "mAP",
    "ttc_mae_seconds",
    "rgtp_mean_fraction_pruned",
    "rgtp_rate_request",
    "latency_ms_mean",
    "latency_ms_median",
    "latency_ms_p90",
    "latency_ms_p95",
    "throughput_samples_per_s",
    "throughput_candidates_per_s",
    "head_benchmark_ms_mean",
    "peak_vram_bytes",
    "peak_vram_reserved_bytes",
    "head_flops",
    "N_mAP",
    "Nv_mAP",
    "N_delta_mAP",
    "All_mAP",
    "N_top5_acc",
    "Nv_top5_acc",
    "N_delta_top5_acc",
    "All_top5_acc",
    "N_top5_mAP",
    "Nv_top5_mAP",
    "N_delta_top5_mAP",
    "All_top5_mAP",
]


def _try_load_trackc_cfg() -> Optional[Any]:
    """Best-effort load of Track C YAML config.

    Returns a config-like object supporting `.get(key, default)` on success, else None.
    """
    try:
        from core import load_config
        return load_config("trackC")
    except Exception:
        return None


def load_metrics(runs_dir: Path, metrics_subdir: str = "metrics") -> List[MetricPoint]:
    metrics_dir = runs_dir / metrics_subdir
    if not metrics_dir.exists():
        raise FileNotFoundError(f"metrics directory not found: {metrics_dir}")
    # Exclude *_summary.json files
    files = sorted(f for f in metrics_dir.glob("trackC_val_rate*.json") if not f.name.endswith("_summary.json"))
    raw_points: List[tuple[datetime, Path, str, Optional[float], str, Dict[str, float]]] = []
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        checkpoint = str(obj.get("checkpoint", "")) or path.name
        rate = obj.get("rgtp_rate_request")
        ts = obj.get("timestamp")
        # Parse timestamp from filename (YYYYMMDD_HHMMSS) or use file modification time
        match = re.search(r'(\d{8})_(\d{6})', path.name)
        if match:
            dt = datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S')
        else:
            dt = datetime.fromtimestamp(path.stat().st_mtime)
        values: Dict[str, float] = {}
        for k, v in obj.items():
            if isinstance(v, (int, float)):
                values[k] = float(v)
        raw_points.append((dt, path, checkpoint, rate, ts, values))
    # Sort by datetime (chronological order)
    raw_points.sort(key=lambda x: x[0])
    # Assign indices based on chronological order
    points: List[MetricPoint] = []
    for i, (dt, path, checkpoint, rate, ts, values) in enumerate(raw_points):
        points.append(MetricPoint(idx=i, checkpoint=checkpoint, rate=rate, timestamp=ts, values=values))
    return points


def _select_keys(points: List[MetricPoint]) -> List[str]:
    """Return an ordered list of metric keys present in at least one file."""
    all_keys = set()
    for p in points:
        all_keys.update(p.values.keys())
    selected = [k for k in IMPORTANT_KEYS_ORDER if k in all_keys]
    for k in sorted(all_keys):
        if k not in selected and not k.endswith("_stats"):
            selected.append(k)
    return selected


def _generate_colors(n: int, seed: int = 42):
    """Generate n distinct random colors avoiding blue (for contrast with trend line)."""
    import random
    rng = random.Random(seed)
    colors = []
    for _ in range(n):
        # Keep blue channel low (<0.3) so points contrast with dark blue trend line
        r, g, b = rng.random(), rng.random(), rng.random() * 0.3
        colors.append((r, g, b))
    return colors


def plot_metrics(points: List[MetricPoint], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[trackC.plots] matplotlib not available ({e}); cannot draw plots.")
        return

    if not points:
        print("[trackC.plots] No metrics files found; nothing to plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = _select_keys(points)
    xs = [p.idx for p in points]

    # Assign a random color to each eval index
    colors = _generate_colors(len(points))

    # Figure layout: metrics subplots + one legend subplot
    n_keys = len(keys)
    n_cols = 2
    n_rows_metrics = (n_keys + n_cols - 1) // n_cols
    # Add one extra row for the legend
    n_rows = n_rows_metrics + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    metric_axes = axes_flat[: n_rows_metrics * n_cols]
    legend_axes = axes_flat[n_rows_metrics * n_cols :]

    for ax, key in zip(metric_axes, keys):
        ys = [p.values.get(key, float("nan")) for p in points]
        # Plot each point with its assigned color, no legend on metric plots
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.plot(x, y, marker="o", color=colors[i], linestyle="None", markersize=8)
        # Connect with a thick dark blue line for trend
        ax.plot(xs, ys, color="darkblue", alpha=0.9, linestyle="-", linewidth=2)
        ax.set_title(key)
        ax.set_xlabel("Run index (chronological order)")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

    # Hide unused metric subplots
    for ax in metric_axes[len(keys):]:
        ax.set_visible(False)

    # Use the legend row: merge both columns into one axis for legend
    for ax in legend_axes:
        ax.set_visible(False)
    # Create a new axis spanning the bottom row for legend
    ax_legend = fig.add_subplot(n_rows, 1, n_rows)
    ax_legend.axis("off")
    for i, p in enumerate(points):
        rate_str = f"rate={p.rate:.2f}" if p.rate is not None else "rate=N/A"
        label = f"idx={p.idx} | {Path(p.checkpoint).name} | {rate_str} | ts={p.timestamp or 'N/A'}"
        ax_legend.scatter([], [], color=colors[i], label=label, s=60)
    ax_legend.legend(loc="center", fontsize=8, frameon=True, ncol=2)
    ax_legend.set_title("Legend: Eval Index → Checkpoint / Rate", fontsize=10)

    fig.tight_layout()
    out_path = out_dir / "trackC_metrics_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[trackC.plots] Saved metric curves + legend → {out_path}")

    # Also produce a figure focusing only on top-5 semantic mAP (same layout, fewer plots)
    top5_keys = [
        "N_top5_mAP",
        "Nv_top5_mAP",
        "N_delta_top5_mAP",
        "All_top5_mAP",
    ]
    present_top5_keys = [k for k in top5_keys if any(k in p.values for p in points)]
    if present_top5_keys:
        n_keys_top5 = len(present_top5_keys)
        n_cols_top5 = 2
        n_rows_metrics_top5 = (n_keys_top5 + n_cols_top5 - 1) // n_cols_top5
        n_rows_top5 = n_rows_metrics_top5 + 1  # extra row for legend
        fig2, axes2 = plt.subplots(
            n_rows_top5,
            n_cols_top5,
            figsize=(6 * n_cols_top5, 3 * n_rows_top5),
            squeeze=False,
        )
        axes_flat2 = [ax for row in axes2 for ax in row]
        metric_axes2 = axes_flat2[: n_rows_metrics_top5 * n_cols_top5]
        legend_axes2 = axes_flat2[n_rows_metrics_top5 * n_cols_top5 :]

        xs_top5 = [p.idx for p in points]
        for ax, key in zip(metric_axes2, present_top5_keys):
            ys = [p.values.get(key, float("nan")) * 100.0 for p in points]
            for i, (x, y) in enumerate(zip(xs_top5, ys)):
                ax.plot(x, y, marker="o", color=colors[i], linestyle="None", markersize=8)
            ax.plot(xs_top5, ys, color="darkblue", alpha=0.9, linestyle="-", linewidth=2)
            ax.set_title(key)
            ax.set_xlabel("Run index (chronological order)")
            ax.set_ylabel("Top-5 mAP (%)")
            ax.grid(True, alpha=0.3)

        # Hide unused metric subplots
        for ax in metric_axes2[len(present_top5_keys) :]:
            ax.set_visible(False)

        # Legend row mirroring the main layout
        for ax in legend_axes2:
            ax.set_visible(False)
        ax_legend2 = fig2.add_subplot(n_rows_top5, 1, n_rows_top5)
        ax_legend2.axis("off")
        for i, p in enumerate(points):
            rate_str = f"rate={p.rate:.2f}" if p.rate is not None else "rate=N/A"
            label = f"idx={p.idx} | {Path(p.checkpoint).name} | {rate_str} | ts={p.timestamp or 'N/A'}"
            ax_legend2.scatter([], [], color=colors[i], label=label, s=60)
        ax_legend2.legend(loc="center", fontsize=8, frameon=True, ncol=2)
        ax_legend2.set_title("Legend: Eval Index → Checkpoint / Rate", fontsize=10)

        fig2.tight_layout()
        out_top5 = out_dir / "trackC_top5_semantic_percent.png"
        fig2.savefig(out_top5, dpi=150)
        plt.close(fig2)
        print(f"[trackC.plots] Saved top-5 semantic mAP subplot grid → {out_top5}")

    summary_path = out_dir / "trackC_metrics_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        header = ["idx", "checkpoint", "rgtp_rate_request", "timestamp"] + keys
        f.write("\t".join(header) + "\n")
        for p in points:
            row: List[str] = [
                str(p.idx),
                Path(p.checkpoint).name,
                "" if p.rate is None else f"{p.rate:.3f}",
                str(p.timestamp or ""),
            ]
            for k in keys:
                v = p.values.get(k)
                row.append("" if v is None else f"{v:.6f}")
            f.write("\t".join(row) + "\n")
    print(f"[trackC.plots] Saved metrics summary -> {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Track C pruning metrics plotting helper.")
    ap.add_argument(
        "--runs_dir",
        type=str,
        default=None,
        help="Path to Track_C runs directory (default: from configs/trackC.yaml -> output.runs_dir).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <runs_dir>/<plots_subdir> from configs/trackC.yaml).",
    )
    args = ap.parse_args()

    cfg = _try_load_trackc_cfg()
    default_runs_dir = str(Path("local_extraction") / "runs" / "Track_C")
    runs_dir_cfg = cfg.get("output.runs_dir", default_runs_dir) if cfg is not None else default_runs_dir
    metrics_subdir = cfg.get("output.metrics_subdir", "metrics") if cfg is not None else "metrics"
    plots_subdir = cfg.get("output.plots_subdir", "plots") if cfg is not None else "plots"

    runs_dir = Path(args.runs_dir) if args.runs_dir is not None else Path(str(runs_dir_cfg))
    out_dir = Path(args.out_dir) if args.out_dir is not None else runs_dir / str(plots_subdir)

    points = load_metrics(runs_dir, metrics_subdir=str(metrics_subdir))
    if not points:
        print(f"[trackC.plots] No trackC_val_rate*.json found under {runs_dir}/{metrics_subdir}.")
        return

    print(f"[trackC.plots] Loaded {len(points)} metric snapshots from {runs_dir}/{metrics_subdir}.")
    plot_metrics(points, out_dir)


if __name__ == "__main__":
    main()
