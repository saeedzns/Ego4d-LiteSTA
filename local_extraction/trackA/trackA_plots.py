#!/usr/bin/env python3
"""
Track A Plotting Script
-----------------------
Generates plots from Track A Stage B summary.json files.

Usage:
    python trackA_plots.py [--summary PATH] [--out DIR]

Example:
    python trackA_plots.py --summary runs/Track_A/trackA_stageB_20251117_184342/summary.json --out runs/Track_A/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_summary(summary_path: Path) -> dict:
    """Load summary.json file."""
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# PLOTS FOR K-SWEEP SUMMARIES (e.g., KSweep_20251115_225146)
# Expected keys: "k_values", "metrics" (list with K, micro_recall, etc.)
# =============================================================================

def plot_k_sweep_metrics(data: dict, out_dir: Path, run_name: str) -> Path:
    """Plot K-sweep recall and F-beta metrics."""
    df_metrics = pd.DataFrame(data["metrics"])
    k_values = data.get("k_values", df_metrics["K"].tolist())

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Recall Subplot
    axes[0].plot(
        df_metrics["K"], df_metrics["micro_recall"],
        marker="o", label="Micro Recall", color="royalblue", linewidth=2
    )
    axes[0].plot(
        df_metrics["K"], df_metrics["macro_recall"],
        marker="s", label="Macro Recall", color="darkorange", linewidth=2, linestyle="--"
    )
    axes[0].set_ylabel("Score")
    axes[0].set_title("Impact of K on Recall", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # F-beta Subplot
    axes[1].plot(
        df_metrics["K"], df_metrics["micro_fbeta"],
        marker="o", label="Micro F-beta", color="forestgreen", linewidth=2
    )
    axes[1].plot(
        df_metrics["K"], df_metrics["macro_fbeta"],
        marker="s", label="Macro F-beta", color="crimson", linewidth=2, linestyle="--"
    )
    axes[1].set_xlabel("K Value")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Impact of K on F-beta Score", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)
    axes[1].set_xticks(k_values)

    plt.tight_layout()
    out_path = out_dir / f"trackA_k_sweep_{run_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confidence_stats(data: dict, out_dir: Path, run_name: str) -> Path:
    """Plot confidence statistics bar chart."""
    conf_stats = data["confidence_stats"]
    labels = ["Median", "Mean", "90th Pctl", "95th Pctl", "Max"]
    values = [
        conf_stats["median_conf"],
        conf_stats["mean_conf"],
        conf_stats["p90_conf"],
        conf_stats["p95_conf"],
        conf_stats["max_conf"],
    ]
    colors = sns.color_palette("Blues", n_colors=len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.8)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval + 0.01,
            f"{yval:.3f}", ha="center", va="bottom", fontweight="bold"
        )

    ax.set_ylabel("Confidence Score (0.0 - 1.0)")
    ax.set_title("Model Prediction Confidence Statistics", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = out_dir / f"trackA_confidence_{run_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_prediction_stats(data: dict, out_dir: Path, run_name: str) -> Path:
    """Plot predictions per image bar chart."""
    pred_stats = data["prediction_stats"]
    labels = ["Median", "Mean", "95th Pctl"]
    values = [
        pred_stats["median_preds_per_image"],
        pred_stats["mean_preds_per_image"],
        pred_stats["p95_preds_per_image"],
    ]
    colors = sns.color_palette("Purples", n_colors=len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.8)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval + 0.1,
            f"{yval:.2f}", ha="center", va="bottom", fontweight="bold"
        )

    ax.set_ylabel("Count per Image")
    ax.set_title("Statistics: Predictions per Image", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = out_dir / f"trackA_preds_per_image_{run_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# PLOTS FOR STAGE B SUMMARIES (e.g., trackA_stageB_20251117_184342)
# Expected keys: "recall_metrics", "head_manifests", "detector"
# =============================================================================

def plot_recall_metrics(data: dict, out_dir: Path, run_name: str) -> Path:
    """Plot recall metrics from Stage B summary."""
    recall = data["recall_metrics"]

    # Summary bar chart
    labels = ["Recall@K", "Mean Best IoU"]
    values = [recall["recall_at_K"], recall["mean_best_iou"]]
    colors = ["royalblue", "darkorange"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.8)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval + 0.02,
            f"{yval:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=12
        )

    ax.set_ylabel("Score")
    ax.set_title(f"Track A Stage B: Recall Metrics (K={data['detector']['K']})", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = out_dir / f"trackA_recall_metrics_{run_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_iou_histogram(data: dict, out_dir: Path, run_name: str) -> Path:
    """Plot IoU distribution histogram from Stage B summary."""
    recall = data["recall_metrics"]
    hist = recall["best_iou_hist"]

    # Sort bins by range
    bins_order = ["0.0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-0.9", "0.9-1.01"]
    labels = [b for b in bins_order if b in hist]
    values = [hist[b] for b in labels]

    # Color gradient from red (low IoU) to green (high IoU)
    colors = sns.color_palette("RdYlGn", n_colors=len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.8)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval + 5,
            f"{int(yval)}", ha="center", va="bottom", fontweight="bold"
        )

    ax.set_xlabel("Best IoU Range")
    ax.set_ylabel("Number of Images")
    ax.set_title("Distribution of Best IoU Scores per Image", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = out_dir / f"trackA_iou_hist_{run_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_head_manifests(data: dict, out_dir: Path, run_name: str) -> Path:
    """Plot head manifest statistics from Stage B summary."""
    hm = data["head_manifests"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original manifest entries
    labels1 = ["Train (Original)", "Val (Original)"]
    values1 = [hm["train_entries"], hm["val_entries"]]
    colors1 = sns.color_palette("Blues", n_colors=2)
    bars1 = axes[0].bar(labels1, values1, color=colors1, edgecolor="black", alpha=0.8)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 500, f"{int(yval):,}", ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylabel("Number of Entries")
    axes[0].set_title("Original Manifest Entries", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Filtered head manifest rows
    labels2 = ["Train (Filtered)", "Val (Filtered)"]
    values2 = [hm["head_train_rows_written"], hm["head_val_rows_written"]]
    colors2 = sns.color_palette("Greens", n_colors=2)
    bars2 = axes[1].bar(labels2, values2, color=colors2, edgecolor="black", alpha=0.8)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 50, f"{int(yval):,}", ha="center", va="bottom", fontweight="bold")
    axes[1].set_ylabel("Number of Rows Written")
    axes[1].set_title("Filtered Head Manifest (Matched Detections)", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Head Manifest Statistics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"trackA_head_manifests_{run_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Track A plots from summary.json")
    parser.add_argument(
        "--summary", type=str,
        default="local_extraction/runs/Track_A/trackA_stageB_20251117_184342/summary.json",
        help="Path to summary.json file"
    )
    parser.add_argument(
        "--out", type=str,
        default="local_extraction/runs/Track_A/plots",
        help="Output directory for plots"
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract run name from summary path (e.g., "20251117_184342")
    run_name = summary_path.parent.name.replace("trackA_stageB_", "")

    print(f"Loading summary from: {summary_path}")
    data = load_summary(summary_path)

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Generate plots
    saved_paths = []

    # --- K-Sweep summary plots (e.g., KSweep runs) ---
    if "metrics" in data and "k_values" in data:
        p = plot_k_sweep_metrics(data, out_dir, run_name)
        saved_paths.append(p)
        print(f"  Saved: {p}")

    if "confidence_stats" in data:
        p = plot_confidence_stats(data, out_dir, run_name)
        saved_paths.append(p)
        print(f"  Saved: {p}")

    if "prediction_stats" in data:
        p = plot_prediction_stats(data, out_dir, run_name)
        saved_paths.append(p)
        print(f"  Saved: {p}")

    # --- Stage B summary plots (trackA_stageB runs) ---
    if "recall_metrics" in data:
        p = plot_recall_metrics(data, out_dir, run_name)
        saved_paths.append(p)
        print(f"  Saved: {p}")

        if "best_iou_hist" in data["recall_metrics"]:
            p = plot_iou_histogram(data, out_dir, run_name)
            saved_paths.append(p)
            print(f"  Saved: {p}")

    if "head_manifests" in data:
        p = plot_head_manifests(data, out_dir, run_name)
        saved_paths.append(p)
        print(f"  Saved: {p}")

    print(f"\nGenerated {len(saved_paths)} plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
