# Track B & Track C Ablation Scripts Reference

This document summarizes the ablation and comparison tools available for Track B and Track C experiments.

---

## Track B Ablation Scripts

### 1. `trackB_ablation_compare.py`

**Purpose:** Compare evaluation results across different configurations (hotspot on/off, CLIP on/off, backbone types).

**Features:**
- Parses `metrics_val_*_summary.json` files from Track B runs
- Generates comparison tables (Core Metrics + Top-5 Metrics)
- Creates bar/line plots for visual comparison
- Groups results by config (backbone, hotspot, CLIP) or shows all checkpoints individually

**Commands:**
```powershell
# Basic comparison (latest runs)
python local_extraction/trackB/trackB_ablation_compare.py

# Compare last N runs
python local_extraction/trackB/trackB_ablation_compare.py --last 8

# Save plots to custom directory
python local_extraction/trackB/trackB_ablation_compare.py --out_dir local_extraction/runs/Track_B/ablation_plots

# Show all checkpoints individually (not grouped by config)
python local_extraction/trackB/trackB_ablation_compare.py --all
```

**Output:**
- Console tables with mAP, accuracy, TTC MAE, N/Nv/N+δ/All metrics
- PNG plots in specified output directory

---

### 2. `summarize_ablation.py`

**Purpose:** Compare VideoMAE vs ResNet18 backbone performance across Track B runs.

> ⚠️ **Note:** This script expects metrics in `<run_dir>/metrics.json` format (per-run directories). 
> The current workflow stores metrics in `runs/Track_B/metrics/metrics_val_*.json` instead.
> **Recommended:** Use `trackB_ablation_compare.py` for comparing metrics from the `metrics/` folder.

**Features:**
- Reads metrics from Track B run directories (expects `<run_id>/metrics.json` structure)
- Automatically categorizes runs by backbone type (resnet18 vs videomae_ego)
- Generates a Markdown comparison table
- Calculates relative performance change between backbones

**Commands:**
```powershell
# Print to console (requires metrics.json in run subdirectories)
python local_extraction/trackB/summarize_ablation.py

# Save to file
python local_extraction/trackB/summarize_ablation.py --output results/ablation_videomae_vs_clip.md

# Custom runs directory
python local_extraction/trackB/summarize_ablation.py --runs_dir local_extraction/runs/Track_B
```

**Output:**
- Markdown table with columns: Model Variant, Video Backbone, Run ID, Accuracy, mAP, TTC MAE, N/Nv/N+δ/All mAP

**Alternative (Recommended):**
Use `trackB_ablation_compare.py` instead, which reads from `metrics/metrics_val_*_summary.json`:
```powershell
python local_extraction/trackB/trackB_ablation_compare.py --last 10
```

---

## Track C Ablation Scripts

### 1. `trackC_compare_metrics.py`

**Purpose:** Compare Track B (unpruned) vs Track C (pruned) metrics side-by-side.

**Features:**
- Loads metrics from both `Track_B/metrics/` and `Track_C/metrics/`
- Prints comparison table with all key metrics
- Supports TSV export for further analysis
- Generates line plots comparing pruning rates

**Commands:**
```powershell
# Compare latest Track B vs Track C (single pair)
python local_extraction/trackC/trackC_compare_metrics.py

# Compare all runs from both tracks
python local_extraction/trackC/trackC_compare_metrics.py --all

# Save comparison to TSV file
python local_extraction/trackC/trackC_compare_metrics.py --all --out comparison.tsv

# Generate comparison plot
python local_extraction/trackC/trackC_compare_metrics.py --all --plot

# Compare specific files
python local_extraction/trackC/trackC_compare_metrics.py --b_metrics path/to/trackB_metrics.json --c_metrics path/to/trackC_metrics.json
```

**Output:**
- TSV table with: track, N_top5_mAP, Nv_top5_mAP, N_delta_top5_mAP, All_top5_mAP, accuracy, mAP, ttc_mae, checkpoint, timestamp
- PNG plot comparing Track B vs Track C over time

---

### 2. `trackC_pruning.py`

**Purpose:** Main Track C evaluation script with RGTP (Rollout-Guided Token Pruning).

**Features:**
- Configurable pruning rates (0.0-0.95)
- Supports both ResNet18 and VideoMAE backbones
- Hotspot priors and CLIP re-ranking toggles
- Latency/VRAM/FLOPs instrumentation

**Commands:**
```powershell
# Default evaluation (uses trackC.yaml config)
python -m trackC.trackC_pruning

# Override pruning rate
python -m trackC.trackC_pruning --rgtp_rate 0.3
python -m trackC.trackC_pruning --rgtp_rate 0.5

# Disable pruning (baseline mode)
python -m trackC.trackC_pruning --no_pruning

# Use VideoMAE backbone
python -m trackC.trackC_pruning --video_backbone videomae_ego

# Override checkpoint
python -m trackC.trackC_pruning --checkpoint local_extraction/runs/Track_B/checkpoints/trackB_best.pt

# Toggle hotspot/CLIP
python -m trackC.trackC_pruning --hotspot on --clip off
python -m trackC.trackC_pruning --hotspot off --clip on
```

**Output:**
- Metrics JSON in `local_extraction/runs/Track_C/metrics/`
- Console summary with accuracy, mAP, TTC MAE, Top-5 metrics, latency stats

---

### 3. `trackC_plots.py`

**Purpose:** Generate visualization plots from Track C metrics.

**Commands:**
```powershell
python local_extraction/trackC/trackC_plots.py
```

**Output:**
- `trackC_metrics_over_time.png` - Metrics progression
- `trackC_top5_semantic_percent.png` - Top-5 semantic metrics bar chart
- `trackC_metrics_summary.tsv` - Summary data

---

### 4. `speed_compare_fair.py`

**Purpose:** Compare training speed WITH vs WITHOUT pre-extracted tokens (fair comparison using same UIDs).

**Commands:**
```powershell
python local_extraction/trackC/speed_compare_fair.py
```

**Output:**
- Speedup ratio (e.g., 8.3× faster with pre-extracted tokens)
- Per-batch timing comparison

---

## Quick Reference: Common Ablation Workflows

### Backbone Ablation (ResNet18 vs VideoMAE)
```powershell
# Run Track B eval with ResNet18
python -m trackB.trackB_eval --video_backbone resnet18

# Run Track B eval with VideoMAE
python -m trackB.trackB_eval --video_backbone videomae_ego

# Summarize results
python -m trackB.summarize_ablation --output results/backbone_ablation.md
```

### RGTP Rate Sweep
```powershell
# Sweep pruning rates
python -m trackC.trackC_pruning --rgtp_rate 0.0
python -m trackC.trackC_pruning --rgtp_rate 0.1
python -m trackC.trackC_pruning --rgtp_rate 0.3
python -m trackC.trackC_pruning --rgtp_rate 0.5

# Compare all results
python local_extraction/trackC/trackC_compare_metrics.py --all --plot
```

### Hotspot + CLIP Ablation
```powershell
# Baseline (no hotspot, no CLIP)
python -m trackC.trackC_pruning --hotspot off --clip off

# + Hotspot only
python -m trackC.trackC_pruning --hotspot on --clip off

# + CLIP only
python -m trackC.trackC_pruning --hotspot off --clip on

# + Both
python -m trackC.trackC_pruning --hotspot on --clip on

# Compare results
python local_extraction/trackB/trackB_ablation_compare.py --last 4
```

---

## Output Locations

| Track | Metrics | Checkpoints | Plots | Error Analysis |
|:------|:--------|:------------|:------|:---------------|
| B | `runs/Track_B/metrics/` | `runs/Track_B/checkpoints/` | `runs/Track_B/plots/` | `runs/Track_B/error_analysis/` |
| C | `runs/Track_C/metrics/` | (uses Track B checkpoints) | `runs/Track_C/plots/` | — |

---

## Error Analysis Tool

### `error_analysis.py`

**Purpose:** Generate comprehensive error analysis from evaluation metrics and predictions.

**Features:**
1. Per-noun accuracy bar charts (worst/best classes)
2. Per-verb accuracy bar charts (worst/best classes)
3. Small object failure analysis (by box area)
4. TTC error distribution histogram
5. Noun confusion matrix (top confusion pairs)
6. Auto-selects failure case overlays for review

**Commands:**
```powershell
# Run with latest metrics and predictions
python local_extraction/trackB/error_analysis.py

# Specify metrics file
python local_extraction/trackB/error_analysis.py --metrics path/to/metrics.json

# Specify predictions file
python local_extraction/trackB/error_analysis.py --predictions path/to/predictions.csv

# Show top/bottom K classes
python local_extraction/trackB/error_analysis.py --top_k 20

# Skip plots (text output only)
python local_extraction/trackB/error_analysis.py --no_plots

# Skip failure gallery
python local_extraction/trackB/error_analysis.py --no_gallery
```

**Output:**
- `noun_worst_classes.png` - Bar chart of worst noun classes by accuracy
- `noun_best_classes.png` - Bar chart of best noun classes by accuracy
- `verb_worst_classes.png` - Bar chart of worst verb classes by accuracy
- `verb_best_classes.png` - Bar chart of best verb classes by accuracy
- `box_size_accuracy.png` - Accuracy by object size (small/medium/large)
- `ttc_error_distribution.png` - Histogram of TTC prediction errors
- `noun_confusions.png` - Top noun confusion pairs
- `failure_gallery/` - Copies of worst failure case overlays
- `error_analysis_*.json` - Full results JSON

---

*Last updated: 2025-12-09*
