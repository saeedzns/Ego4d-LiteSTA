# Track C — How to Run Guide

This guide provides step-by-step instructions for running Track C (RGTP pruning evaluation) on top of a trained Track B checkpoint.

---

## Prerequisites

### 1. Activate the Virtual Environment
Run commands from the repository root with the venv active:

```powershell
cd D:\Thesis\Ego4d-LiteSTA
.\local_extraction\.venv\Scripts\Activate.ps1
```

### 2. Verify Imports
Quick import check:

```powershell
python -c "from local_extraction.trackC import trackC_pruning; print('OK')"
python -c "from local_extraction.trackC import trackC_compare_metrics; print('OK')"
```

### 3. Required Data
Ensure you have:
- **Extracted frames:** `local_extraction/v2/extracted_frames/<uid>/<frame:07d>.jpg`
- **Stage B val manifest:** from `local_extraction/runs/Track_A/trackA_stageB_*/head_val.jsonl`
- **Track B checkpoint:** `local_extraction/runs/Track_B/checkpoints/trackB_best.pt` (or any `trackB_final_*.pt`)

Optional:
- **Pre-extracted tokens:** set `data.tokens_root` in `local_extraction/configs/trackC.yaml` if you want faster inference.

---

## Quick Start

### Default Run (uses `trackC.yaml`)
```powershell
python local_extraction/trackC/trackC_pruning.py
```

Defaults:
- Config: `local_extraction/configs/trackC.yaml`
- Checkpoint: auto-detects latest/best Track B checkpoint
- Val manifest: auto-detects latest Stage B run
- Pruning: enabled with `rgtp.rate` from config

### Baseline (No Pruning)
```powershell
python local_extraction/trackC/trackC_pruning.py --no_pruning
```

### Override RGTP Rate
```powershell
python local_extraction/trackC/trackC_pruning.py --rgtp_rate 0.30
```

### Use a Specific Checkpoint
```powershell
python local_extraction/trackC/trackC_pruning.py --checkpoint "local_extraction/runs/Track_B/checkpoints/trackB_best_mAP_0.3904_20251220_024940.pt"
```

### Switch Backbone (VideoMAE)
```powershell
python local_extraction/trackC/trackC_pruning.py --video_backbone videomae_ego
```

---

## Config Overrides (YAML)

Edit `local_extraction/configs/trackC.yaml` to pin specific inputs:
- `evaluation.checkpoint`
- `evaluation.stageB_run`
- `evaluation.val_manifest`
- `evaluation.candidate_limit`, `evaluation.normalize_ttc`
- `rgtp.rate`, `rgtp.min_keep`, `rgtp.temporal_decay`

Rate sweeps and smoke tests are also configured there:
- `rate_sweep.enabled`, `rate_sweep.rates`
- `smoke_test.enabled`, `smoke_test.max_samples`, `smoke_test.test_rates`

---

## Outputs

All outputs go to `local_extraction/runs/Track_C/`:
```
local_extraction/runs/Track_C/
├── metrics/
│   ├── trackC_val_rateXX_<timestamp>.json
│   └── trackC_val_rateXX_<timestamp>_summary.json
└── plots/  (optional)
```

---

## Utilities

### Compare Track B vs Track C Metrics
```powershell
python local_extraction/trackC/trackC_compare_metrics.py
```

### Plot Track C Metrics Over Time
```powershell
python local_extraction/trackC/trackC_plots.py --runs_dir local_extraction/runs/Track_C --out_dir local_extraction/runs/Track_C/plots
```

---

## CLI Reference

```
usage: trackC_pruning.py [-h] [--config CONFIG] [--video_backbone {resnet18,videomae_ego}]
                         [--rgtp_rate RGTP_RATE] [--checkpoint CHECKPOINT] [--no_pruning]
                         [--hotspot {on,off}] [--clip {on,off}]

options:
  --config CONFIG           Config name to load (default: trackC)
  --video_backbone {resnet18,videomae_ego}
                            Override backbone selection
  --rgtp_rate RGTP_RATE     Override pruning rate (0.0-0.95)
  --checkpoint CHECKPOINT   Override checkpoint path
  --no_pruning              Disable pruning (baseline)
  --hotspot {on,off}        Enable/disable hotspot priors (override config)
  --clip {on,off}           Enable/disable CLIP re-ranking (override config)
```

For full parameter explanations, see `local_extraction/trackC/CONFIGURABLE_OPTIONS.md`.
