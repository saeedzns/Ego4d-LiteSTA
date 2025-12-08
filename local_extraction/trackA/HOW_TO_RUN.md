# Track A — How to Run Guide

This guide provides step-by-step instructions for running Track A (Two-Stage STA Baseline) including Stage A (Detector) and Stage B (Head/Reasoner) pipelines.

---

## Prerequisites

### 1. Activate the Virtual Environment
All commands must be run from the repository root with the virtual environment activated:

```powershell
cd D:\Thesis\Ego4d-LiteSTA
.\local_extraction\.venv\Scripts\Activate.ps1
```

### 2. Verify Installation
Test that imports work correctly:

```powershell
python -c "from local_extraction.trackA.trackA_stageA import trackA_stageA; print('Stage A OK')"
python -c "from local_extraction.trackA.trackA_stageB import trackA_stageB; print('Stage B OK')"
```

### 3. Required Data
Ensure you have:
- **Extracted frames:** `local_extraction/v2/extracted_frames/<uid>/<frame:07d>.jpg`
- **YOLO labels (for oracle/eval):** `local_extraction/v2/yolo_labels_540/{clips|videos}/<uid>/<frame:07d>.txt`
- **YOLO weights (for detector mode):** `local_extraction/toolkit_yolo/runs/.../weights/best.pt`
- **Head manifests (optional):** `local_extraction/v2/manifests/head_*_{clip|video}.json`

---

## Stage A — Detector / Candidate Generation

Stage A generates candidate bounding boxes per frame using either YOLO inference or oracle (ground-truth) mode.

### Configuration
Edit `configs/trackA.yaml` to set detection parameters:

```yaml
stage_a:
  mode: "yolo"          # "yolo" or "oracle"
  k: 6                   # Top-K candidates per image
  last_frame_only: true  # Process only last frame per UID
  max_images: null       # null = all, or integer to cap
```

### Run Stage A (YOLO Detection)
```powershell
python local_extraction/trackA/trackA_stageA/trackA_stageA.py
```

**Note:** This script runs directly without CLI arguments. Configuration is read from `configs/trackA.yaml`.

### Run Stage A (Oracle Mode)
Set environment variables to override config:

```powershell
$env:STAGEA_MODE = "oracle"
$env:STAGEA_K = "10"
$env:STAGEA_MAX_IMAGES = "500"
python local_extraction/trackA/trackA_stageA/trackA_stageA.py
```

### Stage A Outputs
```
local_extraction/runs/Track_A/trackA_stageA_<timestamp>/
├── candidates.jsonl    # One JSON line per image with bounding boxes
├── summary.json        # Run statistics and configuration
└── candidates/         # Per-image CSVs (if save_per_image_csv: true)
```

Example `candidates.jsonl` line:
```json
{"uid":"00be...1705","frame":99,"boxes":[{"x1":161.8,"y1":0.3,"x2":373.9,"y2":244.8,"conf":1.0,"cls":26}]}
```

---

## Stage B — Head/Reasoner Preparation

Stage B processes Stage A candidates: crops ROIs, measures recall against ground-truth, and prepares head manifests for downstream Track B training.

### Configuration
Edit `configs/trackA.yaml` for Stage B settings:

```yaml
stage_b:
  candidates_source: null     # null = auto-detect latest Stage A run
  crop_size: [256, 256]       # [W, H] or null for native size
  keep_top_n: null            # null = keep all candidates
  eval_with_labels: true      # Compute recall metrics
  iou_thresh: 0.5             # IoU threshold for positive match
```

### Run Stage B
```powershell
python local_extraction/trackA/trackA_stageB/trackA_stageB.py
```

Stage B automatically finds the latest Stage A run and processes its candidates.

### Stage B Outputs
```
local_extraction/runs/Track_A/trackA_stageB_<timestamp>/
├── crops/              # Cropped ROI images per candidate
├── manifest.csv        # Per-crop metadata (CSV format)
├── manifest.jsonl      # Per-crop metadata (JSONL format)
├── head_train.jsonl    # Merged head manifest for training
├── head_val.jsonl      # Merged head manifest for validation
└── summary.json        # Run statistics with recall metrics
```

---

## Utility Scripts

### Sanity Check (san_stageB.py)
Inspect the most recent Stage B run and print positive/negative counts:

```powershell
python local_extraction/trackA/trackA_stageB/san_stageB.py
```

Output:
```
Inspecting run: local_extraction\runs\Track_A\trackA_stageB_20251117_184342
head_train.jsonl -> positives 1041 negatives 2311 pos_ratio 0.3106
head_val.jsonl -> positives 547 negatives 1230 pos_ratio 0.3078
Recall metrics: {'images_with_gt': 2323, 'images_with_hit': 1575, 'recall_at_K': 0.678003, ...}
```

### Oracle K-Sweep (oracle_k_sweep.py)
Run Stage A + Stage B for multiple K values to find optimal recall:

```powershell
# Default K values: 1,2,3,5,8,10
python local_extraction/trackA/trackA_stageA/oracle_k_sweep.py

# Custom K values
$env:ORACLE_SWEEP_KS = "4,6,8,10,12"
$env:ORACLE_SWEEP_MAX_IMAGES = "1000"
python local_extraction/trackA/trackA_stageA/oracle_k_sweep.py
```

Output: Table of recall@K metrics for each K value.

### Sweep K Metrics (sweep_k_metrics.py)
Advanced K-sweep with YOLO inference and detailed metrics:

```powershell
python local_extraction/trackA/trackA_stageA/sweep_k_metrics.py
```

Outputs:
- `local_extraction/runs/Track_A/KSweep/StageA_KSweep_<timestamp>/metrics.csv`
- `local_extraction/runs/Track_A/KSweep/StageA_KSweep_<timestamp>/summary.json`

### Generate Plots (trackA_plots.py)
Create visualizations from Track A runs:

```powershell
python local_extraction/trackA/trackA_plots.py --summary "local_extraction/runs/Track_A/trackA_stageB_*/summary.json" --out "local_extraction/runs/Track_A/plots"
```

---

## Environment Variable Overrides

Stage A supports environment variables for quick parameter changes:

| Variable | Description | Example |
|----------|-------------|---------|
| `STAGEA_MODE` | Detection mode | `oracle`, `yolo` |
| `STAGEA_K` | Top-K candidates | `10` |
| `STAGEA_MAX_IMAGES` | Cap images processed | `500` |
| `STAGEA_DEMO_MODE` | Enable demo mode | `1`, `true` |
| `STAGEA_DEMO_N` | Demo sample count | `50` |

Example:
```powershell
$env:STAGEA_MODE = "oracle"
$env:STAGEA_K = "8"
python local_extraction/trackA/trackA_stageA/trackA_stageA.py
```

---

## Common Workflows

### 1. First-Time Setup & Full Pipeline
```powershell
# Activate environment
.\local_extraction\.venv\Scripts\Activate.ps1

# Run Stage A (YOLO detection)
python local_extraction/trackA/trackA_stageA/trackA_stageA.py

# Run Stage B (process candidates, generate head manifests)
python local_extraction/trackA/trackA_stageB/trackA_stageB.py

# Check results
python local_extraction/trackA/trackA_stageB/san_stageB.py
```

### 2. Quick Oracle Recall Test
```powershell
$env:STAGEA_MODE = "oracle"
$env:STAGEA_K = "6"
$env:STAGEA_MAX_IMAGES = "200"
python local_extraction/trackA/trackA_stageA/trackA_stageA.py
python local_extraction/trackA/trackA_stageB/trackA_stageB.py
```

### 3. K-Sweep to Find Optimal K
```powershell
$env:ORACLE_SWEEP_KS = "4,6,8,10,12,15"
$env:ORACLE_SWEEP_MAX_IMAGES = "500"
python local_extraction/trackA/trackA_stageA/oracle_k_sweep.py
```

### 4. Production YOLO Run
```powershell
# Edit configs/trackA.yaml:
#   stage_a.mode: "yolo"
#   stage_a.k: 6
#   stage_a.last_frame_only: true
#   stage_a.max_images: null

python local_extraction/trackA/trackA_stageA/trackA_stageA.py
python local_extraction/trackA/trackA_stageB/trackA_stageB.py
```

### 5. Generate Manifests for Track B
After running Stage A + Stage B, the head manifests are ready:
```
local_extraction/runs/Track_A/trackA_stageB_<timestamp>/head_train.jsonl
local_extraction/runs/Track_A/trackA_stageB_<timestamp>/head_val.jsonl
```

These are automatically used by Track B training.

---

## Troubleshooting

### Import Errors
```powershell
# Make sure you're in repo root and venv is activated
cd D:\Thesis\Ego4d-LiteSTA
.\local_extraction\.venv\Scripts\Activate.ps1
```

### No Images Found
Check that extracted frames exist:
```powershell
ls local_extraction/v2/extracted_frames | Measure-Object
```

### YOLO Weights Not Found
Ensure YOLO weights path is correct in `configs/trackA.yaml`:
```yaml
stage_a:
  yolo:
    weights: "local_extraction/toolkit_yolo/runs/.../weights/best.pt"
```

### Stage B Can't Find Candidates
Stage B auto-detects the latest Stage A run. If needed, set explicitly in `configs/trackA.yaml`:
```yaml
stage_b:
  candidates_source: "local_extraction/runs/Track_A/trackA_stageA_<timestamp>/candidates.jsonl"
```

### Low Recall Metrics
- Increase `K` value for more candidates
- Lower `conf_thresh` in YOLO settings
- Check IoU threshold (`iou_thresh`) in Stage B config

---

## Configuration Files

| File | Description |
|------|-------------|
| `configs/trackA.yaml` | Main configuration for Stage A & B |
| `configs/base.yaml` | Shared base configuration |
| `CONFIGURABLE_OPTIONS.md` | Full parameter reference |

---

## Key Metrics Explained

| Metric | Description |
|--------|-------------|
| `images_with_gt` | Images that have ground-truth boxes |
| `images_with_hit` | Images where at least one candidate matched GT (IoU ≥ threshold) |
| `recall_at_K` | Fraction of GT images with at least one hit |
| `mean_best_iou` | Average of best IoU per image (among GT images) |
| `pos_ratio` | Ratio of positive candidates to total candidates |

---

## Expected Output

After running the full pipeline:

| Stage | Output Files |
|-------|--------------|
| Stage A | `candidates.jsonl`, `summary.json` |
| Stage B | `head_train.jsonl`, `head_val.jsonl`, `crops/`, `summary.json` |

Typical metrics (with K=6, YOLO mode):
- **recall_at_K**: ~65-70%
- **mean_best_iou**: ~0.60-0.65
- **pos_ratio**: ~0.30-0.35
