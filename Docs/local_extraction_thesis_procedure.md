# Local Extraction Thesis Procedure

## 1. Motivation and Scope

This document formalizes the complete end-to-end workflow executed inside the `local_extraction/` workspace for the Ego4D-LiteSTA thesis. It spans environment bootstrapping, dataset materialization, training of Tracks A and B, inference-time Track C pruning, and the bookkeeping required for reproducibility. The emphasis is on practical steps performed on the Windows development machine (PowerShell shell, Python virtual environment) with cross-references to the scripts and configuration toggles stored in the repository.

---

## 2. Repository Layout Summary

```
local_extraction/
├── env/                 # virtual environment scripts + requirements
├── v2/                  # extracted frames, manifests, logs, YOLO labels
├── trackA/              # Stage A detector + Stage B preparation utilities
├── trackB/              # dataset, tokenizer, fusion, head, training + eval
├── trackC/              # RGTP pruning harness
├── reports/, runs/, notebooks/, pipelines/, etc.
└── ...
```

Key subfolders referenced throughout the procedure:

- `v2/extracted_frames/` – canonical frame dumps (JPEG) for Ego4D STA Track B.
- `v2/manifests/` – annotation-first manifests; superseded by Stage B output for Track B training.
- `runs/Track_A/trackA_stageB_<timestamp>/` – Stage B export directory containing `head_train.jsonl` and `head_val.jsonl` used by Track B.
- `runs/Track_B/checkpoints/` – Track B training outputs (epoch checkpoints, best checkpoint, final checkpoint).
- `runs/Track_C/metrics/` – RGTP (pruned inference) evaluation logs.

---

## 3. Environment Preparation

### 3.1 Virtual Environment

1. Navigate to repository root.
2. Execute PowerShell script to build or refresh the virtual environment:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\local_extraction\env\setup_venv.ps1
   .\.venv\Scripts\Activate.ps1
   pip install -r local_extraction\env\requirements.txt
   ```

3. Confirm activation via prompt `(venv)` and Python version compatibility (3.10+ recommended).

### 3.2 Shared Dependencies

- Ultralytics YOLO (`yolov8s.pt`) resident at repo root satisfies Stage A detection.
- PyTorch + torchvision compiled with CUDA support (if GPU available) for efficient tokenization in Track B.

---

## 4. Data Acquisition and Materialization

### 4.1 Extracting Frames and Windows

1. Use `local_extraction/extraction/ego4d_resume_fast_extract.py` or the Colab notebooks (e.g., `local_extraction/notebooks/Ego4D_STA_Colab_Starter_updated.ipynb`) to produce:
   - `v2/extracted_frames/<uid>/<frame:07d>.jpg`
   - `v2/tmp_frame_lists/` and supporting logs for verifying coverage.

2. Store these assets in Drive or local disk; maintain symlink or path references in the scripts.

### 4.2 Generating YOLO Label Proposals

- Optional but recommended: run detection to populate `v2/yolo_labels_540/{clips|videos}/...` for Stage A oracle mode and evaluation.

### 4.3 Stage B Manifest Creation (Track A Stage B)

1. Run Stage A detector (`trackA_stageA.py`) to produce `trackA_stageA_<timestamp>/candidates.jsonl` with top-K proposals.

2. Execute Stage B preparation script to crop ROIs, fuse with annotation semantics, and emit head manifests:

   ```powershell
   python local_extraction\trackA\trackA_stageB\trackA_stageB.py
   ```

   Outputs inside `runs/Track_A/trackA_stageB_<timestamp>/`:
   - `head_train.jsonl`, `head_val.jsonl`
   - Cropped candidate images (`crops/`)
   - Summary JSON and evaluation statistics (optional).

These Stage B manifests become the authoritative input for subsequent Track B training and evaluation.

---

## 5. Track B – Lightweight Fusion Training

### 5.1 Configuration Overview

The core script is `local_extraction/trackB/trackB_train_loader.py`. Configuration is managed through in-file toggles:

```python
class TrainConfig:
    mode: str = 'main'            # 'demo' or 'main'
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3
    train_manifest: str | None = None
    val_manifest: str | None = None
    stageB_run: str | None = None
    amp: bool = False
    candidate_limit: int = 16
    normalize_ttc: bool = True
    ...
```

Shared hyperparameters at the top of the file include `TOKEN_DIM`, `FUSION_LAYERS`, `HEAD_HIDDEN`, `TTC_LOSS_WEIGHT`, etc.

### 5.2 Automatic Stage B Integration

Upon launch the script:

1. Discovers the latest Stage B run via `latest_stageB_run()`.
2. Sets `train_manifest` / `val_manifest` to Stage B outputs unless overrides are provided.
3. Logs the resolved manifest source for traceability.

This allows direct training against the newest Stage B exports without manual path editing.

### 5.3 Demo Mode Validation

Before full training, use demo mode to sanity check the pipeline:

```powershell
python local_extraction\trackB\trackB_train_loader.py  # with TrainConfig.mode='demo'
```

Demo mode performs limited iterations on the DataLoader and prints sample metrics (loss, accuracy preview).

### 5.4 Main Training Execution

Switch `TrainConfig.mode` to `'main'` and run:

```powershell
python local_extraction\trackB\trackB_train_loader.py
```

**Progress Monitoring:**
- Nested `tqdm` bars display epoch progress (`[TrackB main] epoch X/Y`) and overall cumulative progress across epochs.
- Post-epoch summary prints average loss components.
- End-to-end training in `main` mode currently spans roughly six hours of wall-clock time on the local GPU setup (AMP off); budget accordingly for overnight or extended runs.

**Checkpoints:**
- Epoch checkpoints saved to `runs/Track_B/checkpoints/trackB_epoch<idx>_<timestamp>.pt` when `save_epoch_checkpoints=True`.
- Best checkpoint tracking against metric specified in `monitor_metric` (default `mAP`) saved as both timestamped file and `trackB_best.pt`.
- Final checkpoint always saved as `trackB_final_<timestamp>.pt`.

### 5.5 In-Loop Validation

`eval_every` governs validation frequency. When triggered, the script:

1. Builds a validation dataset using Stage B val manifest (auto-discovered).
2. Runs `_evaluate_loader` to compute accuracy, mAP/AP, and TTC MAE.
3. Applies early stopping if `early_stopping_patience` is exceeded.

All metrics print to console and influence best-checkpoint logic.

---

## 6. Track B – Evaluation and Reporting

### 6.1 Evaluation Script

Run `trackB_eval.py` to score checkpoints and produce overlays:

```powershell
python local_extraction\trackB\trackB_eval.py
```

Key features:
- Auto-discovers Stage B manifests unless explicit `EvalConfig` overrides set at top of file.
- Loads most recent `trackB_final_*.pt` if no checkpoint path supplied.
- Provides nested `tqdm` progress bars (`[TrackB eval] samples` + `[TrackB eval] batch`) to track inference.
- Generates metrics JSON under `runs/Track_B/metrics/` and predictions CSV/JSON in `runs/Track_B/predictions/`.
- Optional overlay images saved in `runs/Track_B/overlays/val/` for qualitative review.

### 6.2 Example Metrics Snapshot

Output structure (captured from a run on Stage B manifests):

```json
{
  "accuracy": 0.9278,
  "mAP": 0.0699,
  "ap_per_class": {"1": 0.0699},
  "ttc_mae_seconds": 0.2258,
  "num_candidates": 1662,
  "checkpoint": "local_extraction\\runs\\Track_B\\checkpoints\\trackB_final_20251111_020232.pt",
  "val_manifest": "local_extraction\\runs\\Track_A\\trackA_stageB_20251110_034036\\head_val.jsonl",
  "timestamp": "2025-11-11T01:20:56"
}
```

These numbers serve as the baseline for Track C pruning experiments.

---

## 7. Track C – Training-Free RGTP Pruning

### 7.1 RuntimeConfig Control Panel

`local_extraction/trackC/trackC_pruning.py` exposes an in-file `RuntimeConfig` class for toggles:

```python
class RuntimeConfig:
    checkpoint: Optional[str] = None
    stageB_run: Optional[str] = None
    val_manifest: Optional[str] = None
    pruning_enabled: bool = False
    rgtp_rate: float = 0.5
    min_keep: int = 1
```

Steps:
1. Edit `RuntimeConfig` values to choose checkpoint, Stage B run, manifest, and pruning behavior.
2. Execute the script:

   ```powershell
   python local_extraction\trackC\trackC_pruning.py
   ```

3. The script logs toggles, resolves Stage B run, loads the latest Track B checkpoint (if none specified), and evaluates.

### 7.2 Importance Scoring

The script computes candidate importance by blending:
- **FGTP rollout attention** between the last two temporal slices.
- **Motion energy** derived from token differences (`vid_tokens[:, -1] - vid_tokens[:, -2]`).

Scores are pooled over candidate boxes and pruned using `RGTPConfig.keep_count()` to respect `rgtp_rate` and `min_keep` constraints. Pruned tokens receive a strong negative logit (`LOGIT_FILL = -12.0`), ensuring they minimally influence probability while retaining slot alignment for metric calculation.

### 7.3 Metrics and Logs

Results are written to `runs/Track_C/metrics/trackC_val_rateXX_<timestamp>.json` containing:

- Accuracy, mAP, TTC MAE (mirroring Track B metrics for comparability).
- `rgtp_mean_fraction_pruned` summarizing actual prune fraction (averaged over samples).
- Metadata fields: checkpoint path, val manifest path, toggled prune rate.

Example baseline (pruning disabled):

```text
[trackC] Runtime toggles: {'checkpoint': None, 'stageB_run': None, ...}
[trackC] pruning enabled=False rate=0.50 min_keep=1
...
[trackC] Metrics:
  accuracy: 0.9277978539466858
  mAP: 0.06988885998725891
  ttc_mae_seconds: 0.22583657503128052
  num_candidates: 1662
  rgtp_mean_fraction_pruned: 0.0
```

Subsequent runs with `pruning_enabled=True` and varied `rgtp_rate` generate the latency/accuracy trade-off data for thesis reporting.

---

## 8. Logging, Reporting, and Documentation

### 8.1 Daily Logbook

Maintain entries in `Docs/Ego4d-LiteSTA_Daily_Loop.md` capturing:
- Date, experiment ID, configuration tweaks.
- Observed metrics and qualitative notes.
- Links to produced artifacts (checkpoints, metrics JSON, overlays).

### 8.2 Aggregated Reports

Update or create summary reports in `local_extraction/reports/` (e.g., `sanity_check_report.md`, `label_stats_report.md`) to reflect data quality checks and evaluation highlights.

### 8.3 Thesis Synchronization

Use `Docs/thesis_plan_tracks_ABC_v2.md` to document plan vs execution, particularly sections regarding Track B (fusion) and Track C (RGTP). Append tables or bullet summaries with actual metrics and pruning curves once experiments are complete.

### 8.4 Ongoing Tasks

- Fine-tune the Track A detector using YOLO (e.g., `yolov8s.pt`) on refreshed Ego4D STA crops—remember the detector is single-class (`next_active` noun only)—to lift candidate recall@K before regenerating Stage B manifests. All prior experiments have run on CPU locally; consider offloading this fine-tune to Colab Free for GPU acceleration. Expect roughly two days of wall-clock time on Colab (or an equivalent free GPU service) to reach stable recall; document hyperparameters, data augmentations, timing, and resulting recall metrics for thesis reporting.

---

## 9. Reproducibility Checklist

1. **Environment snapshot:**
   - Active `.venv` freeze via `pip freeze > env/requirements_lock.txt`.
   - GPU / CUDA version notes if applicable.

2. **Data integrity:**
   - Record counts for frames (`len(v2/extracted_frames)`) and manifests (records, candidates) using quick scripts or logs.

3. **Training metadata:**
   - Store `TrainConfig` values, random seeds, and `TokenizerConfig` parameters alongside checkpoints.
   - Preserve console output (PowerShell transcripts) or `runs/Track_B/metrics/*.json` for each experiment.

4. **Evaluation artifacts:**
   - Metrics JSON, prediction CSV/JSON, overlays for Track B.
   - RGTP metrics JSON for Track C sweeps.

5. **Version control:**
   - Commit code changes impacting experiments; capture Git commit hash in reports.

---

## 10. Future Extensions

- Integrate latency measurements (`torch.cuda.Event` timing or external profilers) into Track C runs to quantify real speedups corresponding to `rgtp_rate` adjustments.
- Add CLI wrappers around `RuntimeConfig` for batch experiment scripts while retaining in-file defaults for manual control.
- Explore weighted pruning strategies (class-aware or TTC-aware) by modifying `_candidate_scores()`.
- Consider adding Stage B manifest diffing tools to ensure training/evaluation uses synchronized datasets.

---

## 11. Conclusion

The outlined procedure converts raw Ego4D STA assets into a fully operational two-stage + fusion + pruning system inside the `local_extraction/` workspace. By consolidating configuration toggles, manifest discovery, and progress monitoring into cohesive scripts, the workflow supports reproducible experimentation and thesis-ready documentation. Track A ensures high-recall proposals, Track B lifts reasoning performance with lightweight fusion, and Track C delivers inference-time efficiency without retraining—fulfilling the core objectives in the thesis plan.
