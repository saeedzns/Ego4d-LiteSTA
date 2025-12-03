# Track B TODO — What to Run, In Order

All commands below assume your working directory is the repo root:

`D:\Thesis\Ego4d-LiteSTA` (or `/mnt/d/Thesis/Ego4d-LiteSTA` in WSL).

Use forward slashes in commands so they work on both Windows and WSL:
`python local_extraction/trackB/...`

---

## 0. Prerequisites (run once)

- **Command:** _none (setup step)_
- **Input:**  
  - Extracted frames under `local_extraction/v2/extracted_frames/<uid>/*.jpg`  
  - Manifests either from:  
    - **Base dataset:** `local_extraction/v2/manifests/*.json(.l)` (e.g. `head_train_*`, `head_val_*`), or  
    - **Stage A StageB runs:** `local_extraction/runs/Track_A/trackA_stageB_*/head_{train,val}*.json(.l)`
- **Output:**  
  - Data on disk ready for Track B
- **Why:**  
  - Track B only consumes frames + manifests; it does not extract new clips. Make sure Track A has been run and data is present before training.

If you are unsure, visually confirm that `local_extraction/v2/extracted_frames` and `local_extraction/v2/manifests` both exist and contain subfolders/files.

---

## 1. Core fusion unit test (fast sanity check)

- **Command:**  
  `python local_extraction/trackB/trackB_tests.py`
- **Input:**  
  - No external data; uses random tensors
- **Output:**  
  - Console message like: `[trackB.tests] FGTP invariants PASS`
- **Why:**  
  - Verifies the FGTP block behaves as expected (attention weights sum to 1, residual wiring is correct) before you invest time in training.

---

## 2. Check manifests quality and TTC stats

- **Command:**  
  `python local_extraction/trackB/trackB_manifest_validator.py`
- **Input:**  
  - By default discovers a **base dataset** manifest under `local_extraction/v2/manifests` (e.g. `head_train*.json[l]`, `head_val*.json[l]`, or `val.json[l]`)  
  - You can switch to **Stage A StageB run manifests** by editing `ValidatorConfig` in `local_extraction/trackB/trackB_manifest_validator.py`:
    - Set `use_stageA_run = True` to use the latest `trackA_stageB_*` run under `local_extraction/runs/Track_A`, or  
    - Set `stageA_run` or `manifest_path` to point to a specific run/manifest
- **Output:**  
  - Prints record counts, candidate key coverage, TTC statistics  
  - Writes:  
    - `local_extraction/runs/Track_B/cache/ttc_stats.json`  
    - `local_extraction/runs/Track_B/cache/key_coverage.json`
- **Why:**  
  - Ensures your manifests have the expected keys (`uid`, `frame`, `candidates`, `ttc`, etc.) and that TTC values are present and reasonable before training and evaluation.

---

## 3. End-to-end wiring demo on real frames

- **Command:**  
  `python local_extraction/trackB/trackB_integration_demo.py`
- **Input:**  
  - Frames under `local_extraction/v2/extracted_frames/<uid>/*.jpg`  
  - Uses the first UID and its last frame it can find
- **Output:**  
  - Console prints with shapes, e.g.:  
    - `img_tokens: (N, C)` and `vid_tokens: (T, N, C)`  
    - `fused_img` / `fused_vid` shapes  
    - ROI feature vector shapes and norms
- **Why:**  
  - Confirms that tokenization, fusion, and ROI pooling all work together with your actual frames (not just synthetic data).

Optional additional smoke tests (can be run any time):

- **Fusion-only demo:**  
  - **Command:** `python local_extraction/trackB/trackB_fusion.py`  
  - **Input:** synthetic tokens  
  - **Output:** printed tensor shapes; basic sanity check of the fusion block

- **Tokenizer-only demo:**  
  - **Command:** `python local_extraction/trackB/trackB_tokenizer.py`  
  - **Input:** frames under `local_extraction/v2/extracted_frames`  
  - **Output:** printed token shapes; confirms grid tokenization works on your frames

---

## 4. Quick training demo (small, for debugging)

- **Command:**  
  `python local_extraction/trackB/trackB_train_demo.py`
- **Input:**  
  - Frames under `local_extraction/v2/extracted_frames`  
  - Manifests from either the **base dataset** or a **Stage A StageB run**, controlled by `DemoConfig` inside `trackB_train_demo.py`:  
    - `use_real_labels=True` (default) enables manifest-based labels (otherwise purely synthetic).  
    - If you want a specific file, set `manifest_path` (absolute or relative to repo root).  
    - To pull manifests from a Track A StageB run, set `use_stageA_run_manifests = True` (and optionally `stageA_run` or `stageA_manifest_base`).  
    - Otherwise it falls back to common base manifests under `local_extraction/v2/manifests` (e.g. `head_train_video/clip.{json,jsonl}`).
- **Output:**  
  - Console progress bar over a small number of steps  
  - Uses real labels if manifests are found; otherwise falls back to synthetic boxes/labels/ttc per step
- **Why:**  
  - Cheap way to check that:  
    - Manifests can be parsed for candidates, labels, and TTC  
    - Tokenization + fusion + head + loss functions all run without errors  
    - Basic training dynamics look reasonable before launching full training.

You can open `local_extraction/trackB/trackB_train_demo.py` and adjust `DemoConfig` (e.g. number of steps or forcing synthetic labels) if you need a quicker or more controlled debug run.

---

## 5. Main training with DataLoader + validation (primary run)

- **Command (recommended default):**  
  `python local_extraction/trackB/trackB_train_loader.py`
- **Input:**  
  - Frames: `local_extraction/v2/extracted_frames`  
  - Manifests: by default auto-discovered from Track A Stage B runs or from `local_extraction/v2/manifests`  
  - Config: `TrainConfig` defined inside `trackB_train_loader.py` (epochs, batch size, eval frequency, early stopping, etc.)
- **Output:** under `local_extraction/runs/Track_B/`  
  - `cache/`  
    - `invalid_records.csv`, TTC stats, UID coverage aids for debugging data issues  
  - `checkpoints/`  
    - `trackB_epochE_TIMESTAMP.pt` (per-epoch checkpoints, if enabled)  
    - `trackB_final_TIMESTAMP.pt` (final model)  
    - `trackB_best.pt` + `trackB_best_<metric>_<score>_TIMESTAMP.pt` (best checkpoint based on `TrainConfig.monitor_metric`)  
  - Console logs  
    - Per-epoch training loss and metrics  
    - Periodic validation metrics when `TrainConfig.eval_every > 0`
- **Why:**  
  - This is the main training loop that uses `TrackBDataset` and a proper `DataLoader`.  
  - It performs batched training with label smoothing, optional TTC normalization, and early stopping, and produces the checkpoints used later for evaluation.

If you want to customize training (e.g. number of epochs, which manifests to use, or which metric to monitor), edit the `TrainConfig` class near the top of `trackB_train_loader.py` before running.

---

## 6. Standalone evaluation and overlays (after training)

- **Command:**  
  `python local_extraction/trackB/trackB_eval.py`
- **Input:**  
  - Latest Track B checkpoint from `local_extraction/runs/Track_B/checkpoints`  
    - By default, it auto-selects the latest `trackB_final_*.pt` unless you set `EvalConfig.checkpoint_path` inside `trackB_eval.py`  
  - Validation manifest: auto-discovered from Track A Stage B run or from `local_extraction/v2/manifests`  
  - Frames: `local_extraction/v2/extracted_frames`
- **Output:** under `local_extraction/runs/Track_B/`  
  - `metrics/metrics_val_TIMESTAMP.json`  
    - Contains accuracy, mAP, TTC MAE in seconds, etc.  
  - `predictions/predictions_val_TIMESTAMP.csv` and `.jsonl`  
    - One row per candidate, with predicted probabilities and TTC  
  - `overlays/val/*.jpg`  
    - Validation frames with bounding boxes drawn, class probabilities and TTC predictions (and optional ground truth)
- **Why:**  
  - Provides a clear quantitative and qualitative view of model performance.  
  - The metrics JSON is what you track for experiments; overlays help visually inspect successes and failures.

### 6.1 Optional: multi-task and priors / CLIP

Once you have a stable single‑task baseline:

- Enable multi‑task heads in `TrainConfig` (inside `trackB_train_loader.py`):
  - `use_multi_task_labels = True` to add noun + verb heads.
  - `use_ttc_bins = True` to add TTC‑bin classification.
  - Adjust `loss_w_next`, `loss_w_noun`, `loss_w_verb`, `loss_w_ttc` to balance losses.
- Re‑train and re‑eval:
  - `python local_extraction/trackB/trackB_train_loader.py`
  - `python local_extraction/trackB/trackB_eval.py`
  - Check `N_mAP`, `Nv_mAP`, `N_delta_mAP` and per‑noun/verb stats in the metrics JSON.
- Optional hotspot priors (PEAR‑style) at eval time:
  - Create a JSON file with prior scores for `(noun_id, verb_id)` pairs.
  - Set `EvalConfig.use_hotspot_priors = True`, `hotspot_prior_path = Path("...")`, and `hotspot_alpha` in `trackB_eval.py`.
- Optional CLIP re‑ranking for noun consistency:
  - Install CLIP (`pip install git+https://github.com/openai/CLIP.git`) on your machine.
  - Set `EvalConfig.use_clip_rerank = True`, choose `clip_model`, and `noun_label_path` (taxonomy with labels).
  - Re‑run eval and compare N / N+V / N+δ with and without re‑ranking, especially on ambiguous frames.

---

## 7. Typical run sequence (TL;DR)

From the repo root:

1. **Sanity + data check (once per dataset):**  
   - `python local_extraction/trackB/trackB_tests.py`  
   - `python local_extraction/trackB/trackB_manifest_validator.py`
2. **Quick wiring check on real frames (optional but recommended):**  
   - `python local_extraction/trackB/trackB_integration_demo.py`
3. **Debug-scale training (optional):**  
   - `python local_extraction/trackB/trackB_train_demo.py`
4. **Main training:**  
  - `python local_extraction/trackB/trackB_train_loader.py`
5. **Evaluation + overlays:**  
  - `python local_extraction/trackB/trackB_eval.py`
6. **Optional multi‑task + priors/CLIP experiments:**  
   - Enable toggles in `TrainConfig` / `EvalConfig` and rerun training + eval to collect N / N+V / N+δ curves and per‑noun/verb breakdowns.

Follow this order whenever you change data, manifests, or model settings to catch issues early and keep your Track B experiments reproducible.
