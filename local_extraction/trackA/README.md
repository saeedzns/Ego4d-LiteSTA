# Track A — Full Pipeline (Stage A ➜ Stage B) Local Guide

This document explains every step, toggle, and output for the Track A pipeline that prepares proposals and head manifests for downstream Track B/C. Edit the in-file configs (no CLI needed) and keep everything local.

---

## 1) Prerequisites & Layout
- Frames live under `local_extraction/v2/extracted_frames/<uid>/<frame:07d>.jpg`.
- Optional YOLO GT labels for oracle/recall live under `local_extraction/v2/yolo_labels_540/{clips|videos}/<uid>/<frame:07d>.txt`.
- Optional head manifests (semantic labels) live under `local_extraction/v2/manifests/head_*_{clip|video}.json[l]`.
- Runs are written under `local_extraction/runs/Track_A/`.
- Stage A script: `local_extraction/trackA/trackA_stageA/trackA_stageA.py`
- Stage B script: `local_extraction/trackA/trackA_stageB/trackA_stageB.py`

Quick check (PowerShell):
```powershell
ls local_extraction/v2/extracted_frames | measure
ls local_extraction/v2/manifests
```

---

## 2) Stage A — Detector Proposals (trackA_stageA.py)
**Goal:** produce candidate boxes per frame (either from YOLO inference or oracle labels) and a summary file.

### Key toggles (edit inside the script)
- `RUN_NAME`: used to create `runs/Track_A/<RUN_NAME>/`.
- `VERSION`: usually `v2` (adjust if your frames live elsewhere).
- `SPACE`: `clips` or `videos` (affects oracle label path).
- `DETECTION_MODE`: `oracle` (uses GT labels) or `yolo` (runs Ultralytics YOLO locally).
- `K`: keep top‑K boxes per image.
- `LAST_FRAME_ONLY`: `True` to process only the final frame per UID; `False` to process all frames.
- `MAX_IMAGES`: cap total images for quick smoke tests.
- YOLO-specific: `YOLO_WEIGHTS`, `YOLO_IMGSZ`, `YOLO_CONF`, `YOLO_IOU`.
- `SAVE_PER_IMAGE_CSV`: `True` to emit one CSV per image alongside the JSONL.

### How to run
```powershell
python local_extraction/trackA/trackA_stageA/trackA_stageA.py
```
The script reads frames, generates proposals, and writes outputs under `local_extraction/runs/Track_A/<RUN_NAME>/`.

### Outputs (Stage A run folder)
- `candidates.jsonl`: one line per image, e.g.:
  ```json
  {"uid":"00be...1705","frame":99,"boxes":[{"x1":161.8,"y1":-0.3,"x2":373.9,"y2":244.8,"conf":1.0,"cls":26}]}
  ```
- Optional per-image CSVs in `candidates/` if `SAVE_PER_IMAGE_CSV=True`.
- `summary.json`: counts and settings, e.g.:
  ```json
  {
    "images": 500,
    "avg_boxes_per_image": 4.2,
    "mode": "oracle",
    "K": 10,
    "last_frame_only": true,
    "runtime_sec": 123.4
  }
  ```

### Internal flow (what the code does)
1) Build run dir (`runs/Track_A/<RUN_NAME>/`).
2) Collect frames (last frame per UID by default; or all frames).
3) For each image:
   - Oracle mode: read GT label txt, convert YOLO `cx,cy,w,h` → `x1,y1,x2,y2`, keep top‑K.
   - YOLO mode: run Ultralytics with your weights/thresholds, keep top‑K by confidence.
   - Append to `candidates.jsonl`; optionally write a per-image CSV.
4) After processing: compute counts/averages and write `summary.json`.

### Troubleshooting Stage A
- **No boxes in output:** confirm `DETECTION_MODE` matches available inputs (oracle requires label txt files; YOLO requires weights + `ultralytics` installed).
- **Wrong frames picked:** check `LAST_FRAME_ONLY` and `MAX_IMAGES`; inspect the run folder for sampled frames.
- **Boxes in wrong scale:** ensure label txts match the frame resolution; oracle conversion assumes normalized YOLO format.
- **Performance:** reduce `MAX_IMAGES`, set `LAST_FRAME_ONLY=True`, lower `K`, or switch to oracle for quick recall checks.

---

## 3) Stage B — Crops & Head Manifests (trackA_stageB.py)
**Goal:** take Stage A proposals, crop ROIs, optionally measure recall against GT, and emit head manifests for Track B.

### Key toggles (edit inside the script)
- Inputs:
  - `CANDIDATES_JSONL`: path to Stage A `candidates.jsonl` (auto-detects latest if `None`).
  - `FRAMES_ROOT`, `LABELS_ROOT`, `LABEL_SPACE`, `EVAL_WITH_LABELS`, `IOU_THRESH`.
- Cropping:
  - `KEEP_TOP_N`: keep first N candidates per image (None = all).
  - `CROP_SIZE`: `(W,H)` to resize each crop; `None` keeps native size.
- Limits:
  - `MAX_IMAGES`, `DEMO_MODE`, `DEMO_N`, `PRINT_PROGRESS`.
- Head manifests:
  - `USE_CLIP_MANIFEST`: choose `head_*_clip.json` vs `head_*_video.json`.
  - `HEAD_TRAIN_MANIFEST`, `HEAD_VAL_MANIFEST`: explicit paths if needed.
  - `WRITE_HEAD_TRAIN_VAL`: emit merged `head_train.jsonl` / `head_val.jsonl` in the run dir.
  - `WRITE_SEMANTICS_IN_MANIFEST`: include semantic fields in per-crop manifest rows.
  - `WRITE_BACK_TO_STAGEA_SUMMARY`: write recall numbers back into the Stage A `summary.json`.
- Run dir: `RUN_DIR` defaults to `runs/Track_A/trackA_stageB_<timestamp>/`.

### How to run
```powershell
python local_extraction/trackA/trackA_stageB/trackA_stageB.py
```

### Outputs (Stage B run folder)
- `crops/`: cropped and optionally resized ROI images.
- `manifest.jsonl` / `manifest.csv`: per-crop rows with:
  - `uid`, `frame`, `image_path`, `crop_path`, `bbox`, `bbox_clamped`, `bbox_norm`, `W`, `H`.
  - If semantics are available: `is_positive`, `verb_id`, `noun_id`, `ttc`, `ttc_bin`.
- `head_train.jsonl` / `head_val.jsonl` (if `WRITE_HEAD_TRAIN_VAL=True` and manifests exist):
  - One record per frame, grouped candidates with semantic labels ready for Track B.
- Recall stats (if `EVAL_WITH_LABELS=True`):
  - Recall@K = (# frames with ≥1 hit) / (# frames with ≥1 GT).
  - Mean best IoU, histogram; optionally mirrored into Stage A `summary.json`.

### Sample head manifest record
```json
{
  "uid": "03abc",
  "frame": 120,
  "image": "local_extraction/v2/extracted_frames/03abc/0000120.jpg",
  "candidates": [
    {"bbox": [123.4, 56.7, 200.1, 180.0], "is_positive": 1, "noun_id": 42, "verb_id": 7, "ttc": 0.83, "ttc_bin": 1},
    {"bbox": [10.0, 10.0, 40.0, 50.0], "is_positive": 0, "noun_id": -1, "verb_id": -1, "ttc": 2.5, "ttc_bin": 3}
  ]
}
```

### Internal flow (what the code does)
1) Resolve `CANDIDATES_JSONL` (latest Stage A if None).
2) Build `RUN_DIR` and `crops/`.
3) Load optional head manifests for semantics (clip or video variant).
4) Iterate candidates:
   - Load frame, clamp boxes, optionally limit to `KEEP_TOP_N`.
   - Crop and resize each ROI to `CROP_SIZE`; save under `crops/`.
   - Emit per-crop manifest rows; attach semantics if available.
5) If semantics + `WRITE_HEAD_TRAIN_VAL=True`, merge into grouped `head_train.jsonl` / `head_val.jsonl`.
6) If `EVAL_WITH_LABELS=True`, compute recall@K and IoU stats; optionally patch Stage A summary.

### Troubleshooting Stage B
- **Empty crops/manifest:** check `CANDIDATES_JSONL` path; ensure Stage A run exists and has boxes.
- **Low recall:** inspect label source (`LABEL_SPACE`, `IOU_THRESH`); confirm GT label folder matches frames.
- **Missing semantics:** verify head manifests exist and `USE_CLIP_MANIFEST` is set to the right variant.
- **Cropping errors:** ensure frames exist at `FRAMES_ROOT`; look for permission or path typos.

---

## 4) Hand-off to Track B (and Track C)
- Track B auto-discovers the latest Stage B run via `latest_stageB_run` inside `trackB_dataset.py`.
- The Stage B `head_train.jsonl` / `head_val.jsonl` become the training/validation manifests for Track B.
- Track C pruning uses the same Stage B manifests and the Track B checkpoint; no changes needed in Track A once manifests are produced.

---

## 5) Quick Recipes
- **Fast oracle smoke test (recall@K focus):**
  - Stage A: `DETECTION_MODE='oracle'`, `LAST_FRAME_ONLY=True`, `K=5`, small `RUN_NAME`.
  - Stage B: point `CANDIDATES_JSONL` to that run, set `EVAL_WITH_LABELS=True`, `KEEP_TOP_N=5`.
- **YOLO proposals for real training:**
  - Stage A: `DETECTION_MODE='yolo'`, set weights/thresholds, `K=10`.
  - Stage B: `USE_CLIP_MANIFEST=True`, `WRITE_HEAD_TRAIN_VAL=True`, leave `CROP_SIZE=(256,256)`.
- **Semantics off (binary only):**
  - Stage B: set `WRITE_HEAD_TRAIN_VAL=False`; Track B will still see boxes/ttc if present but can run binary+TTC.
- **Throttle runtime:**
  - Stage A: set `MAX_IMAGES` and `LAST_FRAME_ONLY=True`.
  - Stage B: set `MAX_IMAGES` or `DEMO_MODE=True` with `DEMO_N`.

---

## 6) Outputs Cheat-Sheet (by folder)
- `runs/Track_A/StageA_*` (or custom `RUN_NAME`):
  - `candidates.jsonl`, optional `candidates/*.csv`, `summary.json`.
- `runs/Track_A/trackA_stageB_*`:
  - `crops/`, `manifest.jsonl`/`.csv`, `head_train.jsonl`, `head_val.jsonl`, recall stats.
- `runs/Track_B/` (downstream, for reference):
  - `checkpoints/`, `metrics/`, `predictions/`, `overlays/`, `plots/`.
- `runs/Track_C/` (downstream pruning, for reference):
  - `metrics/` and plots for RGTP experiments.

---

## 7) Field Reference
- Candidates (Stage A):
  - `boxes`: list of `{x1,y1,x2,y2,conf,cls}`; coordinates in pixels.
- Crops (Stage B):
  - `bbox`: original box; `bbox_clamped`: clipped to frame; `bbox_norm`: normalized `[0,1]`.
  - `crop_path`: saved ROI; `W`,`H`: frame size.
- Head manifests (Stage B → Track B):
  - `is_positive` (0/1), `noun_id`, `verb_id`, `ttc`, `ttc_bin`, `bbox`.

---

## 8) Common Pitfalls & Fixes
- **Stage A finds zero images:** verify `VERSION` and frame path; ensure frames are `.jpg` with zero-padded names.
- **YOLO import error:** install `ultralytics` locally and confirm weights path exists.
- **Stage B recall is zero:** make sure GT labels are in the same `SPACE` (`clips` vs `videos`) and frames align.
- **Manifest mixing clip/video:** set `USE_CLIP_MANIFEST` consistently; delete stale files in the run dir before rerunning.
- **Large run folders:** disable per-image CSVs, reduce `K`, and clean old runs under `runs/Track_A/`.

---

## 9) Integration Notes for Track B/C
- Track B expects head manifests with `is_positive` and TTC; noun/verb/TTC-bin fields enable multi-task heads and N/N+V/N+δ metrics.
- If you only need next-active + TTC, you can omit semantics (Track B falls back to binary + TTC regression).
- Track C pruning reuses the same manifests and only needs the Track B checkpoint; no changes to Track A once manifests are ready.

---

## 10) Versioning & Repro Tips
- Always set a descriptive `RUN_NAME` (include mode, K, and date), e.g., `stageA_oracle_K5_2025-11-26`.
- Keep a small text note in the run folder describing any deviations (e.g., different `IOU_THRESH`).
- For reruns, copy the exact toggles used in Stage A/B into your experiment log to make Track B/C results reproducible.

---

## 11) File Map (code)
- `trackA_stageA/trackA_stageA.py` — Stage A runner (oracle/YOLO proposals).
- `trackA_stageA/oracle_k_sweep.py` — sweep K for oracle recall.
- `trackA_stageA/sweep_k_metrics.py` — analyze K vs recall metrics.
- `trackA_stageB/trackA_stageB.py` — Stage B runner (crops + head manifests).
- `trackA_stageB/san_stageB.py` — sanity checks on Stage B outputs.

---

## 12) Minimal End-to-End Checklist
1) Verify frames exist under `local_extraction/v2/extracted_frames/`.
2) Run Stage A (oracle or YOLO) → confirm `candidates.jsonl` + `summary.json`.
3) Run Stage B → confirm `manifest.jsonl` and `head_train/val.jsonl`.
4) Point Track B to the Stage B run (auto-discovery works by default).
5) Train Track B → check `runs/Track_B/checkpoints/trackB_best.pt` and metrics.
6) (Optional) Run Track C pruning and compare metrics/plots.

This guide is intentionally verbose to serve as a single source of truth for Track A. Trim sections locally if you prefer a shorter version.
