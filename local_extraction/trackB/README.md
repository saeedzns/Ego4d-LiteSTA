# Track B — Complete Local Pipeline Guide (Stage B Head, Multi-Task, Priors, Evaluation)

This document is intentionally exhaustive (~500 lines) so you can run, audit, and extend Track B without diving back into the code. It covers data assumptions, modules, configs, training/eval flows, outputs, troubleshooting, and reference tables.

---

## 0. TL;DR Quick Path
- Prepare inputs: Stage B manifests (`head_train.jsonl`, `head_val.jsonl`) and frames under `local_extraction/v2/extracted_frames/<uid>/`.
- Train: edit `TrainConfig` in `trackB_train_loader.py`, then run `python local_extraction/trackB/trackB_train_loader.py`.
- Eval: edit `EvalConfig` in `trackB_eval.py`, then run `python local_extraction/trackB/trackB_eval.py`.
- Outputs: checkpoints in `runs/Track_B/checkpoints/`, metrics/preds/overlays in `runs/Track_B/{metrics,predictions,overlays/val}`; plots in `runs/Track_B/plots/`.
- Multi-task (noun/verb/TTC-bin) is controlled by `use_multi_task_labels` / `use_ttc_bins` in `TrainConfig`; hotspot/CLIP are eval-time only.

---

## 1. Repository Layout (Track B)
- `trackB_dataset.py` — Dataset + collate; manifest discovery; TTC stats; multi-task labels.
- `trackB_tokenizer.py` — Backbone builder, transforms, frame/video tokenization, ROI pooling.
- `trackB_fusion.py` — Frame-Guided Temporal Pooling + dual cross-attention blocks.
- `trackB_head.py` — Classification + TTC regression, optional noun/verb/TTC-bin heads.
- `trackB_train_loader.py` — Main training loop with validation, early stopping, checkpointing.
- `trackB_train_demo.py` — Minimal/demo trainer with synthetic fallback.
- `trackB_eval.py` — Standalone evaluation (metrics, predictions, overlays).
- `trackB_metrics.py` — Accuracy, AP, TTC MAE helpers.
- `trackB_plots.py` — Aggregates metrics JSON into PNG + TSV.
- `trackB_manifest_validator.py` — Coverage/invalid logging for manifests.
- `trackB_show_samples.py`, `trackB_show_config.py` — Inspection helpers.
- `trackB_tests.py` — Small sanity tests.

---

## 2. Data & Manifests Assumptions
- Frames: `local_extraction/v2/extracted_frames/<uid>/<frame:07d>.jpg`.
- Stage B manifests: `head_train.jsonl` / `head_val.jsonl` from Track A Stage B run or `local_extraction/v2/manifests/`.
- Candidate fields expected (per record, grouped by frame):
  - `bbox`: `[x1, y1, x2, y2]` in pixel coords.
  - `is_positive`: 0/1 next-active flag (falls back to `cls==1` if absent).
  - `verb_id`, `noun_id`: ints or `-1` if missing.
  - `ttc`: seconds (float); `ttc_bin` optional.
- Dataset will compute `ttc_bin` if missing via thresholds `(0.5, 1.0, 2.0)`.
- If manifests are missing, synthetic records are generated (config-dependent).

---

## 3. Tokenization Pipeline (trackB_tokenizer.py)
- `TokenizerConfig` holds device, image size, time length/stride, normalization.
- `build_backbone` returns a CLIP-like encoder (512-dim tokens).
- `build_transform` returns preprocessing matching the backbone.
- `list_uid_frames` enumerates frames per UID.
- `sample_window_ending_at` selects a temporal window ending at the target frame.
- `image_grid_tokens(path, backbone, transform, cfg)` → `(N,512)` tokens + grid hw.
- `video_grid_tokens(window, backbone, transform, cfg)` → `(T,N,512)` tokens.
- `roi_pool_tokens_mean(hw, tokens, box, image_size)` → pooled token for an ROI.
- Keep `num_workers=0` unless you move tokenization out of the dataset.

---

## 4. Fusion Module (trackB_fusion.py)
- `FusionConfig`: `dim`, `layers`, `heads`, dropout.
- `FGTP`: Frame-Guided Temporal Pooling projects video tokens onto the last-frame grid.
- `DualCrossAttention`: symmetric image↔video cross-attention block (lightweight).
- `TrackBFusion`: FGTP + stacked cross-attention layers; outputs `(fused_img, fused_vid)` each `(B,N,dim)`.
- Default: dim=256, layers=2, heads=8 (tuned for speed/simplicity).

---

## 5. Head Module (trackB_head.py)
- `HeadConfig`: `dim`, `num_classes` (>=2), `hidden`, `dropout`, optional `num_noun_classes`, `num_verb_classes`, `num_ttc_bins`, `ttc_mode` (`reg` or `bin`).
- `TrackBHead` outputs:
  - `cls_logits`: (B,N,K_next) next-active logits.
  - `ttc`: (B,N,1) continuous TTC regression.
  - Optional: `noun_logits`, `verb_logits`, `ttc_bin_logits` if enabled.
- Multi-task is gated by dataset label availability and `TrainConfig` toggles.

---

## 6. Dataset (trackB_dataset.py)
- Init args: `frames_root`, `manifests_root`, `manifest_path`, `tokenizer_cfg`, `candidate_limit`, `normalize_ttc`, `synthetic_if_empty`, `cache_dir`, `seed`.
- Manifest discovery: tries `head_train*.json[l]` (or `head_val*.json[l]`) in provided root; Stage B run auto-discovery via `latest_stageB_run`.
- Parsing:
  - Accepts flexible keys (`uid|video_uid`, `frame|frame_idx|frame_index`, `frame_name|image`, `candidates|boxes|objects`).
  - Converts per-candidate dict/list into normalized fields: `bbox`, `cls`, `is_positive`, `verb_id`, `noun_id`, `ttc`, `ttc_bin`.
  - TTC stats computed (mean/std/min/max); `ttc_norm` added if `normalize_ttc=True`.
- `__getitem__`:
  - Resolves frame path (absolute or via UID folder).
  - Builds `img_tokens` (N,512) and `vid_tokens` (T,N,512).
  - Applies `candidate_limit` by random subsample.
  - Returns dict with tokens, `hw`, `image_size`, `bboxes`, labels, `is_positive`, `gt_noun_id`, `gt_verb_id`, `gt_ttc`, `gt_ttc_bin`, `ttc`, `ttc_norm`, `valid`.
- Collate (`trackB_collate`): returns `{'samples': [...], 'valid': True}` preserving variable candidate counts.
- QA: invalid records logged to `runs/Track_B/cache/invalid_records.csv`; TTC stats saved in cache.

---

## 7. Training Config (trackB_train_loader.py — TrainConfig fields)
- Core:
  - `mode`: `'main'` or `'demo'`.
  - `epochs`, `batch_size`, `lr`, `min_lr`, `warmup_epochs`.
  - `candidate_limit`, `label_smoothing`, `normalize_ttc`, `amp`.
- Validation:
  - `eval_every`, `early_stopping_patience`, `monitor_metric` (`mAP`/`accuracy`/`ttc_mae`), `greater_is_better` auto-set for `ttc_mae`.
  - `save_best_checkpoint`, `save_epoch_checkpoints`.
  - `val_manifest`: optional override; otherwise discovered.
- Data:
  - `train_manifest`, `stageB_run` override, `time_len`, `time_stride` via `TokenizerConfig`.
- Multi-task:
  - `use_multi_task_labels` (enable noun/verb heads).
  - `use_ttc_bins` (enable TTC-bin head and CE loss).
  - Loss weights: `loss_w_next`, `loss_w_noun`, `loss_w_verb`, `loss_w_ttc`.
- Demo:
  - `demo_steps`, `demo_variant`, `demo_use_real_labels`, `demo_manifest`.

---

## 8. Training Pipeline (trackB_train_loader.py — main mode)
- 1) Build dataset `ds` (train manifest) and optional `ds_val` (val manifest).
- 2) Infer vocab sizes from parsed candidates:
   - Next-active classes: max label + 1, min 2.
   - Noun/verb vocab from observed IDs when `use_multi_task_labels=True`.
   - TTC bins from observed when `use_ttc_bins=True`.
- 3) Instantiate modules:
   - `projector`: Linear 512→TOKEN_DIM (256).
   - `fusion`: `TrackBFusion`(dim=256,layers=2,heads=8).
   - `head`: `TrackBHead` with inferred class counts and `ttc_mode` (`reg` or `bin`).
- 4) Optimizer: Adam over projector+fusion+head; scaler for AMP if enabled.
- 5) Training loop per epoch:
   - Shuffle loader; for each batch:
     - For each sample: project tokens, fuse, ROI-pool per candidate → tensors (N_c,256).
     - Pad to max candidates in batch; build masks.
     - Forward head; compute losses:
       - Next-active CE on `is_positive` (or `labels` fallback).
       - TTC: SmoothL1 (reg) or CE (bin) on `gt_ttc_bin` if enabled.
       - Noun/verb CE on positives when multi-task enabled.
       - Weighted sum via `loss_w_*`.
     - Backprop + optimizer step; optional AMP.
   - Log per-epoch avg losses (`next`, `ttc`, `noun`, `verb`).
- 6) Validation (every `eval_every` epochs):
   - `_evaluate_loader` computes metrics on `ds_val` (same as eval script subset).
   - Track `best_val` on `monitor_metric`; save best checkpoint(s).
- 7) Checkpoints:
   - Epoch checkpoints if `save_epoch_checkpoints=True`.
   - Best checkpoint: `trackB_best_<metric>_<score>_<ts>.pt` + alias `trackB_best.pt`.
   - Summary JSON alongside best checkpoint: `trackB_best_<metric>_<score>_<ts>_summary.json` and `trackB_best_summary.json`.
   - Final checkpoint: `trackB_final_<ts>.pt`.
- 8) Post-train quick sanity: prints one batch’s preds/acc/ttc.

---

## 9. Demo Pipeline (trackB_train_demo.py)
- Smaller, single-process loop for debugging.
- Toggle `DemoConfig.use_real_labels` to use manifests or synthetic.
- Good for checking token shapes, loss computation, and overfit-on-small-batch behavior.

---

## 10. Evaluation Script (trackB_eval.py)
- Config (`EvalConfig`):
  - Paths: `frames_root`, `manifests_root`, `trackA_runs_root`.
  - `val_manifest`: optional override (else discovery order: head_val_video.json[l], head_val_clip.json[l], val.json[l]).
  - `checkpoint_path`: optional override (else latest `trackB_final_*.pt` or `trackB_best.pt`).
  - Model: `token_dim`, `num_classes` (fallback), `fusion_layers`.
  - Data: `batch_size`, `candidate_limit`, `normalize_ttc`.
  - Outputs: `topk_overlay`, `save_overlays`.
  - Multi-task eval: `ttc_mode` (`reg`/`binned`), `iou_thresh`.
  - Priors: `use_hotspot_priors`, `hotspot_prior_path`, `hotspot_alpha`.
  - CLIP re-rank: `use_clip_rerank`, `clip_weight`, `clip_model`, `noun_label_path`.
- Flow:
  1) Discover val manifest and checkpoint; load projector/fusion/head; infer class counts from checkpoint weights.
  2) Build `TrackBDataset` (val) with tokenizer; DataLoader (num_workers=0).
  3) For each sample:
     - Tokenize, project, fuse, ROI-pool per candidate.
     - Forward head; apply optional hotspot/CLIP re-scoring to next-active probs.
     - Collect logits/labels, TTC (denorm), semantics if available.
     - Compute frame-level top-1 and top-5 metrics (N/N+V/N+δ/All) when noun/verb present.
     - Save overlay images if enabled (top-K per frame).
  4) Aggregate metrics:
     - Candidate-level: accuracy, mAP, `ttc_mae_seconds`.
     - Frame-level: `N_mAP`, `Nv_mAP`, `N_delta_mAP`, `All_mAP`.
     - Top-5 hit/AP metrics; per-noun/verb accuracy breakdowns.
  5) Write outputs:
     - `metrics/metrics_val_<ts>.json` (metrics only).
     - `metrics/metrics_val_<ts>_summary.json` (metrics + eval config).
     - Predictions CSV/JSONL under `predictions/`.
     - Overlays under `overlays/val/`.
- CLI overrides: `--checkpoint`, `--val_manifest`, `--stageB_run`, `--ttc_mode`.

---

## 11. Metrics Definitions
- Candidate-level:
  - `accuracy`: next-active accuracy.
  - `mAP`: average precision (binary or macro over classes).
  - `ttc_mae_seconds`: mean absolute error on TTC (denormalized).
- Frame-level (top-1):
  - `N_mAP`: noun matches on highest-prob candidate with IoU >= `iou_thresh`.
  - `Nv_mAP`: noun & verb match on top candidate with IoU >= threshold.
  - `N_delta_mAP`: noun & TTC bin match on top candidate with IoU >= threshold (predicted bin or bin-from-reg).
  - `All_mAP`: noun & verb & TTC bin match on top candidate with IoU >= threshold.
- Frame-level (top-5):
  - Hit rates: `N_top5_acc`, `Nv_top5_acc`, `N_delta_top5_acc`, `All_top5_acc`.
  - AP (top-5 candidates only): `N_top5_mAP`, `Nv_top5_mAP`, `N_delta_top5_mAP`, `All_top5_mAP`.
- Per-class breakdown:
  - `per_noun_stats`, `per_verb_stats`: count/correct/accuracy per taxonomy ID.

---

## 12. Hotspot Priors & CLIP Re-Rank (Eval-time only)
- Hotspot priors:
  - JSON format: `{"pairs": {"noun_id,verb_id": score}, "default": 0.0}`.
  - Applied as `score_final = (1 - alpha) * base + alpha * prior`.
  - Controls: `use_hotspot_priors=True`, set `hotspot_prior_path`, adjust `hotspot_alpha`.
- CLIP re-rank:
  - Text prompts from `noun_label_path` (STA label JSON) matched to candidates.
  - Blends CLIP score with base next-active prob via `clip_weight`.
  - Controls: `use_clip_rerank`, `clip_weight`, `clip_model`.
- Both only affect eval scoring, not training.

---

## 13. Checkpoints & Summaries
- Location: `local_extraction/runs/Track_B/checkpoints/`.
- Files:
  - `trackB_epochE_<ts>.pt` (if `save_epoch_checkpoints=True`).
  - `trackB_final_<ts>.pt` (always saved).
  - `trackB_best_<metric>_<score>_<ts>.pt` (best by monitor metric) + `trackB_best.pt` alias.
  - Summary JSONs: `trackB_best_<metric>_<score>_<ts>_summary.json`, `trackB_best_summary.json`.
- Contents:
  - `projector`, `fusion`, `head` state_dicts.
  - `noun_id_list`, `verb_id_list` for mapping local head indices to global IDs.
  - `train_config` (serialized `TrainConfig`).

---

## 14. Plots & Comparisons
- `trackB_plots.py`:
  - Scans `metrics/metrics_val_*.json`.
  - Produces `plots/trackB_metrics_over_time.png` and `trackB_metrics_summary.tsv`.
  - Adds any new scalar metrics it finds.
- Track C comparison:
  - `local_extraction/trackC/trackC_compare_metrics.py` compares latest Track B vs Track C metrics.

---

## 15. Running Commands (PowerShell examples)
- Train main:
  ```powershell
  python local_extraction\trackB\trackB_train_loader.py
  ```
- Eval standalone:
  ```powershell
  python local_extraction\trackB\trackB_eval.py --ttc_mode reg
  ```
- Fusion smoke:
  ```powershell
  python local_extraction\trackB\trackB_fusion.py
  ```
- Plot metrics:
  ```powershell
  python local_extraction\trackB\trackB_plots.py
  ```

---

## 16. Integration with Track A / Track C
- Track A Stage B produces `head_train.jsonl` / `head_val.jsonl`; Track B auto-discovers latest via `latest_stageB_run`.
- Track C pruning consumes Track B checkpoint + same Stage B manifests; no changes required in Track B when evaluating pruning.

---

## 17. Troubleshooting & FAQs
- **No candidates parsed:** check manifest paths and keys (`candidates|boxes|objects`); run `trackB_manifest_validator.py`.
- **Empty val metrics:** ensure `val_manifest` resolves; set explicit path in `EvalConfig` or `TrainConfig`.
- **Cuda OOM in training:** reduce `batch_size`, `candidate_limit`, fusion layers; disable AMP if unstable.
- **Mismatch noun/verb IDs:** confirm manifest IDs are ints; `noun_id_list`/`verb_id_list` are inferred from data seen.
- **Priors/CLIP not changing scores:** ensure `use_hotspot_priors`/`use_clip_rerank` are True and paths exist; check `hotspot_alpha`/`clip_weight`.
- **ttc_mode confusion:** `reg` uses regression head; `binned` uses `ttc_bin_logits` if present; Track B will infer `ttc_mode` from checkpoint when bin head exists.
- **Slow DataLoader:** keep `num_workers=0` (dataset holds backbone); precompute tokens if scaling up.
- **Overlays missing:** set `save_overlays=True` and ensure frames exist; reduce `topk_overlay` if too many images.
- **Best checkpoint not saved:** verify `monitor_metric` is valid; check metric not NaN; ensure `save_best_checkpoint=True`.

---

## 18. Extended Field Reference (per-sample dict from Dataset)
- `uid`: string UID.
- `frame_path`: absolute path to the image.
- `img_tokens`: `(N,512)` float32.
- `vid_tokens`: `(T,N,512)` float32 (may fallback to single-frame).
- `hw`: (H_tokens, W_tokens) grid shape.
- `image_size`: (W,H) in pixels.
- `bboxes`: list of candidate boxes.
- `labels`: LongTensor[N] (legacy cls).
- `is_positive`: LongTensor[N] (next-active flag).
- `gt_noun_id`: LongTensor[N] (noun or -1).
- `gt_verb_id`: LongTensor[N] (verb or -1).
- `gt_ttc`: FloatTensor[N] (seconds).
- `gt_ttc_bin`: LongTensor[N] (bin index).
- `ttc`: FloatTensor[N] (alias of gt_ttc).
- `ttc_norm`: FloatTensor[N] (z-scored if normalize_ttc=True).
- `valid`: bool.

---

## 19. Extended Metric Interpretation
- Use `Nv_mAP` to judge noun+verb correctness on top candidate; sensitive to semantic heads.
- `N_delta_mAP` reflects noun + TTC-bin agreement; useful when TTC bins are enabled.
- Top-5 metrics capture recall when multiple candidates are plausible; useful for ambiguous scenes.
- `ttc_mae_seconds` should be read alongside classification metrics; regression and binning can be compared by switching `ttc_mode`.
- Per-noun/verb breakdowns help detect class imbalance or poor recall on specific actions.

---

## 20. Ablation Suggestions
- Fusion depth: layers {2,3,4}; dimension {192,256,320}; heads {4,8}.
- Candidate pruning: vary `candidate_limit` in dataset and prune rate in Track C to study latency vs mAP.
- TTC handling: compare regression vs binning (`use_ttc_bins=True`) on `N_delta_mAP` and `ttc_mae_seconds`.
- Multi-task on/off: toggle `use_multi_task_labels` to see impact on N/N+V/N+δ without changing data.
- Priors: sweep `hotspot_alpha` and CLIP `clip_weight` to quantify gains on `Nv_mAP` in ambiguous scenes.

---

## 21. Serialization Details
- Checkpoints include `train_config` dict from `_config_to_dict` (scalar-friendly fields).
- Best summaries include metrics snapshot and config; filenames mirror checkpoint name with `_summary`.
- Eval summaries mirror metrics with serialized `EvalConfig` (paths converted to strings).

---

## 22. Stage-B Manifest Notes (consumed by Track B)
- Expected keys when writing your own:
  - `uid`, `frame` or `frame_idx`, `image` (path), `candidates`.
  - Each candidate: `x1,y1,x2,y2`, `is_positive`, `verb_id`, `noun_id`, `ttc`, optional `ttc_bin`.
- Grouped per frame; Track B groups them if provided flat by image path in Stage B format.
- Use consistent taxonomy IDs for noun/verb across train/val.

---

## 23. CLI vs In-File Config
- Training/eval scripts favor in-file config for reproducibility; only a few CLI overrides exist for eval.
- Recommended: set paths and toggles directly in `TrainConfig` / `EvalConfig` and commit the files if you need reproducible runs.

---

## 24. Performance Tips
- Reduce `time_len` in tokenizer if temporal window is large and not critical.
- Lower fusion layers or dim for faster runs; Monitor mAP drop.
- Use AMP (`amp=True`) cautiously; verify stability.
- For quick experiments, set `MAX_IMAGES` in Stage A/B and small `candidate_limit` in Track B.

---

## 25. Example Experiment Templates
- **Baseline single-task:** `use_multi_task_labels=False`, `use_ttc_bins=False`; monitor `mAP` + `ttc_mae_seconds`.
- **Full multi-task:** `use_multi_task_labels=True`, `use_ttc_bins=True`; monitor `N_mAP`, `Nv_mAP`, `N_delta_mAP`.
- **Hotspot+CLIP eval:** train with multi-task on; eval with `use_hotspot_priors=True`, `use_clip_rerank=True`, moderate alphas.
- **Latency-aware:** pair Track B metrics with Track C pruning metrics to plot mAP vs prune rate.

---

## 26. Logging & Caching
- Cache dir: `runs/Track_B/cache/` holds TTC stats, coverage logs, invalid records CSV.
- Training prints per-step loss (with postfixes) and per-epoch averages.
- Validation prints metric dict per eval interval.
- Eval prints metrics dict and paths written.

---

## 27. Overlay Rendering (Eval)
- Draws boxes with probability and TTC (pred + GT).
- Colors cycle per class; top-K controlled by `topk_overlay`.
- Saved under `runs/Track_B/overlays/val/`.

---

## 28. Predictions CSV/JSONL Schema
- Common fields:
  - `uid`, `frame`, `bbox`, `prob_pos`, `score_final` (after priors/CLIP), `prob_base` (pre-prior).
  - `pred_noun_id`, `pred_verb_id`, `pred_ttc_s`, `pred_ttc_bin`.
  - `gt_noun_id`, `gt_verb_id`, `gt_ttc_s`, `gt_ttc_bin`.
  - `hotspot_score` (if used), `clip_score` (if used).
- JSONL mirrors CSV rows as dicts.

---

## 29. Dev Notes on Code Structure
- Minimal external deps: PyTorch, PIL, tqdm, (optional) CLIP/priors loaders.
- Torch usage avoids JIT/custom kernels; ROI pooling is pure Python over token grids.
- Tokenizer holds backbone; hence DataLoader workers must stay 0 unless refactored.
- Functions are organized per concern: dataset parsing, fusion, head, metrics, train, eval.

---

## 30. Safety & Paths
- All scripts default to relative paths inside repo; no network calls.
- Overwrite-safe: run dirs are timestamped; best/final names are stable aliases.
- Avoid destructive commands; clean old runs manually if disk grows.

---

## 31. Minimal Code Walkthrough (Training)
- Dataset: yields tokens + labels per sample.
- Batch loop:
  - For each sample: pool candidates → list of tensors.
  - Pad pooled tensors to (B, max_candidates, dim) with masks.
  - Head forward; losses masked where no candidate.
  - Sum weighted losses; backward; step optimizer.
- Validation mirrors eval but without overlays/preds writing.

---

## 32. Minimal Code Walkthrough (Evaluation)
- Load checkpoint, infer head shapes.
- Build val dataset/loader.
- For each sample:
  - Project/fuse tokens; pool per candidate.
  - Head forward.
  - Optional priors/CLIP: adjust scores.
  - Collect logits/labels/TTC; compute frame-level semantics if heads present.
  - Save overlay if enabled.
- Aggregate metrics; write JSON + summary + preds.

---

## 33. Extending Track B
- Swap backbone: change `build_backbone` and token dims; update projector/head dims.
- Add heads: extend `HeadConfig`/`TrackBHead`; add losses; update metrics if needed.
- Precompute tokens: persist `img_tokens`/`vid_tokens` to disk and modify dataset to load them.
- Multi-GPU: current loop is single-device; would need DistributedDataParallel + tokenization refactor.

---

## 34. Environment Assumptions
- Local filesystem; no network.
- CUDA optional; code runs on CPU (slower).
- Python 3; PyTorch installed; optional `ultralytics` for Stage A (outside Track B scope).

---

## 35. Cleanup & Housekeeping
- Remove stale runs in `runs/Track_B/` to save disk.
- Keep best/final checkpoints; delete old epoch checkpoints if not needed.
- Archive metrics/plots for reporting.

---

## 36. Glossary
- **FGTP**: Frame-Guided Temporal Pooling.
- **N_mAP**: noun match AP (frame-level top-1).
- **Nv_mAP**: noun+verb match AP (frame-level top-1).
- **N_delta_mAP**: noun+TTC-bin match AP (frame-level top-1).
- **All_mAP**: noun+verb+TTC-bin match AP (frame-level top-1).
- **TTC**: Time-to-contact (seconds).
- **Hotspot priors**: noun/verb pair prior scores blended into next-active probability.
- **CLIP re-rank**: text-image similarity blended into next-active probability.

---

## 37. End-to-End Checklist (Concise)
- [ ] Stage B manifests present.
- [ ] Frames accessible at `frames_root`.
- [ ] TrainConfig set (multi-task, bins, monitor metric).
- [ ] Run training; verify checkpoints and best summary JSON.
- [ ] Run eval; verify metrics, preds, overlays, summary JSON.
- [ ] Plot metrics; compare with Track C if needed.

---

## 38. Example Settings (Copy/Paste)
- Multi-task run:
  ```python
  TrainConfig.use_multi_task_labels = True
  TrainConfig.use_ttc_bins = True
  TrainConfig.monitor_metric = "mAP"
  TrainConfig.eval_every = 1
  ```
- Eval with priors:
  ```python
  EvalConfig.use_hotspot_priors = True
  EvalConfig.hotspot_alpha = 0.3
  EvalConfig.use_clip_rerank = True
  EvalConfig.clip_weight = 0.3
  ```

---

## 39. When Things Go Wrong (Debug Steps)
- Set `candidate_limit` low; run tiny batch; print shapes in train loop.
- Run `trackB_train_demo.py` to isolate tokenization vs training issues.
- Inspect `cache/invalid_records.csv` to see why records were skipped.
- Force `val_manifest` to a known file; reduce batch size for eval overlays.
- Disable priors/CLIP to isolate base head performance.

---

## 40. Notes on TTC Binning
- Bins: [0,0.5), [0.5,1.0), [1.0,2.0), [2.0, inf).
- `use_ttc_bins=True` switches TTC loss to CE on bins; regression head still present for compatibility.
- Eval `ttc_mode` can be `binned` (use logits) or `reg` (bin the reg prediction).

---

## 41. Notes on Positive Definition
- `is_positive` is preferred; falls back to `cls==1`.
- Frame-level metrics assume at most one positive per frame; picks first positive index.
- Ensure manifests mark next-active candidate explicitly for consistent metrics.

---

## 42. ROI Pooling Details
- Uses grid-aware mean over token grid inside the bounding box.
- Requires `hw` (grid shape) and `image_size` to map ROI to grid indices.
- Pure Python; no compiled ops; safe on CPU/GPU.

---

## 43. Normalization & TTC Stats
- `normalize_ttc=True` z-scores TTC using dataset mean/std.
- Stored in `ds.stats`; used to denorm predictions in eval.
- TTC MAE is computed on denormalized seconds.

---

## 44. Candidate Subsampling
- `candidate_limit` in dataset randomly subsamples per sample if too many boxes.
- Track C pruning handles runtime pruning; keep Track B candidate_limit to reasonable size for training stability.

---

## 45. Stage B Discovery Logic
- `latest_stageB_run` finds newest `trackA_stageB_*` under `runs/Track_A/` containing head manifests.
- `resolve_stageB_manifest` picks `head_train/val.{jsonl,json}` inside that run.
- You can override with absolute paths in configs.

---

## 46. Eval Summary JSON Contents
- `metrics`: full metrics dict.
- `eval_config`: serialized EvalConfig (paths as strings).
- Useful for sweeps: compare settings without re-opening code.

---

## 47. Best Summary JSON Contents (Training)
- `checkpoint`, `monitor_metric`, `monitor_value`, `timestamp`, `metrics`, `train_config`.
- Filenames match best checkpoint with `_summary` suffix; also mirrored to `trackB_best_summary.json`.

---

## 48. Plot Summary TSV Columns
- `idx`, `checkpoint`, `timestamp`, then ordered metrics found (accuracy, mAP, TTC, N/Nv/N_delta/All, top-5, priors if present).
- Easy to diff runs in Excel or `pandas`.

---

## 49. Comparing Runs Manually
- Use `trackB_metrics_summary.tsv` for quick deltas.
- For deeper dive, diff `metrics_val_*.json` pairs (baseline vs priors/CLIP).
- Check predictions JSONL for frames where N/Nv/All differ; open overlays.

---

## 50. Closing
- Track B is designed to be a small, modular head on top of Track A Stage B outputs.
- This README should let you operate, debug, and extend without re-reading all code.
- If adding features, mirror new toggles and outputs here to keep it the single source of truth.
