# Track C — RGTP Pruning Evaluation (Detailed Guide)

This README explains the Track C pruning/evaluation pipeline end-to-end. It is deliberately detailed (~200+ lines) so you can run, audit, and extend Track C without digging into the code.

---

## 1) Purpose
Track C applies **Run-time Guided Token Pruning (RGTP)** on top of a trained Track B checkpoint. It prunes low-importance candidate tokens before the head, then evaluates accuracy/mAP/TTC and semantic metrics (N/N+V/N+δ), producing both quality metrics and pruning stats (latency/VRAM/FLOPs when enabled).

---

## 2) Inputs & Dependencies
- Track B checkpoint: `local_extraction/runs/Track_B/checkpoints/trackB_best.pt` (or any `trackB_final_*.pt` / best variant).
- Stage B manifests: from Track A Stage B run (`head_val.jsonl/json`) or `local_extraction/v2/manifests/`.
- Frames: `local_extraction/v2/extracted_frames/<uid>/<frame:07d>.jpg`.
- Python: torch, tqdm; optional CUDA for latency/VRAM measurements. No network calls.

---

## 3) Key Files
- `trackC_pruning.py` — main evaluation + pruning harness.
- `trackC_compare_metrics.py` — compares Track B vs Track C metrics JSON.
- `trackC_plots.py` — plots Track C metrics over time (includes runtime metrics).

---

## 4) Runtime Config (edit in `trackC_pruning.py`)
`RuntimeConfig` toggles (no CLI by default):
- Paths: `checkpoint`, `stageB_run`, `val_manifest` (None → auto-discover).
- Pruning: `pruning_enabled`, `rgtp_rate` (fraction to drop), `min_keep`.
- Instrumentation: `measure_latency`, `measure_vram`, `measure_flops`, `bench_warmup`, `bench_iters`, `bench_samples`.

Tips:
- Set `checkpoint` if you want a specific Track B model; otherwise latest/final is auto-picked.
- Use `pruning_enabled=False` or `rgtp_rate=0` to emulate Track B baseline.

---

## 5) Evaluation Config (EvalConfig)
- Paths: `frames_root`, `manifests_root`, `trackA_runs_root`, `checkpoint`, `stageB_run`, `val_manifest`.
- Data: `batch_size`, `candidate_limit`, `normalize_ttc`, `num_workers` (keep 0).
- Defaults point to `local_extraction/v2` assets and Track B runs/checkpoints.

---

## 6) RGTP Config (RGTPConfig)
- `enabled`: True/False.
- `rate`: fraction of candidate tokens to prune (clamped [0, 0.95]).
- `min_keep`: minimum candidates to keep per frame.
- `temporal_decay`: blend rollout vs motion energy for importance.

Importance scoring:
- Combines FGTP rollout attention (t-1) with motion energy (t-1 → t).
- Each candidate gets a score by ROI-pooling the blended grid.
- Lowest scores are dropped; dropped tokens receive a strong negative logit fill.

---

## 7) Instrumentation (InstrumentationConfig)
- `enabled`: set automatically if any of `measure_latency/vram/flops` are True.
- `record_vram`: peak CUDA memory (no-op on CPU).
- `record_flops`: one-off FLOPs profile of head forward pass.
- `use_cuda_events`: latency via CUDA events when available; falls back to wall clock.
- `bench_warmup`, `bench_iters`, `bench_samples`: control micro-benchmark of head forward (`torch.utils.benchmark`).

Runtime outputs (added to metrics JSON):
- Latency stats: mean/median/p90/p95 per sample (ms).
- Throughput: samples/s and candidates/s.
- Head micro-benchmark mean (ms) if enabled.
- Peak VRAM allocated/reserved (bytes) if CUDA.
- Head FLOPs (one-off) if profiling succeeds.

---

## 8) Outputs
All under `local_extraction/runs/Track_C/`:
- Metrics: `metrics/trackC_val_rateXX_<timestamp>.json`
  - Accuracy, mAP, TTC MAE, N/N+V/N+δ/All (top-1), top-5 hit/AP, per-noun/verb stats.
  - Pruning stats: requested rate, mean fraction pruned, counts.
  - Runtime stats (latency/throughput/VRAM/FLOPs) when instrumentation is on.
- Summary: `metrics/trackC_val_rateXX_<timestamp>_summary.json`
  - Contains metrics + serialized eval/pruning/instrumentation configs.
- Plots: from `trackC_plots.py` → `plots/trackC_metrics_over_time.png` and TSV summary.

---

## 9) How to Run (PowerShell examples)
- Default (pruning on, auto-discover checkpoint/manifest):
  ```powershell
  python local_extraction\trackC\trackC_pruning.py
  ```
- Disable pruning (baseline timing/metrics):
  ```powershell
  # set RuntimeConfig.pruning_enabled = False or rgtp_rate = 0 in file, then run
  python local_extraction\trackC\trackC_pruning.py
  ```
- Custom checkpoint/manifest:
  - Set `RuntimeConfig.checkpoint = "local_extraction/runs/Track_B/checkpoints/trackB_best.pt"`
  - Set `RuntimeConfig.val_manifest = "local_extraction/runs/Track_A/<run>/head_val.jsonl"`

---

## 10) Internal Flow (trackC_pruning.py)
1) Resolve Stage B run/val manifest (latest Track A Stage B if not provided).
2) Resolve Track B checkpoint (explicit or latest `trackB_final_*.pt`).
3) Load projector/fusion/head; infer class vocab and optional heads from checkpoint weights.
4) Build `TrackBDataset` (val) with tokenizer; DataLoader (num_workers=0).
5) For each sample:
   - Project img/vid tokens → fused tokens (via Track B fusion).
   - ROI-pool per candidate.
   - Compute RGTP scores (rollout + motion), build keep mask; drop low-score tokens.
   - Run head on kept tokens; fill dropped positions with negative logits/zeros (and multi-task logits if present).
   - Collect logits, TTC (denorm), labels, semantics; update frame-level N/N+V/N+δ metrics if heads present.
   - Record latency/bench/FLOPs/VRAM if enabled.
6) Aggregate metrics; compute prune rate; attach runtime stats; include checkpoint/manifest paths.
7) Save metrics JSON and summary JSON; print key metrics.

---

## 11) Metrics Tracked
- Candidate-level: `accuracy`, `mAP`, `ttc_mae_seconds`, `num_candidates`.
- Pruning: `rgtp_enabled`, `rgtp_rate_request`, `rgtp_mean_fraction_pruned`, `samples_total`, `samples_without_candidates`.
- Frame-level (top-1): `N_mAP`, `Nv_mAP`, `N_delta_mAP`, `All_mAP` (when noun/verb/bin available).
- Frame-level (top-5 hit): `N_top5_acc`, `Nv_top5_acc`, `N_delta_top5_acc`, `All_top5_acc`.
- Frame-level (top-5 AP): `N_top5_mAP`, `Nv_top5_mAP`, `N_delta_top5_mAP`, `All_top5_mAP`.
- Per-class breakdowns: `per_noun_stats`, `per_verb_stats`.
- Runtime: `latency_ms_mean`, `latency_ms_median`, `latency_ms_p90`, `latency_ms_p95`, `throughput_samples_per_s`, `throughput_candidates_per_s`, `head_benchmark_ms_mean`, `peak_vram_bytes`, `peak_vram_reserved_bytes`, `head_flops` (when available).

---

## 12) Plotting (trackC_plots.py)
- Scans `metrics/trackC_val_rate*.json`.
- Plots all important scalar metrics including runtime fields (latency/throughput/VRAM/FLOPs) when present.
- Writes:
  - `plots/trackC_metrics_over_time.png`
  - `plots/trackC_metrics_summary.tsv` (tabular snapshot with metric columns found).
- Usage:
  ```powershell
  python local_extraction\trackC\trackC_plots.py --runs_dir local_extraction/runs/Track_C --out_dir local_extraction/runs/Track_C/plots
  ```

---

## 13) Comparing Track B vs Track C
- Script: `trackC_compare_metrics.py`
- Defaults: picks latest `metrics_val_*.json` (Track B) and `trackC_val_rate*.json` (Track C).
- Prints side-by-side table for key metrics (accuracy, mAP, TTC, N/Nv/Nδ/All, top-5).
- Override with `--b_metrics` / `--c_metrics` to compare specific runs.

---

## 14) Common Workflows
- **Baseline vs Pruned:** run once with pruning off (`pruning_enabled=False`), once with pruning on (e.g., `rgtp_rate=0.5`), compare `N_mAP`, `mAP`, and latency stats.
- **Rate Sweep:** edit `RuntimeConfig.rgtp_rate` across runs (e.g., 0.25/0.5/0.75), then use plots/compare scripts to build mAP-latency curves.
- **Multi-task Semantics:** if Track B checkpoint has noun/verb/bin heads, check `Nv_mAP`, `All_mAP` to ensure pruning preserves semantics.
- **Latency Focus:** enable instrumentation; watch `latency_ms_*` and throughput vs `rgtp_mean_fraction_pruned`.

---

## 15) Troubleshooting
- **No val manifest found:** set `RuntimeConfig.val_manifest` explicitly; ensure Stage B run exists or manifests root has `head_val*.json[l]`.
- **Checkpoint missing:** set `RuntimeConfig.checkpoint` to an existing Track B checkpoint; otherwise train Track B first.
- **CUDA OOM:** reduce `candidate_limit`, disable pruning (to debug), or run on CPU; pruning itself should reduce head load when rate > 0.
- **Latency not recorded:** ensure `measure_latency=True`; CUDA events require CUDA; CPU falls back to wall clock.
- **VRAM fields zero:** likely CPU run; set `measure_vram=False` if CPU-only to avoid confusion.
- **FLOPs missing:** set `measure_flops=True` and ensure a CUDA-capable build; profiling may be expensive.

---

## 16) RGTP Scoring Details
- Rollout: uses fusion FGTP attention to score spatial tokens at t-1.
- Motion: squared difference between last two temporal slices.
- Blend: `temporal_decay * rollout + (1 - temporal_decay) * motion`.
- ROI pooling: averages blended grid over each candidate box → score.
- Pruning: keep top-K by score, where `K = max(min_keep, round(N * (1 - rate)))`.
- Dropped candidates: assigned `LOGIT_FILL` (very negative) so their next-active prob is near zero; TTC set to zero; multi-task logits filled likewise.

---

## 17) Frame-Level Semantic Metrics in Track C
- Requires Track B checkpoint with noun/verb heads (and TTC bin head if `ttc_mode='binned'`):
  - `N_mAP`: noun correct + IoU >= 0.5 on top candidate.
  - `Nv_mAP`: noun+verb correct + IoU >= 0.5.
  - `N_delta_mAP`: noun + TTC bin correct + IoU >= 0.5 (bin from logits or reg → bin).
  - `All_mAP`: noun+verb+TTC bin correct + IoU >= 0.5.
- Top-5 versions treat top-5 candidates per frame as predictions for AP/hit metrics.

---

## 18) File Map
- `trackC_pruning.py`: main eval + pruning + instrumentation + metrics writing.
- `trackC_compare_metrics.py`: diff Track B vs Track C metrics.
- `trackC_plots.py`: plot Track C metrics including runtime fields.
- Metrics dir: `runs/Track_C/metrics/`.
- Plots dir: `runs/Track_C/plots/` (created by trackC_plots.py).

---

## 19) Sample Metrics JSON Excerpt
```json
{
  "accuracy": 0.59,
  "mAP": 0.33,
  "ttc_mae_seconds": 0.19,
  "rgtp_mean_fraction_pruned": 0.50,
  "N_mAP": 0.25,
  "Nv_mAP": 0.05,
  "N_delta_mAP": 0.22,
  "latency_ms_mean": 12.5,
  "throughput_samples_per_s": 80.0
}
```

---

## 20) Tips for Reproducibility
- Use descriptive run names for Track B checkpoints and Track A Stage B runs.
- Keep pruning rates and instrumentation toggles in the summary JSON to trace exact settings.
- For sweeps, script multiple runs changing only `rgtp_rate`; then plot with `trackC_plots.py`.

---

## 21) Notes on Candidates & Pruning Interaction
- `candidate_limit` (EvalConfig/Dataset) caps candidates before pruning; RGTP prunes after scoring.
- If rate is high and `min_keep` low, many candidates get `LOGIT_FILL`; metrics still count them but with near-zero prob.
- If a sample has 0 candidates, it is skipped and counted in `samples_without_candidates`.

---

## 22) Hand-off from Track B
- Track C loads Track B checkpoint and infers head shapes (noun/verb/bin) from weight shapes.
- Uses Track B tokenizer/fusion/head directly; no retraining.
- `ttc_mode` defaults to `bin` if bin head present, else `reg`.

---

## 22.1) VideoMAE Backbone Support

Track C supports both video backbone variants:

### ResNet18 (Default)
- Per-frame 2D encoding with ImageNet-pretrained backbone
- Projector dimension: 512 → 256 tokens

### VideoMAE (Ego-Only)
- Spatiotemporal 3D encoding with Ego4D-pretrained backbone
- Projector dimension: 768 → 256 tokens

### Usage

```powershell
# Default (ResNet18 backbone)
python -m trackC.trackC_pruning

# With VideoMAE backbone
python -m trackC.trackC_pruning --video_backbone videomae_ego

# With specific checkpoint (must match backbone)
python -m trackC.trackC_pruning --video_backbone videomae_ego --checkpoint path/to/videomae_checkpoint.pt
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Config name (default: trackC) |
| `--video_backbone` | `resnet18` or `videomae_ego` |
| `--rgtp_rate` | Override pruning rate (0.0-0.95) |
| `--checkpoint` | Override checkpoint path |
| `--no_pruning` | Disable pruning (baseline mode) |

### Important Notes
- Checkpoint must match the backbone: ResNet18 checkpoints won't load with VideoMAE and vice versa.
- VideoMAE evaluation requires the VideoMAE pretrained encoder weights.
- See `trackB/videomae_readme.md` for VideoMAE architecture details.

---

## 23) Overlay Generation (Why Track C doesn’t do it)
- Track C focuses on pruning/time vs quality; overlays are already produced in Track B eval.
- If you need overlays under pruning, you can adapt Track B eval to apply RGTP scoring before head forward; Track C currently omits overlays to keep runtime slim.

---

## 24) Extending Track C
- Different pruning strategies: swap scoring function (`_candidate_scores`).
- Structured pruning: keep top-K per grid region; requires changing ROI pooling strategy.
- Multi-head latency analysis: add more granular timers around tokenizer/fusion/head.
- Batch-mode pruning: vectorize pruning across batch for speed (currently per-sample).

---

## 25) Cleanup & Storage
- Metrics/plots accumulate under `runs/Track_C/`; delete old runs to save disk.
- Runtime summaries are duplicated in summary JSON; plots pick up new fields automatically.

---

## 26) Glossary
- RGTP: Run-time Guided Token Pruning.
- LOGIT_FILL: very negative logit assigned to pruned candidates.
- N/Nv/Nδ/All: noun; noun+verb; noun+TTC-bin; noun+verb+TTC-bin metrics at frame level.
- TTC: Time-to-contact (seconds).
- FGTP: Frame-Guided Temporal Pooling (from Track B fusion).

---

## 27) Minimal Checklist
- [ ] Track B checkpoint available.
- [ ] Val manifest resolvable (Stage B run or explicit path).
- [ ] Pruning toggles set (`pruning_enabled`, `rgtp_rate`, `min_keep`).
- [ ] Instrumentation toggles set if latency/VRAM/FLOPs needed.
- [ ] Run script; confirm metrics + summary JSON.
- [ ] Plot metrics if comparing multiple runs.

This README should give you everything needed to operate and extend Track C locally. Trim sections if you want a shorter version for your own notes.
