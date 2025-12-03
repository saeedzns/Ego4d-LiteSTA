# Runs Configs & Results

This document presents evaluation results and configuration details for Track A, B, and C experiments on the **Ego4D-STA v2 validation split**.

Results in **Top-5 mAP (%)** — higher is better. **N** = noun-only, **N+V** = noun + verb, **N+δ** = noun + TTC-bin, **All** = noun + verb + TTC-bin.

---

## 📊 Comparison with Related Work (Ego4D-STA v2 Val)

| Model | N | N+V | N+δ | All | Source |
|:------|:---:|:---:|:---:|:---:|:-------|
| FRCNN+SF [17] | 21.00 | 7.45 | 7.07 | 2.98 | Literature |
| InternVideo [4] | 19.45 | 8.00 | 6.97 | 3.25 | Literature |
| StillFast [42] | 20.26 | 10.37 | 7.26 | 3.96 | Literature |
| GANO v2 [50] | 20.52 | 10.42 | 7.28 | 3.99 | Literature |
| STAformer | 24.85 | 13.45 | 7.41 | 4.90 | Literature |
| STAformer + AFF | 27.03 | 14.36 | 8.72 | 5.04 | Literature |
| STAformer + MH | 27.51 | 14.68 | 9.63 | 5.50 | Literature |
| **STAformer + MH + AFF** | **29.39** | **15.38** | **9.94** | **5.67** | Literature |
| — | — | — | — | — | — |
| **Ours: Track B (1122_0428)** | 10.94 | 2.83 | 9.20 | 2.44 | This work |
| **Ours: Track B (1122_1131)** | 13.69 | 3.46 | 11.81 | 3.05 | This work |
| **Ours: Track B (1122_1735)** | **17.43** | **8.48** | **14.92** | **8.66** | This work |
| Ours: Track C (1126_151858) | 11.06 | 2.99 | 9.95 | 2.98 | This work (pruned) |

> **Note:** Our best Track B checkpoint (`1122_1735`, bin TTC) achieves competitive N+δ (14.92%) and All (8.66%) metrics, outperforming several baselines on TTC-related metrics while using a lightweight two-stage architecture.

> **TTC Training Details:**  
> - **Track B (1122_1735)** is the only checkpoint that trains a TTC classification head (`use_ttc_bins=True`).  
> - **Other checkpoints** use TTC regression during training; at eval time, TTC MAE is computed alongside TTC-binned metrics (N+δ, All).  
> - This means time-to-contact is regressed in continuous time (consistent with Ego4D annotations) and discretized into bins only at evaluation for N+δ and All metrics, matching the benchmark protocol and remaining comparable to prior Ego4D-STA work.

---

## 📊 Key Metrics Summary (Top-5 mAP %)

### Track B — Fusion Head (Unpruned)

| Checkpoint | N | N+V | N+δ | All | Timestamp |
|:-----------|:---:|:---:|:---:|:---:|:----------|
| `trackB_best_1122_0428.pt` | 10.94 | 2.83 | 9.20 | 2.44 | 2025-11-22 20:09 |
| `trackB_best_1122_1131.pt` | 13.69 | 3.46 | 11.81 | 3.05 | 2025-11-22 20:21 |
| `trackB_best_1122_1735.pt` | **17.43** | **8.48** | **14.92** | **8.66** | 2025-11-22 20:30 |

> **Best overall:** `trackB_best_1122_1735.pt` (bin TTC, equal loss weights)

### Track C — RGTP Pruning (rate=0.5)

| Checkpoint | N | N+V | N+δ | All | File |
|:-----------|:---:|:---:|:---:|:---:|:-----|
| `trackB_best_1122_1131.pt` | 5.85 | 3.50 | 5.19 | 3.50 | trackC_val_rate50_20251123_020030 |
| `trackB_best_1122_1131.pt` | 8.10 | 2.16 | 7.03 | 2.16 | trackC_val_rate50_20251123_024455 |
| `trackB_best_1122_1131.pt` | **11.06** | **2.99** | **9.95** | **2.98** | trackC_val_rate50_20251126_151858 |
| `trackB_best_1122_1131.pt` | 11.06 | 2.99 | 9.95 | 2.98 | trackC_val_rate50_20251126_192806 |

> **Note:** Track C uses pruning with `rgtp_rate=0.5` on top of Track B checkpoint.

---

## 🔧 Track B Checkpoints — Training Config

| Checkpoint | TTC Mode | Loss Weights (next, noun, verb, ttc) | Notes |
|:-----------|:---------|:-------------------------------------|:------|
| `trackB_best_1122_0428.pt` | reg | 1.0, 1.0, 1.0, 1.0 | Baseline equal weights |
| `trackB_best_1122_1131.pt` | reg | 1.5, 0.25, 0.25, 1.0 | Emphasize next-active + TTC |
| `trackB_best_1122_1735.pt` | bin (classification) | 1.0, 1.0, 1.0, 1.0 | TTC as classification bins |

---

## 📋 Track A Summaries

| Run | Config Highlights | Notes |
|:----|:------------------|:------|
| KSweep_20251115_225146 | K∈{4,6,8,10,12,15}; IoU=0.5; last_frame_only | Recall sweep for K selection |
| stageA_20251117_175534 | mode=yolo; K=6; imgsz=960; conf=0.05 | 2323 images; avg 2.21 boxes/img |
| stageB_20251117_184342 | eval_with_labels=True; IoU=0.5 | 5129 crops; head manifests generated |

---

## 📈 Full Metrics Reference (Top-5 mAP %)

### Track B — Top-5 mAP by Checkpoint

| Checkpoint | N | N+V | N+δ | All |
|:-----------|:---:|:---:|:---:|:---:|
| 1122_0428 | 10.94 | 2.83 | 9.20 | 2.44 |
| 1122_1131 | 13.69 | 3.46 | 11.81 | 3.05 |
| 1122_1735 | **17.43** | **8.48** | **14.92** | **8.66** |

### Track C — Top-5 mAP by Run (Pruning rate=0.5)

| Run | N | N+V | N+δ | All |
|:----|:---:|:---:|:---:|:---:|
| 1123_020030 | 5.85 | 3.50 | 5.19 | 3.50 |
| 1123_024455 | 8.10 | 2.16 | 7.03 | 2.16 |
| 1126_151858 | **11.06** | **2.99** | **9.95** | **2.98** |
| 1126_192806 | 11.06 | 2.99 | 9.95 | 2.98 |

---

## 📁 Artifact Locations

| Track | Metrics | Checkpoints | Plots |
|:------|:--------|:------------|:------|
| A | `runs/Track_A/*/summary.json` | — | — |
| B | `runs/Track_B/metrics/metrics_val_*.json` | `runs/Track_B/checkpoints/trackB_best_*.pt` | `runs/Track_B/plots/` |
| C | `runs/Track_C/metrics/trackC_val_rate*.json` | (uses Track B checkpoint) | `runs/Track_C/plots/` |

---

## Notes

- All metrics are **Top-5 mAP (%)** on **Ego4D-STA v2 validation split**.
- **Track B** metrics reflect eval-time config only; training hyperparams documented in checkpoint table above.
- **Track C** applies RGTP token pruning at inference; all runs use `rgtp_rate=0.5` on `trackB_best_1122_1131.pt`.
- **Best Track B checkpoint** for top-5 semantic metrics: `trackB_best_1122_1735.pt` (bin TTC mode).
- Metric columns: **N** = Noun, **N+V** = Noun+Verb, **N+δ** = Noun+TTC, **All** = Noun+Verb+TTC.
