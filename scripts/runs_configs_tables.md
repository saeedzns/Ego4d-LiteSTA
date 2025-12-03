# Runs Configs & Results

This document presents evaluation results and configuration details for Track A, B, and C experiments.

---

## 📊 Key Metrics Summary (Top-5 mAP)

### Track B — Fusion Head (Unpruned)

| Checkpoint | N | N+V | N+δ | All | Timestamp |
|:-----------|:---:|:---:|:---:|:---:|:----------|
| `trackB_best_1122_0428.pt` | **0.1094** | 0.0283 | 0.0920 | 0.0244 | 2025-11-22 20:09 |
| `trackB_best_1122_1131.pt` | 0.1369 | 0.0346 | 0.1181 | 0.0305 | 2025-11-22 20:21 |
| `trackB_best_1122_1735.pt` | **0.1743** | **0.0848** | **0.1492** | **0.0866** | 2025-11-22 20:30 |

> **Best overall:** `trackB_best_1122_1735.pt` (bin TTC, equal loss weights)

### Track C — RGTP Pruning (rate=0.5)

| Checkpoint | N | N+V | N+δ | All | File |
|:-----------|:---:|:---:|:---:|:---:|:-----|
| `trackB_best_1122_1131.pt` | 0.0585 | 0.0350 | 0.0519 | 0.0350 | trackC_val_rate50_20251123_020030 |
| `trackB_best_1122_1131.pt` | 0.0810 | 0.0216 | 0.0703 | 0.0216 | trackC_val_rate50_20251123_024455 |
| `trackB_best_1122_1131.pt` | **0.1106** | **0.0299** | **0.0995** | **0.0298** | trackC_val_rate50_20251126_151858 |
| `trackB_best_1122_1131.pt` | 0.1106 | 0.0299 | 0.0995 | 0.0298 | trackC_val_rate50_20251126_192806 |

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

## 📈 Full Metrics Reference (Top-5 mAP)

### Track B — Top-5 mAP by Checkpoint

| Checkpoint | N | N+V | N+δ | All |
|:-----------|:---:|:---:|:---:|:---:|
| 1122_0428 | 0.1094 | 0.0283 | 0.0920 | 0.0244 |
| 1122_1131 | 0.1369 | 0.0346 | 0.1181 | 0.0305 |
| 1122_1735 | **0.1743** | **0.0848** | **0.1492** | **0.0866** |

### Track C — Top-5 mAP by Run (Pruning rate=0.5)

| Run | N | N+V | N+δ | All |
|:----|:---:|:---:|:---:|:---:|
| 1123_020030 | 0.0585 | 0.0350 | 0.0519 | 0.0350 |
| 1123_024455 | 0.0810 | 0.0216 | 0.0703 | 0.0216 |
| 1126_151858 | **0.1106** | **0.0299** | **0.0995** | **0.0298** |
| 1126_192806 | 0.1106 | 0.0299 | 0.0995 | 0.0298 |

---

## 📁 Artifact Locations

| Track | Metrics | Checkpoints | Plots |
|:------|:--------|:------------|:------|
| A | `runs/Track_A/*/summary.json` | — | — |
| B | `runs/Track_B/metrics/metrics_val_*.json` | `runs/Track_B/checkpoints/trackB_best_*.pt` | `runs/Track_B/plots/` |
| C | `runs/Track_C/metrics/trackC_val_rate*.json` | (uses Track B checkpoint) | `runs/Track_C/plots/` |

---

## Notes

- **Track B** metrics reflect eval-time config only; training hyperparams documented in checkpoint table above.
- **Track C** applies RGTP token pruning at inference; all runs use `rgtp_rate=0.5` on `trackB_best_1122_1131.pt`.
- **Best Track B checkpoint** for top-5 semantic metrics: `trackB_best_1122_1735.pt` (bin TTC mode).
- Metric columns: **N** = Noun, **N+V** = Noun+Verb, **N+δ** = Noun+TTC, **All** = Noun+Verb+TTC.
