# Short-Term Object Interaction Anticipation – Benchmark Results Summary

This document summarizes reported results from several works related to Ego4D Short-Term Object Interaction Anticipation (STA) and related benchmarks. For each paper, we include the main quantitative tables and a short explanation of what each table shows.

---

## 1. AFF-ttention! Affordances and Attention Models for Short-Term Object Interaction Anticipation

### 1.1 Ego4D-STA v2 – Validation Split

Results in Top-5 mAP (%) on the validation split of Ego4D-STA v2. Higher is better. **N** = noun-only, **N+V** = noun + verb, **N+δ** = noun + TTC-bin, **All** = noun + verb + TTC-bin.

| Model               | N     | N+V   | N+δ   | All   |
|---------------------|-------|-------|-------|-------|
| FRCNN+SF [17]       | 21.00 | 7.45  | 7.07  | 2.98  |
| InternVideo [4]     | 19.45 | 8.00  | 6.97  | 3.25  |
| StillFast [42]      | 20.26 | 10.37 | 7.26  | 3.96  |
| GANO v2 [50]        | 20.52 | 10.42 | 7.28  | 3.99  |
| STAformer           | 24.85 | 13.45 | 7.41  | 4.90  |
| STAformer + AFF     | 27.03 | 14.36 | 8.72  | 5.04  |
| STAformer + MH      | 27.51 | 14.68 | 9.63  | 5.50  |
| **STAformer + MH + AFF** | **29.39** | **15.38** | **9.94** | **5.67** |
| Gain (rel %) vs 2nd best | +43.3 | +47.6 | +36.5 | +42.1 |

**Explanation.**  
This table compares a range of baselines (FRCNN+SF, InternVideo, StillFast, GANO v2) against STAformer and its variants on Ego4D-STA v2 validation. Adding multi-head attention (MH) and the proposed affordance attention (AFF) on top of STAformer yields the best performance across all semantic metrics (N, N+V, N+δ, All), with a large relative gain over the second-best model.

---

### 1.2 EPIC-Kitchens – Validation Split

Results in Top-5 mAP (%) on the validation split of EPIC-Kitchens.

| Model           | N     | N+V   | N+δ  | All  |
|-----------------|-------|-------|------|------|
| StillFast [42]  | 21.24 | 12.41 | 6.22 | 3.28 |
| STAformer       | 24.16 | 15.55 | 7.08 | 4.31 |
| **STAformer + AFF** | **26.19** | **16.49** | **7.18** | **4.69** |  
| Gain (rel %) vs 2nd best | +23.3 | +32.8 | +15.4 | +42.9 |

**Explanation.**  
On EPIC-Kitchens, STAformer already improves over StillFast in all metrics. Incorporating affordance attention (AFF) further boosts performance, especially on joint metrics such as N+V and All. The relative gain row shows the percentage improvement of STAformer+AFF compared to the second-best model (plain STAformer).

---

## 2. Short-Term Object Interaction Anticipation with Disentangled Object Detection (SOIA-DOD)

### 2.1 Ablation: Number of Object Candidates

Top-5 mAP (%) on Ego4D validation for different numbers of object candidates per frame used as transformer queries.

| # Candidates | Noun (N) | N+V   | N+TTC | Overall |
|-------------|----------|-------|-------|---------|
| 5           | 30.14    | 14.54 | 9.219 | 4.91    |
| 10          | 30.65    | 15.22 | 9.222 | 4.98    |
| 20          | 30.94    | 14.88 | 8.86  | 4.87    |

**Explanation.**  
This ablation studies how many potential active object candidates are passed to the transformer. Increasing from 5 to 10 candidates slightly improves most metrics (especially N+V and Overall). Pushing to 20 candidates gives a marginal gain on noun-only mAP but degrades N+TTC and Overall, suggesting that too many noisy proposals can hurt anticipation quality.

---

### 2.2 Comparison with Baselines on Ego4D v2 Test

Top-5 mAP (%) on the Ego4D v2 **test** set.

| Method          | Noun (N) | N+V   | N+TTC | Overall |
|----------------|----------|-------|-------|---------|
| StillFast [6]  | 25.06    | 13.29 | 9.14  | 5.12    |
| GANOv2 [7]     | 25.67    | 13.60 | 9.02  | 5.16    |
| Zarrio*        | 33.50    | 17.26 | 11.77 | 6.75    |
| EgoVideo*      | 31.08    | 16.18 | 12.41 | 7.21    |
| **SOIA-DOD (ours)** | **34.89** | **17.61** | 10.91 | 6.22    |

**Explanation.**  
This table evaluates multiple anticipation methods on the official Ego4D test set. Methods marked with * (Zarrio, EgoVideo) use stronger video backbones. SOIA-DOD achieves the best noun and N+V scores and competitive N+TTC and Overall performance, showing that disentangling object detection from anticipation helps improve short-term interaction prediction.

---

## 3. Egocentric Object-Interaction Anticipation with Retentive and Predictive Learning (EgoIR / EgoIP)

Results in Top-5 mAP (%) on Ego4D-STA v2. Columns correspond to different label combinations: **b+n** (binary + noun), **b+v** (binary + verb), **b+n+t** (binary + noun + TTC), **b+n+v** (binary + noun + verb), **b+n+v+t** (binary + noun + verb + TTC).

| Model (mAP)        | b+n   | b+v   | b+n+t | b+n+v | b+n+v+t |
|--------------------|-------|-------|-------|-------|---------|
| **One-stage training** |       |       |       |       |         |
| StillFast          | 20.26 | –     | 7.16  | 10.37 | 3.96    |
| STAformer          | 24.85 | –     | 7.41  | 13.45 | 4.90    |
| StillFast + D EgoIR | 25.75 | 12.35 | 9.30  | 14.41 | 5.56    |
| StillFast + D EgoIP | **26.00** | 11.92 | **9.62** | **14.44** | **5.74** |
| **Two-stage training** |       |       |       |       |         |
| SlowFast           | 21.00 | –     | 7.04  | 7.45  | 2.98    |
| EgoAnticipator†    | 22.54 | 10.89 | 9.67  | 10.28 | 5.41    |
| EgoAnticipator     | 23.52 | 11.18 | 10.29 | 11.04 | 5.60    |

† uses teacher–student configuration with ϕ_M = ϕ_S.

**Explanation.**  
This table compares baseline models (StillFast, STAformer, SlowFast) with the proposed retentive/predictive variants EgoIR and EgoIP, as well as EgoAnticipator. Adding the retentive–predictive learning (D EgoIR / D EgoIP) on top of StillFast improves all metrics, especially those that combine binary and semantic supervision (b+n+t, b+n+v, b+n+v+t). Two-stage training with EgoAnticipator further refines performance compared to SlowFast.

---

## 4. StillFast: An End-to-End Approach for Short-Term Object Interaction Anticipation

Results in Top-5 mAP (%) on Ego4D v2 **validation** and **test** splits.

| Split | Method           | Ver | Noun (N) | N+V   | N+TTC | Overall |
|-------|------------------|-----|----------|-------|-------|---------|
| Val   | FRCNN+SF [17]    | v2  | 21.00    | 7.45  | 7.04  | 2.98    |
| Val   | StillFast (ours) | v2  | 20.26    | 10.37 | 7.16  | 3.96    |
| Test  | FRCNN+SF [17]    | v2  | 26.15    | 9.45  | 8.69  | 3.61    |
| Test  | StillFast (ours) | v2  | 25.06    | 13.29 | 9.14  | 5.12    |

**Explanation.**  
The table compares the original FRCNN+SF pipeline with the end-to-end StillFast model. On both validation and test splits, StillFast trades a tiny drop in noun-only accuracy for substantial gains on joint metrics (N+V, N+TTC, Overall), demonstrating the benefit of learning detection and anticipation jointly in a single network.

---

## 5. Guided Attention for Next Active Object (GANOv2) – Ego4D STA Challenge

### 5.1 Comparison with Baselines on Ego4D v2

Top-5 mAP (%) on Ego4D v2 validation and test splits.

| Method        | Data Split | Noun (N) | N+V   | N+TTC | Overall |
|--------------|------------|----------|-------|-------|---------|
| FRCNN+SF [14] | Val        | 21.00    | 7.45  | 7.04  | 2.98    |
| StillFast [2] | Val        | 20.26    | 10.37 | 7.16  | 3.96    |
| **GANOv2 (ours)** | Val    | 20.52    | 10.42 | 7.28  | 3.99    |
| FRCNN+SF [14] | Test       | 26.15    | 9.45  | 8.69  | 3.61    |
| StillFast [2] | Test       | 25.06    | 13.29 | 9.14  | 5.12    |
| **GANOv2 (ours)** | Test   | 25.67    | 13.60 | 9.02  | 5.16    |

**Explanation.**  
Here GANOv2 is evaluated against FRCNN+SF and StillFast. On the validation set it offers a small but consistent improvement over StillFast across all metrics. On the test set, GANOv2 improves Noun and N+V slightly compared to StillFast while maintaining comparable N+TTC and Overall, showing that guided attention can refine predictions without redesigning the whole architecture.

---

### 5.2 Guided Fusion Layer Ablation

Ablation over which 3D CNN output layer uses Guided Attention Fusion, measured on Ego4D v2 validation (Top-5 mAP %).

| Guided Fusion Layer(s) | Noun (N) | N+V   | N+TTC | Overall |
|------------------------|----------|-------|-------|---------|
| Layer 1                | 18.70    | 9.42  | 6.27  | 3.22    |
| Layer 4                | 20.47    | 10.40 | 7.20  | 3.96    |
| **All layers**         | **20.52**| **10.42** | **7.28** | **3.99** |

**Explanation.**  
This ablation shows how the choice of fusion layer affects performance. Using guided attention only at early layers (Layer 1) underperforms. Applying it at a deeper layer (Layer 4) helps, and applying guided fusion across **all** layers gives the best scores on all metrics, indicating that multi-layer guidance better captures both low-level and high-level cues for next-active object anticipation.

---

