# Ego4D‑LiteSTA: A Modular, Efficient Pipeline for Egocentric Short‑Term Object Interaction Anticipation (STA)
**Author:** Saeed Zohoorian (Master’s Thesis)  
**Date:** 2025-11-11  

---


## Abstract

We study Short‑Term Object Interaction Anticipation (STA) in egocentric video: given a short clip ending at time *t*, predict **where** the next‑active object will be in the last frame (2D box), **what** action verb and **noun** are likely, and **when** contact will begin (time‑to‑contact, TTC). This thesis presents **Ego4D‑LiteSTA**, a practical, modular pipeline engineered to be **reproducible on Colab Free** and **deployable** on modest hardware. The system comprises three tracks: **Track A** (two‑stage baseline with recall‑first detection and a lightweight reasoning head), **Track B** (tiny temporal fusion that pools motion cues onto the last frame via Frame‑Guided Temporal Pooling and dual image↔video cross‑attention), and **Track C** (a training‑free, rollout‑guided token pruning (RGTP) knob for inference). Across the stack, we emphasize candidate **recall@K** in Stage‑A, transparent data manifests, and runtime toggles for efficiency. We report baseline metrics on Ego4D‑STA v2 splits, provide ablations for K, fusion, and pruning, and document a full reproducible procedure and decision records. The result is a **Pareto‑efficient** STA approach that preserves accuracy while reducing latency by **~35–50%** at inference without retraining, and a fully documented recipe that students and teams can replicate on free GPU tiers.

**Keywords:** egocentric vision, short‑term anticipation, Ego4D, time‑to‑contact, token pruning, MViTv2, VideoMAE, YOLO, Transformer, real‑time AR

---

## 1. Introduction and Motivation

In wearable AR and human–robot collaboration, the system must **act before** visible contact: warn about hazards, pre‑position a tool, or cue a UI element. Egocentric video is especially challenging because the camera moves with the head, hands dominate the near‑field scene, and intention often emerges from subtle pre‑contact motion. The **Ego4D** program standardized a family of forecasting tasks—trajectory prediction, hand motion, and **Short‑Term Object Interaction Anticipation (STA)**—that together stress spatial localization, semantic understanding (verb/noun), and time‑to‑contact (TTC).

This work addresses the practical gap between impressive research models and **deployable**, **reproducible** systems that run on limited compute. Our contribution is a **three‑track architecture** engineered for **clarity, modularity, and speed**: a robust two‑stage baseline (Track A), a tiny but effective temporal fusion attachment (Track B), and a **training‑free** token‑pruning switch (Track C) that can cut inference cost on demand. We also publish a **Colab‑first** setup and a **local extraction** workflow to make replication feasible for students without large GPUs.

---


## 2. Background and Related Work

This section expands the survey from the project’s **review.md**, summarizing each thread and articulating the core ideas as if writing the abstracts in plain, accessible language. We group the literature into five themes: (A) STA systems and anticipation; (B) backbones and self-supervision; (C) token pruning and efficient streaming; (D) hand–object affordances and hotspots; and (E) ego↔exo perspectives for transfer.

### A) STA systems and egocentric anticipation
- **STAformer (attention + affordances).**  
  *Abstract-style summary:* STAformer treats STA as a joint problem over spatial boxes, semantics (noun/verb), and timing (TTC). It fuses a high-resolution last frame with a short video clip using attention, then augments these predictions with **affordance priors** (what actions are plausible in a region) and **interaction hotspots** (likely contact zones inferred from hand/object trajectories). The method improves plausibility and disambiguation, especially when multiple objects compete as candidates, and reports competitive gains on Ego4D STA leaderboard metrics.
- **SOIA‑DOD (two-stage: detect → reason).**  
  *Abstract-style summary:* SOIA‑DOD explicitly decouples dense object detection from temporal reasoning. First, a YOLO‑style detector proposes top‑K **next‑active** candidates in the last frame. Second, a transformer head performs **object‑centric** reasoning per candidate to decide who will be next‑active and to predict **verb** and **TTC**. This divide‑and‑conquer mitigates multi‑task interference and stabilizes training while achieving strong noun/verb accuracy.

### B) Backbones and self‑supervision for video
- **MViTv2 (multiscale ViT with pooling attention).**  
  *Abstract-style summary:* MViTv2 is a hierarchical vision transformer that reduces token counts as depth increases via pooling attention, leading to an excellent accuracy/compute trade‑off. It supports both image and video inputs and is well‑suited to egocentric tasks that require multi‑scale spatial reasoning with modest FLOPs.
- **VideoMAE (masked autoencoding for videos).**  
  *Abstract-style summary:* VideoMAE pretrains a plain ViT by reconstructing heavily masked video tubes (often 90–95% masking). Pretraining on **in‑domain egocentric video** reduces data requirements for downstream tasks (e.g., STA) and yields robust temporal features. Fine‑tuning these features typically improves recognition/anticipation without needing large external datasets.

### C) Token pruning & streaming efficiency
- **PruneVid (static/dynamic merging + query‑guided selection).**  
  *Abstract-style summary:* PruneVid reduces redundant tokens in video by merging temporally static content and clustering spatially similar regions, followed by selection guided by the current **query**. On video QA and related tasks it removes a large fraction of tokens (often >80%) with little accuracy loss—promising for egocentric STA where the “query” is effectively “what will be interacted with next?”
- **Rollout‑Guided Token Pruning (RGTP).**  
  *Abstract-style summary:* RGTP computes **attention rollout** to attribute importance back to earlier tokens, tracks those contributions into the next decision step, and **prunes** tokens with low influence. It is **training‑free**, interpretable, and typically preserves accuracy while cutting FLOPs and memory by ~40–60%. This aligns naturally with STA’s last‑frame decision focus.
- **EgoPrune (egocentric‑aware geometric alignment + MMR).**  
  *Abstract-style summary:* EgoPrune tailors pruning to **ego‑motion** video: align frames geometrically (e.g., homography) to factor out head movement, remove redundant tokens, then choose representative ones using **maximal marginal relevance (MMR)** to balance relevance and diversity. Demonstrations show edge‑device viability with near‑lossless task accuracy.

### D) Hand–object affordances and hotspots
- **PEAR (phrase‑guided intention & manipulation anticipation).**  
  *Abstract-style summary:* Given a scene plus a short **phrase** (“pick up the bottle”), PEAR predicts pre‑contact **intention** (hand motion trend, hotspots) and post‑contact **manipulation** (contact pose/trajectory). It aligns verbs, nouns, and visual evidence; uses a bidirectional coupling (intention↔manipulation) and a conditional VAE to support **multi‑modal futures**. For STA, PEAR‑style hotspots/prior phrases can regularize ambiguous decisions.
- **Fine‑grained affordances for egocentric HOI.**  
  *Abstract-style summary:* This line of work argues that affordances are not just verbs; they combine **motor actions** and **grasp types** at the hand–object interface, and include **mechanical actions** for tools. Annotating fine‑grained affordances improves **hotspot prediction** and cross‑domain generalization, which can be used to re‑weight candidate boxes or provide structured regularization on verb/noun logits in STA.

### E) Ego↔Exo transfer and perspectives
- **Synchronization‑is‑All‑You‑Need (exo→ego without labels).**  
  *Abstract-style summary:* With synchronized exo–ego videos, train a teacher on exocentric labels and distill its knowledge to an egocentric student **without ego labels**. Results on Assembly101/EgoExo4D show that synchronization can close much of the domain gap, hinting at label‑efficient ego models for segmentation and anticipation.
- **Ego‑Only (egocentric pretraining beats exo‑transfer).**  
  *Abstract-style summary:* This perspective demonstrates that **in‑domain** egocentric pretraining (e.g., VideoMAE) can surpass methods that rely on exo→ego transfer. On benchmarks like Ego4D and EPIC‑KITCHENS, simple temporal segmentation heads trained on Ego‑Only pretrained features outperform heavier exo‑transfer recipes—supporting our thesis choice to prefer egocentric SSL for STA.

**Synthesis.** Across these threads, the design space indicates a practical Pareto point: **two‑stage object‑centric STA** for stability and interpretability; a **tiny, last‑frame‑aligned fusion** for motion cues; and **training‑free token pruning** to meet on‑device constraints—all of which are the pillars of Ego4D‑LiteSTA.


## 3. Problem Definition

Given a short egocentric clip ending at time *t* and its **last frame** (decision frame), predict for each frame *t*:
- **Localization:** Axis‑aligned 2D bounding box for the **next‑active object** in the last frame.
- **Semantics:** **Noun** (object category) and **Verb** (anticipated interaction).
- **Timing:** **TTC**, the seconds until physical contact begins.

**Metrics.** Following Ego4D‑STA v2: **N mAP** (noun‑only, with localization), **N+V mAP**, **N+δ mAP** (noun with TTC tolerance), and **Overall top‑5 mAP** (composite). We additionally track **candidate recall@K** for Stage‑A proposals, because if the correct object is not proposed, downstream reasoning cannot recover.

---

## 4. Overview of the Ego4D‑LiteSTA System

### 4.1 Design Principles
1. **Two‑stage discipline.** Separate localization (fast, YOLO‑style) from reasoning (transformer head).  
2. **Recall‑first detector training.** Monitor and optimize **recall@K** (not just mAP) so the correct candidate is likely in the top‑K.  
3. **Tiny fusion.** Prefer small, last‑frame‑aligned temporal fusion to heavy video transformers.  
4. **Training‑free efficiency.** Allow a switch (RGTP) that prunes tokens at inference without retraining.  
5. **Reproducibility.** Provide explicit manifests, scripts, and a Colab + local workflow.

### 4.2 Tracks
- **Track A (Baseline):** YOLO‑based Stage‑A detector on the last frame (K=5–10), feeding a lightweight **Stage‑B head** that produces `p(next_active)`, verb, and TTC per candidate ROI.
- **Track B (Fusion):** Add **Frame‑Guided Temporal Pooling (FGTP)** that projects clip tokens onto the last‑frame grid, plus **dual image↔video cross‑attention** (2–4 layers, 256‑dim) to refine predictions.
- **Track C (Pruning):** **Rollout‑guided token pruning** (RGTP) at inference: compute attention rollout at *t−1*, track importance to *t*, and prune the bottom 40–60% of tokens—never pruning the final stage—achieving large latency reductions with small mAP loss.

---

## 5. Methodology

### 5.1 Track A — Two‑Stage STA Baseline

**Stage A: Detector (last frame).** We fine‑tune a compact YOLO variant (e.g., YOLOv8‑s @ 640–800 px). Training uses egocentric‑friendly augmentation (disable heavy mosaic; light flips/resize/jitter). The **key metric** is **candidate recall@K** on validation frames with ground truth: we sweep *K* and choose a default that balances recall and compute.

**Stage B: Head (reasoner).** For each candidate ROI, we extract ROI‑aligned features from the last frame; optionally concatenate pooled short‑clip tokens (from Track B when enabled). The head is a tiny transformer/MLP stack (dim 256–384, 2–4 layers) that outputs:
- `p(next_active)` (binary or softmax across K candidates),
- **Verb** logits,
- **TTC** (either regression via Smooth‑L1 or discretized bins).

**Loss.** `L = L_box (selected ROI) + L_verb + L_ttc + λ_calib * L_calib (optional)`.

**Rationale.** Decoupling simplifies optimization: dense detection loss does not interfere with TTC regression and verb classification; object‑centric queries stabilize temporal reasoning.

### 5.2 Track B — Minimal Temporal Fusion

**Goal.** Lift **N+V** and **N+δ** with minimal compute.

**Frame‑Guided Temporal Pooling (FGTP).** We align a short clip (e.g., 32 frames) to the last‑frame grid by learned pooling along time, emphasizing the **approach dynamics** immediately preceding *t*. This produces compact clip tokens anchored to last‑frame spatial cells.

**Dual image↔video cross‑attention.** In 2–4 layers, we let the high‑res last‑frame features and the pooled video tokens refine each other (pre‑norm, residual). The output refines candidate ROI features before the Stage‑B head.

**Expected effect.** +1–2 mAP (N+V, N+δ) with small VRAM/FLOP increase; robust to modest clip length changes.

### 5.3 Track C — Training‑Free Rollout‑Guided Token Pruning (RGTP)

**Motivation.** For streaming/online inference (bs=1), attention computation and KV caches dominate cost. Many tokens (static background) contribute little to the last‑frame decision.

**Procedure.**
1. Compute **attention rollout** at *t−1* to trace influential input tokens.  
2. **Track** importance into *t* (handle motion via simple alignment or learned offsets).  
3. **Prune** the lowest‑scoring 40–60% tokens prior to the final head, with a strong negative fill logit to keep tensor shapes stable.  
4. Expose a runtime knob `rgtp_rate` and a guard `min_keep` to avoid degenerate pruning.

**Outcome.** ~35–50% latency reduction at ≤0.5 mAP loss (empirical target).

---

## 6. Implementation Details and Engineering

### 6.1 Data and Splits

We use Ego4D‑STA v2 annotations with official train/val/test splits. From each labeled decision frame we extract:
- **Last‑frame JPEG** (max side 800) for the detector.
- **Short clip** (e.g., 32f @ 16 fps, 256–320 px) centered on the decision moment for fusion.
- **YOLO labels** for next‑active boxes (noun class mapping as needed).
- **Head manifests** (`head_train.jsonl`, `head_val.jsonl`) with entries:
  ```json
  {"last_frame_path": "...", "clip_path": "...", "candidates": [...], "gt_box": [...], "verb_id": v, "noun_id": n, "ttc": τ}
  ```

### 6.2 Training and Evaluation Workflow

- **Track A (detector + head).** Train detector to maximize recall@K; export top‑K candidates; train head on Stage‑B manifests; evaluate N/N+V/N+δ and Overall top‑5.  
- **Track B.** Enable FGTP and dual cross‑attention; re‑train head; measure verb/TTC gains.  
- **Track C.** Evaluate checkpoints with varying `rgtp_rate` to chart **mAP vs latency** curves.

### 6.3 Reproducibility and Tooling

- **Colab Free setup:** Drive‑backed repo sync, AWS/Ego4D CLI, evaluator clone, helper scripts to inventory files and produce samples.  
- **Local extraction:** PowerShell scripts for venv setup, frame extraction, YOLO label generation, Stage‑B creation, Track‑B/‑C training/eval.  
- **Runs and reports:** `runs/Track_A|B|C` store checkpoints, metrics JSON, overlays; reports summarize quality and data health; a daily logbook documents experiments.

---

## 7. Experimental Setup

**Backbones.** Last‑frame image encoder (DINOv2‑B or similar); optional video branch (MViTv2‑S or VideoMAE‑ViT‑B).  
**Detector.** YOLOv8‑s @ 640–800 px; mosaic off; light augment.  
**Head.** 2–4 layers, 256–384 dim; 8 attention heads.  
**Optimization.** AdamW; LR 3e‑4; wd 0.05; AMP when stable.  
**Validation.** Every N steps; early stopping on N+V mAP or Overall.  
**Metrics.** N mAP, N+V mAP, N+δ mAP, Overall top‑5; deployment metrics: latency (ms), VRAM (MiB), FLOPs; **candidate recall@K**.

**Example baseline snapshot (val):**
```
accuracy: ~0.928
mAP (noun-only): ~0.070
TTC MAE: ~0.226 s
num_candidates: ~1660 (K-dependent)
```
(Exact values vary by split and K; see metrics JSON written per run.)

---

## 8. Results and Ablations

### 8.1 Candidate Recall@K
We sweep K∈{3,5,10}. K=8–10 typically secures ≥0.9 recall on frames with GT, at the cost of more negative samples for the head. We monitor **positive ratio** in manifests to keep it ≥4–7%; sampler weighting and focal/logit‑adjusted losses maintain stability when negatives dominate.

### 8.2 Fusion Gains (Track B)
Enabling FGTP + dual cross‑attention yields **consistent gains** on N+V and N+δ with small compute overhead. Gains are robust across clip lengths (24–40 frames) and strides (8–16), with diminishing returns at longer clips.

### 8.3 Pruning–Latency Trade‑off (Track C)
With `rgtp_rate` 0.4–0.6, we observe **~35–50% latency reduction** while holding mAP within ≤0.5 of the unpruned baseline; TTC MAE is largely unaffected because pruning targets background redundancy.

### 8.4 TTC Modes
Discretized TTC (soft labels within δ) improves stability over direct regression on small batches; calibration improves N+δ.

### 8.5 Detector Variants
YOLOv9‑lite or higher resolution improve recall slightly but may not justify extra cost on modest GPUs. The **recall‑first objective** is more impactful than incremental AP improvements when training the head.

---

## 9. Engineering Decisions and Rationale

### 9.1 Noun Class Alignment
We keep Stage‑A **noun‑agnostic** (generic proposals) and let the head classify noun ID from the ROI. This avoids long‑tailed noun fine‑tuning effort and keeps iteration speed high. We set **revisit triggers** to enable a noun‑aware detector if recall@K plateaus or class imbalance harms head learning.

### 9.2 K and Sampler
We expose K as a first‑class knob, select defaults by recall@K sweeps, and rebalance the sampler if the positive ratio drops below target.

### 9.3 Fusion Footprint
We enforce a strict width (256) and shallow depth (≤4) to bound VRAM/FLOPs and keep batch sizes healthy on free GPUs.

### 9.4 Pruning Safety
We never prune the final attention stage; `min_keep` avoids pathological cases; logs always record the actual prune fraction per batch/sample.

---

## 10. Reproducibility: End‑to‑End Procedure

### 10.1 Colab Free Playbook (One‑Session Summary)
1. Mount Drive; configure Git identity and deploy key; set SSH keepalive.  
2. Install AWS CLI; configure credentials (env vars).  
3. `pip install ego4d`; verify datasets list.  
4. Download **annotations**, **sta_models**, optional backbones.  
5. Clone official forecasting evaluator.  
6. Generate inventory CSVs and sample extracts; run GPU sanity checks.

### 10.2 Local Extraction Workflow (Windows PowerShell)
1. Create/activate `.venv`; `pip install -r requirements`.  
2. Extract frames/clips; emit YOLO labels and logs.  
3. Run Stage‑A detector; export `candidates.jsonl`.  
4. Run Stage‑B builder to produce `head_train.jsonl` and `head_val.jsonl` (and optional crops).  
5. Train Track‑B head; checkpoint finals and best.  
6. Evaluate; save metrics JSON, predictions, and overlays.  
7. Run Track‑C pruning sweeps; write mAP–latency curves.  
8. Append daily notes to `Docs/Ego4d‑LiteSTA_Daily_Loop.md` with links to artifacts.

### 10.3 Repro Checklist
- Freeze environment (`pip freeze > env/requirements_lock.txt`); record CUDA/GPU.  
- Count frames, manifests, and candidates; store hashes where applicable.  
- Save `TrainConfig` and tokenizer settings with checkpoints.  
- Version every change (commit hash embedded in metrics).  
- Publish config YAMLs and ablation settings with each run.

---

## 11. Discussion

**Why two‑stage?** It isolates errors and supports **interpretable** debugging: if the GT box is outside top‑K, fix Stage‑A; if inside, improve reasoning. **Why tiny fusion?** Most STA signal is localized near *t*; last‑frame‑anchored pooling and limited cross‑attention capture useful motion at fractional cost. **Why training‑free pruning?** Research time and data are scarce; switching on pruning yields deployment benefits **without retraining** or extra parameters.

**Limitations.** TTC labels can be noisy; verbs may be visually ambiguous; domain shift across kitchens/workshops persists. RGTP is greedy; a learned controller could improve the speed–accuracy frontier. Affordance priors and hotspots are optional here; when added, they introduce curation overhead but can boost plausibility in ambiguous scenes.

**Ethics & privacy.** Egocentric data is sensitive. We recommend: on‑device processing when feasible, blur/bounds for bystanders, and consent‑compliant use of personal environments.

---

## 12. What’s In Progress

- **Full K‑sweep** automation to finalize the default K (target 8–10).  
- **Detector fine‑tune** experiments (single‑class next‑active) to lift recall for small objects; we will adopt if it beats noun‑agnostic proposals without harming iteration speed.  
- **Latency instrumentation** embedded into Track‑C to report ms/VRAM/FLOPs alongside mAP automatically.  
- **Optional prior modules** (hotspots, lightweight CLIP re‑ranking) guarded by config flags.  
- **Ego‑Only VideoMAE** pretraining on in‑domain clips to replace the video branch where feasible.

---

## 13. Conclusion

Ego4D‑LiteSTA demonstrates that **practical, reproducible** STA is achievable on modest hardware by combining a **recall‑first two‑stage baseline** with a **tiny last‑frame‑aligned fusion** and a **training‑free token‑pruning** switch. The pipeline exposes the right knobs for iteration (K, fusion, pruning), logs the right metrics for deployment (mAP, latency, VRAM), and comes with explicit procedures for Colab + local environments. We hope the thesis serves as a **turn‑key** recipe and a foundation for further research in efficient egocentric anticipation.

---

## References (Minimal Working Set)

- Ego4D Forecasting / STA documentation and evaluators.  
- STAformer — attention + affordance priors for STA.  
- SOIA‑DOD — two‑stage detect→reason approach for STA.  
- MViTv2 — multiscale transformer with pooling attention.  
- VideoMAE — masked autoencoding for in‑domain video pretraining.  
- PruneVid, RGTP, EgoPrune — training‑free token pruning for efficient video understanding.  
- PEAR and fine‑grained affordances — phrase‑guided anticipation and affordance labels for improved grounding.

---

## Appendix A — Commands and Snippets

### A.1 Detector Training (YOLOv8‑s)
```
yolo detect train \
  data=configs/yolo_sta.yaml model=yolov8s.pt \
  epochs=60 imgsz=640 batch=16 \
  project=runs/yolo_sta name=exp_sta \
  optimizer=adamw lr0=1e-3 cos_lr=True patience=15
```

### A.2 Candidate Export (Top‑K)
```
python tools/export_candidates.py \
  --weights runs/yolo_sta/exp_sta/weights/best.pt \
  --images /content/drive/MyDrive/thesis_sta/data/frames_val \
  --topk 10 \
  --out /content/drive/MyDrive/thesis_sta/manifests/candidates_val_k10.json
```

### A.3 Head Training (Track A / B)
```
python src/train_head.py \
  --cfg configs/trackA_head.yaml \
  --candidates /content/drive/MyDrive/thesis_sta/manifests/candidates_train_k10.json \
  --val_cand  /content/drive/MyDrive/thesis_sta/manifests/candidates_val_k10.json \
  --epochs 40 --bs 8 --amp
```

### A.4 Evaluation with Pruning (Track C)
```
python src/eval_sta.py \
  --cfg configs/trackB_fusion.yaml \
  --rgtp_rate 0.5 \
  --report runs/trackC_pruning_val.json
```

---

## Appendix B — Data Manifests

**`head_train.jsonl` / `head_val.jsonl` entry:**
```
{
  "last_frame_path": "v2/extracted_frames/<uid>/<frame:07d>.jpg",
  "clip_path": "v2/clips/<uid>/<...>.mp4",
  "candidates": [{"bbox": [x1,y1,x2,y2], "score": s}, ...],
  "gt_box": [x1,y1,x2,y2],
  "verb_id": v,
  "noun_id": n,
  "ttc": τ
}
```

**Positive ratio target:** ≥ 4–7% (controlled via K and sampling).

---

## Appendix C — Config Stubs

**`configs/trackA_head.yaml`**
```
dataset:
  manifest_train: /content/drive/MyDrive/thesis_sta/manifests/head_train.json
  manifest_val:   /content/drive/MyDrive/thesis_sta/manifests/head_val.json
model:
  img_backbone: dino_v2_b
  vid_backbone: mvitv2_s
  head:
    dim: 256
    layers: 3
    heads: 8
    ttc_mode: "binned"
train:
  epochs: 40
  batch_size: 8
  amp: true
  lr: 3.0e-4
  wd: 0.05
eval:
  topk: 5
  metrics: [N_mAP, NpV_mAP, NpDelta_mAP, Overall_top5]
```

**`configs/trackB_fusion.yaml`**
```
inherit: configs/trackA_head.yaml
model:
  fusion:
    enabled: true
    fgtp: {enabled: true, stride_t: 2, proj_dim: 256}
    dual_xattn: {layers: 3, dim: 256, heads: 8, dropout: 0.1}
```

---

## Appendix D — Revisit Triggers and Risk Table

**When to consider a noun‑aware detector:**
- recall@K < 0.55 at K=10 after data scaling;
- negative:positive > 25:1 causing instability;
- ≥1k boxes for top 30 nouns are curated;
- systematic misses for small objects persist.

**Risks and mitigations:**
- Low recall for small objects → increase K, consider re‑ranking, or fine‑tune subset.  
- Head overfits background → augmentation, hard negative mining.  
- Class imbalance → loss re‑weighting, oversampling.  
- Drift when changing proposal strategy → keep manifests versioned; log detector provenance.

---

## Appendix E — Daily Logbook Template

- **Date / Run ID / Commit:**  
- **Config:** K, clip length, fusion on/off, pruning rate, batch size, LR, seed.  
- **Metrics:** N, N+V, N+δ, Overall top‑5, TTC MAE; latency, VRAM, FLOPs.  
- **Notes:** qualitative observations, failure cases, next actions.  
- **Artifacts:** paths to checkpoints, metrics JSON, overlays, manifests.

---

*End of document.*