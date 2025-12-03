
# Thesis Implementation Plan (v2): Egocentric STA in 8 Weeks — Tracks A / B / C

**Author:** Saeed  
**Generated:** 2025-11-01 20:27  
**Scope:** Short‑Term Object Interaction Anticipation (STA) on Ego4D with a **two‑stage baseline (Track A)**, a **lightweight fusion upgrade (Track B)**, and a **training‑free inference efficiency module (Track C)**. Everything is engineered to run reproducibly on **Google Colab Free**.

---

## 0) Executive Summary — What’s New (Novelty & Contributions)

### One‑line idea
We build a **practical, modular STA system** that anyone can reproduce on **Colab Free**, and we make it **significantly faster at inference without retraining** by inserting a **training‑free token‑pruning step**. We keep accuracy competitive by **decoupling detection from reasoning** and adding a **tiny temporal fusion** that focuses motion onto the last frame.

### The 3 novelty pillars
1. **Object‑centric two‑stage STA with recall‑first training.**  
   We explicitly optimize and monitor **candidate recall@K** in Stage‑A (detector) so Stage‑B (reasoner) always sees the box it needs. This makes training stable on egocentric video (hand jitter, head motion) and mirrors deployable AR pipelines (propose → reason → act).

2. **Minimal temporal fusion that piggybacks on the last frame.**  
   Instead of a heavy video transformer, we align a short clip onto the decision frame using **Frame‑Guided Temporal Pooling (FGTP)** and a **tiny dual image↔video cross‑attention** (2–4 layers, width 256). This reliably lifts **N+V** and **N+δ** with a tiny compute footprint and preserves the two‑stage discipline.

3. **Training‑free, rollout‑guided token pruning specialized to STA timing.**  
   At inference, we use **attention rollout at t−1**, track token importance to t, and prune 40–60% of low‑value tokens **without any retraining**. We target **~35–50% latency reductions** with ≤0.5 mAP loss. This converts the same trained model into a “near real‑time” one by flipping a switch.

### Additional contributions that strengthen the package
- **Colab‑first engineering:** small backbones, AMP, resumable runs, Drive persistence—so results are repeatable by students/teams without big GPUs.
- **Exact data manifests & evaluators:** a clear, scriptable path from Ego4D STA to {frames, clips, YOLO labels, head manifests}, removing ambiguity that often blocks replication.
- **Deployment‑oriented evaluation:** beyond mAP, we chart **latency, VRAM, FLOPs, and mAP vs pruning‑rate curves** to expose the speed/quality frontier.

---

## 1) Problem Addressing & Motivation

**STA task.** Given a short egocentric clip ending at time t, anticipate **where** the next‑active object will be in the last frame (bbox), **what** verb (and noun) the wearer is likely to perform, and **when** (time‑to‑contact, TTC).

**Why it’s hard.**
- Subtle and short‑range motions with egomotion noise (head‑mounted camera).
- Multi‑task interference among {localization, verb, TTC} if trained end‑to‑end.
- Real‑time constraints matter for AR/robotics; many accurate models are too heavy.

**Project goal (2 months).**
1) Deliver a strong, stable two‑stage STA baseline.  
2) Add **tiny fusion** for verbs/TTC gains at small compute.  
3) Achieve **meaningful latency savings with no retraining** via token pruning.  
4) Ensure the entire stack is **reproducible on Colab Free**.

---

## 2) Solution Overview (Tracks A / B / C)

### Track A — Two‑Stage STA (Baseline)
- **Stage A: Detector (last frame)** → top‑K candidate boxes (K=5–10).  
  *Model:* YOLOv8‑s (or YOLOv9‑lite) at 640–800 px.  
  *Objective:* Maximize **candidate recall@K**, not only mAP.
- **Stage B: Head (reasoner)** → per candidate: `p(next_active)`, `verb`, `ttc` (binned or reg).  
  *Model:* lightweight transformer or CNN‑MLP over ROI features (+optional video tokens).  
  *Loss:* GIoU+L1 on selected box + CE(verb) + SmoothL1/CE(TTC).

**Why two‑stage?** It stabilizes training on egocentric data, isolates localization errors, and yields interpretable failure modes (proposals vs reasoning).

---

### Track B — Lightweight Fusion (STAformer‑Lite spirit)
We keep Track A intact and add two tiny modules:
1. **Frame‑Guided Temporal Pooling (FGTP).**  
   Projects short‑clip tokens onto the **last‑frame grid**, aligning motion with the decision instant.
2. **Dual Image↔Video Cross‑Attention (2–4 layers @ 256‑dim).**  
   Lets appearance sharpen motion and vice versa. Works as a drop‑in “on/off” switch.

**Outcome:** +1–2 mAP on **N+V** and **N+δ** with small FLOPs/VRAM increase.

---

### Track C — Inference‑Time Token Pruning (Training‑Free)
1. Compute **attention rollout** at t−1 to estimate token importance.  
2. **Track** importance to t (account for motion).  
3. **Prune** 40–60% of least important tokens before the head (never prune last layer).

**Outcome:** ~**35–50% lower latency** and reduced memory, while keeping mAP within ≤0.5 of the full model—no retraining required.

---

## 3) How This Compares to Prior Work (Clarity, Not Just Claims)

| Aspect | This Work | STAformer‑style (ZARRIO) | SOIA‑DOD | Heavy Video Transformers | Training‑Free Pruning (generic VLM) |
|---|---|---|---|---|---|
| Pipeline | **Two‑stage** (detector → head), recall‑first | Single‑stage attention stack w/ affordances/hotspots | **Two‑stage** (YOLO then transformer) | Mostly heavy, end‑to‑end | Usually after a big encoder |
| Temporal modeling | **Tiny FGTP + dual CA**, last‑frame aligned | Larger attention & affordance memory | Minimal temporal fusion | Heavy temporal attention | Often generic, not STA‑aware |
| Efficiency | **Training‑free pruning** tailored to STA timing | N/A (accuracy‑first) | N/A (accuracy‑first) | Slow, high VRAM | Generic pruning (not last‑frame‑aware) |
| Reproducibility | **Colab‑first**, exact manifests, recall@K | Complex infra (affordance db, hotspots) | Two‑stage but heavier training | Large compute / complex recipes | Adds on top of large models |
| Deployability | **Flip‑a‑switch speedup** w/o retrain | Extra modules to train/curate | Strong but train‑heavy | Hard on edge devices | Gains not specialized to STA |

**Takeaway:** We occupy a **Pareto point**: solid accuracy from two‑stage + tiny fusion, and **unique deployability** via **training‑free**, STA‑aware pruning.

---

## 4) Colab Free Strategy (Reproducibility)

- **Persist to Drive:** `runs/`, `checkpoints/`, `logs/`, `results/*.json` saved every N steps.
- **Small backbones:** YOLOv8‑s; MViTv2‑S or ViT‑B for clips (optional in Track B).
- **AMP + grad accumulation** when needed; resumeable training.
- **Pre‑extract** last frames/clips once and reuse.

**Boilerplate:**
```python
from google.colab import drive; drive.mount('/content/drive')
PROJ='/content/drive/MyDrive/thesis_sta'; import os; os.makedirs(PROJ, exist_ok=True)
%cd {PROJ}
!pip install -U ultralytics==8.3.0 timm==1.0.11 einops yacs tqdm pyyaml opencv-python pycocotools matplotlib
```

---

## 5) Data Requirements & Conversion

- **Annotations:** Ego4D STA (v2) with splits and TTC.
- **Artifacts we create:**
  - **Last frames** (JPEG, max side 800) for detector.
  - **Short clips** (e.g., 32f @ 16 fps, 256–320 px) for optional video branch.
  - **YOLO labels** for next‑active bboxes (noun class).
  - **Head manifests**: JSON entries per sample:  
    `{last_frame_path, clip_path?, candidates?, gt_box, verb_id, noun_id, ttc}`

**Extraction script outline:**
```bash
python tools/prepare_sta_data.py \
  --ann /content/drive/MyDrive/ego4d/sta_v2.json \
  --video_root /content/drive/MyDrive/ego4d/videos \
  --out_root /content/drive/MyDrive/thesis_sta/data \
  --clip_len 32 --fps 16 --img_max 800 --clip_size 320 --splits train,val,test
```

---

## 6) Track A: Training & Inference Details

### Detector (Stage A)
- **Model:** YOLOv8‑s (img=640 or 800), **mosaic off** (egocentric geometry), light flips/resize/jitter.
- **Key metric:** **candidate recall@K** ≥ 95% on val.

**Train command (example):**
```bash
yolo detect train \
  data=configs/yolo_sta.yaml model=yolov8s.pt \
  epochs=60 imgsz=640 batch=16 \
  project=runs/yolo_sta name=exp_sta \
  optimizer=adamw lr0=1e-3 cos_lr=True patience=15
```

**Export top‑K candidates for Stage B:**
```bash
python tools/export_candidates.py \
  --weights runs/yolo_sta/exp_sta/weights/best.pt \
  --images /content/drive/MyDrive/thesis_sta/data/frames_val \
  --topk 10 \
  --out /content/drive/MyDrive/thesis_sta/manifests/candidates_val_k10.json
```

### Head (Stage B)
- **Inputs:** ROI features for top‑K boxes (+ optional pooled video tokens).  
- **Head:** 2–4 layers, dim 256–384, 8 heads; outputs `p(next_active)`, `verb`, `ttc`.
- **Loss:** `L = L_box + L_verb + L_ttc (+ L_calib optional)`

**Train command:**
```bash
python src/train_head.py \
  --cfg configs/trackA_head.yaml \
  --candidates /content/drive/MyDrive/thesis_sta/manifests/candidates_train_k10.json \
  --val_cand  /content/drive/MyDrive/thesis_sta/manifests/candidates_val_k10.json \
  --epochs 40 --bs 8 --amp
```

---

## 7) Track B: Fusion (FGTP + Dual Cross‑Attention)

- **FGTP:** Learned pooling from short‑clip tokens to last‑frame grid (stride_t=2).  
- **Dual CA:** (Image→Video) + (Video→Image) with pre‑norm & residuals.

**Config switch (example):**
```yaml
model:
  fusion:
    enabled: true
    fgtp: {enabled: true, stride_t: 2, proj_dim: 256}
    dual_xattn: {layers: 3, dim: 256, heads: 8, dropout: 0.1}
```

**Expected:** +1–2 mAP on N+V, N+δ at modest compute increase.

---

## 8) Track C: Training‑Free Pruning (RGTP‑style)

- **Rollout at t−1 → importance scores**; **track** to t; **prune** bottom 40–60%.  
- **Never prune** the final stage; expose a **runtime knob** (`--rgtp_rate`).

**Eval command:**
```bash
python src/eval_sta.py \
  --cfg configs/trackB_fusion.yaml \
  --rgtp_rate 0.5 \
  --report runs/trackC_pruning_val.json
```

**Goal:** retain mAP within ≤0.5 of full model while cutting latency by ~35–50%.

---

## 9) Metrics & Logging

- **Primary:** N mAP, N+V mAP, N+δ mAP, Overall top‑5.  
- **Deployment:** latency (ms, bs=1), VRAM (MiB), FLOPs; **candidate recall@K**.  
- **Ablations:** K∈{3,5,10}, clip length, TTC (bins vs reg), fusion on/off, **pruning rate**∈{0.4,0.5,0.6}.  
- **Repro:** seeds, git commit, config YAML dump, artifact hashes; save JSON to Drive.

---

## 10) Week‑by‑Week Plan (Colab‑feasible)

- **W1:** Data extract + manifests + eval scripts. Dry‑run Track A on a small subset.  
- **W2:** Detector @640 px; export top‑K; verify recall@K≥95%.  
- **W3:** Train Head (Track A). Get first N / N+V / N+δ / Overall top‑5.  
- **W4:** Add Fusion (Track B). Re‑eval; plot verb/TTC gains & attention maps.  
- **W5:** Integrate RGTP (Track C). Measure latency vs mAP curves.  
- **W6:** Optional priors (hotspots/affordances) as lightweight re‑weighting.  
- **W7:** Full ablations; TTC calibration; robustness across scenes/users.  
- **W8:** Final tables/plots, README, seeds/config bundles; write thesis chapter.

---

## 11) Risks & Mitigations

- **Colab disconnects:** frequent Drive checkpoints, `--resume`.  
- **OOM:** lower batch, 256‑dim tokens, 320‑px clips, AMP + grad accumulation.  
- **I/O bottlenecks:** cache frames/clips; pack to NPZ shards.  
- **Training instabilities:** warmups, freeze early layers, binned TTC + label smoothing.

---

## 12) Repository Layout (Suggested)

```
repo/
  configs/
    yolo_sta.yaml
    trackA_head.yaml
    trackB_fusion.yaml
  tools/
    prepare_sta_data.py
    export_candidates.py
  src/
    datasets/ego4d_sta.py
    detector/train_yolo.py
    models/{backbones.py, head.py, fusion.py, pruning_rgtp.py}
    train_head.py
    eval_sta.py
  runs/  -> Drive symlink
  logs/  -> Drive symlink
  README.md
```

---

## 13) Config Stubs

**`configs/trackA_head.yaml`**
```yaml
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
    ttc_mode: "binned"   # or "reg"
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
```yaml
inherit: configs/trackA_head.yaml
model:
  fusion:
    enabled: true
    fgtp: {enabled: true, stride_t: 2, proj_dim: 256}
    dual_xattn: {layers: 3, dim: 256, heads: 8, dropout: 0.1}
```

---

## 14) Deliverables

- **Checkpoints:** detector best.pt; head (trackA.pt, trackB.pt).  
- **Results:** `val_metrics.json` (N/N+V/N+δ/Overall top‑5), `latency.json` (with/without RGTP).  
- **Figures:** pipeline diagram, attention visualization, **mAP vs latency** curves.  
- **Thesis text:** novelty & comparisons (this doc), methodology, ablations, limitations.

---

## 15) Limitations & Future Work

- The pruning policy is **greedy & training‑free**; learning a tiny controller may unlock more savings.  
- FGTP/dual‑CA is intentionally small; richer temporal cues (audio, hands) could add gains if compute allows.  
- Affordance memories & hotspots are optional add‑ons; we avoid them to keep training simple on Colab, but they can be plugged later as priors.

---

### TL;DR
- **Track A**: robust two‑stage baseline with **recall‑first** detector → stable training.  
- **Track B**: **tiny fusion** (FGTP + dual CA) → better verbs/TTC at small cost.  
- **Track C**: **training‑free pruning** → **35–50% latency cut** with ≤0.5 mAP drop.  
All of this is **Colab‑reproducible** end‑to‑end.
