# Thesis Implementation Plan: Egocentric STA in 8 Weeks (Tracks A/B/C)

**Author:** Saeed  
**Date:** (fill when exporting)  
**Scope:** Short‑Term Object Interaction Anticipation (STA) on Ego4D with a **two‑stage baseline (Track A)**, a **lightweight fusion upgrade (Track B)**, and an **inference‑time efficiency module (Track C)**, designed to run on **Google Colab Free** (T4/V100/A100 *as available*) with strict time and memory budgets.

---

## 1) Problem Addressing & Motivation

**Problem.** In egocentric AR/robotics, we must **predict the next object interaction before contact** to enable proactive assistance (warn, prepare tools, cue UI). Given a short video clip ending at the current frame, **STA** requires predicting:
- **Where**: 2D box(es) of the **next‑active object(s)** in the current frame.
- **What**: **Noun** (object) and **Verb** (interaction).
- **When**: **Time‑to‑Contact (TTC)** until the interaction starts.

**Challenges.**
- Hand/object motion is subtle and short‑range; background is dynamic due to head motion.
- Multi‑task interference (localization vs verb vs TTC) hurts training stability.
- Real‑time constraints (edge devices) require **low latency** and **small memory**.

**Goal.** Build a **robust, efficient STA pipeline** in 2 months that:
1. Achieves strong mAP on **N**, **N+V**, **N+δ**, **Overall top‑5**.
2. Demonstrates **novel efficiency** via **training‑free token pruning** without accuracy loss.
3. Is **fully reproducible** on **Colab Free** (including data prep, training, eval, and logging to Google Drive).

---

## 2) How We Will Solve It (Tracks A, B, C)

### Track A — Two‑Stage Object‑Centric STA (Baseline + Stability)
1. **Stage A: Detector** on the **last frame** to produce top‑K **candidate boxes** (next‑active object proposals).  
   - Model: **YOLOv8‑s (or YOLOv9‑lite)**, fine‑tuned on STA targets.
   - Output: top‑K boxes + scores + class logits (K=5–10).
2. **Stage B: Transformer Head** that reasons over candidates (+ optional short clip features) to predict:  
   **p(next‑active)**, **verb**, **TTC**, and select final box.
   - Losses: Box (GIoU+L1), CE(verb), TTC (binned CE or Smooth‑L1).

**Why:** Decoupling detection from temporal/semantic reasoning makes training **stable** and **interpretable**; detector controls recall; head focuses on future interaction.

---

### Track B — Lightweight Fusion (STAformer‑Lite)
**Add two small modules** before the head, keeping Track A intact:
1. **Frame‑Guided Temporal Pooling (FGTP).** Project short‑clip video tokens onto the last‑frame grid so the head sees motion cues aligned to the decision frame.
2. **Dual Image↔Video Cross‑Attention (tiny).** 2–4 transformer layers (width 256), to let image tokens refine video tokens and vice‑versa.

**Why:** Small compute, consistent gains on **N+V** and **N+δ** by better aligning appearance and motion around the interaction instant.

---

### Track C — Inference‑Time Efficiency (Rollout‑Guided Token Pruning, RGTP)
**Training‑free** pruning for the **video branch** at inference:
1. Compute **attention rollout** on frame *t−1* to trace **important input tokens**.
2. **Track** them to frame *t*; prune 40–60% least‑useful tokens.
3. Run the same head with fewer tokens → **lower FLOPs/latency**, ~no mAP loss.

**Why:** Delivers a **real‑time story** with measurable savings (latency, memory) without retraining.

---

## 3) Colab Free: Practical Execution Strategy

**Constraints.**
- GPU type/time varies; sessions may disconnect after inactivity (~90 min) or long runs (shorter for Free).
- Limited disk (≈100GB ephemeral), RAM (~12–16GB), and GPU VRAM (T4≈16GB).

**Key Tactics.**
- **Mount Google Drive** as persistent storage for datasets, checkpoints, logs.
- Use **small models** (YOLOv8‑s, MViTv2‑S / ViT‑B video encoder).
- Train in **short epochs**; **resume** via Drive checkpoints.
- **Mixed‑precision (AMP)** and **gradient accumulation** for memory.
- **Pre‑extract clips/frames** once to Drive, reuse across runs.
- **Save artifacts frequently** to Drive (every n epochs).

**Colab Cell Boilerplate:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Project root on Drive
import os, sys
PROJ = '/content/drive/MyDrive/thesis_sta'
os.makedirs(PROJ, exist_ok=True)
%cd {PROJ}

# Clone your repo (once)
!git clone https://github.com/saeedzns/Ego4d-LiteSTA.git repo || echo 'repo exists'
%cd repo

# Dependencies
!pip install -U pip
!pip install ultralytics==8.3.0 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install timm==1.0.11 einops yacs tqdm pyyaml opencv-python pycocotools matplotlib

# (Optional) wandb login
#import wandb; wandb.login()
```

**Session Recovery:**
- Always **sync to Drive** after each chunk:
  - `runs/` (YOLO), `logs/`, `checkpoints/`, `results/*.json`.
- Use `--resume` flags for detectors and heads.

---

## 4) Data Requirements (Exactly What We Need)

**Dataset:** Ego4D STA v2 (Short‑Term Object Interaction Anticipation). We need:
1. **Annotations JSON** with next‑active targets, verbs, nouns, TTC, and evaluation splits.
2. **High‑resolution last frames** for each clip (or the original videos to extract them).
3. **Short video clips** around the decision frame: 1–2s (e.g., 32 frames @ 16 fps).  
   - Store as MP4 per sample or as tensor archives (WebDataset/tar).

**Conversion for our pipeline:**
- **Detector training set** in **YOLO format** (txt per image) or **COCO JSON** (bbox + noun class).
- **Head training manifest** mapping each sample to:  
  {{last_frame_path, clip_path, candidate_boxes (during training), gt_box, verb_id, noun_id, ttc}}

**Disk Budget on Drive (estimate):**
- Extracted last frames (JPEG, 800 px side): ~40–80 GB (depends on subset size).
- Clips (1–2s @ 16 fps, 256–320 px): ~80–150 GB (subset).  
**Tip:** Start with **subset** (e.g., 20–30% of train) to iterate, then scale.

**Colab Extraction Notebook Steps:**
```python
# 1) Parse Ego4D STA annotations (place JSON on Drive beforehand).
# 2) Extract last frames at target size (e.g., max_side=800).
# 3) Extract clips centered on decision frames (length=32 frames @ 16 fps).
# 4) Build YOLO txt files for detector (noun classes); write COCO if preferred.
# 5) Build head-manifest.json for Stage B.

!python tools/prepare_sta_data.py   --ann /content/drive/MyDrive/ego4d/sta_v2.json   --video_root /content/drive/MyDrive/ego4d/videos   --out_root /content/drive/MyDrive/thesis_sta/data   --clip_len 32 --fps 16 --img_max 800 --clip_size 320   --splits train,val,test
```

---

## 5) Track A: Implementation Details

### A.1 Detector (Stage A)
- **Model:** YOLOv8‑s (or YOLOv9‑lite) at **img=640** or **800**.
- **Labels:** next‑active object box + **noun class id**.
- **Augment:** mosaic disabled (egocentric geometry), random flip/resize, light color jitter.

**Train:**
```bash
yolo detect train   data=configs/yolo_sta.yaml   model=yolov8s.pt   epochs=60 imgsz=640 batch=16   project=runs/yolo_sta name=exp_sta   optimizer=adamw lr0=1e-3 cos_lr=True   patience=15
```
_YOLO configs (`configs/yolo_sta.yaml`) set train/val image dirs and label dirs._

**Export Top‑K Candidates (for Stage B):**
```bash
python tools/export_candidates.py   --weights runs/yolo_sta/exp_sta/weights/best.pt   --images /content/drive/MyDrive/thesis_sta/data/frames_val   --topk 10   --out /content/drive/MyDrive/thesis_sta/manifests/candidates_val_k10.json
```
**Sanity:** Candidate **recall@K** ≥ 95% on val.

### A.2 Transformer Head (Stage B)
- **Inputs per sample:** 
  - ROIAlign features for top‑K candidate boxes from last frame (**image branch**: DINOv2‑B or ResNet‑50).
  - (**Optional**) pooled **video tokens** from the clip (**video branch**: MViTv2‑S or ViT‑B).
- **Architecture:** 2–4 transformer layers, width 256–384, heads 8.
- **Outputs:** per‑candidate `p(next_active)`, `verb_logits`, `ttc_logits` (binned) and/or `ttc_reg`.

**Loss:**  
`L = L_box(GIoU+L1 for selected) + L_verb(CE) + L_ttc(binned CE or SmoothL1) + λ * L_calib(optional)`

**Train:**
```bash
python src/train_head.py   --cfg configs/trackA_head.yaml   --candidates /content/drive/MyDrive/thesis_sta/manifests/candidates_train_k10.json   --val_cand  /content/drive/MyDrive/thesis_sta/manifests/candidates_val_k10.json   --epochs 40 --bs 8 --amp
```
**Eval:** compute **N**, **N+V**, **N+δ**, **Overall top‑5** on val.

---

## 6) Track B: Fusion Upgrade (STAformer‑Lite)

### B.1 Frame‑Guided Temporal Pooling (FGTP)
- Extract video tokens (stride‑8/16) → project onto last‑frame grid by learned pooling with positional alignment.

### B.2 Dual Cross‑Attention (tiny)
- **Block:** (Image→Video CA) + (Video→Image CA) + FFN; repeat 2–4×.
- Keep width 256; pre‑norm; residual connections.

**Config switch:**
```yaml
model:
  fusion:
    enabled: true
    fgtp: { enabled: true, stride_t: 2, proj_dim: 256 }
    dual_xattn: { layers: 3, dim: 256, heads: 8, dropout: 0.1 }
```
**Expected gains:** +1–2 mAP on **N+V** and **N+δ** at small FLOPs increase.

---

## 7) Track C: RGTP Inference‑Time Pruning

### C.1 Mechanics
- **At t−1:** Compute attention rollout → token importance.
- **Track:** Propagate importance to t via token correspondences (positional/ROI flow).
- **Prune:** Drop bottom 40–60% tokens from the **video branch** (never prune final layer).

**Eval with pruning:**
```bash
python src/eval_sta.py   --cfg configs/trackB_fusion.yaml   --rgtp_rate 0.5   --report runs/trackC_pruning_val.json
```
**Targets:** ~**35–50% latency reduction** with **≤0.5 mAP** drop.

---

## 8) Metrics, Logging, and Reproducibility

**Primary metrics:** **N mAP**, **N+V mAP**, **N+δ mAP**, **Overall top‑5**.  
**Secondary:** FLOPs, **latency (ms)** per sample (bs=1), GPU memory, candidate recall@K.  
**Logging:** JSON dumps to Drive; optional **wandb**.  
**Seeds:** set for PyTorch/NumPy; log exact git commit and configs.  
**Ablations:** 
- K ∈ {3,5,10}, clip length, TTC bins vs regression, fusion on/off, priors on/off, pruning rate ∈ {0.4, 0.5, 0.6}.

---

## 9) Week‑by‑Week Checklist (Colab‑feasible)

- **W1:** Data extraction + manifests + evaluator.
- **W2:** Detector train @640 px; export top‑K; measure recall.
- **W3:** Head (Track A) training + first val results.
- **W4:** Fusion (Track B) small blocks; re‑eval; visualize attention.
- **W5:** RGTP (Track C) integration + latency/mAP tradeoff.
- **W6:** (Optional) Hotspot prior + zone‑level affordance histograms.
- **W7:** Full ablations + cross‑scene robustness + TTC calibration.
- **W8:** Final tables/plots, code clean‑up, README, seeds/configs zip.

---

## 10) Risks & Mitigations (Colab)

- **GPU disconnects:** save checkpoints to Drive every N steps; use `--resume`.
- **VRAM OOM:** reduce batch, clip size, token dim (256), enable AMP, gradient accumulation.
- **Slow I/O:** keep frames/clips on Drive; cache tensors (npz) for hot loops.
- **Training instability:** warmup epochs, freeze lower backbone stages, binned TTC with label smoothing.

---

## 11) Repository Layout (suggested)

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
  runs/  # -> symlinked to Drive
  logs/  # -> symlinked to Drive
  README.md
```

---

## 12) Example Config Stubs

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
    ttc_mode: "binned"  # or "reg"
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

## 13) Deliverables

- **Checkpoints:** Detector (best.pt), Head (trackA.pt, trackB.pt).
- **Results:** `val_metrics.json` (all metrics), `latency.json` (with/without RGTP), plots (mAP vs latency).
- **Code:** Clean repo with configs and Colab notebooks.
- **Thesis assets:** Figures (pipeline, attention maps, pruning visualization), tables, ablation summaries.

---

## 14) Optional Add‑Ons (time permitting)

- **Hotspot UNet** (weakly supervised from hand/object approach frames) to re‑weight boxes.
- **Zone‑level affordance prior** (noun/verb histograms per scene cluster) as logit prior with temperature τ∈[2,5].
- **INT8 quantization** of the head for further latency reduction.

---

### TL;DR
- **Track A** (two‑stage) → strong, stable baseline on Colab.  
- **Track B** (tiny fusion) → +1–2 mAP on verbs/TTC.  
- **Track C** (RGTP) → ~40–50% latency cut at ≤0.5 mAP drop.  
All within **8 weeks** on **Colab Free** with Drive persistence.
