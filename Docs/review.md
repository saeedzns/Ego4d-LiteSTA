# Review of Egocentric Short‑Term Anticipation (STA), Forecasting, and Efficient Video Understanding for Wearable AR/Robotics

> **Scope.** This chapter surveys the Ego4D forecasting/anticipation tasks and winning baselines; two strong STA systems (STAformer, SOIA‑DOD); efficient video backbones and self‑supervised pretraining (MViTv2, VideoMAE); token pruning for real‑time streaming (PruneVid, Rollout‑Guided Token Pruning, EgoPrune); hand–object anticipation with affordances (PEAR; fine‑grained affordances); and ego↔exo transfer perspectives (survey, Synchronization‑is‑All‑You‑Need, Ego‑Only). It closes with practical design choices and a reproducible recipe for my thesis experiments.

---

## 1) Why egocentric *forecasting* and *anticipation*?

**Problem.** In AR and human‑robot collaboration, systems must *act before* visible contact: e.g., warn about hazards, pre‑position a tool, or cue a UI. The egocentric (first‑person) view emphasizes hands, near‑field objects, and the wearer’s motion—quite different from third‑person datasets. Forecasting therefore requires: (i) modeling the camera‑wearer’s intent and locomotion; (ii) anticipating *which* object will be touched *where*, *how* (verb), and *when* (time‑to‑contact, TTC); and (iii) long‑horizon action sequences for planning.

**Applications.** Assistive AR, tele‑operation, mobile manipulation, safety monitoring, and on‑device guidance.

---

## 2) Ego4D forecasting benchmark (tasks, labels, protocol)

### 2.1 Tasks (STA is highlighted)
- **Locomotion prediction.** Predict feasible ego‑trajectories on the ground plane over a short horizon.
- **Hand movement prediction.** Predict the future locations of the camera‑wearer’s hands.
- **Short‑Term Object Interaction Anticipation (STA).** *Given a video clip,* predict at the most recent frame: (a) **next‑active object(s)** with 2D boxes; (b) **verb** (how it will be interacted with); (c) **noun** (object class); and (d) **TTC** (seconds until interaction begins).
- **Long‑term action anticipation.** Predict the *sequence* of upcoming actions.

### 2.2 Annotations (how supervision is built)
- Action narration → curated **verb/noun taxonomies** (refined beyond raw narrations).
- For each interaction: **pre‑condition** frame and **contact** frame; **boxes** on active objects; **hand** and **object** boxes at −1.5s, −1.0s, −0.5s to encode approach dynamics; and **ego‑trajectory** via structure‑from‑motion (SfM). 

### 2.3 Metrics for STA
- **N mAP** (noun‑only AP over objects) — localization + object category.
- **N+V mAP** (noun+verb) — adds verb correctness.
- **N+δ mAP** — noun with **TTC tolerance**.
- **Overall top‑5 mAP** — composite (boxes + N/V + TTC).

> **Takeaway.** STA is a *multi‑signal* problem: spatial (boxes), semantic (N,V), and temporal (TTC) with partial observability and heavy egocentric motion.

---

## 3) STA baselines & two recent systems

### 3.1 Baseline families
- **Two‑branch “Still+Fast” style.** High‑res last frame + low‑res clip; fuse spatio‑temporal cues for object/verb/TTC.
- **Temporal fusion with detection priors.** Detect objects/hands first, then guide attention; or detect candidates then classify future interaction (two‑stage).
- **Language‑aided fusion.** Optionally incorporate narrations/summaries.

### 3.2 **STAformer (ZARRIO @ Ego4D STA)** — attention + affordances

**Goal.** Jointly predict next‑active boxes, N/V labels, TTC; *augment* predictions with human‑behavior priors via **affordances**.

**Inputs.** (i) A *still* last frame (high‑res appearance), (ii) a *short* video snippet. 

**Backbones.** DINOv2 (image) + TimeSformer (video).

**Key architectural ideas**
1) **Frame‑Guided Temporal Pooling Attention.** Pool video tokens *onto the last frame as reference.* This focuses temporal evidence where it matters—at the decision frame. 
2) **Dual Image↔Video Cross‑Attention.** Symmetric refinement—image tokens inform video tokens and vice‑versa to consolidate appearance + motion.
3) **Multi‑scale fusion + Fast‑RCNN head** adapted for STA (boxes + N/V + TTC heads).

**Affordance‑driven grounding**
- **Environment affordances as persistent memory.** Build a database over **EgoTopo zones**: for each topological zone, aggregate likely verbs/nouns from STA labels + narrations. At inference, match the observed scene to similar zones → produce **prior distributions** over verbs/nouns → *refine* STAformer’s posteriors. 
- **Interaction hotspots.** Predict a heatmap of likely contact regions from **hands + object trajectories** (trained following HOI hotspot literature). Use it to **re‑weight** per‑box confidence (spatial prior).

**Performance (qualitative).** Large relative gains vs. prior year’s winner on val; competitive test scores on N mAP, N+V mAP, N+δ mAP, and Overall top‑5.

**Strengths.** Clear, modular fusion; explicit affordance grounding improves *plausibility* under ambiguity; hotspot prior sharpens localization.

**Limitations.** Requires building/serving the zone‑level affordance memory; assumes availability of pre‑extracted hands/trajectories; risk of bias toward seen environments if matching is brittle.

### 3.3 **SOIA‑DOD** — *detect → (verb, TTC)* in two stages

**Goal.** Reduce multi‑task interference by **decoupling** localization from interaction reasoning.

**Stage A. Potential next‑active detection**
- Fine‑tune **YOLOv9** on last‑frame targets to detect **candidate active objects**.
- Keep top‑K boxes → turn each into a **query token** (box + class embedding).

**Stage B. Transformer reasoning**
- Concatenate **visual tokens** (from the frame/clip) with **object‑query tokens** → Transformer encoder.
- Predict per‑query: **next‑active probability**, **verb class**, **TTC**.

**Why it helps.** Separating detection stabilizes training (dense box supervision doesn’t fight with TTC regression); queries enforce *object‑centric* temporal reasoning. 

**Strengths.** Strong next‑active and noun/verb accuracy; simple to scale; detector can be swapped.

**Limitations.** Two‑stage latency; detector quality bounds recall; TTC learned only from short clips (limited long‑range cues).

---

## 4) Efficient backbones & pretraining you can leverage

### 4.1 **MViTv2** — multiscale ViT with pooling attention
- **What.** A hierarchical ViT with *pooling attention* (shrinks tokens across stages), better than windowed attention in accuracy/compute; adds *decomposed relative positional embeddings* and *residual pooling connections*.
- **Why relevant.** Serves as a unified backbone for images *and* videos (PySlowFast / Detectron2). Good accuracy–FLOPs trade‑off; pairs naturally with STA (video clip branch) and with two‑stage detectors.
- **Practice.** Start with S or B variants; keep temporal stride modest (e.g., 8–16) for motion; couple with FPN‑style neck if doing dense detection.

### 4.2 **VideoMAE** — data‑efficient self‑supervised pretraining
- **What.** Masked tube reconstruction with *extreme masking (90–95%)* using a light decoder; trains **plain ViT** on videos *without* extra datasets.
- **Why relevant.** If you don’t have massive exocentric pretraining, VideoMAE on **in‑domain** egocentric clips is a powerful way to bootstrap robust video features for STA, action detection, and long‑term anticipation.
- **Practice.** Pretrain on Ego4D subsets (hours ≫ labels), then fine‑tune for STA heads; keep augmentation simple; monitor reconstruction loss + downstream mAP.

---

## 5) Token pruning & streaming efficiency (toward on‑device STA)

**Motivation.** Egocentric video is redundant (static background, slow object change). Attention cost is quadratic in token count; streaming inference also bloats KV caches. Three *training‑free* directions:

### 5.1 **PruneVid** — prune by *static/dynamic* merging + question‑guided selection
- **Idea.** (i) Merge **temporally static** tokens across frames; (ii) cluster **spatially similar** tokens; (iii) inside the LLM/VLM, select tokens most relevant to the **current query** using mid‑layer attention. 
- **Outcome.** Prunes ≳80% tokens with minimal performance drop on video QA benchmarks; reduces FLOPs and memory substantially.
- **Relevance to STA.** Treat “What will be interacted with next?” as a query: keep tokens near moving hands and candidate objects; drop static background.

### 5.2 **Rollout‑Guided Token Pruning (RGTP)** — propagate importance over time
- **Idea.** Use **attention rollout** to trace which *input tokens* contributed to the last frame’s predictions; **track** them to the next frame; prune the rest. 
- **Properties.** Training‑free, interpretable; up to ~60–65% FLOPs reduction on action recognition and video detection without accuracy loss.
- **Relevance to STA.** Naturally focuses compute on **approach trajectories** (hands→object), ideal for short‑term anticipation.

### 5.3 **EgoPrune** — egocentric‑aware geometric alignment + MMR selection
- **Idea.** For *ego‑motion* video: (i) select keyframes; (ii) align frames by **homography**/**perspective** to account for head movement; (iii) filter redundant tokens after alignment; (iv) finally select tokens by **maximal marginal relevance (MMR)** balancing *query relevance* and *diversity*. 
- **Outcome.** Preserves ~99% task accuracy while cutting FLOPs/memory/latency; demonstrated on edge (Jetson Orin) embodied setups.

> **Design note.** For *lightweight real‑time STA*, combine: VideoMAE‑pretrained ViT (or MViTv2‑S/B) + RGTP at inference + a compact two‑stage head (SOIA‑DOD‑style). This keeps accuracy while controlling latency on a single GPU/edge device.

---

## 6) Hand–object anticipation & affordances

### 6.1 **PEAR** — phrase‑guided anticipation of *intention* and *manipulation*
- **Task.** Given an image (current scene) + a *phrase* (e.g., “pick up bottle”), predict: (i) **intention** before contact (hand motion trend + interaction hotspots), and (ii) **manipulation** after contact (hand pose with contact + manipulation trajectory).
- **Key ideas.**
  - **Cross‑align verbs, nouns, images** to reduce intention ambiguity (verbs ↔ motion patterns; nouns ↔ functional regions).
  - **Dynamic bidirectional constraints** between intention ↔ manipulation via a Deep‑Equilibrium‑style module; residual connections refine intention using manipulation cues.
  - **C‑VAE decoders** model multi‑modality (multiple plausible futures).
- **Why relevant to STA.** Hotspots and motion trends can condition STA box ranking and TTC; phrase priors map to verb distributions.

### 6.2 **Fine‑grained affordances** for HOI (annotations & utility)
- **Problem.** “Affordance” often conflated with verb labels (e.g., *cut*, *take*), missing **grasp types** and **motor capacity**; goal‑oriented labels leak intent.
- **Contribution.** Define affordance as **(motor action × grasp type)** at the *hand–object interface*; add **mechanical actions** for tool–object relations. Annotate EPIC‑KITCHENS with fine‑grained affordances; show improved **hotspot prediction** and **cross‑domain generalization**.
- **Use in STA.** Use affordance priors as **structured regularizers** on verb/noun logits; use predicted hotspots to **re‑weight** candidate boxes (as in STAformer).

---

## 7) Ego↔Exo transfer and perspectives

### 7.1 Short survey (ego–exo joint learning)
- Joint ego–exo modeling is promising: exocentric views provide holistic context; egocentric views capture fine hand–object signals. Tasks include action recognition, correspondence, pose, retrieval, temporal segmentation, and anticipation.

### 7.2 **Exo→Ego without labels via synchronization**
- **Problem.** Exo‑trained temporal segmentation fails on Ego due to domain gap. 
- **Method.** Use *unlabeled synchronized* Exo–Ego pairs to **distill** features/heads from the Exo teacher to an Ego student—no Ego labels needed.
- **Result.** Matches supervised Ego training on Assembly101/EgoExo4D; suggests synchronization is a powerful bridge.
- **Relevance.** For long‑term anticipation/action segmentation sub‑tasks.

### 7.3 **Ego‑Only** (no exo at all)
- **Claim.** With **MAE‑style pretraining** on egocentric videos and simple temporal segmentation fine‑tuning, we can surpass exo‑transfer methods for Ego action detection/recognition (Ego4D, EPIC‑K, Charades‑Ego).
- **Lesson.** If you have *enough egocentric unlabeled video*, prefer **in‑domain** self‑supervised pretraining (e.g., VideoMAE), then fine‑tune.

> **Synthesis.** For STA, which is compact and near‑term, *Ego‑Only* pretraining + STA‑specific heads is a strong default. For longer‑horizon segmentation/action plans, **synchronized exo–ego** can still help.

---

## 8) Related works (extended)

**STA & anticipation.** Early STA used two‑branch “StillFast” variants; guided‑attention (GANOv2) fused detections; multimodal fusion (TransFusion) added language. Recent trend: *object‑centric* querying (SOIA‑DOD) and *behavior priors* (STAformer). TTC regression varies from direct regression to discretized bins with tolerant mAP (N+δ).

**Egocentric action detection/segmentation.** Beyond Ego‑Only and Sync‑is‑All‑You‑Need, strong foundations include long‑sequence transformers, temporal proposal methods (ActionFormer), and Ego4D tasks (Moments Queries, PNR temporal localization). EgoVLP/EgoSchema show that large text‑video pretraining helps episodic queries and QA, but STA remains low‑latency and geometry‑centric.

**Backbones.** Multiscale transformers (MViTv2) are compute‑savvy for dense video; Image→Video adapters (TimeSformer/XViT) remain popular for clip‑level reasoning; DINOv2 is common for high‑res last‑frame features.

**Self‑supervision.** VideoMAE family: variants for detection, long‑form modeling, and masked conditioning; data quality and domain match out‑weigh size.

**Affordances & hotspots.** Hotspot prediction from HOI trajectories and contact cues; fine‑grained affordance labels improve spatial priors; phrases/prompts connect language to manipulation constraints (PEAR).

**Efficiency.** Token merging/pruning (ToMe, KV‑cache pruning) → video‑aware methods (Eventful Transformers, PruneVid, RGTP, EgoPrune); geometric alignment is *especially* helpful under head motion.

**Workshops/benchmarks.** Ego4D/Ego‑Exo4D challenges at EgoVis (CVPR’24/’25) sustain standardized evaluation of STA and related tasks.

---

## 9) Design choices for my thesis (actionable)

### 9.1 Data & splits
- Use Ego4D STA v2 annotations + high‑res last‑frame images; extract 1–2s clips at 16–32 fps centered on the decision frame. Ensure train/val/test follow official splits.

### 9.2 Backbones & pretraining
- Start with **MViTv2‑S** (video branch) + **DINOv2‑B** (image branch). In parallel, run **VideoMAE‑ViT‑B** pretraining for an Ego‑Only variant (replace MViTv2 with ViT‑B video encoder at fine‑tune time).

### 9.3 Model heads
- **SOIA‑DOD‑style** two‑stage: fine‑tune YOLO‑X/YOLOv9 for next‑active *candidates*, K=5–10; Transformer head predicts {verb, TTC} and final selection.
- **STAformer‑style** fusion ablation: add frame‑guided pooling and dual cross‑attention; add **affordance priors** and **hotspot re‑weighting** modules.

### 9.4 Losses & metrics
- Multi‑task: box (GIoU+L1), noun CE, verb CE, TTC smooth‑L1 (and/or soft labels over δ‑bins). Report **N mAP**, **N+V mAP**, **N+δ mAP**, **Overall top‑5**.

### 9.5 Efficiency for real‑time
- At inference: **RGTP** pruning at 40–60% rate on the video branch; optional PruneVid/EgoPrune for VLM variants.
- Quantize heads to FP16/INT8; cap resolution on video branch; run detector every *n* frames and track between.

### 9.6 Affordances & phrases (optional)
- Train a hotspot head using hand/object tracks; build a lightweight **zone‑level affordance cache** from train split narrations and STA labels; optionally accept **simple phrases** (verb+noun) to bias logits.

### 9.7 Reproducibility checklist
- Fix seeds; log exact dataset hashes; save evaluation JSONs; publish config files (backbone, clip length, TTC binning, pruning rates); include ablations (no affordance, no hotspot, no pruning).

---

## 10) Pitfalls & open problems
- **TTC noise.** Human annotations have jitter; prefer tolerant bins and calibration.
- **Verb granularity.** Fine‑grained verbs may be visually ambiguous; affordances/hotspots mitigate but don’t fully solve.
- **Domain shift.** Kitchens vs workshops: environment priors may overfit; include *cross‑scene* validation.
- **Latency budgeting.** STA heads must run under tight budgets on edge; pruning and two‑stage caching are essential.
- **Long‑horizon coupling.** STA and long‑term anticipation could be trained jointly (shared encoder + multi‑head), but sequencing loss design remains open.

---

## 11) Abbreviations
**STA** short‑term anticipation; **TTC** time‑to‑contact; **HOI** hand–object interaction; **MMR** maximal marginal relevance; **RGTP** rollout‑guided token pruning; **MAE** masked autoencoder; **mAP** mean average precision.

---

## 12) Minimal bib (for the thesis reference list)
- Ego4D dataset/benchmarks; Forecasting/STA docs.
- STAformer (affordances + attention); SOIA‑DOD (YOLOv9 + transformer).
- MViTv2; VideoMAE.
- PruneVid; RGTP; EgoPrune.
- PEAR (phrase‑based HOI anticipation); Fine‑grained affordances for egocentric videos.
- Ego–Exo short survey; Synchronization‑is‑All‑You‑Need; Ego‑Only.

> *End of chapter.*
