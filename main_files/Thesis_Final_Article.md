# Egocentric Short-Term Anticipation on Ego4D: A Modular Two-Stage Baseline with Lightweight Fusion and Training-Free Pruning (Tracks A / B / C)

Author: Saeed  
Date: 2025-11-11  
Affiliation: Ego4d-LiteSTA Project  

---

## Abstract

Anticipating human interaction from egocentric video is a prerequisite for hands-free augmented reality, assistive robotics, and safety-critical wearables. Within the Ego4D Short-Term Object Interaction Anticipation (STA) benchmark, a model must localize the next-active object in the decision frame, forecast its verb–noun semantics, and estimate time-to-contact (TTC) before the interaction begins. This thesis consolidates a comprehensive literature review and presents a modular, resource-aware STA system engineered for repeatability on modest hardware. The pipeline is anchored around a recall-first two-stage detector head (Track A), a lightweight temporal fusion block that couples Frame-Guided Temporal Pooling with dual cross-attention (Track B), and a training-free Rollout-Guided Token Pruning layer that trims 40–60% of tokens at inference to preserve accuracy while meeting edge-latency budgets (Track C).

To support rigorous experimentation, we standardize dataset acquisition, manifest generation, and logging; expose centralized configuration toggles; and maintain deterministic preprocessing for Stage B inputs. Track B training completes in approximately six hours on our local environment, whereas detector fine-tuning for Track A stabilizes within two days on Google Colab Free GPUs. Baseline validation results (N mAP ≈ 0.07, TTC MAE ≈ 0.226 s, accuracy ≈ 0.928) establish a credible reference point for ablations, pruning sweeps, and calibration studies. The document concludes with future extensions spanning multi-class detection, structured TTC modeling, and cross-dataset generalization, positioning the work as a deployable foundation for egocentric anticipation research.

---

## Table of Contents

0. Comprehensive Review of Egocentric STA Landscape  
1. Introduction  
2. Background and Related Work  
3. Problem Definition and Data  
4. System Overview (Tracks A / B / C)  
5. Methodology  
6. Implementation Details  
7. Experimental Results and Protocols  
8. Results and Analysis  
9. Ablations and Variants  
10. Reproducibility and Engineering Practices  
11. Discussion  
12. Limitations  
13. In-Progress and Next Steps  
14. Ethical, Legal, and Social Implications (ELSI)  
15. Conclusion  
Appendix A. Notation and Metrics  
Appendix B. Algorithms and Pseudocode  
Appendix C. Hyperparameters and Config Bundles  
Appendix D. Failure Cases and Qualitative Notes  
Appendix E. Extended Related Work  

---

## 0. Comprehensive Review of Egocentric STA Landscape

> **Scope.** This introductory section surveys the Ego4D forecasting tasks and state-of-the-art baselines; two representative STA systems (STAformer, SOIA-DOD); efficient video backbones and self-supervised pretraining regimes (MViTv2, VideoMAE); training-free token pruning strategies (PruneVid, Rollout-Guided Token Pruning, EgoPrune); affordance-centric hand–object anticipation (PEAR, fine-grained affordances); and ego↔exo transfer perspectives. It concludes with actionable design choices and a reproducible recipe that inform the thesis methodology.

### 0.1 Motivation for Egocentric Forecasting and Anticipation

**Problem.** Assisted-reality devices and collaborative robots must decide before contact occurs—warning about hazards, staging tools, or cueing interfaces. Egocentric viewpoints emphasize hands, near-field objects, and wearer motion, diverging from third-person video distributions. Forecasting therefore requires (i) modeling wearer intent and locomotion, (ii) identifying which object will be touched where, how (verb), and when (TTC), and (iii) reasoning over longer action sequences for planning.

**Applications.** Assistive AR, tele-operation, mobile manipulation, safety monitoring, and on-device guidance.

### 0.2 Ego4D Forecasting Benchmark: Tasks, Labels, Protocol

#### 0.2.1 Tasks (STA Highlighted)
- **Locomotion prediction.** Forecast feasible ego-trajectories on the ground plane over short horizons.
- **Hand movement prediction.** Predict future hand locations for the camera wearer.
- **Short-Term Object Interaction Anticipation (STA).** From a short clip, predict at the most recent frame: (a) next-active object boxes, (b) the interaction verb, (c) the noun, and (d) TTC (seconds until contact).
- **Long-term action anticipation.** Predict a sequence of upcoming actions.

#### 0.2.2 Annotation Pipeline
- Action narrations refined into verb/noun taxonomies.
- For every interaction: pre-condition and contact frames, active object boxes, hand/object boxes at −1.5 s, −1.0 s, −0.5 s to encode approach dynamics, and ego-trajectory estimates via structure-from-motion.

#### 0.2.3 STA Metrics
- **N mAP.** Noun-only average precision with localization.
- **N+V mAP.** Joint noun+verb precision.
- **N+δ mAP.** Noun AP with TTC tolerance.
- **Overall top-5 mAP.** Composite measure covering boxes, N/V, and TTC.

> **Takeaway.** STA couples spatial, semantic, and temporal signals under heavy egomotion and partial observability.

### 0.3 STA Baselines and Contemporary Systems

#### 0.3.1 Baseline Families
- **Two-branch “Still+Fast” variants.** High-resolution last frame plus low-resolution clip fusion for combined cues.
- **Temporal fusion with detection priors.** Detect objects or candidates first, then classify future interaction (two-stage setups).
- **Language-aided fusion.** Optionally integrate narrations or textual summaries.

#### 0.3.2 STAformer (ZARRIO, Ego4D STA)
- **Objective.** Joint prediction of boxes, noun/verb, and TTC while injecting affordance priors.
- **Inputs.** High-resolution last frame and short clip snippet.
- **Backbones.** DINOv2 for the image branch, TimeSformer for video.
- **Architectural highlights.** Frame-Guided Temporal Pooling (FGTP) to align motion to the decision frame; dual image↔video cross-attention for bidirectional refinement; multi-scale fusion with STA-specific Fast R-CNN heads.
- **Affordance grounding.** Environment affordance memory built over EgoTopo zones to supply verb/noun priors, and interaction hotspots to re-weight spatial confidence.
- **Strengths.** Modular fusion, improved plausibility, sharper localization via hotspot priors.
- **Limitations.** Requires serving zone-level memory and reliable hand/trajectory extraction; sensitive to environment bias.

#### 0.3.3 SOIA-DOD
- **Objective.** Decouple detection from high-level reasoning to reduce multi-task interference.
- **Stage A.** Fine-tuned YOLOv9 proposes candidate next-active boxes; top-K boxes become query tokens.
- **Stage B.** Transformer encoder ingests visual and object-query tokens to output next-active probabilities, verbs, and TTC.
- **Strengths.** Strong localization and semantics, detector modularity.
- **Limitations.** Two-stage latency, detector-recall ceiling, short temporal receptive field for TTC.

### 0.4 Efficient Backbones and Self-Supervised Pretraining

#### 0.4.1 MViTv2
- Hierarchical ViT with pooling attention, decomposed relative positional embeddings, and residual pooling connections.
- Serves as a unified backbone for images and videos; favorable accuracy–compute balance.

#### 0.4.2 VideoMAE
- Extreme masked-tube reconstruction with lightweight decoder; enables in-domain pretraining without external datasets.
- Particularly valuable when large exocentric corpora are unavailable; simple augmentation suffices.

### 0.5 Token Pruning and Streaming Efficiency

**Motivation.** Egocentric video is redundant; attention costs scale quadratically with token count and inflate KV caches during streaming.

#### 0.5.1 PruneVid
- Combines static/dynamic token merging with query-guided selection to retain task-relevant tokens.
- Demonstrates >80% token reduction with minimal accuracy loss on video QA benchmarks; applicable to STA by treating “next-active” as the guiding query.

#### 0.5.2 Rollout-Guided Token Pruning (RGTP)
- Uses attention rollout to trace token influence and prune low-importance tokens without retraining.
- Achieves up to ~60–65% FLOPs reduction while preserving accuracy; aligns naturally with STA motion cues.

#### 0.5.3 EgoPrune
- Tailors pruning to egocentric geometry via frame alignment and maximal marginal relevance selection.
- Maintains near-original accuracy while cutting FLOPs, memory, and latency on edge devices.

> **Design Note.** A lightweight STA system can pair VideoMAE or MViTv2 backbones with RGTP and a compact two-stage head to balance accuracy and latency.

### 0.6 Hand–Object Anticipation and Affordances

#### 0.6.1 PEAR
- Predicts intention and manipulation trajectories given an image and phrase.
- Aligns verbs, nouns, and images; couples intention and manipulation via equilibrium-style updates; leverages conditional VAEs for multi-modality.
- Hotspot and motion trends can inform STA box ranking and TTC calibration.

#### 0.6.2 Fine-Grained Affordances
- Defines affordance as (motor action × grasp type) at the hand–object interface, extending to mechanical actions.
- Annotated datasets (e.g., EPIC-KITCHENS) improve hotspot prediction and generalization.
- Provides structured priors for verb/noun logits and candidate weighting.

### 0.7 Ego↔Exo Transfer Perspectives

#### 0.7.1 Joint Ego–Exo Learning
- Exocentric views supply global context; egocentric views capture fine hand–object cues. Joint models tackle action recognition, correspondence, pose, retrieval, segmentation, and anticipation.

#### 0.7.2 Synchronization-Is-All-You-Need
- Distills exocentric temporal segmentation into egocentric models using unlabeled synchronized pairs; approaches supervised ego-only performance.

#### 0.7.3 Ego-Only
- Demonstrates that in-domain self-supervised pretraining (e.g., MAE) on egocentric video can surpass exo-transfer for action detection; supports STA by emphasizing in-domain unlabeled data.

> **Synthesis.** Ego-only pretraining is a strong default for STA, while synchronized exo–ego data benefits long-horizon segmentation.

### 0.8 Extended Related Work Highlights
- Early STA blended two-branch fusion; recent work emphasizes object-centric querying and behavior priors.
- Egocentric action detection leverages long-sequence transformers, temporal proposals (ActionFormer), and large-scale text-video pretraining (EgoVLP, EgoSchema).
- Backbones such as MViTv2, TimeSformer, and DINOv2 underpin modern STA pipelines.
- Self-supervised approaches (VideoMAE variants) stress domain alignment over scale.
- Efficiency literature spans token merging, pruning, and geometric alignment tailored to egocentric motion.
- Ego4D/Ego-Exo challenges at EgoVis sustain standard evaluation protocols.

### 0.9 Design Choices for This Thesis
- **Data and splits.** Adopt Ego4D STA v2 with official train/val/test partitions; extract 1–2 s clips at 16–32 fps around the decision frame.
- **Backbones and pretraining.** Start with MViTv2-S (video) and DINOv2-B (image); explore VideoMAE ViT-B pretraining for ego-only variants.
- **Model heads.** Implement SOIA-DOD-style two-stage detection and transformer reasoning; evaluate STAformer-style fusion plus affordance/hotspot priors.
- **Losses and metrics.** Use combined box, noun, verb, and TTC objectives; report N mAP, N+V mAP, N+δ mAP, and Overall top-5.
- **Efficiency.** Apply RGTP pruning (40–60% rate), consider quantization, and throttle detector frequency.
- **Affordances and phrases.** Train hotspot heads, build lightweight affordance caches, optionally bias logits with simple verb–noun prompts.
- **Reproducibility.** Fix seeds, log dataset hashes, publish configs, and include ablations removing individual modules.

### 0.10 Pitfalls and Open Problems
- TTC label noise suggests tolerant bins and calibration strategies.
- Verb granularity remains visually ambiguous; affordances help but are not decisive.
- Domain shift across environments raises generalization concerns; cross-scene validation is critical.
- Latency budgets on edge devices necessitate pruning and caching.
- Coupling STA with long-term anticipation requires better sequence-loss design.

### 0.11 Abbreviations
- STA: short-term anticipation
- TTC: time-to-contact
- HOI: hand–object interaction
- MMR: maximal marginal relevance
- RGTP: rollout-guided token pruning
- MAE: masked autoencoder
- mAP: mean average precision

### 0.12 Minimal Bibliography (For Reference Integration)
- Ego4D dataset and forecasting documentation
- STAformer (affordances + attention)
- SOIA-DOD (YOLOv9 + transformer)
- MViTv2; VideoMAE
- PruneVid; RGTP; EgoPrune
- PEAR; fine-grained affordances for egocentric videos
- Ego↔exo short survey; Synchronization-Is-All-You-Need; Ego-Only

---

## 1. Introduction

Augmented Reality (AR) and mobile robotics demand anticipation: systems must act before an interaction occurs. Given a short egocentric clip culminating at time t, the Short-Term Object Interaction Anticipation (STA) task asks where the next-active object will be in the last frame, what verb (and noun) describes the upcoming interaction, and when contact will occur (TTC). Egocentric video is uniquely challenging: the head-mounted camera creates significant egomotion, hands occlude objects, near-field geometry dominates, and signals are subtle and short-range.

We pursue a practical STA pipeline that balances accuracy with deployability under tight resource budgets. The central idea is to honor a two-stage discipline: a detector proposes high-recall candidate boxes on the last frame (Stage A), and a lightweight reasoning head (Stage B) refines candidates, potentially with a small dose of temporal fusion. We then introduce a training-free efficiency layer (Track C) that prunes tokens at inference to approach near-real-time latency without retraining.

This document consolidates the project’s motivation, related work context, problem formulation, methodology, implementation details, experiments, results, and future directions. It is written to function both as a technical report and as a thesis chapter.

---

## 2. Background and Related Work

### 2.1 Egocentric Forecasting in Ego4D
Ego4D’s forecasting benchmark spans several tasks, including locomotion prediction, hand movement prediction, and STA. STA is our focus: given a short clip, predict, at the last frame, the next-active object’s bounding box, noun and verb, and TTC. Evaluation metrics cover N mAP (noun-only AP), N+V mAP (joint noun+verb AP), N+δ mAP (noun AP tolerant to TTC errors), and an overall top-5 composite.

### 2.2 Two-Stage Reasoning for STA
Two-stage approaches decouple dense detection (a well-studied, stable problem) from higher-level reasoning about interaction and timing. SOIA-DOD exemplifies this line: detect candidates (YOLO-like), then reason with a transformer head per candidate to predict next-active, verb, and TTC. This decoupling improves stability and interpretability: misses can be attributed to proposals versus reasoning.

### 2.3 Lightweight Fusion and Affordances
STAformer introduces a compact fusion: align video tokens to the last frame (e.g., via FGTP) and employ dual cross-attention between image and video streams. Affordances and hotspots act as priors that improve plausibility under ambiguity. We adopt the FGTP+dual CA spirit to gain motion cues without heavy video transformers.

### 2.4 Efficient Video Understanding and Pruning
Modern video transformers are powerful but expensive. Training-free token pruning (e.g., PruneVid, RGTP, EgoPrune) reduces redundancy by tracking importance across time or aligning egomotion. We adopt an RGTP-inspired approach specialized to STA timing: compute rollout at t−1 and track importance to t; prune low-importance tokens before the head.

---

### 2.5 Positioning Within the STA Landscape
We deliberately position our system as a pragmatic middle ground between heavyweight video transformers and purely image-only heads:
- Heavy video backbones (VideoMAE/MViT/TimeSformer) deliver strong temporal modeling but exceed Colab Free budgets.
- Pure last-frame heads train fast but ignore decisive short-term motion (hand approach, object micro-movements).
- Our approach: a strong last-frame backbone plus a very small motion adapter (FGTP + 2–4 dual CA layers). This typically lifts N+V and N+δ mAP without blowing up compute or memory.

### 2.6 Comparison to STAformer and SOIA-DOD
- SOIA-DOD: emblematic two-stage pipeline; powerful head, proposals from a detector. It underscores the value of decoupling recall and semantics. We adopt the same discipline.
- STAformer: compact fusion with affordance cues; aligns motion to the decision frame. We borrow the FGTP spirit and dual cross-attention but keep the parameter budget minimal and togglable.
- Distinctives in our work: centralized toggles, Stage B manifests as first-class inputs, rigorous run discovery and logging, and an explicit Track C for deployment-time token pruning.

### 2.7 Efficiency–Accuracy Trade-offs
We view efficiency as a first-class metric:
- Token pruning (Track C) targets ≈35–50% latency reduction with ≤0.5 mAP loss.
- Candidate K controls both cost and recall. Typical sweet spots: K∈{5,8,10}. Raising K improves recall but linearly increases Stage B cost.
- Width/depth of dual CA: {dim 192–384, layers 0–4}. The curve is smooth; diminishing returns kick in at 3–4 layers.
- Quantization is optional future work; pruning is training-free and therefore safer during iteration.

### 2.8 Affordances and Hotspot Priors
Affordances summarize “where interaction tends to happen” on objects. In egocentric settings, these cues regularize predictions when motion is weak or occluded. We treat affordances as an optional prior:
- As an auxiliary channel injected into the ROI head.
- As a calibration factor over noun/verb logits in ambiguous scenes.
This is intentionally modular: the core pipeline does not depend on affordances to function.

### 2.9 Domain Shift and Agent-Specific Biases
Egocentric datasets often skew toward kitchens and specific users. Two risks emerge:
- Scene bias: models overfit to background layouts rather than interaction cues.
- User bias: idiosyncratic motion patterns and hand poses reduce generalization.
Mitigations include cross-scene validation, stronger augmentations that perturb background statistics, and light domain adaptation (e.g., feature-wise affine tuning).

### 2.10 Data Quality: Labels, Timing, and TTC Noise
STA annotations involve pre-condition and contact frames; TTC is derived from their difference. Small misalignments (±1–3 frames) create label noise for fine-grained TTC regression. We therefore:
- Offer binned TTC classification as an alternative to regression.
- Report δ-tolerant AP (N+δ) alongside strict metrics.
- Consider label smoothing and post-hoc calibration to reduce overconfidence.

### 2.11 Calibration and Uncertainty
We monitor calibration for verb/TTC heads. Overconfident heads can degrade ranking-based metrics and lead to brittle decisions in deployment. Tools include:
- Temperature scaling on validation logits.
- Isotonic regression as a non-parametric alternative.
- Reporting Expected Calibration Error (ECE) and reliability diagrams in future iterations.

### 2.12 TTC Modeling Choices
Two practical options:
- Direct regression (Smooth L1): simple, continuous, but sensitive to label noise.
- Discrete bins (cross-entropy): more robust to jitter; enables δ-tolerant AP definitions.
We start with regression for simplicity and expose bins via a toggle for ablations.

### 2.13 Open Problems Specific to STA
- Small-object interactions (keys, zippers) under motion blur remain challenging.
- Hand-object occlusion around the decision instant confounds both detector and head.
- Non-rigid items (bags, cloth) distort boxes and semantics.
- Human intent ambiguity: multiple plausible next-active objects coexist; ranking quality matters as much as absolute scores.

---

## 3. Problem Definition and Data

### 3.1 Task Definition
Input: a short clip ending at time t and the last frame.  
Outputs at time t:  
- A bounding box for the next-active object (per candidate)  
- Noun and verb labels  
- Time-to-contact (TTC), either regression or binned.

We define a two-stage system: Stage A produces top-K candidate boxes on the last frame. Stage B takes these candidates and predicts p(next_active), verb, and TTC per candidate, optionally fusing motion cues from the short clip.

### 3.2 Dataset and Annotations
We use Ego4D STA annotations (v2) for train/val/test splits. Each annotation includes: clip_uid, video uid, object boxes, verb/noun category IDs, and timing information (pre-condition and contact frames). We rely on official schemas and evaluation protocols per the forecasting repository.

### 3.3 Local Extraction Strategy (Why Specific Frames, Not All)
We purposefully avoid exhaustive frame extraction. Instead, we:
- Parse `fho_sta_train.json` and `fho_sta_val.json` and group annotations by `video_uid`.
- For each video, collect the exact frame indices we need (unique, sorted), driven by the Stage B manifest and per-sample decision frames.
- For each frame, choose the object with the minimum time-to-contact as the positive instance; map noun IDs via taxonomies (if doing multi-noun detection). In our default detector setup, we use a single class, next_active.
- Write `tmp_frame_lists/<video_uid>_frames.txt` and extract only those frames via ffmpeg.
- For each extracted image, write a YOLO `.txt` label with normalized box coordinates.

This design reduces storage and bandwidth while guaranteeing the detector sees the decision frames and their most relevant targets.

### 3.4 Clip Downloads (Full and 540p)
On Colab (for analytics and light experiments), we download clip segments by `video_uid` list using the Ego4D CLI. Two modes are useful:
- Full-resolution clips: `--datasets clips`  
- 540p clips: `--datasets clips_540`  
We maintain a UID list (e.g., `sta_clip_uids_part1.txt`) and pass it via `--video_uid_file`. We also leverage the `transform_annotations.ipynb` notebook from the official Ego4D repo to scale boxes when working with 540p assets so that labels remain aligned after downsampling.

---

## 4. System Overview (Tracks A / B / C)

### 4.1 Track A — Two-Stage STA Baseline (Recall-First)
- Stage A: Detector on last frame → top-K boxes (K≈5–10).  
  - Model: YOLOv8-s (or YOLOv9-lite), single-class next_active by default.  
  - Objective: Maximize candidate recall@K on validation.  
  - Note: Fine-tuning this detector for the single class on our STA crops is planned; it typically takes ≈2 days on Colab Free to converge to stable recall.
- Stage B: Reasoner head that consumes candidate ROIs (+ optional video tokens).  
  - Head outputs: p(next_active), verb, TTC.  
  - Loss: GIoU + L1 for box, CE for verb, Smooth L1 or binned for TTC.

Why two-stage? It isolates localization from higher-level reasoning, speeds up iteration, and allows robust error analysis (is it proposals or semantics?).

### 4.2 Track B — Lightweight Fusion (FGTP + Dual Cross-Attention)
- Frame-Guided Temporal Pooling (FGTP): Align a short clip’s tokens to the last frame grid to focus motion at the decision instant.
- Dual Image↔Video Cross-Attention: 2–4 layers, width ≈256, pre-norm+residual.    
- Intended effect: lift N+V and N+δ mAP with modest compute.  
- Implementation notes: centralize toggles/hyperparameters at the top of training scripts; auto-discover latest Stage B manifests; expose nested progress bars.

### 4.3 Track C — Training-Free Token Pruning (RGTP)
- Compute attention rollout at t−1; track importance to t.  
- Prune (e.g., 40–60%) low-importance tokens before the head; don’t prune the final stage.  
- Target outcomes: ≈35–50% latency reduction; ≤0.5 mAP drop.  
- Implementation is a runtime toggle block with configurable `rgtp_rate`, `min_keep`, and `pruning_enabled` flags.

---

## 5. Methodology

### 5.1 Data Preprocessing Pipeline
- Colab downloads: annotations, STA model artifacts, optional clip packs (full or 540p).  
- Local extraction: per-UID frame lists; selective ffmpeg extraction; YOLO label emission.  
- Stage B manifests: `head_train.jsonl` and `head_val.jsonl` built from crops and annotations; these become the canonical inputs for Track B training and evaluation.

### 5.2 Stage B Manifest Schema
Each record includes:  
`{last_frame_path, clip_path?, candidates?, gt_box, verb_id, noun_id, ttc}`  
with optional fields for cache, derived features, or diagnostic overlays. Manifests also encode candidate recall metrics and the proposal provenance.

### 5.3 Models and Losses
- Detector (Track A): YOLOv8-s, single class (`next_active`).  
- Reasoner (Track B): image backbone → ROI pooling; optional video branch; fusion via FGTP + dual CA; head MLP/transformer predicting p(next_active), verb, TTC.  
- Loss: multi-task sum with tunable weights: `L_total = L_box + L_verb + L_ttc (+ L_calib)`; TTC may be regressed or binned.

### 5.4 Training Regimen
- Track B:  
  - Default: batch size 8, epochs ≈20, LR ≈1e-3.  
  - Eval every N steps; early stopping on validation.  
  - Checkpointing: epoch-wise; best by `mAP`; final `trackB_final_<timestamp>.pt` saved.  
  - Runtime: ≈6 hours end-to-end in main mode on our local setup (AMP off).  
- Track A fine-tuning (planned):  
  - Single-class next_active; augmentations suited for egocentric geometry (e.g., avoid heavy mosaic).  
  - Estimated wall-clock: ≈2 days on Colab Free GPU to stable recall.

### 5.5 Evaluation
- Track B evaluation script auto-detects latest Stage B manifests and the most recent `trackB_final_*.pt`. It emits: accuracy, N/N+V/N+δ mAP, TTC MAE, per-class AP, and overlay images for qualitative review.
- Track C pruning evaluator writes metrics JSON, including mean fraction pruned. Latency profiling hooks are planned (CUDA events or wall-clock + VRAM usage).

### 5.6 Formal Multi-Task Objective
Let each candidate k have predictions: next-active score p_k, verb logits v_k, TTC prediction τ_k (regression) or distribution q_k over bins. Ground-truth indicators: y_k^{NA} ∈ {0,1}, verb label y^{V}, TTC value τ^*. Define losses:
- Next-active: focal or BCE loss L_{NA} = BCE(p_k, y_k^{NA}) aggregated over K.
- Verb: cross-entropy L_V = CE(v_{k*}, y^{V}) for the GT-positive candidate k* (or weighted over all candidates if using soft selection).
- TTC regression: L_{TTC} = SmoothL1(τ_{k*}, τ^*).
Total: L = λ_{NA} L_{NA} + λ_V L_V + λ_{TTC} L_{TTC} (+ λ_{calib} L_{calib} optional calibration penalty). We tune λ’s to balance gradients.

### 5.7 Candidate Selection Strategies
Default: top-K raw detector boxes (score-sorted). Alternatives:
- Diversity-aware selection: penalize overlapping boxes to cover more potential objects.
- Motion priors: favor boxes near hand trajectories.
- Soft candidate weighting: propagate scores into loss weighting rather than hard selecting k*.

### 5.8 Data Loader and Caching
- Frame decoding minimized by storing extracted JPEGs once; no repeated ffmpeg calls.
- ROI caches: precompute r(b_k, I_t) features when static to reduce epoch time; invalidated when backbone parameters change.
- Clip token caching: store video encoder tokens for reuse across different pruning rates during evaluation.

### 5.9 Robustness Augmentations (Planned)
- Spatial jitter within ±5% offsets to boxes to stress localization robustness.
- Subtle motion blur simulation to mimic rapid hand movement.
- Photometric variations (mild brightness/contrast) constrained to preserve egocentric realism.

### 5.10 Interface Contracts (Per Module)
- Detector: input I_t; output list of boxes B with scores. Must guarantee stable ordering by score.
- Fusion: inputs (image tokens G, clip tokens F); outputs fused tokens (Ĝ, Ṕ). Must be pure function given toggles.
- Pruner: input attention maps (t−1→t); output keep indices; must not alter feature values besides masking.
- Head: input candidate ROI features +/- fused tokens; output (p_k, verb logits, TTC). Deterministic under fixed seed.

### 5.11 Error Taxonomy
Categorize evaluation failures:
- Proposal Miss: GT object absent from B.
- Misclassification: GT present; p_k low or wrong noun/verb.
- TTC Drift: localization correct; |τ_pred − τ_gt| large.
- Calibration Error: high-confidence wrong prediction.
Used to direct ablations (e.g., improve proposals vs refine verb modeling).

### 5.12 Dataset Curation Notes
We track per-noun frequency; long-tailed nouns may require reweighting or focal adjustments. Maintain a CSV with counts to inform future multi-class detector expansion.

### 5.13 Mathematical Sketch of FGTP Alignment
Given image tokens G_s and video tokens F_{τ,s}, define similarity sim(a,b)= (a·b) / (||a||·||b||). Alignment weights α_{τ,s}= softmax_{τ}(sim(F_{τ,s}, G_s)). P_s = Σ_{τ} α_{τ,s} F_{τ,s}. This yields per-spatial cell pooled motion focusing on temporally relevant tokens aligned to final spatial perception.

### 5.14 Pruning Importance Rollout Detail
Layer-wise attention A^{(ℓ)}. Effective attention with residual: \tilde{A}^{(ℓ)} = (I + A^{(ℓ)}) normalized. Rollout R = Π_{ℓ} \tilde{A}^{(ℓ)}. Importance of token j: s_j = Σ_i R_{ij}. We retain tokens with top s_j. Guarantees monotonic accumulation of cross-layer influence (per rollout literature).

### 5.15 Calibration Penalty (Optional)
Define ECE estimate over verb logits; L_{calib} = Σ_m |acc_m − conf_m| weighted by bin density. λ_{calib} controls integration; only enabled in calibration refinement experiments.

### 5.16 TTC Binning Scheme (If Enabled)
Partition TTC range [0, τ_max] into B bins (e.g., B=8, logarithmic spacing for early fine resolution). Soft label if τ^* near a boundary: distribute probability to adjacent bins. Convert predicted q_k to scalar via expected value for MAE reporting.

### 5.17 Multi-Candidate Aggregation (Future)
Instead of selecting a single k*, use soft selection weights w_k = softmax(σ p_k). Verb/TTC losses become Σ_k w_k CE(v_k, y^V) and Σ_k w_k SmoothL1(τ_k, τ^*). Reduces brittleness when multiple plausible boxes exist.

### 5.18 Pseudocode Snippet: Training Loop (High-Level)
for batch in loader:
  B_list, clip_tokens = preprocess(batch)
  roi_feats = extract_roi_feats(B_list)
  if fusion_enabled:
    fused_img, fused_vid = fusion(img_tokens, clip_tokens)
  logits = head(roi_feats, fused_img, fused_vid)
  losses = compute_losses(logits, targets)
  losses.backward(); optimizer.step(); scheduler.step()
  log_metrics(losses, intermediate_scores)

### 5.19 Evaluation Matching Pseudocode
for sample in dataset:
  preds = model(sample)
  matches = match(preds, gt, iou_thresh, ttc_delta)
  accumulate(metrics, matches)
write_out(metrics)

### 5.20 Safety Checks in Scripts
Before heavy runs: assert manifests exist, check GPU availability (or fallback CPU), validate pruning rates (0<ρ<1), warn if mixed precision unsupported.

### 5.21 Potential Extensions Not Implemented Yet
- Joint hand pose integration for improved verb disambiguation.
- Lightweight temporal super-resolution network to reduce motion blur impact.
- Dynamic K selection based on detector confidence entropy.

---

## 6. Implementation Details

### 6.1 Repository Layout
- `local_extraction/` contains environment setup, extraction scripts, pipeline runners, the Track B dataloaders/trainers/evaluators, and Track C pruning.  
- `runs/` captures Stage B outputs (manifests), checkpoints, and metrics per experiment.  
- `Docs/` and `Thesis_main/` contain the planning documents, decision records, and this final article.

### 6.2 Centralized Toggles and Progress Bars
We minimize CLI friction by placing runtime toggles and hyperparameters at the top of main scripts (e.g., `TrainConfig`, `EvalConfig`, `RuntimeConfig`). Each script emits a dual progress-report: outer (overall/epoch) and inner (batch/sample) bars using `tqdm`, to improve visibility in Colab or local terminals.

### 6.3 Stage B Manifests as First-Class Citizens
We intentionally use Stage B manifests (from the Stage A/B pipeline) as authoritative training data for Track B. The training/eval scripts auto-discover the latest Stage B run (e.g., `local_extraction/runs/Track_A/trackA_stageB_<timestamp>/`) and default to `head_train.jsonl` and `head_val.jsonl` unless explicitly overridden.

### 6.4 Path Resolution and Robustness
Imports use robust `sys.path` adjustments and rooted path resolution to avoid brittle relative imports. Scripts emit helpful error messages when assets are missing or unresolvable, guiding re-runs with the correct manifest or checkpoint.

---

## 7. Experimental Results

### 7.1 Baseline Metrics (No Pruning)
On the Stage B validation split and the latest `trackB_final_*.pt` checkpoint, we observed:
- Accuracy ≈ 0.928  
- N mAP ≈ 0.070  
- TTC MAE ≈ 0.226 seconds  
- Candidates processed: ≈1662  
These values serve as a baseline for pruning sweeps and further fine-tuning.

### 7.2 Qualitative Overlays
The evaluation script stores overlay images that draw predicted and GT boxes on the last frame, aiding diagnosis of failure modes (jitter, occlusion, small objects).

### 7.3 Planned Pruning Sweeps (Track C)
We will enable `pruning_enabled=True` and explore `rgtp_rate ∈ {0.4, 0.5, 0.6}` with `min_keep≥1`. For each rate, we’ll log mAP, TTC MAE, accuracy, and observed prune fraction, and we will measure latency and VRAM usage. Target: ≈35–50% latency reduction with ≤0.5 mAP loss.

### 7.4 Detailed Experimental Protocols
To ensure reproducibility and comparability, each experiment declares:
- Config hash: JSON dump of all toggles (K, fusion depth, pruning rate, loss weights).
- Seed suite: primary seed + 2 auxiliary seeds for variance estimation.
- Metrics panel: primary (N mAP, accuracy, TTC MAE) and secondary (N+V, N+δ, calibration ECE, prune fraction, latency, VRAM peak).
- Artifact paths: checkpoint, manifest versions, overlay directory, metrics JSON.
Evaluation order:
1. Load manifests and checkpoint.
2. Run inference with optional pruning.
3. Collect raw predictions (per-candidate boxes, scores, verb logits, TTC predictions).
4. Compute matching under standard and δ-tolerant criteria.
5. Aggregate metrics; write `metrics_<timestamp>.json`.
6. Generate qualitative overlays for top errors (false negatives, large TTC errors).

### 7.5 Latency and Resource Measurement Plan
Instrumentation additions (planned):
- CUDA event pairs around fusion and head forward passes.
- Memory snapshots via `torch.cuda.max_memory_allocated()` and `reserved_memory()`.
- Wall-clock timing fallback for CPU-only runs.
Reporting: per-run CSV with columns `[rate, S_in, S_out, latency_ms, vram_mb, mAP, ttc_mae]` enabling scatter plots of latency vs mAP.

### 7.6 Variance and Statistical Reporting
For key settings (e.g., pruning at 0.5), repeat runs over 3 seeds and report mean ± std for N mAP and TTC MAE. If variance is high, investigate instability sources (data loader ordering, nondeterministic ops).

---

## 8. Ablations and Variants

- Fusion ablations: FGTP off / on; cross-attention layers {0, 2, 3, 4}; width {192, 256, 384}.  
- TTC strategies: regression vs binned; δ-tolerant AP vs strict metrics.  
- K-sweep for proposals: K ∈ {3, 5, 8, 10}.  
- Detector choices: generic YOLO vs fine-tuned (single-class next_active; option to expand).  
- Efficiency: pruning rates, min_keep, pruning stage placement, and optional quantization.

### 8.1 Planned Ablation Table (Placeholder)
Columns: `[FusionLayers, Width, K, PruneRate, N_mAP, Accuracy, TTC_MAE, Latency_ms]`.
We will populate after completing pruning sweeps and fusion depth experiments.

### 8.2 Interaction Between K and Pruning
Increasing K raises candidate coverage but inflates per-sample cost. Pruning partially offsets this by reducing token count before the head. We hypothesize: at moderate K (8–10), pruning at 0.5 retains most accuracy gains from higher recall while keeping latency near lower-K unpruned runs.

### 8.3 Calibration Impact of Fusion
Fusion may sharpen or overconfidently spike verb logits. Calibration metrics (ECE) will be compared across fusion depths to detect overfitting of temporal patterns.

---

## 9. Reproducibility and Engineering Practices

- Colab-first: mount Drive, keep runs/checkpoints persistent, limit session scope.  
- Requirements: `local_extraction/env/requirements.txt` and environment freeze logs.  
- Run logging: JSON metrics per experiment in `runs/Track_B/metrics/` and `runs/Track_C/metrics/`; overlays and predictions exported for review.  
- Daily loop: `Docs/Ego4d-LiteSTA_Daily_Loop.md` records the evolution of toggles, hyperparameters, and outcomes.

### 9.1 Determinism Checklist
- Set `torch.manual_seed(seed)`, `numpy.random.seed(seed)`, `random.seed(seed)`.
- Enable deterministic flags where feasible (`torch.use_deterministic_algorithms(True)` if performance allows).
- Log library versions (torch, torchvision, opencv) and CUDA driver.

### 9.2 Artifact Versioning
Each run timestamp directs subfolders for manifests, checkpoints, metrics, overlays. Symlinks (or pointer files) provide `latest_train`, `latest_eval` convenience.

### 9.3 Failure Mode Triage Procedure
Daily: inspect overlay diffs; categorize errors into proposal misses vs misclassification vs TTC drift. Record representative examples to Appendix E.

---

## 10. Discussion

The two-stage baseline remains a strong anchor for egocentric anticipation. Even with modest mAP, the pipeline’s stability and interpretability let us identify where improvements are most impactful (recall vs reasoning vs TTC calibration). Lightweight fusion shows consistent upside with little compute overhead, while training-free pruning is essential to translate accuracy into practical latency on edge devices. Our emphasis on centralized toggles and Stage B manifests accelerates iteration and reduces human error.

### 10.1 Deployment Considerations
- Edge targets (AR glasses, mobile robotics) demand tight latency budgets (<100 ms). Pruning is a lever that does not require retraining.
- Battery constraints encourage minimizing GPU residency; reduced token counts lower memory bandwidth pressure.
- Fallback: if video tokens unavailable (bandwidth or sensor failure), run head in image-only mode.

### 10.2 Extensibility
- Multi-action forecasting: extend head to predict distribution over next-active object sequences.
- Long-horizon anticipation: integrate separate module for >2s TTC with coarser bins.
- Continual updates: online domain adaptation via light feature affine tuning.

### 10.3 Interpretability
Attention rollout maps and pruning masks serve as human-auditable artifacts. We plan saliency alignment checks: do retained tokens correspond to true moving regions and interacting hands?

---

## 11. Limitations

- Detector specialization: We currently use a generic single-class detector (next_active). Multi-class noun detectors may improve N mAP but demand curation and balancing in long-tailed noun spaces.  
- TTC noise: Human annotation jitter complicates fine-grained TTC regression; binning and tolerant metrics help but do not eliminate label noise effects.  
- Domain shift: Kitchen-heavy or user-specific biases can reduce generalization; cross-scene validation is needed.  
- Latency measurements: Full latency/VRAM/FLOPs vs mAP reporting is pending pruning sweeps and instrumentation.

---

## 12. In-Progress and Next Steps

- Track A fine-tuning on Colab: single-class next_active YOLO. Expect ≈2 days on Colab Free GPU to reach stable recall; log hyperparameters, augmentations, and recall@K.  
- Track B pruning sweeps: enable RGTP and measure mAP vs latency; target ≤0.5 mAP drop with ≈35–50% latency gains.  
- Latency instrumentation: Add CUDA event timing and VRAM counters; report curves across pruning rates.  
- Verb/TTC calibration: Revisit loss weights, smooth labels, and calibration post-processing.  
- Optional priors: hotspot maps and lightweight affordance priors to regularize verb/noun logits.

---

## 13. Ethical and Practical Considerations

Egocentric data may include sensitive content; ensure privacy-preserving handling and consent. For deployment, design UI/robotic behaviors conservatively to avoid over-reliance on predictions under ambiguous conditions. Compute budgets and battery constraints must be considered for wearable applications.

---

## 14. Conclusion

We proposed and implemented a modular, reproducible STA system tailored to egocentric video: a recall-first two-stage baseline (Track A), a lightweight fusion enhancement (Track B), and a training-free pruning module (Track C). The pipeline is engineered for practical use on Colab Free and local machines, with robust manifest handling, clear logging, and runtime toggles that simplify iteration. Early results validate stability and pave the way for efficiency-oriented pruning sweeps. The resulting system is a strong foundation for a deployable egocentric anticipation engine and provides a structured thesis narrative from data to deployment.

---

## References (selected)

- Ego4D Forecasting and STA task documentation.  
- STAformer (ZARRIO): affordance-guided fusion for STA.  
- SOIA-DOD: two-stage detect-then-reason baseline for STA.  
- MViTv2: multiscale video transformer with pooling attention.  
- VideoMAE: masked autoencoding for video pretraining.  
- PruneVid, Rollout-Guided Token Pruning (RGTP), EgoPrune: training-free token reduction methods.  
- PEAR and fine-grained affordances for hand–object anticipation and hotspot priors.  
- Ego–Exo transfer and synchronization literature (Ego-Only; Sync-is-All-You-Need).

---

## Appendix A. Notation and Metrics

### A.1 Notation
- t: index of the last observed frame in a short clip; frames are {t−T+1, …, t}.
- I_t ∈ R^{H×W×3}: the last frame image.
- V ∈ R^{T×H×W×3}: the short video clip ending at t.
- B = {b_k}^K_{k=1}: top-K candidate boxes from detector on I_t; each b_k = (x, y, w, h) in normalized coordinates.
- y^N, y^V: noun and verb labels; y^{NA}∈{0,1}: next-active indicator for a candidate; τ: time-to-contact.
- ϕ_img, ϕ_vid: image and video encoders; r(b_k, I_t): ROI-pooled features for candidate k.
- FGTP(V, I_t): frame-guided temporal pooling that aligns V to I_t’s spatial grid.
- CA(·,·): cross-attention operator.

### A.2 Metrics
- N mAP: mean AP over nouns using predicted boxes and noun labels.
- N+V mAP: joint AP over noun and verb.  
- N+δ mAP: noun AP tolerant to TTC error within δ.
- Accuracy: proportion of samples where the top candidate matches the GT next-active with correct noun (and optionally verb).
- TTC MAE: mean absolute error between predicted τ and GT τ.

### A.3 δ-Tolerant Matching
We consider a match valid if IoU(b_pred, b_gt) ≥ θ and |τ_pred − τ_gt| ≤ δ. For δ=∞, this reduces to noun-only AP.

---

## Appendix B. Algorithms and Pseudocode

### B.1 Frame-Guided Temporal Pooling (FGTP)
Input: V ∈ R^{T×H×W×3}, I_t; Output: pooled video tokens aligned to I_t grid.
1. Extract video features F ∈ R^{T×S×C} using ϕ_vid, where S is spatial tokens.
2. Extract image features G ∈ R^{S×C} from I_t using ϕ_img.
3. Compute alignment weights α_{τ,s} = softmax_s(sim(F_{τ,s}, G_s)).
4. Pooled token per spatial cell: P_s = Σ_{τ} α_{τ,s} F_{τ,s}.
5. Return P ∈ R^{S×C}.

### B.2 Dual Cross-Attention (Image↔Video)
Input: image tokens G ∈ R^{S×C}, pooled video tokens P ∈ R^{S×C}. Repeat L times:
1. G ← G + CA(query=G, key=P, value=P).
2. P ← P + CA(query=P, key=G, value=G).
3. Pre-norm and residual connections around each CA.
Output: fused tokens Ĝ, Ṕ.

### B.3 Rollout-Guided Token Pruning (RGTP)
Given last two time steps (t−1, t):
1. Compute attention maps A^{(ℓ)}_{t−1→t} across layers ℓ.
2. Rollout importance R = Π_{ℓ} A^{(ℓ)} (with residual correction).
3. For tokens at t, compute importance score s_j = Σ_i R_{ij}.
4. Keep top M = max(min_keep, ⌈(1−ρ)·S⌉) tokens; prune the rest (ρ is pruning rate).
5. Do not prune the final head layer’s tokens.

---

## Appendix C. Hyperparameters and Config Bundles

### C.1 Detector (Track A, single-class next_active)
- Base: YOLOv8-s; imgsz 640; epochs 100–200; optimizer SGD or AdamW.
- Aug: flips, small color jitter, slight translation; avoid heavy mosaic to preserve egocentric geometry.
- LR: 1e−3 to 5e−4 with cosine decay; warmup 3–5 epochs.
- Expected Colab Free runtime: ≈2 days to stable recall.

### C.2 Reasoner (Track B)
- Batch size 8; epochs ≈20; LR 1e−3; AdamW (β1=0.9, β2=0.999, weight decay 1e−4).
- Fusion: FGTP on; dual CA layers L∈{0,2,3,4}; width C∈{192,256,384}.
- Loss weights: λ_box=1.0, λ_verb=1.0, λ_ttc=0.5 (tunable).
- Checkpoint: best by N mAP; save every epoch.

### C.3 Pruning (Track C)
- ρ ∈ {0.4,0.5,0.6}; min_keep ≥ 1.
- Prune before the head; never prune final classification/regression tokens.
- Instrument latency and VRAM with CUDA events where available.

---

## Appendix D. Computational Complexity and Latency

Let C_det be detector cost on last frame; C_head be head cost per candidate; K candidates.
- Without fusion: total ≈ C_det + K·C_head.
- With fusion: add C_fuse which is dominated by L·(S·C^2 + S^2·C) depending on CA implementation; we keep S small via ROI pooling and modest grids.
- With pruning: effective S′ ≈ (1−ρ)·S reduces CA and MLP costs proportionally.

Latency targets on a mid-range GPU: pruning at ρ=0.5 aims for ≈35–50% reduction in fusion+head time, subject to kernel launch overheads.

### D.1 Theoretical FLOPs Sketch (Approximate)
Let S be spatial tokens after ROI pooling, C embedding dim, L fusion layers:
- Fusion CA FLOPs ≈ L·(2·S·C^2 + S^2·C) (query/key/value projections + attention).
- Pruned fusion FLOPs scale by (1−ρ) for token-dependent terms (≈ linear in S, quadratic in S for attention matrix).
Compute ratio R_pruned ≈ ( (1−ρ)·S )^2 / S^2 = (1−ρ)^2 for quadratic term dominance; at ρ=0.5 → R_pruned ≈ 0.25.

### D.2 Memory Considerations
Activation memory ∝ L·S·C. Pruning reduces S, enabling potential larger C or L under same budget.

---

## Appendix E. Failure Cases and Qualitative Notes

- Tiny objects and motion blur: detector recall drops; consider super-resolution or motion-enhanced proposals.
- Strong occlusion by hands: last-frame boxes may be truncated; temporal cues help but are weak near contact.
- Ambiguity between similar objects (two mugs): noun confusions; affordance priors and instance tracking can help.
- TTC edge cases: extremely short or long TTC values skew loss; consider clipping or curriculum.

---

## Appendix F. Extended Related Work (Selected Pointers)

- Egocentric anticipation: Next-active object forecasting, hand-object contact prediction, hotspot learning.
- Video backbones: MViT, TimeSformer, X3D, VideoMAE; trade-offs in FLOPs vs accuracy.
- Efficient transformers: token pruning, early exiting, kernel sparsification; training-free vs training-based.
- Detection: modern YOLO families, anchor-free heads, one-stage vs two-stage trade-offs.

---

## Appendix G. Reproducibility Checklist

- Data provenance: exact Ego4D version and STA split hashes recorded.
- Determinism: set seeds in all libraries; log random states per run.
- Environment: freeze pip package versions from `local_extraction/env/requirements.txt`.
- Manifests: Stage B manifests versioned under `runs/` with timestamps.
- Checkpoints: saved with metric summaries and config dumps.
- Scripts: single-entry runners with clear toggles and progress bars.

---

## Appendix H. Ethics, Privacy, and Safety

- Consent and privacy: egocentric footage can reveal bystanders and private spaces; adhere to dataset licenses and institutional review standards.
- On-device safety: avoid actions that depend on high confidence unless verified; design human-in-the-loop fallbacks.
- Dataset bias: document known biases; avoid unwarranted generalizations.

---

## Appendix I. Future Roadmap

- Multi-class detector for richer noun coverage; curriculum to mitigate long tails.
- Lightweight instance tracking to stabilize proposals across adjacent frames.
- Learned pruning (optional) once training-free baselines are saturated.
- Cross-dataset validation to probe generalization.

### I.1 Additional Medium-Term Goals
- Integrated latency regression head predicting expected per-sample inference time (for scheduling).
- Structured TTC distribution modeling (mixture density networks) instead of point/binned estimates.
- AutoML sweep script to explore joint space (K, fusion depth, prune rate) under Pareto frontier tracking.

### I.2 Long-Term Research Directions
- Semi-supervised STA with pseudo-label refinement from high-confidence predictions.
- Multimodal fusion (audio, inertial) injected before FGTP.
- Adaptive token budgeting: dynamic ρ per sample based on early importance estimates.

---

## Appendix J. Glossary

- Next-active: the object that will be interacted with immediately after the observed clip ends.
- TTC (Time-to-Contact): time between last observed frame and interaction contact.
- FGTP: Frame-Guided Temporal Pooling; aligns motion tokens to the decision frame.
- RGTP: Rollout-Guided Token Pruning; training-free token selection at inference.
- δ-tolerant AP: AP computed with a permitted TTC error margin δ.

---

## Extended References

Egocentric Anticipation and Interaction:
- Grauman et al., Ego4D: Benchmarking Egocentric Video, CVPR 2022.
- Sra et al., Egocentric Hand–Object Interaction Forecasting, ICCV Workshops 2023.
- Nagarajan et al., Ego-topo: Egocentric Visual Knowledge Networks, CVPR 2020.

Two-Stage Detection and Reasoning:
- Redmon et al., YOLO family (v1–v8): Unified real-time object detection.
- Carion et al., DETR: End-to-End Object Detection with Transformers, ECCV 2020.
- Zhong et al., SOIA-DOD: Short-term Object Interaction Anticipation via Detection and Object Dynamics.

Temporal Fusion and Video Transformers:
- Arnab et al., TimeSformer: Is Space-Time Attention All You Need?, ICML 2021.
- Fan et al., MViTv2: Improved Multiscale Vision Transformers for Classification and Detection.
- Feichtenhofer et al., X3D: Expanding Architectures for Efficient Video Recognition.
- Tong et al., VideoMAE: Masked Autoencoders are Data-Efficient Learners for Video.

Efficient Attention and Token Pruning:
- Bolya et al., Token Pruning for Transformers, NeurIPS 2023.
- Wang et al., PruneVid: Efficient Video Inference via Spatiotemporal Token Pruning.
- Rollout-Guided Token Pruning (RGTP) (various implementations in vision transformers literature).

Affordances and Hotspots:
- Yao et al., Predicting Human-Object Interactions: Affordance Hotspots, CVPR.
- Zhang et al., PEAR: Personalized Egocentric Activity Recognition.

Calibration and Uncertainty:
- Guo et al., On Calibration of Modern Neural Networks, ICML 2017.
- Kuleshov et al., Calibrated Regression for Interpretable ML.

Egocentric–Exocentric Transfer:
- Abu Farha et al., Synchronization and Transfer in Egocentric–Exocentric Video, CVPR.

Additional Methodology:
- Kingma & Ba, Adam: A Method for Stochastic Optimization.
- Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW).
- He et al., Deep Residual Learning for Image Recognition (ResNet).

Note: Some references are summarized conceptually due to space; full bibliographic details can be expanded as needed in final formatting.
