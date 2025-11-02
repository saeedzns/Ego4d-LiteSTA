# Glossary (Thesis Plan v2, Review, and Notebooks)

Short, plain-language definitions of terms used across thesis_plan_tracks_ABC_v2.md, review.md, and the Colab notebooks. Each item is two lines: what it is and why it matters here.

## Dataset & Splits
- Ego4D — A large egocentric (first-person) video dataset with rich tasks and labels.
  Used for forecasting/anticipation (STA, LTA, etc.) and evaluation in this thesis.

- FHO (Forecasting Hands and Objects) — Ego4D task family centered on hand–object interactions.
  Covers STA, LTA, OSCC‑PNR, SCOD, Hands, NLQ, VQ, AV, and Moments splits.

- Video UID / Clip UID / Clip ID — Stable identifiers for full videos and subclips.
  They link annotations, features, boxes, and metrics back to the same data.

## Core Tasks
- STA (Short‑Term Object Interaction Anticipation) — Predict next‑active object box, verb, noun, and time‑to‑contact at the last frame.
  Central thesis task; requires spatial, semantic, and temporal reasoning together.

- LTA (Long‑Term Anticipation) — Forecast upcoming actions further into the future.
  Complements STA with longer horizons and sequence‑level reasoning.

- OSCC‑PNR (Object State Change Classification — Point of No Return) — Detect the decisive frame when an object’s state changes.
  Supervision centers on the PNR frame and short clips around the transition.

- SCOD (State Change Object Detection) — Localize the object undergoing state change at key pre/PNR/post frames.
  Trains detectors to find hands/objects/tools precisely around state change.

- NLQ (Natural Language Queries) — Given a question about a video, find the matching time span.
  Bridges text and time with query templates, times, and frames.

- VQ (Visual Queries) — Track the referred object in space–time from a visual crop.
  Provides a response trajectory (per‑frame boxes) and a query frame/crop.

- AV (Audio‑Visual Segments) — Segments where the camera wearer speaks or interacts socially.
  Aligns voice/talking segments with video frames for cross‑modal modeling.

- Moments — Dense temporal action labels across a clip from multiple annotators.
  Useful for temporal segmentation, retrieval, and evaluation of action coverage.

## Annotations & Fields
- Time/Frame Fields — `video_start/end_{sec,frame}`, `clip_start/end_{sec,frame}` define ranges.
  Convert with FPS from `video_metadata`; keep clip vs video time bases distinct.

- Action Windows — `action_*` fields (sec/frame) describe the target interaction interval.
  Clip‑scoped variants (`action_clip_*`) align to the sampled clip’s timeline.

- Context Windows — `interval_start/end_{sec,frame}` provide a broader temporal context.
  Used by LTA/STA to encode what’s visible around the action.

- Key Frames — `pre_45`, `pre_30`, `pre_15`, `pre_frame`, `contact_frame`, `post_frame`, `pnr_frame`.
  Anchor supervision around imminent contact and the decisive PNR moment.

- Boxes — `box` as [x1,y1,x2,y2], or `bbox` as {x,y,width,height}, or `boxes` list per frame.
  Represent hand/object locations; ensure you use the expected convention per file.

- Objects (STA) — Per‑object record with `box`, `verb_category_id`, `noun_category_id`, `time_to_contact`.
  Drives multi‑head training: localization, verb/noun classification, and TTC prediction.

- Narrations & Taxonomies — Narration text aligned to time; noun/verb taxonomies normalize labels.
  Enable mapping free‑form text to consistent class vocabularies.

- Manifests — CSV pointers to canonical S3 locations of full JSONs and artifacts.
  Ensure reproducible downloads and exact file linkage beyond the samples.

## Models & Features
- YOLOv8 / YOLOv9‑lite — Real‑time object detectors used on the last frame (Stage‑A).
  Provide top‑K candidate boxes with high recall to feed the reasoning head.

- MViTv2 — Modern video transformer backbone with efficiency improvements.
  Appears in related work; a candidate for compact video features if needed.

- VideoMAE — Self‑supervised video pretraining via masked autoencoding.
  Supplies robust video features with fewer labels; relevant to baselines.

- TimeSformer — Early pure‑transformer video backbone (space/time attention).
  Used by some STA systems to capture motion cues from short clips.

- DINOv2 — Strong image backbone used for high‑res last‑frame features.
  Lifts appearance quality for box/verb/noun heads in two‑branch designs.

- Omnivore Video SwinL FP16 — Precomputed action features (Swin‑based) stored as .pt.
  Allow feature‑based baselines without end‑to‑end video training in Colab.

- Fast R‑CNN Head — Multi‑task ROI head for box, classes, and sometimes TTC.
  Adapted for STA to jointly predict next‑active box, verb, noun, and timing.

## Fusion & Attention
- Frame‑Guided Temporal Pooling (FGTP) — Pools short‑clip tokens onto the last‑frame grid.
  Aligns motion evidence with the decision frame using a lightweight mapping.

- Dual Image↔Video Cross‑Attention — Let image tokens refine video tokens and vice‑versa.
  Adds 2–4 layers at small width to lift N+V and N+δ with modest compute.

- Attention Rollout — Aggregate attention across layers to estimate token importance.
  Used to score tokens at t−1, track them to t, and prune low‑value ones.

## Efficiency & Pruning
- Token Pruning — Drop low‑importance tokens before the head to save FLOPs/VRAM.
  Done at inference time; target large latency cuts with minimal mAP drop.

- Rollout‑Guided Token Pruning (RGTP) — Use attention rollout to rank tokens for pruning.
  Training‑free switch that aims for ~35–50% latency reduction within ≤0.5 mAP.

- PruneVid / EgoPrune — Prior pruning strategies for streaming/egocentric video.
  Related art that motivates our training‑free, STA‑specific pruning strategy.

## Losses & Metrics
- mAP / AP — Mean Average Precision for detection/anticipation across classes.
  Evaluated as N mAP (noun), N+V mAP (noun+verb), N+δ mAP (noun with TTC tolerance).

- Overall top‑5 mAP — Composite measure used in STA leaderboards.
  Captures performance across boxes, labels, and timing jointly.

- GIoU / L1 / SmoothL1 / CE — Box overlap and regression losses plus cross‑entropy.
  Combined as L_box + L_verb + L_ttc (+ optional calibration loss) for STA.

- Candidate Recall@K — Fraction of GT objects covered by the top‑K proposals.
  Key stability metric for two‑stage STA; must be high so the head sees the right box.

- TTC (Time‑to‑Contact) — Seconds until interaction starts; can be binned or regressed.
  The temporal head predicts TTC at the decision frame alongside N/V.

## Engineering & Repro
- Two‑Stage STA — Stage‑A detector proposes boxes; Stage‑B head reasons per box.
  Stabilizes training and makes failures interpretable (proposals vs reasoning).

- ROI Features — Per‑box features pooled from a backbone for head prediction.
  Feed verb/noun/TTC heads; optionally concatenate pooled video tokens.

- FLOPs / VRAM / Latency — Compute, memory, and speed metrics for deployment.
  Reported alongside mAP to show the speed/quality tradeoff for AR/robotics.

- AMP (Automatic Mixed Precision) — Float16/32 mixed training/inference for speed and memory.
  Important on Colab GPUs to fit models and reach real‑time.

- Colab + Drive — Google Colab runtime with Google Drive for persistent storage.
  All pipelines/notebooks assume Drive mounts for data, logs, and checkpoints.

- COCO / YOLO Formats — Common detection label formats (JSON vs TXT per image).
  Detector training accepts either; manifests map STA samples to these labels.

- Manifests & Seeds — Exact file lists and RNG seeds saved with results.
  Enable full reproduction of experiments and consistent comparisons.

- Calibration Loss (`L_calib`) — Optional term to calibrate predicted probabilities.
  Helps match scores to likelihoods; useful when ranking top‑K candidates.

- Binning vs Regression (TTC) — Classify TTC buckets or regress a continuous value.
  Binning is simpler/robust; regression can be more precise given enough data.

- STAformer — A recent STA system using dual cross‑attention and affordance priors.
  Serves as inspiration for lightweight fusion and affordance‑aware reasoning.

- SOIA‑DOD — A competitive STA approach (Second‑Order Interaction Anticipation + Detections).
  Represents the state of the art the thesis compares against qualitatively.

- PEAR (Affordance) — Method using fine‑grained affordances to guide anticipation.
  Motivates adding priors like hotspots and zone‑based expectations.

- EgoTopo — Topological zones of a scene used as environmental context.
  Zone priors (verbs/nouns per zone) can regularize STA predictions.

