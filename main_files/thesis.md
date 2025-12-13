# Tesi

*Document Type: DOCX*

## Table of Contents

- [Chapter 1: Introduction](#chapter-1-introduction)
  - [1.1 Background](#11-background)
  - [1.2 Problem Statement](#12-problem-statement)
  - [1.3 Objectives of the Study](#13-objectives-of-the-study)
    - [1.3.1 Contributions](#131-contributions)
  - [1.4 Scope and Limitations](#14-scope-and-limitations)
  - [1.5 Research Methodology](#15-research-methodology)
  - [1.6 Structure of the Thesis](#16-structure-of-the-thesis)
- [Chapter 2 – Literature Review](#chapter-2--literature-review)
  - [2.1 Ego4D Forecasting Benchmark and STA Task](#21-ego4d-forecasting-benchmark-and-sta-task)
    - [2.1.1 Forecasting benchmark and STA definition](#211-forecasting-benchmark-and-sta-definition)
    - [2.1.2 Ego4D dataset characteristics](#212-ego4d-dataset-characteristics)
  - [2.2 Baseline STA Challenge Solutions](#22-baseline-sta-challenge-solutions)
    - [2.2.1 STAformer and AFF-ttention](#221-staformer-and-aff-ttention)
    - [2.2.2 SOIA-DOD: disentangled detection and anticipation](#222-soia-dod-disentangled-detection-and-anticipation)
  - [2.3 Efficient Video Backbones and Pre-Training](#23-efficient-video-backbones-and-pre-training)
    - [2.3.1 Multiscale Vision Transformers (MViTv2)](#231-multiscale-vision-transformers-mvitv2)
    - [2.3.2 VideoMAE: data-efficient video self-supervision](#232-videomae-data-efficient-video-self-supervision)
  - [2.4 Token-Level Efficiency for Video and Egocentric Streams](#24-token-level-efficiency-for-video-and-egocentric-streams)
    - [2.4.1 PruneVid](#241-prunevid)
    - [2.4.2 Rollout-Guided Token Pruning (RGTP)](#242-rollout-guided-token-pruning-rgtp)
    - [2.4.3 EgoPrune](#243-egoprune)
  - [2.5 Hand–Object Interaction, Affordances, and Hotspots](#25-handobject-interaction-affordances-and-hotspots)
    - [2.5.1 PEAR](#251-pear)
    - [2.5.2 Fine-grained affordance annotation](#252-fine-grained-affordance-annotation)
  - [2.6 Ego–Exo Transfer, Ego-Only Learning, and Retrieval-Augmented Models](#26-egoexo-transfer-ego-only-learning-and-retrieval-augmented-models)
    - [2.6.1 Ego–exo joint learning survey](#261-egoexo-joint-learning-survey)
    - [2.6.2 Synchronization-based exo→ego transfer](#262-synchronization-based-exoego-transfer-for-temporal-segmentation)
    - [2.6.3 Ego-Only](#263-ego-only)
    - [2.6.4 Early motion transfer](#264-early-motion-transfer-between-egocentric-and-exocentric-views)
    - [2.6.5 REAR](#265-retrieval-augmented-egocentric-action-recognition-rear)
  - [2.7 Community Resources and Challenge Ecosystem](#27-community-resources-and-challenge-ecosystem)
    - [2.7.1 EgoVis workshop](#271-egovis-workshop-and-challenge-ecosystem)
    - [2.7.2 Curated lists](#272-curated-lists-and-awesome-repositories)
  - [2.8 Summary and Positioning of Ego4D-LiteSTA](#28-summary-and-positioning-of-ego4d-litesta)
- [Chapter 3: Problem Definition and Data](#chapter-3-problem-definition-and-data)
  - [3.1 Task Definition (STA)](#31-task-definition-sta)
  - [3.2 Dataset, Splits, and Annotations (Ego4D-STA v2)](#32-dataset-splits-and-annotations-ego4d-sta-v2)
  - [3.3 Metrics](#33-metrics)
  - [3.4 Data Manifests and Label Alignment Decisions](#34-data-manifests-and-label-alignment-decisions)
- [Chapter 4: System Overview (Tracks A / B / C)](#chapter-4-system-overview-tracks-a--b--c)
  - [4.1 Design Principles](#41-design-principles)
  - [4.2 Track A Overview (Proposals + Manifests)](#42-track-a-overview-proposals--manifests)
  - [4.3 Track B Overview (Lightweight Fusion Head)](#43-track-b-overview-lightweight-fusion-head)
  - [4.4 Track C Overview (Training-Free Pruning)](#44-track-c-overview-training-free-pruning)
- [Chapter 5: Methodology](#chapter-5-methodology)
  - [5.1 Track A: Candidate Generation and Recall@K](#51-track-a-candidate-generation-and-recallk)
  - [5.2 Track B: Tokenization, Fusion (FGTP + Dual Cross-Attention), and Heads](#52-track-b-tokenization-fusion-fgtp--dual-cross-attention-and-heads)
  - [5.3 Track C: Rollout-Guided Token Pruning (RGTP)](#53-track-c-rollout-guided-token-pruning-rgtp)
- [Chapter 6: Implementation and Engineering](#chapter-6-implementation-and-engineering)
  - [6.1 Local/Colab Workflow and Tooling](#61-localcolab-workflow-and-tooling)
  - [6.2 Configuration System and Reproducible Runs](#62-configuration-system-and-reproducible-runs)
  - [6.3 Data Validation and Failure Handling](#63-data-validation-and-failure-handling)
- [Chapter 7: Experimental Setup](#chapter-7-experimental-setup)
  - [7.1 Baselines and Compared Variants](#71-baselines-and-compared-variants)
  - [7.2 Training Protocols and Hyperparameters](#72-training-protocols-and-hyperparameters)
  - [7.3 Evaluation Protocol](#73-evaluation-protocol)
- [Chapter 8: Results](#chapter-8-results)
  - [8.1 Track A Results (Recall@K)](#81-track-a-results-recallk)
  - [8.2 Track B Results (N / N+V / N+δ, TTC)](#82-track-b-results-n--nv--n-δ-ttc)
  - [8.3 Track C Results (Accuracy–Latency Pareto)](#83-track-c-results-accuracylatency-pareto)
- [Chapter 9: Ablations and Analysis](#chapter-9-ablations-and-analysis)
  - [9.1 Proposal K Sweeps](#91-proposal-k-sweeps)
  - [9.2 Fusion Depth and Token Dimensionality](#92-fusion-depth-and-token-dimensionality)
  - [9.3 TTC Modeling (Regression vs Bins)](#93-ttc-modeling-regression-vs-bins)
  - [9.4 Priors (Hotspots, CLIP) and Their Impact](#94-priors-hotspots-clip-and-their-impact)
- [Chapter 10: Discussion, Limitations, and Future Work](#chapter-10-discussion-limitations-and-future-work)
- [Chapter 11: Reproducibility Checklist](#chapter-11-reproducibility-checklist)
- [Chapter 12: Ethical, Legal, and Social Implications (ELSI)](#chapter-12-ethical-legal-and-social-implications-elsi)
- [Chapter 13: Conclusion](#chapter-13-conclusion)
- [References](#references)
- [Appendices](#appendices)


  





# Chapter 1: Introduction

## 1.1 Background

Short‑Term Object Interaction Anticipation (STA) in egocentric video asks a system to predict, one or two seconds before contact, **where** the next‑active object will be in the last frame, **what** verb and noun describe the interaction, and **when** (time‑to‑contact, TTC) the interaction begins. Ego4D provides the largest, most diverse STA benchmark with detailed verb/noun/TTC annotations. Unlike third‑person action recognition, egocentric footage is dominated by wearer motion, hand jitter, and cluttered near‑field scenes, making spatial localization and temporal reasoning tightly coupled. Recent STA solutions lean on heavy video transformers or multimodal fusion, achieving accuracy at the cost of compute, latency, and reproducibility on modest hardware. This motivates **Ego4D‑LiteSTA**: a modular, lightweight, and reproducible three‑track pipeline engineered to preserve accuracy while targeting low computational overhead.

## 1.2 Problem Statement

STA couples spatial detection, temporal reasoning, and semantic understanding in a resource‑constrained setting. High‑performing models typically depend on transformer video encoders with high FLOPs, dense tokens with redundancy, and tightly coupled multi‑task losses that can conflict. Deployment on wearable or embedded hardware is further limited by latency and VRAM budgets. The problem addressed here is to build a **lightweight, modular, and efficient STA pipeline** that maintains competitive accuracy while reducing computation and simplifying reproducibility on free or low‑cost GPUs.

## 1.3 Objectives of the Study

This thesis pursues six objectives:

1) **Decompose STA into modular stages** that can be optimized independently while remaining end‑to‑end reproducible.  
2) **Develop a high‑recall proposal mechanism** on the decision frame (Track A) to reliably surface potential next‑active objects, prioritizing recall@K over noun‑specific detection in the baseline.  
3) **Design a lightweight temporal–spatial fusion head** (Track B) using Frame‑Guided Temporal Pooling and dual cross‑attention, with token caching to cut training cost.  
4) **Integrate a training‑free token‑pruning layer** (Track C) to reduce inference compute and latency with minimal accuracy loss.  
5) **Evaluate on Ego4D‑STA v2** with baselines and ablations (K, fusion, priors, pruning), reporting **top‑5** N/N+V/N+δ mAP, **top‑5 Overall mAP**, TTC MAE, and latency/VRAM trade‑offs.  
6) **Document decisions, configurations, and procedures** (manifests, run logs, and evaluation scripts) to keep the pipeline auditable, comparable across variants, and easy to extend.

### 1.3.1 Contributions

The main contributions of this thesis are:

1) A **modular three‑track STA pipeline** (Tracks A/B/C) that cleanly separates proposals, reasoning, and efficiency so each component can be evaluated and improved independently.
2) A **recall@K‑first proposal stage** on the decision frame (Track A), paired with manifest construction to make downstream training and evaluation explicit and reproducible.
3) A **lightweight temporal fusion head** (Track B) built around Frame‑Guided Temporal Pooling and dual image↔video cross‑attention to incorporate short‑term motion cues with minimal compute.
4) A **training‑free inference knob** (Track C) based on rollout‑guided token pruning (RGTP) that enables controlled latency/VRAM reduction without retraining.
5) A **reproducible experimentation workflow** (configs, logging, manifests, and evaluation protocol) suitable for Colab‑class hardware, enabling systematic ablations and fair comparisons.

## 1.4 Scope and Limitations

Scope:
- Short‑term object interaction anticipation on Ego4D‑STA v2 with decision‑frame boxes, verb/noun labels, and TTC.
- Three tracks only: Stage‑A proposals/manifests, Track B temporal head, Track C inference‑time pruning.
- Compute and workflow: experiments are run primarily in a local environment; GPU‑heavy training steps (Track A detector fine‑tuning and VideoMAE pretraining) are executed on a cloud notebook when needed, while Track B/Track C training, evaluation, and analysis are performed locally.
- Metrics: recall@K (proposals), **top‑5** N/N+V/N+δ mAP, **top‑5 Overall mAP**, TTC MAE, and latency/VRAM under pruning.

Limitations:
- No long‑term anticipation or language grounding beyond optional CLIP reranking; focus is strictly STA.
- Detector is kept noun‑agnostic in the baseline; class‑specific detectors and large priors are future work.
- Results are limited to Ego4D; cross‑dataset generalization is discussed but not experimentally covered.
- Hyperparameter sweeps are bounded by low‑compute constraints; large model variants are out of scope.

## 1.5 Research Methodology

The study proceeds in five phases:

1) **Survey and design choices**: Review STA literature (two‑stage baselines, fusion, token pruning) and record key engineering decisions (e.g., noun‑agnostic proposals as the baseline for Track A).
2) **Data preparation and validation**: Build local manifests and optional YOLO labels; generate head_train/val JSONL; run sanity checks on label alignment (noun/verb/TTC) and paths; optionally pre‑extract video tokens to accelerate Track B training.
3) **Modeling and training**: Fine‑tune Track A detector weights and pretrain VideoMAE weights when needed using a cloud notebook workflow; run the remaining training (Track B heads) and all evaluations locally, keeping Track C pruning as a training‑free evaluation step.
4) **Evaluation protocol**: Evaluate on Ego4D‑STA v2 validation with consistent metrics (all mAP metrics reported in the benchmark’s **top‑5** protocol: N/N+V/N+δ and Overall; plus TTC MAE) and deployment‑oriented measurements (latency/VRAM), using fixed configs where possible.
5) **Ablations and reproducibility**: Sweep proposal K, fusion settings, TTC modeling choices, priors (e.g., hotspots/CLIP where applicable), and pruning rates; log run configs and outputs so results can be reproduced and audited.

## 1.6 Structure of the Thesis

**Chapter 2: Literature Review** surveys STA systems, efficient backbones, token pruning, affordances/hotspots, and ego/exo perspectives, grounding the design choices.  
**Chapter 3: Problem Definition and Data** formalizes the STA task, dataset splits and annotations, metrics, and the manifest/label‑alignment decisions that determine what supervision is available to each track.  
**Chapter 4: System Overview (Tracks A/B/C)** presents the pipeline at a high level, motivating the separation into proposals, lightweight fusion, and training‑free pruning.  
**Chapter 5: Methodology** details the modeling approach per track: proposal generation and recall@K (Track A), tokenization/fusion/head design (Track B), and rollout‑guided pruning at inference (Track C).  
**Chapter 6: Implementation and Engineering** describes the local/Colab workflow, configuration system, and reproducible run artifacts (manifests, logs, checkpoints, metrics).  
**Chapter 7: Experimental Setup** defines baselines, training protocols, hyperparameters, and the evaluation procedure used throughout the thesis.  
**Chapter 8: Results** reports results for each track, including accuracy and resource trade‑offs.  
**Chapter 9: Ablations and Analysis** studies sensitivity to key design choices (K, fusion depth, TTC modeling, priors, and pruning rate).  
**Chapters 10–13** discuss limitations and future work, provide a reproducibility checklist and ELSI considerations, and conclude with the main contributions.

Taken together, the thesis moves from defining the STA problem and its constraints (Chapters 1–3), to specifying the proposed pipeline (Chapter 4), to detailing how each track is implemented and evaluated (Chapters 5–7), and finally to presenting results, analysis, and implications for deployable egocentric anticipation (Chapters 8–13).


# Chapter 2 – Literature Review

This chapter reviews the main research threads that inform the Ego4D‑LiteSTA thesis:
short‑term object interaction anticipation in egocentric video, efficient video backbones and
self‑supervised pre‑training, token‑level efficiency for streaming video, hand–object affordances,
and cross‑view ego–exo transfer. For each line of work, we summarize the core ideas, highlight
architectural patterns, and extract concrete design choices that can be adapted to a lightweight,
real‑time STA pipeline.

Throughout the chapter, we refer to the Ego4D forecasting benchmark and dataset paper as the
foundational source, then connect them to recent challenge solutions (STAformer, SOIA‑DOD)
and efficiency‑oriented methods (MViTv2, VideoMAE, PruneVid, RGTP, EgoPrune, PEAR,
REAR, etc.).

---

## 2.1 Ego4D Forecasting Benchmark and STA Task

### 2.1.1 Forecasting benchmark and STA definition

The Ego4D forecasting benchmark defines four future‑prediction tasks based on the Ego4D
dataset: locomotion prediction, hand movement prediction, short‑term object interaction
anticipation (STA), and long‑term action anticipation.
The STA task is the most relevant for this thesis, as it directly formalizes the problem of predicting
what object the camera wearer will interact with, how, and when.

In STA, the model is given a short egocentric video clip and must predict, for the last observed
frame, (i) the bounding boxes of next‑active objects, (ii) a verb–noun pair describing the
upcoming interaction, and (iii) a scalar time‑to‑contact (TTC) value indicating when the
interaction will start relative to the current frame.
This joint spatial–semantic–temporal formulation makes STA considerably more challenging than
pure detection or classification.

From a data perspective, the Ego4D forecasting annotations are derived by aligning dense
egocentric narrations with frame‑level annotations of hand–object contact, pre‑contact context,
and hand trajectories.
For each interaction, annotators provide a pre‑condition frame, a contact frame, and multiple
earlier frames sampled at fixed temporal offsets (e.g., −0.5 s, −1.0 s, −1.5 s).
Active objects and hands are annotated with bounding boxes, and each interaction is labelled with
a verb category and noun category drawn from curated taxonomies.

These design choices have several implications for a “lite” STA variant:

- The problem is inherently **multi‑task** (detection + verb classification + noun classification +
  regression), which can be hard to optimize end‑to‑end under limited compute.
- The **time‑to‑contact** component is naturally noisy, as annotators approximate continuous
  human behavior with discrete frames; this makes TTC prediction relatively fragile.
- The dataset is **long‑tailed** in both verbs and nouns, which favors architectures that decouple
  object localization from fine‑grained verb/noun modeling and that can leverage priors or
  retrieval.

For Ego4D‑LiteSTA, a practical consequence is that we can selectively down‑scope some parts of
the official benchmark (e.g., treat TTC as optional in early experiments, or focus on top‑k next
active object prediction) while staying faithful to the original problem definition.

### 2.1.2 Ego4D dataset characteristics

The Ego4D dataset itself contains 3,670 hours of egocentric video recorded by 931 camera wearers
across 74 locations and 9 countries.
The footage spans daily‑life activities in household, workplace, leisure, outdoor, and social
settings.
Most recordings are long‑form, unscripted, and captured “in the wild”, rather than short,
trimmed clips.
This produces a strong mismatch between Ego4D and the short, highly curated third‑person clips
used for classic video benchmarks such as Kinetics.

From a modeling perspective, the dataset is challenging for several reasons:

- **Egocentric viewpoint** – the hands and immediate workspace dominate the field of view,
  while the actor is rarely visible.
- **High camera motion** – head movements and locomotion introduce strong ego‑motion,
  which complicates object tracking and stable localization.
- **Long‑term temporal context** – interactions are embedded in long continuous streams, not
  isolated snippets.
- **Diverse environments and objects** – the variety of kitchens, offices, tools, and household
  items increases domain shift across sequences.

The Ego4D paper also frames five benchmark families:
episodic memory, hand–object interaction, social interaction, audio‑visual conversation, and
forecasting.
STA resides in the forecasting group, but is tightly connected to hand–object interaction tasks
(hands and objects, hotspots, and affordances) and to long‑term temporal reasoning.

For Ego4D‑LiteSTA, the dataset properties argue strongly in favor of:

- Treating **hand proximity and object context** as primary cues for next interaction.
- Designing architectures that can **reuse features** across tracks (STA vs. other Ego4D tasks).
- Exploiting **self‑supervised pre‑training** on egocentric video to reduce the need for
  large exocentric datasets.

---

## 2.2 Baseline STA Challenge Solutions

Work on the Ego4D STA benchmark has produced several strong baselines and challenge
solutions that inspire the design of Ego4D‑LiteSTA, especially in terms of decomposition of the
task, use of attention, and integration of affordances.

### 2.2.1 STAformer and AFF-ttention

STAformer, introduced by the ZARRIO team, is an attention‑based architecture tailored to the
Ego4D STA challenge.
The model processes an image–video pair: the last frame at high resolution and a short sequence of
frames as a video clip.
STAformer extracts DINOv2 features from the still image and TimeSformer features from the
video, then fuses them via several specialized attention modules:

- **Frame‑Guided Temporal Pooling Attention** – projects video features onto the last‑frame
  reference, emphasizing temporal information that matters at the current time step.
- **Dual Image–Video Cross‑Attention** – refines both image and video features by letting each
  attend to the other, encouraging consistency between static and dynamic cues.
- **Multi‑scale feature fusion** – aggregates information from different spatial resolutions.
- **Fast R‑CNN‑style detection head** – adapted to predict bounding boxes, verb and noun
  probabilities, and time‑to‑contact for each candidate box.

On top of STAformer, the AFF‑ttention work extends the model with **environment affordance**
and **interaction hotspot** modules.
The environment affordance model uses an EgoTopo‑like representation: videos are decomposed
into topological zones, and a database of zones is used as a persistent memory of what interactions
are feasible in each region (e.g., “countertop near the sink”, “corner with a kettle”).
Given a new input, the system matches the observed scene to the database and refines verb/noun
probabilities based on historically observed affordances.
The interaction hotspot module predicts a spatial map of likely contact regions on the current
frame, using hand and object trajectories, and then re‑weights STA predictions based on how close
 candidate boxes are to hotspot areas.

Experimentally, STAformer with affordances reaches 33.5 N‑mAP, 17.25 N+V‑mAP, 11.77
N+δ‑mAP, and 6.75 Overall top‑5 mAP on the v2 test set, improving substantially over previous
challenge winners.

For Ego4D‑LiteSTA, STAformer and AFF‑ttention suggest several principles:

- Explicitly modeling **image–video fusion** around the last frame is effective.
- **Affordance priors** can compensate for limited model capacity by encoding “what is usually
  done where.”
- Hotspot prediction anchors STA to **hand‑centric spatial priors**, which is valuable when
  bounding box proposals are noisy.

However, the full STAformer + AFF pipeline is computationally heavy:
it relies on high‑capacity vision transformers (DINOv2, TimeSformer), large affordance databases,
and additional modules for hotspots.
This motivates the LiteSTA strategy of approximating similar behavior with simpler backbones
(e.g., YOLOv8‑s) and focusing on egocentric‑specific cues rather than global scene semantics.

### 2.2.2 SOIA-DOD: disentangled detection and anticipation

SOIA‑DOD (Short‑term Object Interaction Anticipation with Disentangled Object Detection) takes
a different perspective: instead of one monolithic network that jointly solves detection, verb/noun
classification, and TTC regression, it decomposes the problem into two stages:

1. **Potential active object detection** – a fine‑tuned YOLOv9 model detects all candidate active
   objects in the last frame.
   The top‑k high‑confidence boxes are kept as potential next‑active objects.
2. **Interaction and TTC prediction** – a transformer‑based encoder takes both visual tokens and
   object queries (encoding box coordinates and class labels) and predicts, for each candidate,
   the probability of being the next‑active object, the interaction class (verb), and TTC.

This decomposition simplifies optimization: YOLOv9 focuses purely on localization and object
class, while the transformer learns to refine which candidate is truly next‑active and when the
interaction will occur.
On the Ego4D STA challenge, SOIA‑DOD achieves state‑of‑the‑art performance for predicting the
next active object and interaction, ranking third in overall top‑5 mAP including TTC.

For Ego4D‑LiteSTA, SOIA‑DOD provides a clear reference design that aligns well with a
resource‑constrained regime:

- Use a **lightweight detector** (e.g., YOLOv8‑s instead of YOLOv9) fine‑tuned for next‑active
  candidate generation.
- Keep a **compact transformer head** to operate on a small set of queries, which keeps
  complexity roughly linear in the number of proposals rather than quadratic in spatial tokens.
- Treat TTC as an **auxiliary regression task**, which can be scaled back or approximated if
  necessary.

Compared to STAformer, SOIA‑DOD is closer in spirit to the proposed Track A/B/C structure in
Ego4D‑LiteSTA, where Stage A is a YOLO‑based detection stage and later stages reuse these
candidates for anticipation.

---

## 2.3 Efficient Video Backbones and Pre-Training

Running STA at scale over long egocentric streams requires backbones that provide a good
accuracy–efficiency trade‑off and can exploit self‑supervised pre‑training.
Two key lines of work are Multiscale Vision Transformers (MViTv2) and VideoMAE.

### 2.3.1 Multiscale Vision Transformers (MViTv2)

MViTv2 generalizes vision transformers into a **multiscale hierarchy** that can serve as a single
backbone for image classification, object detection, and video recognition.
Instead of operating at a fixed resolution with global attention at all layers, MViTv2 progressively
reduces spatial (and temporal) resolution across stages, using “pooling attention” to aggregate
information while controlling compute.

Two main architectural improvements over the original MViT are highlighted:

- **Decomposed relative positional embeddings** – inject shift‑invariant position information into
  attention without exploding parameter count.
- **Residual pooling connections** – compensate for information loss when applying pooling
  strides inside attention blocks.

These changes lead to strong results across domains:

- Up to 88.8% top‑1 on ImageNet‑1K when pre‑trained on ImageNet‑21K.
- 58.7 AP_box on COCO detection.
- 86.1% accuracy on Kinetics‑400 for video classification.

For a lightweight STA pipeline, MViTv2 is not necessarily the final backbone (YOLOv8‑s is a more
natural fit for real‑time detection), but the design ideas are instructive:

- A **feature hierarchy** with decreasing resolution is critical for efficiency and for integrating
  local and global information.
- Pooling attention offers an alternative to windowed attention, and the authors report better
  accuracy/compute trade‑offs than Swin‑style local windows.
- The same backbone can power image‑only tasks (detection on last frame) and video‑based tasks
  (clip‑level motion reasoning), which is analogous to Track A (image‑centric) vs. Track B/C
  (clip‑centric) in Ego4D‑LiteSTA.

An interesting extension for future work would be to swap YOLOv8‑s with an MViTv2‑based
detector or to use an MViT feature extractor for the temporal head, while keeping the rest of the
LiteSTA pipeline unchanged.

### 2.3.2 VideoMAE: data-efficient video self-supervision

VideoMAE is a masked autoencoder framework designed specifically for self‑supervised video
pre‑training with vanilla ViT backbones.
It applies a very high masking ratio (90–95%) to video tubes (spatio‑temporal patches) and tasks
the model with reconstructing the missing content from a small subset of visible tokens.
Because video has strong temporal redundancy, such aggressive masking makes reconstruction
challenging and encourages the encoder to learn robust, high‑level representations instead of
memorizing low‑level patterns.

Key findings from VideoMAE are particularly relevant for an egocentric STA setting:

- **Very high masking ratios work** – unlike images, where ~75% masking is typical, videos can
  tolerate 90–95% while still training successfully.
- **Data efficiency** – VideoMAE achieves strong performance even when pre‑training on relatively
  small video datasets (3k–4k clips), provided they are in‑domain.
- **No extra data requirement** – the method reaches state‑of‑the‑art performance on several video
  benchmarks (Kinetics‑400, Something‑Something‑V2, UCF101, HMDB51) without using
  large external datasets.

This is echoed by the “Ego‑Only” work (discussed later), which demonstrates that egocentric action
detection can achieve state‑of‑the‑art results by performing MAE‑style pre‑training directly on
egocentric data, without exocentric transfer.

For Ego4D‑LiteSTA, VideoMAE suggests a pre‑training path for the temporal head:

- Use a **tube‑masked autoencoder** on Ego4D STA clips (or on Ego4D more broadly) to pre‑train
  a compact ViT or MViT backbone.
- Fine‑tune this encoder for STA‑specific supervision (verb/noun/TTC).
- Keep the heavy reconstruction decoder only during pre‑training; at inference, use only the
  encoder, which keeps runtime manageable.

Even if the final LiteSTA implementation uses a simpler temporal module (e.g., 3D CNN or small
transformer), the underlying principle is the same: leverage **self‑supervision on in‑domain
egocentric videos** to avoid heavy exocentric pre‑training.

---

## 2.4 Token-Level Efficiency for Video and Egocentric Streams

Running STA in a streaming or near‑real‑time setting motivates methods that operate not only on
efficient backbones, but also on **efficient token sets**.
Recent work explores training‑free pruning and merging strategies that exploit spatio‑temporal
redundancy in video tokens.

### 2.4.1 PruneVid

PruneVid introduces a training‑free method for pruning visual tokens in multimodal video‑language
models.
Its goal is to reduce the computational burden of attention in large language models that ingest long
sequences of video tokens.

The method has two main steps:

1. **Intrinsic redundancy reduction** – temporally static regions are detected and merged across
   frames; spatially similar tokens are then clustered and merged within frames, compressing both
   static background and redundant object regions.
2. **Question‑guided attention pruning** – during LLM processing, attention maps are used to
   identify tokens most relevant to the query; tokens with consistently low attention are pruned
   across layers, and only their key‑value caches are kept where necessary.

Experiments show that PruneVid can prune over 80% of visual tokens while maintaining
competitive QA performance, reducing FLOPs by 74–80% and significantly lowering memory
usage.

For Ego4D‑LiteSTA, PruneVid is conceptually relevant in two ways:

- It demonstrates that **temporal and spatial redundancy** can be aggressively exploited, which is
  also true in egocentric STA clips where backgrounds and non‑manipulated regions change
  slowly.
- It suggests a **two‑stage pruning strategy** (data‑level compression + task‑guided token
  selection) that could be adapted to a compact STA transformer head without any re‑training.

In practice, a LiteSTA implementation could, for example, merge background tokens across
frames in the temporal encoder, and then keep only tokens near hands or candidate boxes, mimicking
PruneVid’s behavior but in a much smaller model.

### 2.4.2 Rollout-Guided Token Pruning (RGTP)

Rollout‑Guided Token Pruning (RGTP) focuses on efficient video understanding with frame‑by‑frame
vision transformers, explicitly leveraging **attention rollout** and **token tracking** across time to
guide pruning decisions.

The main idea is to estimate the importance of each input token in the current frame by tracing its
contribution from previous frames’ predictions:

- Attention rollout is used to propagate relevance from output tokens back to earlier layers, giving
  a measure of how much each token contributed to past predictions.
- Token tracking then aligns tokens between consecutive frames (e.g., via motion or positional
  correspondence), so that importance scores can be propagated over time.
- Tokens with consistently low importance are pruned before the attention blocks, leading to large
  FLOP reductions without retraining.

RGTP is training‑free, interpretable, and shows up to 65% FLOP reduction on ImageNet VID and
60% on EPIC‑Kitchens action recognition, with negligible accuracy degradation.

For Ego4D‑LiteSTA, RGTP is particularly relevant because EPIC‑Kitchens is also an egocentric
dataset with hand–object interactions.
Adapting a rollout‑guided pruning strategy to the STA temporal head could further reduce compute:
only tokens that historically matter for predicting interactions (e.g., hand regions, moving objects)
would be kept in later layers.

### 2.4.3 EgoPrune

EgoPrune focuses specifically on **egomotion videos** in embodied agents and proposes a
perspective‑aware, training‑free token pruning method tailored to first‑person streams.
The method introduces three components:

1. **Keyframe selection** – adapted from EmbodiedR, to reduce temporal redundancy by sampling
   only a subset of frames that capture important changes.
2. **Perspective‑Aware Redundancy Filtering (PARF)** – uses homography‑based perspective
   transformations to align visual tokens across frames and then removes redundant tokens that
   correspond to stable regions in the scene.
3. **Maximal Marginal Relevance (MMR)‑based token selection** – jointly considers visual–text
   relevance and intra‑frame diversity to retain tokens that are both informative for the query and
   non‑redundant.

EgoPrune demonstrates that incorporating egomotion‑specific geometry (e.g., camera motion,
scene structure) into pruning can outperform generic video pruning methods at the same pruning
ratios, both in accuracy and in latency on edge devices.

For Ego4D‑LiteSTA, the main takeaway is that **egocentric geometry matters for efficiency**.
If the STA temporal head operates on patches or tokens, we can:

- Treat hand‑centric, near‑field regions as high‑priority tokens.
- Use simple motion cues (optical flow or YOLO‑based object tracks) to drop far‑field,
  background tokens that remain static over time.
- Apply diversity‑aware selection when we have many overlapping candidate boxes or patches in
  similar areas.

Although EgoPrune targets video‑language models, its principles align well with a “lite” STA
design that must run on modest GPUs or even edge hardware.

---

## 2.5 Hand–Object Interaction, Affordances, and Hotspots

STA is fundamentally about **anticipating hand–object interactions**.
Several recent works directly address how to model hand motion trends, interaction hotspots, and
affordances.

### 2.5.1 PEAR

PEAR (Phrase‑Based Hand‑Object Interaction Anticipation) proposes a model that jointly forecasts
both **interaction intention** and **interaction manipulation** over a future time window, given a
pre‑interaction image and a natural language phrase such as “pick up bottle”.

The authors argue that existing work often predicts only pre‑contact intention (e.g., contact
hotspots) while ignoring detailed manipulation trajectories and hand poses after contact, which
makes predictions incomplete and less constrained.
PEAR addresses two sources of uncertainty:

- **Intention uncertainty** – high variability in hand motion patterns and object functional
  attributes.
- **Manipulation uncertainty** – difficulty in matching pre‑contact intention to post‑contact
  manipulation elements (trajectories, poses).

To reduce intention uncertainty, PEAR performs **cross‑alignment of verbs, nouns, and images**.
It leverages image–text encoders to align nouns with object affordances and verbs with motion
patterns, and then uses cross‑attention modules to constrain the space of plausible intentions.

To mitigate manipulation uncertainty, PEAR introduces a **dynamic bidirectional constraint**
between intention and manipulation:

- A deep equilibrium model jointly predicts hand motion trends, hotspots, manipulation
  trajectories, and hand poses.
- Residual connections from manipulation back to intention help refine early predictions so that
  they remain consistent with the final manipulation.

A conditional VAE (C‑VAE) decoder introduces controlled randomness, capturing human variability
while keeping predictions plausible.

For Ego4D‑LiteSTA, PEAR suggests several ideas that can be simplified and reused:

- Cross‑alignment between **verb/noun labels and visual features** can help disambiguate fine‑grained
  categories in long‑tailed vocabularies.
- Modeling **interaction hotspots** and hand motion trends explicitly is beneficial, even if we
  keep a shorter prediction horizon than PEAR.
- Phrase‑based conditioning hints at a future extension where STA predictions could be guided by
  higher‑level task descriptions or textual priors.

### 2.5.2 Fine-grained affordance annotation

Another line of work revisits how affordances are defined and annotated in hand–object interaction
datasets.
The “Fine‑grained Affordance Annotation” paper argues that many existing datasets conflate
affordance with object functionality or high‑level actions (verbs like “cut”, “take”, “turn off”) and
ignore human motor capacity and grasp types.

To address this, the authors propose an annotation scheme where affordance labels are constructed
as combinations of **goal‑irrelevant motor actions** and **grasp types**, focusing on the
hand–object interface itself, and introduce **mechanical action** labels to describe interactions
between tools and target objects.
They apply this scheme to EPIC‑KITCHENS, generating labels that distinguish between affordances
(e.g., how an object can be grasped or manipulated) and goal‑level actions (e.g., “turn off tap”).

The paper evaluates the new annotations on three tasks:

- Affordance recognition.
- Hand–object interaction hotspot prediction with affordances as weak supervision.
- Cross‑domain affordance generalization.

Results show that affordance‑centric labels yield better generalization and finer‑grained hotspot
prediction than action labels alone.

For Ego4D‑LiteSTA, this reinforces the importance of separating:

- **What the object can afford** (affordance, grasp, contact regions).
- **What the user is about to do** (verb, goal, task).

Even if the thesis does not introduce new affordance annotations, the ideas support design choices
such as:

- Using hand proximity and object category as strong priors for **next‑active object selection**.
- Optionally integrating external affordance priors (e.g., from EPIC‑KITCHENS) into the STA
  head as an additional score or bias.

---

## 2.6 Ego–Exo Transfer, Ego-Only Learning, and Retrieval-Augmented Models

The broader egocentric literature has explored two complementary directions:
(1) transferring knowledge from large exocentric datasets to egocentric tasks, and
(2) training “ego‑only” models directly on egocentric data without exocentric transfer.
Both perspectives are useful for positioning Ego4D‑LiteSTA in the design space.

### 2.6.1 Ego–exo joint learning survey

A recent short survey reviews joint egocentric–exocentric learning, highlighting datasets that
contain paired ego–exo views (e.g., CMU‑MMAC, Charades‑Ego, Assembly101, EgoExo4D) and
the corresponding tasks: action recognition, proficiency estimation, action anticipation, pose
estimation, correspondence, temporal segmentation, frame retrieval, and alignment.

The survey’s central argument is that exocentric videos provide complementary cues—full‑body
pose, global context—that can help interpret egocentric hand–object interactions, and that joint
ego–exo modeling can unlock richer representations for downstream tasks.
This viewpoint supports the idea that STA could benefit from exocentric priors in principle, but
also emphasizes the practical difficulty of collecting synchronized ego–exo data at scale.

Ego4D‑LiteSTA positions itself closer to the “ego‑only” side (especially when working on
student‑scale hardware and free Colab), but the survey’s taxonomy is helpful for framing potential
extensions, such as using exocentric retrieval to provide additional context for rare interactions.

### 2.6.2 Synchronization-based exo→ego transfer for temporal segmentation

“Synchronization is All You Need” proposes a method to adapt a temporal action segmentation
(TAS) model from an exocentric dataset to an egocentric setting using **unlabeled synchronized
ego–exo video pairs**.
Instead of collecting labeled egocentric videos, the method uses existing exocentric labels plus
unlabeled synchronized pairs and applies knowledge distillation from an exocentric teacher to an
egocentric student at both feature and model levels.

Experiments on Assembly101 and EgoExo4D show that this approach can bridge much of the
performance gap between exocentric‑only models and fully supervised egocentric models, improving edit scores substantially without using any egocentric labels.

While the task (TAS) differs from STA, the key lesson is that **synchronization can act as a
powerful supervision signal** in ego–exo transfer.
For Ego4D‑LiteSTA, a related idea could be to use synchronized streams (e.g., multi‑view or
multi‑sensor data) for future work, but the base thesis keeps the pipeline simpler and does not rely
on exocentric labels.

### 2.6.3 Ego-Only

The Ego‑Only work directly challenges the assumption that exocentric pre‑training is required for
egocentric action detection.
Instead, it shows that with a sufficiently strong self‑supervised pre‑training strategy—specifically,
a masked autoencoder finetuned for temporal segmentation—one can train competitive egocentric
models using only egocentric video data.

Ego‑Only emphasizes several differences between egocentric and exocentric data that make naive
transfer problematic:

- Egocentric videos are long‑form and dominated by hand–object interactions and near‑field
  views, whereas exocentric clips are short, trimmed, and show full bodies and global context.
- Action classes in Ego4D and EPIC‑KITCHENS are fine‑grained and long‑tailed, reflecting
  real‑world distributions.
- Temporal localization is central in egocentric action detection, whereas many exocentric
  datasets focus on clip‑level classification.

The proposed pipeline has three stages:

1. MAE‑style pre‑training on egocentric videos.
2. Temporal segmentation fine‑tuning.
3. Action detection with an off‑the‑shelf temporal detector (e.g., ActionFormer).

Ego‑Only achieves state‑of‑the‑art results on Ego4D, EPIC‑Kitchens‑100, and Charades‑Ego
without any exocentric data.

For Ego4D‑LiteSTA, Ego‑Only provides a strong conceptual justification for focusing on **ego‑only
pre‑training and fine‑tuning**, especially under compute constraints and when exocentric data is
not central to the problem being solved.
Combining Ego‑Only principles with VideoMAE‑style pre‑training is a natural fit for a lightweight
STA head.

### 2.6.4 Early motion transfer between egocentric and exocentric views

EgoTransfer represents an early attempt to model “mirror neurons” by learning mappings between
motion features in egocentric and exocentric videos.
The authors record time‑synchronized ego–exo video pairs, extract motion features (e.g., optical
flow descriptors) in both views, and train linear or non‑linear models to predict egocentric motion
from exocentric motion and vice versa.

Evaluation is done via cross‑view retrieval: given a motion feature in one view, retrieve its
corresponding feature in the other view from a set of candidates.
Results show that the learned mappings can successfully transfer motion across views, suggesting
that there is a stable relationship between egocentric and exocentric motion patterns.

For Ego4D‑LiteSTA, EgoTransfer is mostly of historical and conceptual interest.
It reinforces the idea that exocentric information could help interpret egocentric motion and vice
versa, but it also highlights the increasing complexity and data requirements of such approaches.
Given the thesis focus on a practical, lightweight STA pipeline, the design remains ego‑centric
and does not rely on motion transfer from exocentric sources.

### 2.6.5 Retrieval-augmented egocentric action recognition (REAR)

REAR (Retrieval‑Augmented Egocentric Action Recognition) proposes to augment egocentric
representations with **retrieved exocentric video features** without requiring synchronized ego–exo
pairs.

The framework uses a dual‑branch architecture:

- A target branch that encodes the egocentric video.
- A retrieval branch that retrieves semantically related exocentric features from a large corpus
  based on similarity in a joint embedding space.

A cross‑view integration module performs staged fusion and attention‑based alignment of
egocentric and exocentric features.
To address long‑tailed class distributions, REAR introduces a **class‑adaptive selector** that varies
the number of retrieved examples per class and uses logit‑adjusted cross‑entropy (LACE) for
training the classifiers.

Experiments on three egocentric benchmarks show that REAR improves object (noun) recognition
and tail‑class performance, demonstrating the value of exocentric retrieval as an auxiliary
knowledge source.

For Ego4D‑LiteSTA, REAR’s architecture is heavier than what is realistic for a Colab‑based,
single‑GPU pipeline, but the ideas are relevant conceptually:

- Retrieval‑augmented modeling is a promising direction for handling **rare verbs/nouns** in STA.
- Class‑adaptive retrieval reflects the intuition that **tail classes need more external help** than
  head classes.
- A future extension of LiteSTA could attach a small retrieval branch that uses external data
  (exocentric or egocentric) to refine verb/noun scores for ambiguous cases.

---

## 2.7 Community Resources and Challenge Ecosystem

The egocentric vision community has converged around a set of large‑scale datasets and workshops
that provide both data and benchmarks relevant to STA and related tasks.

### 2.7.1 EgoVis workshop and challenge ecosystem

The Joint Egocentric Vision (EgoVis) workshop at CVPR 2024 brings together multiple datasets
and challenges in egocentric perception, including Ego4D, EgoExo4D, EPIC‑Kitchens, HoloAssist,
Aria Digital Twin, and Aria Synthetic Environments.

For Ego4D specifically, the workshop hosts challenges on:

- Visual queries (2D and 3D).
- Natural language queries and moment queries.
- EgoTracks and goal step prediction.
- PNR temporal localization, localization and tracking.
- Short‑term and long‑term anticipation (including STA).

This ecosystem demonstrates that STA is part of a broader **anticipation and assistance** agenda,
where models should not only recognize what is happening but also predict what will happen next,
locate relevant moments, and interact with language.

For a thesis focusing on Ego4D‑LiteSTA, the workshop context reinforces the relevance of:

- Designing methods that could be plugged into the official STA challenge if desired.
- Keeping an eye on **multi‑task and multi‑modal extensions**, such as combining STA with
  natural language queries or goal step prediction.

### 2.7.2 Curated lists and “awesome” repositories

Finally, community‑curated resources like “Awesome Egocentric Action Understanding” aggregate
recent work on egocentric action recognition, anticipation, representation learning, and multi‑view
modeling.
These lists are useful for situating Ego4D‑LiteSTA within the broader literature, identifying new
baselines, and ensuring that design choices are informed by the current state of the art.

---

## 2.8 Summary and Positioning of Ego4D-LiteSTA

The literature reviewed in this chapter converges on several key themes that directly influence the
design of the Ego4D‑LiteSTA pipeline:

1. **STA is a multi‑task, hand‑centric, future‑prediction problem**  
   The Ego4D forecasting benchmark and STA task definition make clear that next‑active object
   prediction requires joint reasoning about spatial localization, verb/noun semantics, and time‑to‑contact.
   Challenge solutions like STAformer and SOIA‑DOD show that
   decomposing the problem into detection + anticipation, and exploiting hand and affordance
   cues, is an effective strategy.

2. **Efficient backbones and self‑supervised pre‑training are essential**  
   MViTv2 and VideoMAE demonstrate that multiscale transformers and high‑mask‑ratio MAEs
   can provide strong representations that are both accurate and efficient.
   Ego‑Only confirms that, for egocentric tasks, in‑domain MAE pre‑training can replace heavy
   exocentric pre‑training entirely.

3. **Token‑level efficiency can dramatically reduce compute**  
   PruneVid, RGTP, and EgoPrune show that aggressive token pruning and merging—informed by
   temporal redundancy, attention rollout, and egomotion geometry—can reduce FLOPs by 60–80%
   with minimal accuracy loss.
   While Ego4D‑LiteSTA primarily targets efficiency via model choice (YOLOv8‑s) and pipeline
   design, these methods point to additional gains available through token‑level optimization.

4. **Affordances and hand–object hotspots provide powerful priors**  
   PEAR and fine‑grained affordance annotations demonstrate that modeling where and how the
   hand will interact with an object can significantly improve anticipation quality.
   For a lightweight STA system, even simple approximations—such as focusing on objects near the
   hands and encoding object‑specific priors—can provide a large boost.

5. **Ego–exo transfer is useful but not mandatory**  
   Surveyed works on ego–exo joint learning, synchronization‑based transfer, and retrieval‑augmented
   models show that exocentric data can help, especially for rare classes and global context.
   However, Ego‑Only, VideoMAE, and the Ego4D dataset itself make a strong case that **ego‑only
   pipelines are viable and competitive**, which aligns with the practical constraints of this thesis.

Within this landscape, Ego4D‑LiteSTA positions itself as a **practical, lightweight STA pipeline**
that:

- Uses a YOLOv8‑s detector for next‑active candidate generation (Stage A).
- Adds compact temporal and semantic heads for STA on top of detected candidates (Tracks B and C).
- Leverages egocentric‑focused design choices inspired by STAformer, SOIA‑DOD, VideoMAE,
  Ego‑Only, and affordance‑based works.
- Is designed to run end‑to‑end on free Google Colab GPUs, with transparent logging of runs,
  metrics, and errors for reproducible research.

This literature review thus provides both the theoretical foundation and the practical design space
within which Ego4D‑LiteSTA is developed and evaluated.



# Chapter 3: Problem Definition and Data

## 3.1 Task Definition (STA)

## 3.2 Dataset, Splits, and Annotations (Ego4D-STA v2)

## 3.3 Metrics

## 3.4 Data Manifests and Label Alignment Decisions


# Chapter 4: System Overview (Tracks A / B / C)

## 4.1 Design Principles

## 4.2 Track A Overview (Proposals + Manifests)

## 4.3 Track B Overview (Lightweight Fusion Head)

## 4.4 Track C Overview (Training-Free Pruning)


# Chapter 5: Methodology

## 5.1 Track A: Candidate Generation and Recall@K

## 5.2 Track B: Tokenization, Fusion (FGTP + Dual Cross-Attention), and Heads

## 5.3 Track C: Rollout-Guided Token Pruning (RGTP)


# Chapter 6: Implementation and Engineering

## 6.1 Local/Colab Workflow and Tooling

## 6.2 Configuration System and Reproducible Runs

## 6.3 Data Validation and Failure Handling


# Chapter 7: Experimental Setup

## 7.1 Baselines and Compared Variants

## 7.2 Training Protocols and Hyperparameters

## 7.3 Evaluation Protocol


# Chapter 8: Results

## 8.1 Track A Results (Recall@K)

## 8.2 Track B Results (N / N+V / N+δ, TTC)

## 8.3 Track C Results (Accuracy–Latency Pareto)


# Chapter 9: Ablations and Analysis

## 9.1 Proposal K Sweeps

## 9.2 Fusion Depth and Token Dimensionality

## 9.3 TTC Modeling (Regression vs Bins)

## 9.4 Priors (Hotspots, CLIP) and Their Impact


# Chapter 10: Discussion, Limitations, and Future Work


# Chapter 11: Reproducibility Checklist


# Chapter 12: Ethical, Legal, and Social Implications (ELSI)


# Chapter 13: Conclusion


# References


# Appendices


