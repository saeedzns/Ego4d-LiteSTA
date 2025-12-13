
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
anticipation (STA), and long‑term action anticipation citeturn0search0.
The STA task is the most relevant for this thesis, as it directly formalizes the problem of predicting
what object the camera wearer will interact with, how, and when.

In STA, the model is given a short egocentric video clip and must predict, for the last observed
frame, (i) the bounding boxes of next‑active objects, (ii) a verb–noun pair describing the
upcoming interaction, and (iii) a scalar time‑to‑contact (TTC) value indicating when the
interaction will start relative to the current frame citeturn0search0.
This joint spatial–semantic–temporal formulation makes STA considerably more challenging than
pure detection or classification.

From a data perspective, the Ego4D forecasting annotations are derived by aligning dense
egocentric narrations with frame‑level annotations of hand–object contact, pre‑contact context,
and hand trajectories citeturn0search0.
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
across 74 locations and 9 countries citeturn0search1.
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
forecasting citeturn0search1.
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

### 2.2.1 STAformer and AFF‑ttention (ZARRIO @ Ego4D STA)

STAformer, introduced by the ZARRIO team, is an attention‑based architecture tailored to the
Ego4D STA challenge citeturn0search2.
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
and **interaction hotspot** modules citeturn0search2.
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
challenge winners citeturn0search2.

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

### 2.2.2 SOIA‑DOD: disentangled detection and anticipation

SOIA‑DOD (Short‑term Object Interaction Anticipation with Disentangled Object Detection) takes
a different perspective: instead of one monolithic network that jointly solves detection, verb/noun
classification, and TTC regression, it decomposes the problem into two stages citeturn0search3:

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
next active object and interaction, ranking third in overall top‑5 mAP including TTC citeturn0search3.

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

## 2.3 Efficient Video Backbones and Pre‑Training

Running STA at scale over long egocentric streams requires backbones that provide a good
accuracy–efficiency trade‑off and can exploit self‑supervised pre‑training.
Two key lines of work are Multiscale Vision Transformers (MViTv2) and VideoMAE.

### 2.3.1 Multiscale Vision Transformers (MViTv2)

MViTv2 generalizes vision transformers into a **multiscale hierarchy** that can serve as a single
backbone for image classification, object detection, and video recognition citeturn0search4.
Instead of operating at a fixed resolution with global attention at all layers, MViTv2 progressively
reduces spatial (and temporal) resolution across stages, using “pooling attention” to aggregate
information while controlling compute.

Two main architectural improvements over the original MViT are highlighted citeturn0search4:

- **Decomposed relative positional embeddings** – inject shift‑invariant position information into
  attention without exploding parameter count.
- **Residual pooling connections** – compensate for information loss when applying pooling
  strides inside attention blocks.

These changes lead to strong results across domains:

- Up to 88.8% top‑1 on ImageNet‑1K when pre‑trained on ImageNet‑21K.
- 58.7 AP\_box on COCO detection.
- 86.1% accuracy on Kinetics‑400 for video classification citeturn0search4.

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

### 2.3.2 VideoMAE: data‑efficient video self‑supervision

VideoMAE is a masked autoencoder framework designed specifically for self‑supervised video
pre‑training with vanilla ViT backbones citeturn0search5.
It applies a very high masking ratio (90–95%) to video tubes (spatio‑temporal patches) and tasks
the model with reconstructing the missing content from a small subset of visible tokens.
Because video has strong temporal redundancy, such aggressive masking makes reconstruction
challenging and encourages the encoder to learn robust, high‑level representations instead of
memorizing low‑level patterns.

Key findings from VideoMAE are particularly relevant for an egocentric STA setting citeturn0search5:

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

## 2.4 Token‑Level Efficiency for Video and Egocentric Streams

Running STA in a streaming or near‑real‑time setting motivates methods that operate not only on
efficient backbones, but also on **efficient token sets**.
Recent work explores training‑free pruning and merging strategies that exploit spatio‑temporal
redundancy in video tokens.

### 2.4.1 PruneVid: training‑free visual token pruning for video LLMs

PruneVid introduces a training‑free method for pruning visual tokens in multimodal video‑language
models citeturn0search6.
Its goal is to reduce the computational burden of attention in large language models that ingest long
sequences of video tokens.

The method has two main steps citeturn0search6:

1. **Intrinsic redundancy reduction** – temporally static regions are detected and merged across
   frames; spatially similar tokens are then clustered and merged within frames, compressing both
   static background and redundant object regions.
2. **Question‑guided attention pruning** – during LLM processing, attention maps are used to
   identify tokens most relevant to the query; tokens with consistently low attention are pruned
   across layers, and only their key‑value caches are kept where necessary.

Experiments show that PruneVid can prune over 80% of visual tokens while maintaining
competitive QA performance, reducing FLOPs by 74–80% and significantly lowering memory
usage citeturn0search6.

For Ego4D‑LiteSTA, PruneVid is conceptually relevant in two ways:

- It demonstrates that **temporal and spatial redundancy** can be aggressively exploited, which is
  also true in egocentric STA clips where backgrounds and non‑manipulated regions change
  slowly.
- It suggests a **two‑stage pruning strategy** (data‑level compression + task‑guided token
  selection) that could be adapted to a compact STA transformer head without any re‑training.

In practice, a LiteSTA implementation could, for example, merge background tokens across
frames in the temporal encoder, and then keep only tokens near hands or candidate boxes, mimicking
PruneVid’s behavior but in a much smaller model.

### 2.4.2 Rollout‑Guided Token Pruning (RGTP)

Rollout‑Guided Token Pruning (RGTP) focuses on efficient video understanding with frame‑by‑frame
vision transformers, explicitly leveraging **attention rollout** and **token tracking** across time to
guide pruning decisions citeturn0search7.

The main idea is to estimate the importance of each input token in the current frame by tracing its
contribution from previous frames’ predictions:

- Attention rollout is used to propagate relevance from output tokens back to earlier layers, giving
  a measure of how much each token contributed to past predictions.
- Token tracking then aligns tokens between consecutive frames (e.g., via motion or positional
  correspondence), so that importance scores can be propagated over time.
- Tokens with consistently low importance are pruned before the attention blocks, leading to large
  FLOP reductions without retraining.

RGTP is training‑free, interpretable, and shows up to 65% FLOP reduction on ImageNet VID and
60% on EPIC‑Kitchens action recognition, with negligible accuracy degradation citeturn0search7.

For Ego4D‑LiteSTA, RGTP is particularly relevant because EPIC‑Kitchens is also an egocentric
dataset with hand–object interactions.
Adapting a rollout‑guided pruning strategy to the STA temporal head could further reduce compute:
only tokens that historically matter for predicting interactions (e.g., hand regions, moving objects)
would be kept in later layers.

### 2.4.3 EgoPrune: perspective‑aware pruning for egomotion videos

EgoPrune focuses specifically on **egomotion videos** in embodied agents and proposes a
perspective‑aware, training‑free token pruning method tailored to first‑person streams citeturn0search8.
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
ratios, both in accuracy and in latency on edge devices citeturn0search8.

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

### 2.5.1 PEAR: phrase‑based hand‑object interaction anticipation

PEAR (Phrase‑Based Hand‑Object Interaction Anticipation) proposes a model that jointly forecasts
both **interaction intention** and **interaction manipulation** over a future time window, given a
pre‑interaction image and a natural language phrase such as “pick up bottle” citeturn0search9.

The authors argue that existing work often predicts only pre‑contact intention (e.g., contact
hotspots) while ignoring detailed manipulation trajectories and hand poses after contact, which
makes predictions incomplete and less constrained citeturn0search9.
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

### 2.5.2 Fine‑grained affordance annotation for egocentric HOI

Another line of work revisits how affordances are defined and annotated in hand–object interaction
datasets.
The “Fine‑grained Affordance Annotation” paper argues that many existing datasets conflate
affordance with object functionality or high‑level actions (verbs like “cut”, “take”, “turn off”) and
ignore human motor capacity and grasp types citeturn0search10.

To address this, the authors propose an annotation scheme where affordance labels are constructed
as combinations of **goal‑irrelevant motor actions** and **grasp types**, focusing on the
hand–object interface itself, and introduce **mechanical action** labels to describe interactions
between tools and target objects citeturn0search10.
They apply this scheme to EPIC‑KITCHENS, generating labels that distinguish between affordances
(e.g., how an object can be grasped or manipulated) and goal‑level actions (e.g., “turn off tap”).

The paper evaluates the new annotations on three tasks:

- Affordance recognition.
- Hand–object interaction hotspot prediction with affordances as weak supervision.
- Cross‑domain affordance generalization.

Results show that affordance‑centric labels yield better generalization and finer‑grained hotspot
prediction than action labels alone citeturn0search10.

For Ego4D‑LiteSTA, this reinforces the importance of separating:

- **What the object can afford** (affordance, grasp, contact regions).
- **What the user is about to do** (verb, goal, task).

Even if the thesis does not introduce new affordance annotations, the ideas support design choices
such as:

- Using hand proximity and object category as strong priors for **next‑active object selection**.
- Optionally integrating external affordance priors (e.g., from EPIC‑KITCHENS) into the STA
  head as an additional score or bias.

---

## 2.6 Ego–Exo Transfer, Ego‑Only Learning, and Retrieval‑Augmented Egocentric Models

The broader egocentric literature has explored two complementary directions:
(1) transferring knowledge from large exocentric datasets to egocentric tasks, and
(2) training “ego‑only” models directly on egocentric data without exocentric transfer.
Both perspectives are useful for positioning Ego4D‑LiteSTA in the design space.

### 2.6.1 Survey of egocentric and exocentric methods

A recent short survey reviews joint egocentric–exocentric learning, highlighting datasets that
contain paired ego–exo views (e.g., CMU‑MMAC, Charades‑Ego, Assembly101, EgoExo4D) and
the corresponding tasks: action recognition, proficiency estimation, action anticipation, pose
estimation, correspondence, temporal segmentation, frame retrieval, and alignment citeturn0search11.

The survey’s central argument is that exocentric videos provide complementary cues—full‑body
pose, global context—that can help interpret egocentric hand–object interactions, and that joint
ego–exo modeling can unlock richer representations for downstream tasks citeturn0search11.
This viewpoint supports the idea that STA could benefit from exocentric priors in principle, but
also emphasizes the practical difficulty of collecting synchronized ego–exo data at scale.

Ego4D‑LiteSTA positions itself closer to the “ego‑only” side (especially when working on
student‑scale hardware and free Colab), but the survey’s taxonomy is helpful for framing potential
extensions, such as using exocentric retrieval to provide additional context for rare interactions.

### 2.6.2 Synchronization‑based exo‑to‑ego transfer for temporal segmentation

“Synchronization is All You Need” proposes a method to adapt a temporal action segmentation
(TAS) model from an exocentric dataset to an egocentric setting using **unlabeled synchronized
ego–exo video pairs** citeturn0search12.
Instead of collecting labeled egocentric videos, the method uses existing exocentric labels plus
unlabeled synchronized pairs and applies knowledge distillation from an exocentric teacher to an
egocentric student at both feature and model levels.

Experiments on Assembly101 and EgoExo4D show that this approach can bridge much of the
performance gap between exocentric‑only models and fully supervised egocentric models, improving edit scores substantially without using any egocentric labels citeturn0search12.

While the task (TAS) differs from STA, the key lesson is that **synchronization can act as a
powerful supervision signal** in ego–exo transfer.
For Ego4D‑LiteSTA, a related idea could be to use synchronized streams (e.g., multi‑view or
multi‑sensor data) for future work, but the base thesis keeps the pipeline simpler and does not rely
on exocentric labels.

### 2.6.3 Ego‑Only: egocentric action detection without exocentric transfer

The Ego‑Only work directly challenges the assumption that exocentric pre‑training is required for
egocentric action detection citeturn0search13.
Instead, it shows that with a sufficiently strong self‑supervised pre‑training strategy—specifically,
a masked autoencoder finetuned for temporal segmentation—one can train competitive egocentric
models using only egocentric video data.

Ego‑Only emphasizes several differences between egocentric and exocentric data that make naive
transfer problematic citeturn0search13:

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
without any exocentric data citeturn0search13.

For Ego4D‑LiteSTA, Ego‑Only provides a strong conceptual justification for focusing on **ego‑only
pre‑training and fine‑tuning**, especially under compute constraints and when exocentric data is
not central to the problem being solved.
Combining Ego‑Only principles with VideoMAE‑style pre‑training is a natural fit for a lightweight
STA head.

### 2.6.4 Early motion transfer between egocentric and exocentric views

EgoTransfer represents an early attempt to model “mirror neurons” by learning mappings between
motion features in egocentric and exocentric videos citeturn0search14.
The authors record time‑synchronized ego–exo video pairs, extract motion features (e.g., optical
flow descriptors) in both views, and train linear or non‑linear models to predict egocentric motion
from exocentric motion and vice versa.

Evaluation is done via cross‑view retrieval: given a motion feature in one view, retrieve its
corresponding feature in the other view from a set of candidates.
Results show that the learned mappings can successfully transfer motion across views, suggesting
that there is a stable relationship between egocentric and exocentric motion patterns citeturn0search14.

For Ego4D‑LiteSTA, EgoTransfer is mostly of historical and conceptual interest.
It reinforces the idea that exocentric information could help interpret egocentric motion and vice
versa, but it also highlights the increasing complexity and data requirements of such approaches.
Given the thesis focus on a practical, lightweight STA pipeline, the design remains ego‑centric
and does not rely on motion transfer from exocentric sources.

### 2.6.5 Retrieval‑augmented egocentric action recognition (REAR)

REAR (Retrieval‑Augmented Egocentric Action Recognition) proposes to augment egocentric
representations with **retrieved exocentric video features** without requiring synchronized ego–exo
pairs citeturn0search15.

The framework uses a dual‑branch architecture:

- A target branch that encodes the egocentric video.
- A retrieval branch that retrieves semantically related exocentric features from a large corpus
  based on similarity in a joint embedding space.

A cross‑view integration module performs staged fusion and attention‑based alignment of
egocentric and exocentric features.
To address long‑tailed class distributions, REAR introduces a **class‑adaptive selector** that varies
the number of retrieved examples per class and uses logit‑adjusted cross‑entropy (LACE) for
training the classifiers citeturn0search15.

Experiments on three egocentric benchmarks show that REAR improves object (noun) recognition
and tail‑class performance, demonstrating the value of exocentric retrieval as an auxiliary
knowledge source citeturn0search15.

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
Aria Digital Twin, and Aria Synthetic Environments citeturn0search16.

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

## 2.8 Summary and Positioning of Ego4D‑LiteSTA

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
