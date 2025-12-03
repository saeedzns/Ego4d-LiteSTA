# Exo-Transfer Baseline in Ego4D-LiteSTA

This document explains where the project uses models pretrained on **non-egocentric (third-person)** data, constituting an **exo-to-ego transfer** baseline.

---

## Overview

Exo-transfer refers to transferring knowledge from models pretrained on exocentric (third-person) datasets—such as COCO or ImageNet—to egocentric (first-person) video understanding tasks. This is a common baseline approach when egocentric-pretrained models are unavailable or to establish a performance reference.

In Ego4D-LiteSTA, **both Track A and Track B** rely on exo-pretrained components:

| Track | Component | Pretrained Source | Fine-tuned On |
|:------|:----------|:------------------|:--------------|
| **A** | YOLOv8-s detector | **COCO** (80-class, third-person) | Ego4D STA single-class (`next_active`) |
| **B** | ResNet18 backbone | **ImageNet** (third-person) | Frozen (no fine-tuning) |

---

## Track A — YOLOv8-s Detector (COCO Pretrained)

### What it is

The Stage A detector uses **YOLOv8-s** initialized from `yolov8s.pt`, which is pretrained on the **COCO dataset** (80 object classes, third-person images).

### Training config

From `local_extraction/toolkit_yolo/runs/sta_yolov8s_singlecls_20251113_002330/args.yaml`:

```yaml
model: yolov8s.pt      # COCO-pretrained weights
pretrained: true       # loads COCO weights
data: .../dataset.yaml # fine-tuned on Ego4D STA
epochs: 30
imgsz: 800
single_cls: false      # (note: labels are single-class "next_active")
```

### Where it is used

- **Training:** `local_extraction/toolkit_yolo/Yolo_Ego4d.ipynb` or Colab
- **Inference:** `local_extraction/trackA/trackA_stageA/trackA_stageA.py`

```python
YOLO_WEIGHTS = ".../sta_yolov8s_singlecls_20251113_002330/weights/best.pt"
```

### Transfer rationale

COCO provides a strong object detection prior (localization, scale handling, NMS). Fine-tuning on Ego4D STA adapts the detector to:
- Egocentric viewpoint (hands, near-field objects)
- Single-class task (`next_active` object only)
- Different aspect ratios and motion blur typical of wearable cameras

---

## Track B — ResNet18 Backbone (ImageNet Pretrained)

### What it is

The Track B tokenizer uses a **ResNet18** backbone pretrained on **ImageNet** (1000-class third-person image classification) to extract spatial grid tokens from frames.

### Code location

From `local_extraction/trackB/trackB_tokenizer.py`:

```python
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        import torchvision.models as models
        m = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # ... layer extraction ...
        
        # Frozen for feature extraction
        for p in self.parameters():
            p.requires_grad_(False)
```

### How it is used

- Extracts `(B, 512, 7, 7)` feature maps from 224×224 input frames
- Provides `img_tokens` (last frame) and `vid_tokens` (temporal window) for fusion
- **Frozen** — no gradient updates during Track B training

### Transfer rationale

ImageNet features provide:
- General visual representations (edges, textures, object parts)
- Stable, well-understood feature space for downstream heads
- Zero training cost for the backbone (frozen)

The fusion head and prediction heads learn egocentric-specific patterns on top of these frozen features.

---

## Contrast with Ego-Only Pretraining

The literature (see `Docs/review.md`, Section 7.3 "Ego-Only") suggests that **in-domain egocentric pretraining** (e.g., VideoMAE on Ego4D unlabeled clips) can outperform exo-transfer for action detection tasks.

However, for this thesis:
- **Exo-transfer is the baseline** — establishes what off-the-shelf third-person models can achieve
- Future work could compare against Omnivore (egocentric-pretrained) or VideoMAE backbones
- The downloaded `omnivore_video_swinl_fp16` features are available but not currently used in the pipeline

---

## Summary

| Aspect | Track A (Detector) | Track B (Tokenizer) |
|:-------|:-------------------|:--------------------|
| **Model** | YOLOv8-s | ResNet18 |
| **Pretrained on** | COCO (80 classes, third-person) | ImageNet (1000 classes, third-person) |
| **Fine-tuned?** | Yes (30 epochs on Ego4D STA) | No (frozen) |
| **Output** | Candidate boxes + confidence | Grid tokens (512-d, 7×7) |
| **Role** | Stage A object proposals | Feature extraction for fusion |

This exo-transfer setup provides a reproducible, resource-efficient baseline. The key contribution of Ego4D-LiteSTA is demonstrating that lightweight fusion (Track B) and training-free pruning (Track C) can improve egocentric STA performance on top of these standard pretrained components.
