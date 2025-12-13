# Track B Training Notebook - Technical Documentation

> **Generated for Thesis Reference**  
> Last Updated: December 6, 2025

---

## Table of Contents

1. [VideoMAE Pretrained Checkpoint Usage](#1-videomae-pretrained-checkpoint-usage)
2. [Backbone Architecture Comparison](#2-backbone-architecture-comparison)
3. [System Architecture Flow](#3-system-architecture-flow)
4. [Configuration Details](#4-configuration-details)

---

## 1. VideoMAE Pretrained Checkpoint Usage

### 1.1 Checkpoint Location

```
local_extraction/
└── videomae_local_part/
    └── checkpoints_ego_scratch/
        └── videomae_ego_scratch_last.pt   # ~305.9 MB
```

### 1.2 Where It's Used in the Notebook

The `videomae_ego_scratch_last.pt` checkpoint is referenced in **two key cells**:

| Cell # | Lines | Purpose | Description |
|--------|-------|---------|-------------|
| **Cell 6** | 113-168 | Backbone Selection | Validates checkpoint exists; sets `BACKBONE = "videomae_ego"` |
| **Cell 14** | 384-430 | Tokenizer Build | Loads weights into VideoMAE encoder via `tok_cfg.videomae_weights_path` |

### 1.3 Why Use This Checkpoint?

The `videomae_ego_scratch_last.pt` contains a **VideoMAE encoder pretrained on Ego4D egocentric videos** using self-supervised Masked Autoencoding (MAE). 

#### Motivation

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DOMAIN ALIGNMENT PRINCIPLE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Training Data Domain  ───────►  Better Feature Extraction          │
│                                                                      │
│   ┌─────────────────┐        ┌─────────────────────────────────┐    │
│   │  Ego4D Videos   │   ══►  │  Egocentric hand-object features │    │
│   │  (egocentric)   │        │  optimized for STA task          │    │
│   └─────────────────┘        └─────────────────────────────────┘    │
│                                                                      │
│   ┌─────────────────┐        ┌─────────────────────────────────┐    │
│   │  ImageNet       │   ══►  │  Generic object features         │    │
│   │  (exocentric)   │        │  domain gap for egocentric       │    │
│   └─────────────────┘        └─────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Key Benefits

| Aspect | VideoMAE (Ego-pretrained) | ResNet18 (ImageNet) |
|--------|---------------------------|---------------------|
| **Pretraining Domain** | Egocentric video (Ego4D) | Exocentric images (ImageNet) |
| **Modality** | Spatio-temporal (video understanding) | Spatial only (single image) |
| **Feature Dimension** | 768-d | 512-d |
| **Spatial Grid** | 14×14 = 196 tokens | 7×7 = 49 tokens |
| **Temporal Context** | 16 frames | 8 frames (repeated single-frame) |
| **Domain Gap** | ✅ None (in-domain) | ⚠️ Significant (exo→ego transfer) |
| **Hand-Object Understanding** | ✅ Learned from egocentric POV | ❌ Limited |

---

## 2. Backbone Architecture Comparison

### 2.1 Architectural Differences

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RESNET18 BACKBONE                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Input: Single Frame (224×224×3)                                     │
│                    │                                                  │
│                    ▼                                                  │
│   ┌────────────────────────────────┐                                 │
│   │  Conv Layers (ResNet18 stem)   │                                 │
│   │  Layer1 → Layer2 → Layer3      │                                 │
│   └────────────────────────────────┘                                 │
│                    │                                                  │
│                    ▼                                                  │
│   ┌────────────────────────────────┐                                 │
│   │     Layer4 Feature Map         │                                 │
│   │     Shape: 7×7×512             │                                 │
│   └────────────────────────────────┘                                 │
│                    │                                                  │
│                    ▼                                                  │
│   Output: 49 tokens × 512 dimensions                                  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────┐
│                       VIDEOMAE BACKBONE                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Input: Video Clip (16×224×224×3)                                    │
│                    │                                                  │
│                    ▼                                                  │
│   ┌────────────────────────────────┐                                 │
│   │   Patch Embedding (3D)         │                                 │
│   │   patch=16×16, tubelet=2       │                                 │
│   │   → 8×14×14 = 1568 patches     │                                 │
│   └────────────────────────────────┘                                 │
│                    │                                                  │
│                    ▼                                                  │
│   ┌────────────────────────────────┐                                 │
│   │   Transformer Encoder          │                                 │
│   │   12 layers, 12 heads          │                                 │
│   │   embed_dim = 768              │                                 │
│   └────────────────────────────────┘                                 │
│                    │                                                  │
│                    ▼                                                  │
│   ┌────────────────────────────────┐                                 │
│   │   Temporal Pooling             │                                 │
│   │   8×14×14 → 14×14              │                                 │
│   └────────────────────────────────┘                                 │
│                    │                                                  │
│                    ▼                                                  │
│   Output: 196 tokens × 768 dimensions                                 │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Feature Comparison Table

| Property | ResNet18 | VideoMAE (Ego) |
|----------|----------|----------------|
| **Architecture Type** | CNN (Convolutional) | ViT (Transformer) |
| **Input Shape** | `(B, 3, 224, 224)` | `(B, 3, 16, 224, 224)` |
| **Output Shape** | `(B, 512, 7, 7)` | `(B, 768, 14, 14)` |
| **Token Count** | 49 | 196 |
| **Token Dimension** | 512 | 768 |
| **Parameters** | ~11M | ~86M |
| **Pretrained On** | ImageNet (1.2M images) | Ego4D (egocentric videos) |
| **Temporal Modeling** | ❌ None | ✅ 16-frame context |
| **Computation** | Fast (single frame) | Slower (video processing) |

---

## 3. System Architecture Flow

### 3.1 Track B Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRACK B: TOKENIZATION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌─────────────┐
     │  Raw Frame  │
     │  (224×224)  │
     └──────┬──────┘
            │
            ▼
┌───────────────────────┐      ┌─────────────────────────────────────────────┐
│   BACKBONE SELECTION  │      │  Cell 6: BACKBONE = "videomae_ego"          │
│   (Cell 6)            │◄─────│  or      BACKBONE = "resnet18"              │
└───────────┬───────────┘      └─────────────────────────────────────────────┘
            │
            ▼
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌────────┐    ┌───────────┐
│ResNet18│    │ VideoMAE  │
│        │    │   Ego     │
└───┬────┘    └─────┬─────┘
    │               │
    │    ┌──────────┘
    │    │  Load: videomae_ego_scratch_last.pt
    │    │  (Cell 14: tok_cfg.videomae_weights_path)
    │    │
    ▼    ▼
┌──────────────────────────────┐
│      build_backbone(cfg)     │
│      (trackB_tokenizer.py)   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Backbone Features        │
│  ResNet: (B, 512, 7, 7)      │
│  VideoMAE: (B, 768, 14, 14)  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      ROI Align / Pooling     │
│  Extract hand-object regions │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│        Token Fusion          │
│   (trackB_fusion.py)         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Prediction Head          │
│  Noun + Verb + TTC + Δ       │
│   (trackB_head.py)           │
└──────────────────────────────┘
```

### 3.2 VideoMAE Wrapper Integration

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    VideoMAEBackboneWrapper Class                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   class VideoMAEBackboneWrapper(nn.Module):                                   │
│       """Wraps VideoMAE encoder with ResNet-compatible interface"""          │
│                                                                               │
│       ┌─────────────────────────────────────────────────────────────────┐    │
│       │  __init__(encoder, checkpoint_path, freeze=True)                │    │
│       │                                                                  │    │
│       │  1. Load VideoMAE encoder architecture                          │    │
│       │  2. Load pretrained weights from checkpoint_path                │    │
│       │  3. If freeze=True: set requires_grad=False for all params      │    │
│       └─────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│       ┌─────────────────────────────────────────────────────────────────┐    │
│       │  forward(x) → features                                          │    │
│       │                                                                  │    │
│       │  Input:  x shape (B, T, C, H, W) or (B, C, H, W)                │    │
│       │  Output: features shape (B, 768, 14, 14)                        │    │
│       │                                                                  │    │
│       │  Processing:                                                     │    │
│       │    1. Ensure 5D input (add temporal dim if needed)              │    │
│       │    2. Rearrange: (B,T,C,H,W) → (B,C,T,H,W)                      │    │
│       │    3. Pass through encoder                                       │    │
│       │    4. Pool temporal dimension                                    │    │
│       │    5. Reshape to (B, D, H, W) for ROI compatibility             │    │
│       └─────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Configuration Details

### 4.1 TokenizerConfig for VideoMAE

```python
tok_cfg = TokenizerConfig()
tok_cfg.video_backbone = "videomae_ego"  # or "resnet18"
tok_cfg.device = "cuda"  # or "cpu"

# VideoMAE-specific settings (only used when backbone == "videomae_ego")
tok_cfg.videomae_weights_path = "path/to/videomae_ego_scratch_last.pt"
tok_cfg.videomae_freeze_encoder = True   # Use as frozen feature extractor
tok_cfg.videomae_patch_size = 16         # Spatial patch size
tok_cfg.videomae_tubelet_size = 2        # Temporal patch size
tok_cfg.videomae_embed_dim = 768         # Transformer hidden dimension
tok_cfg.videomae_depth = 12              # Number of transformer layers
tok_cfg.videomae_num_heads = 12          # Attention heads per layer
```

### 4.2 Configuration Decision Tree

```
                        ┌──────────────────┐
                        │  Start Notebook  │
                        └────────┬─────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  BACKBONE = ?          │
                    └────────────┬───────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                      │
              ▼                                      ▼
    ┌─────────────────┐                   ┌─────────────────┐
    │  "resnet18"     │                   │  "videomae_ego" │
    └────────┬────────┘                   └────────┬────────┘
             │                                      │
             │                                      ▼
             │                            ┌─────────────────────┐
             │                            │ Checkpoint exists?  │
             │                            └──────────┬──────────┘
             │                                       │
             │                          ┌────────────┴────────────┐
             │                          │                         │
             │                          ▼                         ▼
             │                    ┌───────────┐            ┌───────────┐
             │                    │    YES    │            │    NO     │
             │                    └─────┬─────┘            └─────┬─────┘
             │                          │                        │
             │                          │                        ▼
             │                          │              ┌──────────────────┐
             │                          │              │ Fallback to      │
             │                          │              │ resnet18         │
             │                          │              └────────┬─────────┘
             │                          │                       │
             │                          ▼                       │
             │                   ┌────────────────┐             │
             │                   │ Configure:     │             │
             │                   │ - weights_path │             │
             │                   │ - freeze=True  │             │
             │                   │ - embed_dim    │             │
             │                   │ - depth, heads │             │
             │                   └───────┬────────┘             │
             │                           │                      │
             └───────────────────────────┼──────────────────────┘
                                         │
                                         ▼
                              ┌────────────────────┐
                              │  build_backbone()  │
                              │  build_transform() │
                              └────────────────────┘
```

### 4.3 Freeze Encoder Rationale

| Setting | `freeze_encoder = True` | `freeze_encoder = False` |
|---------|-------------------------|--------------------------|
| **Training Mode** | Feature extraction only | Fine-tuning |
| **Gradient Flow** | No gradients through encoder | Full backprop |
| **Memory Usage** | Lower (no activation storage) | Higher |
| **Training Speed** | Faster | Slower |
| **Use Case** | Transfer learned representations | Adapt to new domain |
| **Recommended When** | Limited data, quick experiments | Large dataset, full training |

---

## 5. Notebook Cell Reference

### Quick Reference Table

| Cell # | Title/Purpose | Key Variables | Dependencies |
|--------|---------------|---------------|--------------|
| 3 | Environment Setup | `REPO_ROOT`, `LOCAL_EXT` | - |
| 4 | Path Configuration | `PATHS` dict | Cell 3 |
| 6 | **Backbone Selection** | `BACKBONE`, `BACKBONE_SPECS` | Cell 4 (needs `PATHS['videomae_weights']`) |
| 10 | Module Imports | Clears sys.modules cache | Cell 3 |
| 14 | **Tokenizer Build** | `tok_cfg`, `backbone`, `transform` | Cells 6, 10 |
| 15 | Tokenization Test | `img_tokens`, `vid_tokens` | Cell 14 |

---

## 6. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: videomae_ego_scratch_last.pt` | Checkpoint not downloaded/trained | Run VideoMAE pretraining or switch to `resnet18` |
| `TypeError: build_backbone() got unexpected argument` | Cached old module version | Clear sys.modules (Cell 10 does this) |
| `ModuleNotFoundError: transformers` | HuggingFace not installed | Uses custom loader fallback automatically |
| CUDA OOM with VideoMAE | 768-d, 196 tokens = more memory | Use `freeze_encoder=True` or reduce batch size |

---

*This document is auto-generated for thesis reference. For the latest code, see `notebooks/TrackB_Training_Local_Test.ipynb`.*
