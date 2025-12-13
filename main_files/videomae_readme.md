# VideoMAE: Ego-Only Video Backbone for STA

This document describes the **Ego-only VideoMAE** video backbone variant, which is compared against the baseline **exocentric-transfer** approach using ResNet18.

## Overview

### Baseline: Exocentric Transfer (ResNet18)

The original Track B pipeline uses a **2D image backbone** (ResNet18) pretrained on ImageNet:

```
┌──────────────────────────────────────────────────────────────┐
│  Per-frame Encoding (ResNet18)                               │
│  ─────────────────────────────                               │
│  • Each video frame encoded independently                    │
│  • ResNet18 pretrained on ImageNet (exocentric images)       │
│  • Output: (B, T, N, 512) where N = spatial locations        │
│  • No temporal modeling in backbone                          │
│  • Temporal fusion happens later in TrackBFusion             │
└──────────────────────────────────────────────────────────────┘
```

**Pros:**
- Fast inference (2D convolutions only)
- Benefits from ImageNet's large-scale supervision
- Simple integration with existing pipelines

**Cons:**
- No native temporal modeling
- Pretrained on *exocentric* (third-person) images
- May not capture ego-specific motion patterns

### Ego-Only VideoMAE

Our alternative uses **VideoMAE** self-supervised pretrained on egocentric video:

```
┌──────────────────────────────────────────────────────────────┐
│  Spatiotemporal Encoding (VideoMAE)                          │
│  ──────────────────────────────────                          │
│  • T-frame clips encoded as a video                          │
│  • VideoMAE pretrained on Ego4D clips (self-supervised)      │
│  • 3D patch embedding with tubelet_size=2                    │
│  • Transformer encoder with temporal attention               │
│  • Output: (B, N_patches, 768)                               │
│  • Native spatiotemporal tokens                              │
└──────────────────────────────────────────────────────────────┘
```

**Pros:**
- Native temporal modeling in backbone
- Pretrained on *in-domain* egocentric video
- Self-supervised pretraining captures ego motion patterns
- Potentially better for STA which requires temporal reasoning

**Cons:**
- Heavier compute (3D attention)
- Requires pretraining stage
- Different token structure than per-frame encoding

## Architecture Details

### VideoMAE Encoder

```
Input: (B, T, 3, H, W)  e.g. (4, 16, 3, 224, 224)
       │
       ▼
┌─────────────────────────────────┐
│  Patch Embedding (3D)           │
│  • patch_size: 16×16            │
│  • tubelet_size: 2              │
│  • embed_dim: 768               │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Positional Embedding (3D)      │
│  • Learnable parameters         │
│  • Shape: (1, num_patches, 768) │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Transformer Blocks (×12)       │
│  • Multi-head attention         │
│  • MLP with GELU                │
│  • LayerNorm (pre-norm)         │
└─────────────────────────────────┘
       │
       ▼
Output: (B, num_patches, 768)

num_patches = (T/2) × (H/16) × (W/16)
            = 8 × 14 × 14 = 1568 for T=16, H=W=224
```

### Self-Supervised Pretraining

VideoMAE uses masked autoencoding:

1. **Masking**: 75% of video patches are masked (tube masking)
2. **Encoding**: Only visible patches go through encoder
3. **Decoding**: Lightweight decoder reconstructs masked patches
4. **Loss**: MSE between predicted and original patch pixels

```python
# Pretraining config (videomae_pretrain.py)
PretrainConfig:
    num_frames: 16       # Frames per clip
    mask_ratio: 0.75     # 75% masked
    epochs: 100          # Pretraining epochs
    batch_size: 16       # Per-GPU batch size
    learning_rate: 1.5e-4
```

### Token Flow Comparison

```
                    ResNet18 Path                    VideoMAE Path
                    ─────────────                    ─────────────
Input               (B, T, 3, H, W)                  (B, T, 3, H, W)
                         │                                │
Backbone            ResNet18 per-frame              VideoMAE encoder
                         │                                │
Raw Tokens          (B, T, N, 512)                  (B, N_patches, 768)
                         │                                │
Projector           Linear(512 → 256)               Linear(768 → 256)
                         │                                │
Projected           (B, T, N, 256)                  (B, T, N, 256)*
                         │                                │
                    └────────┬────────┘
                             │
                      TrackBFusion
                             │
                      TrackBHead
                             │
                      Predictions

* VideoMAE tokens are reshaped to (B, T, N, 256) for fusion compatibility
```

## Configuration

### Baseline Config (`trackB_resnet18_baseline.yaml`)

```yaml
model:
  tokenizer:
    video_backbone: "resnet18"
    img_size: 224
    num_frames: 16
    
  projector:
    in_dim: 512    # ResNet18 feature dim
    out_dim: 256
```

### VideoMAE Config (`trackB_videomae_ego.yaml`)

```yaml
model:
  tokenizer:
    video_backbone: "videomae_ego"
    img_size: 224
    num_frames: 16
    videomae_weights: "videomae_ego_encoder.pt"
    
  projector:
    in_dim: 768    # VideoMAE feature dim
    out_dim: 256
    
  videomae:
    embed_dim: 768
    depth: 12
    num_heads: 12
    patch_size: 16
    tubelet_size: 2
```

## Usage

### 1. Pretrain VideoMAE (optional if weights exist)

```bash
python -m trackB.videomae.videomae_pretrain --epochs 100
```

Weights saved to: `runs/VideoMAE/videomae_ego_encoder.pt`

### 2. Train Track B with Baseline

```bash
python -m trackB.trackB_train_loader --config trackB_resnet18_baseline
```

### 3. Train Track B with VideoMAE

```bash
python -m trackB.trackB_train_loader --config trackB_videomae_ego
```

### 4. Compare Results

```bash
python -m trackB.summarize_ablation --output results/ablation_videomae_vs_clip.md
```

## Expected Results

| Model Variant | Video Backbone | mAP | TTC MAE | Notes |
|---------------|----------------|-----|---------|-------|
| Exo-transfer (base) | resnet18 | ~X% | ~Y s | ImageNet pretraining |
| Ego-only VideoMAE | videomae_ego | ~X% | ~Y s | Ego4D pretraining |

*Results will be populated after training runs.*

## Motivation

### The Domain Gap Problem

Traditional vision models are pretrained on:
- **ImageNet**: Static images, third-person perspective
- **Kinetics**: Web videos, varied viewpoints

Egocentric video has unique characteristics:
- First-person perspective with hand-object interactions
- Rapid viewpoint changes from head motion
- Frequent occlusions and motion blur
- Task-focused attention patterns

### In-Domain vs Exocentric Transfer

**Hypothesis**: Self-supervised pretraining on in-domain egocentric video captures:
1. Ego-specific motion patterns
2. Hand-object interaction dynamics
3. Temporal structure of egocentric activities

This may lead to better performance on STA compared to:
- Transferring from exocentric (third-person) image datasets
- Using generic video representations

## Files

```
trackB/
├── videomae/
│   ├── __init__.py              # Module exports
│   ├── videomae_model.py        # VideoMAE encoder/decoder
│   ├── videomae_pretrain_dataset.py  # Ego4D clip dataset
│   └── videomae_pretrain.py     # Self-supervised training
├── trackB_tokenizer.py          # Extended for VideoMAE
├── trackB_train_loader.py       # CLI --config support
├── trackB_tests.py              # Smoke tests for both paths
└── summarize_ablation.py        # Comparison script

configs/
├── trackB.yaml                  # Base config
├── trackB_resnet18_baseline.yaml   # Baseline preset
└── trackB_videomae_ego.yaml     # VideoMAE preset

results/
└── ablation_videomae_vs_clip.md # Comparison table
```

## References

- **VideoMAE**: [Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)
- **Ego4D**: [Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://ego4d-data.org/)
- **MAE**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

## Notes

1. **Baseline is retained**: The ResNet18 pipeline is kept as a strong baseline for comparison
2. **Ablation study**: Direct comparison on STA v2 val with controlled variables
3. **Fusion/Head unchanged**: Only the video backbone differs between variants
4. **Reproducibility**: Same random seeds, training hyperparameters (where applicable)
