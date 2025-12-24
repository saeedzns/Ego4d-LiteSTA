# TrackB Configurable Options Reference

This document lists **all configurable elements** in TrackB that you can modify for different training and evaluation experiments.

---

## 📁 Config Files

Located in `local_extraction/configs/`:

| Config File | Purpose |
|-------------|---------|
| `trackB.yaml` | Main TrackB config (default) |
| `trackB_resnet18_baseline.yaml` | ResNet18 baseline preset |
| `trackB_videomae_ego.yaml` | VideoMAE egocentric preset |
| `trackC.yaml` | TrackC pruning config |
| `base.yaml` | Shared base settings |

**Usage:**
```bash
python -m trackB.trackB_train_loader --config trackB
python -m trackB.trackB_train_loader --config trackB_videomae_ego
```

---

## ✅ Compatibility Guide (Backbone / Tokens / Encoder)

Some Track B config keys are **free knobs** (change them anytime), while others are **compatibility knobs** that must match your backbone, pre-extracted tokens, and/or encoder weights.

This section explains how to tell which is which.

### How to identify “safe vs compatibility” variables

Use this rule of thumb:

- If a key controls **optimization / losses / evaluation toggles** (e.g., `training.*`, `multi_task.*`, `evaluation.*`), it is usually a **safe knob**.
- If a key controls **tensor shapes / encoder construction / token formats** (e.g., `model.tokenizer.*`, `model.projector.in_dim`, `data.tokens_root`), it is a **compatibility knob**.

Practical method:

1) Start from a known-good preset:
   - ResNet baseline: `--config trackB_resnet18_baseline`
   - VideoMAE: `--config trackB_videomae_ego`
2) Only change **safe knobs** first.
3) If you change a compatibility knob, do a quick smoke run:
   - `python local_extraction/trackB/trackB_train_loader.py --config <preset> --demo --epochs 1`
   - `python local_extraction/trackB/trackB_eval.py --config <preset> --checkpoint <path>`

### Safe knobs (generally compatible across backbones)

You can change these without worrying about encoder/token compatibility:

- Training: `training.epochs`, `training.batch_size`, `training.lr`, `training.min_lr`, `training.warmup_epochs`, `training.label_smoothing`, `training.eval_every`, `training.early_stopping.*`, `training.amp`
- Data limits: `training.candidate_limit`, `training.normalize_ttc`
- Multi-task: `multi_task.enabled`, `multi_task.ttc_mode`, `multi_task.ttc_thresholds`, `multi_task.loss_weights.*`
- Eval toggles (usually): hotspot / CLIP flags and weights under `evaluation.*` (these affect scoring, not shapes)

### Compatibility knobs (must match the backbone / tokens)

These keys can cause hard errors or silent mismatch if set inconsistently:

1) **Backbone selection**
- `model.tokenizer.video_backbone`
  - `"resnet18"`: 2D per-frame tokens
  - `"videomae_ego"`: 3D VideoMAE tokens

2) **Pre-extracted tokens**
- `data.tokens_root`
  - If set: the tokenizer will load tokens from disk.
  - The tokens in that folder must match the chosen `video_backbone` (and expected token shapes).
  - If the folder is missing or malformed, Track B falls back to on-the-fly extraction (slower).

3) **Projector input dimension (shape-critical)**
- `model.projector.in_dim`
  - Must match the token feature dimension produced by the backbone/tokens:
    - ResNet18 tokens are typically 512
    - VideoMAE base is typically 768
  - If mismatched, you’ll get a shape error when projecting tokens.

4) **VideoMAE encoder construction (VideoMAE-only)**
Only relevant when `video_backbone: "videomae_ego"`:

- `model.tokenizer.videomae.weights_path` (must exist, otherwise results degrade)
- `model.tokenizer.videomae.embed_dim`, `depth`, `num_heads`, `patch_size`, `tubelet_size`, `num_frames`
  - These must match how the encoder checkpoint was trained.
  - If they do not match, loading weights can fail or produce incompatible outputs.

5) **Temporal window parameters**
- `model.tokenizer.time_len`, `model.tokenizer.time_stride`
  - These affect what frame window is sampled.
  - For VideoMAE, `time_len` typically must match the pretrained model’s expectation (often 16).

### Recommended workflow

- If you’re using **ResNet18**:
  - Prefer changing: `training.*`, `multi_task.*`, `evaluation.*`, `training.candidate_limit`
  - Be cautious changing: `model.projector.in_dim` (should stay 512)

- If you’re using **VideoMAE**:
  - Start from [local_extraction/configs/trackB_videomae_ego.yaml](../configs/trackB_videomae_ego.yaml)
  - Prefer changing: `training.*`, `multi_task.*`, `training.candidate_limit`
  - Avoid changing (unless you know your encoder checkpoint): `videomae.*` block and `projector.in_dim`

### Quick “did I break compatibility?” signals

- Immediate runtime error mentioning shapes (e.g., `mat1 and mat2 shapes cannot be multiplied`) → `model.projector.in_dim` mismatch.
- Errors loading model weights → `videomae.*` parameters don’t match the checkpoint.
- Eval/training runs but metrics collapse unexpectedly → often wrong `weights_path` (random init) or tokens from a different backbone.

---

## 🧠 Model Architecture

### Backbone / Tokenizer

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `model.tokenizer.video_backbone` | `"resnet18"`, `"videomae_ego"` | `"resnet18"` | Feature extraction backbone |
| `model.tokenizer.pretrained` | `true/false` | `true` | Use pretrained weights |
| `model.tokenizer.freeze_backbone` | `true/false` | `true` | Freeze backbone weights |
| `model.tokenizer.img_size` | int | `224` | Input image size |
| `model.tokenizer.time_len` | int | `8` (resnet), `16` (videomae) | Frames in temporal window |
| `model.tokenizer.time_stride` | int | `2` | Step between frames |

### VideoMAE-Specific

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `model.tokenizer.videomae.hf_model_name` | HuggingFace model name | `"MCG-NJU/videomae-base"` | HF pretrained model |
| `model.tokenizer.videomae.weights_path` | path | `null` | Custom weights (overrides HF) |
| `model.tokenizer.videomae.embed_dim` | `768/1024/1280` | `768` | Embedding dimension |
| `model.tokenizer.videomae.patch_size` | int | `16` | Spatial patch size |
| `model.tokenizer.videomae.tubelet_size` | int | `2` | Temporal patch size |
| `model.tokenizer.videomae.depth` | int | `12` | Transformer depth |
| `model.tokenizer.videomae.num_heads` | int | `12` | Attention heads |
| `model.tokenizer.videomae.freeze_encoder` | `true/false` | `true` | Freeze encoder |

### Projector

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `model.projector.in_dim` | int | `512` (resnet), `768` (videomae) | Input dimension |
| `model.projector.out_dim` | int | `256` | Output token dimension |

### Fusion Module (FGTP)

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `model.fusion.dim` | int | `256` | Token dimension |
| `model.fusion.layers` | int | `2` | Number of fusion layers |
| `model.fusion.heads` | int | `8` | Attention heads |
| `model.fusion.dropout` | float | `0.1` | Dropout rate |

### Prediction Head

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `model.head.dim` | int | `256` | Input dimension |
| `model.head.hidden` | int | `256` | Hidden layer size |
| `model.head.num_classes` | int | `2` | Binary next-active classes |
| `model.head.dropout` | float | `0.1` | Dropout rate |

---

## 🏋️ Training Configuration

### Hyperparameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `training.epochs` | int | `20` | Number of training epochs |
| `training.batch_size` | int | `8` | Batch size |
| `training.lr` | float | `1e-3` | Learning rate |
| `training.min_lr` | float | `1e-5` | Minimum LR (for scheduler) |
| `training.warmup_epochs` | float | `1.0` | Warmup period |

### Regularization

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `training.label_smoothing` | float | `0.05` | Label smoothing factor |
| `training.weight_decay` | float | `0.01` | Weight decay |

### Data

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `training.candidate_limit` | int | `16` | Max candidates per frame |
| `training.normalize_ttc` | `true/false` | `true` | Normalize TTC to [0,1] |

### Checkpointing

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `training.save_epoch_checkpoints` | `true/false` | `true` | Save each epoch |
| `training.save_best_checkpoint` | `true/false` | `true` | Save best model |

### Early Stopping

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `training.early_stopping.enabled` | `true/false` | `true` | Enable early stopping |
| `training.early_stopping.patience` | int | `5` | Epochs without improvement |
| `training.early_stopping.monitor` | `"mAP"`, `"accuracy"`, `"ttc_mae"` | `"mAP"` | Metric to monitor |
| `training.early_stopping.mode` | `"max"`, `"min"` | `"max"` | Optimization direction |

### Evaluation During Training

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `training.eval_every` | int | `1` | Evaluate every N epochs |

### Mixed Precision

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `training.amp` | `true/false` | `false` | Automatic mixed precision |

---

## 🎯 Multi-Task Learning

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `multi_task.enabled` | `true/false` | `true` | Enable multi-task heads |
| `multi_task.predict_noun` | `true/false` | `true` | Predict noun class |
| `multi_task.predict_verb` | `true/false` | `true` | Predict verb class |
| `multi_task.predict_ttc` | `true/false` | `true` | Predict time-to-contact |
| `multi_task.ttc_mode` | `"reg"`, `"bin"` | `"reg"` | TTC as regression or classification |
| `multi_task.ttc_bins` | int | `5` | Number of TTC bins (if `bin` mode) |
| `multi_task.ttc_thresholds` | list | `[0.5, 1.0, 2.0]` | TTC bin boundaries |

### Loss Weights

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `multi_task.loss_weights.next_active` | float | `1.0` | Binary next-active loss weight |
| `multi_task.loss_weights.noun` | float | `1.0` | Noun classification loss weight |
| `multi_task.loss_weights.verb` | float | `1.0` | Verb classification loss weight |
| `multi_task.loss_weights.ttc` | float | `1.0` | TTC loss weight |

---

## 📊 Evaluation Configuration

### Basic Settings

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `evaluation.batch_size` | int | `8` | Eval batch size |
| `evaluation.compute_map` | `true/false` | `true` | Compute mean AP |
| `evaluation.compute_ttc_mae` | `true/false` | `true` | Compute TTC MAE |

### Hotspot Priors (PEAR-style)

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `evaluation.hotspot_priors.enabled` | `true/false` | `true` | Use hotspot priors |
| `evaluation.hotspot_priors.path` | path | `local_extraction/v2/hotspot_priors_train_logodds_min10.json` | Priors file |
| `evaluation.hotspot_priors.alpha` | float | `0.3` | Prior blending weight |

### CLIP Re-ranking

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `evaluation.clip_rerank.enabled` | `true/false` | `true` | Use CLIP re-ranking |
| `evaluation.clip_rerank.model` | `"ViT-B/32"`, `"ViT-L/14"`, etc. | `"ViT-B/32"` | CLIP model variant |
| `evaluation.clip_rerank.weight` | float | `0.3` | CLIP score blending weight |

### Visualization

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `evaluation.save_overlays` | `true/false` | `true` | Save visualization overlays |
| `evaluation.topk_overlay` | int | `3` | Number of top predictions to visualize |

---

## 📂 Data Sources

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `data.stageB_run` | path or `null` | `null` (auto-detect) | Stage B run directory |
| `data.train_manifest` | path or `null` | `null` (auto-detect) | Training manifest |
| `data.val_manifest` | path or `null` | `null` (auto-detect) | Validation manifest |
| `data.tokens_root` | path or `null` | `null` | Pre-extracted tokens directory |

---

## ✂️ TrackC: RGTP Pruning

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `rgtp.enabled` | `true/false` | `true` | Enable RGTP pruning |
| `rgtp.rate` | float (0.0-0.95) | `0.1` | Fraction of tokens to prune |
| `rgtp.min_keep` | int | `2` | Minimum tokens to keep |
| `rgtp.temporal_decay` | float | `0.6` | Rollout vs motion blend factor |
| `rgtp.logit_fill` | float | `-12.0` | Logit value for pruned tokens |

### Instrumentation

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `instrumentation.record_latency` | `true/false` | `true` | Measure inference latency |
| `instrumentation.record_vram` | `true/false` | `true` | Record VRAM usage |
| `instrumentation.record_flops` | `true/false` | `true` | Profile FLOPs |

---

## 🖥️ CLI Overrides

Training script supports these CLI arguments:

```bash
python -m trackB.trackB_train_loader \
    --config trackB              # Config preset
    --epochs 30                  # Override epochs
    --batch_size 16              # Override batch size
    --lr 0.0005                  # Override learning rate
    --demo                       # Quick demo mode
```

Evaluation script supports:

```bash
python -m trackB.trackB_eval \
    --checkpoint path/to/checkpoint.pt
```

TrackC pruning:

```bash
python -m trackC.trackC_pruning \
    --config trackC              # Config preset
    --checkpoint path/to/checkpoint.pt
    --rgtp_rate 0.3              # Override pruning rate
    --no_pruning                 # Disable pruning (baseline)
    --video_backbone videomae_ego
```

---

## 🔧 Recommended Ablation Experiments

### 1. Backbone Comparison
```bash
# ResNet18 baseline
python -m trackB.trackB_train_loader --config trackB_resnet18_baseline

# VideoMAE egocentric
python -m trackB.trackB_train_loader --config trackB_videomae_ego
```

### 2. Learning Rate Sweep
```bash
python -m trackB.trackB_train_loader --lr 0.0001
python -m trackB.trackB_train_loader --lr 0.0005
python -m trackB.trackB_train_loader --lr 0.001
python -m trackB.trackB_train_loader --lr 0.005
```

### 3. Multi-Task Ablation
Edit `trackB.yaml`:
```yaml
multi_task:
  enabled: false  # Disable all auxiliary tasks
```

### 4. Loss Weight Tuning
Edit `trackB.yaml`:
```yaml
multi_task:
  loss_weights:
    next_active: 2.0   # Increase primary task weight
    noun: 0.5
    verb: 0.5
    ttc: 0.5
```

### 5. Pruning Rate Sweep (TrackC)
```bash
python -m trackC.trackC_pruning --rgtp_rate 0.0   # Baseline
python -m trackC.trackC_pruning --rgtp_rate 0.1
python -m trackC.trackC_pruning --rgtp_rate 0.3
python -m trackC.trackC_pruning --rgtp_rate 0.5
```

---

## 📋 Your Best ResNet18 Config

From `trackB_best.pt` checkpoint:

```yaml
training:
  epochs: 20
  batch_size: 8
  lr: 0.001
  min_lr: 1e-05
  warmup_epochs: 1.0
  label_smoothing: 0.05
  candidate_limit: 16
  normalize_ttc: true
  amp: false
  early_stopping:
    patience: 5
    monitor: "mAP"
    mode: "max"

multi_task:
  enabled: true
  ttc_mode: "reg"  # regression
  loss_weights:
    next_active: 1.0
    noun: 1.0
    verb: 1.0
    ttc: 1.0

model:
  tokenizer:
    video_backbone: "resnet18"
```

**Result: mAP 35.80%**
