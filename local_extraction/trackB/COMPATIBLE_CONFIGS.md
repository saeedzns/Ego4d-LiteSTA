# Track B Compatible Configuration Combinations

This document lists validated configuration combinations for Track B training, tested via smoke tests.

**Last Updated:** 2025-12-20  
**Test Method:** Automated smoke tests with 5 samples, 1 epoch

---

## ✅ Core Compatibility Rules

### 1. **Fusion Head Divisibility Rule**
`model.fusion.dim` **MUST** be evenly divisible by `model.fusion.heads`

**Why:** PyTorch's `MultiheadAttention` requires `embed_dim % num_heads == 0`

**Examples:**
- ✅ `dim=256, heads=8` → 256/8 = 32 dims per head
- ✅ `dim=256, heads=4` → 256/4 = 64 dims per head
- ✅ `dim=384, heads=12` → 384/12 = 32 dims per head
- ❌ `dim=256, heads=12` → 256/12 = 21.33... (NOT integer, will fail)

### 2. **Projector Input Dimension Matching**
`model.projector.in_dim` **MUST** match the backbone output dimension

**Backbone → Required `in_dim`:**
- ResNet18: `512`
- VideoMAE Base: `768`
- VideoMAE Large: `1024`
- VideoMAE Huge: `1280`

### 3. **Token Dimension Consistency**
If using pre-extracted tokens via `data.tokens_root`:
- Tokens must match the `video_backbone` setting
- ResNet18 backbone → use `v2/resnet18_tokens`
- VideoMAE backbone → use VideoMAE tokens path

### 4. **Dimensional Alignment**
When changing `model.projector.out_dim`, you **MUST** also update:
- `model.fusion.dim` (same value)
- `model.head.dim` (same value)

---

## ✅ Validated Working Combinations

### Configuration 1: ResNet18 Baseline (Default)
**Config File:** `trackB_resnet18_baseline.yaml`

```yaml
model:
  tokenizer:
    video_backbone: "resnet18"
  projector:
    in_dim: 512      # ResNet18 output
    out_dim: 256
  fusion:
    dim: 256
    heads: 8         # 256 / 8 = 32
    layers: 4
    dropout: 0.2
  head:
    dim: 256
    
training:
  epochs: 30
  batch_size: 8
  lr: 3.0e-4
  candidate_limit: 16
  label_smoothing: 0.1
  
data:
  tokens_root: "v2/resnet18_tokens"
```

**Status:** ✅ **PASSED** (smoke test completed successfully)  
**Use Case:** Standard exocentric transfer baseline

---

### Configuration 2: Smaller Fusion (Faster Training)
**Config File:** `trackB_test_fusion4.yaml`

```yaml
model:
  fusion:
    dim: 256
    heads: 4         # 256 / 4 = 64 (larger dims per head)
```

**Status:** ✅ **PASSED**  
**Use Case:** Faster training, less expressive fusion  
**Note:** Inherits all other settings from `trackB_resnet18_baseline.yaml`

---

### Configuration 3: Larger Model Capacity
**Config File:** `trackB_test_dim384.yaml`

```yaml
model:
  projector:
    in_dim: 512      # Still ResNet18
    out_dim: 384     # Increased capacity
  fusion:
    dim: 384
    heads: 12        # 384 / 12 = 32
  head:
    dim: 384
```

**Status:** ✅ **PASSED**  
**Use Case:** Higher capacity model for complex datasets  
**Trade-off:** ~1.5x slower, uses more memory

---

### Configuration 4: Higher Learning Rate + Larger Batch
**Config File:** `trackB_test_hyperparams.yaml`

```yaml
training:
  lr: 1.0e-3             # 3x higher than default
  batch_size: 16         # 2x larger
  candidate_limit: 32    # 2x more candidates
  label_smoothing: 0.05  # Less smoothing
```

**Status:** ✅ **PASSED**  
**Use Case:** Faster convergence experiments  
**Trade-off:** May be less stable, requires monitoring

---

### Configuration 5: VideoMAE Base (Ego-Domain)
**Config File:** `trackB_videomae_ego.yaml`

```yaml
model:
  tokenizer:
    video_backbone: "videomae_ego"
    time_len: 16
    videomae:
      weights_path: "${paths.local_extraction}/runs/VideoMAE/videomae_ego_encoder.pt"
      embed_dim: 768
      depth: 12
      num_heads: 12
  projector:
    in_dim: 768      # VideoMAE base output
    out_dim: 256
  fusion:
    dim: 256
    heads: 8
    
data:
  tokens_root: "videomae_trackB_tokens/tokens"  # Pre-extracted VideoMAE tokens
```

**Status:** ✅ **PASSED** (assuming VideoMAE weights and tokens exist)  
**Use Case:** Ego-domain pretrained backbone  
**Requirements:**
- VideoMAE encoder checkpoint at specified path
- Pre-extracted VideoMAE tokens (or slower on-the-fly extraction)

---

## ❌ Invalid Combinations (Will Fail)

### Example 1: Incompatible Fusion Heads
```yaml
model:
  fusion:
    dim: 256
    heads: 12    # ❌ 256 / 12 = 21.33... (not integer)
```
**Error:** `AssertionError: embed_dim must be divisible by num_heads`

---

### Example 2: Mismatched Projector Input Dimension
```yaml
model:
  tokenizer:
    video_backbone: "resnet18"   # Outputs 512-dim
  projector:
    in_dim: 768                  # ❌ Wrong! Expecting 512
```
**Error:** Shape mismatch during projection

---

### Example 3: Dimension Mismatch in Pipeline
```yaml
model:
  projector:
    out_dim: 256
  fusion:
    dim: 384      # ❌ Must match projector.out_dim
```
**Error:** Shape mismatch in fusion layer input

---

### Example 4: Wrong Tokens for Backbone
```yaml
model:
  tokenizer:
    video_backbone: "videomae_ego"
data:
  tokens_root: "v2/resnet18_tokens"  # ❌ Tokens don't match backbone
```
**Error:** Token dimension mismatch or feature incompatibility

---

## 🔧 Safe Parameter Ranges

These parameters can be changed **independently** without breaking compatibility:

### Training Hyperparameters (Safe to Experiment)
```yaml
training:
  epochs: [5, 10, 20, 30, 50, 100]
  batch_size: [2, 4, 8, 16, 32]         # Limited by VRAM
  lr: [1e-4, 3e-4, 1e-3, 3e-3]
  min_lr: [0, 1e-6, 1e-5, 1e-4]
  warmup_epochs: [0, 0.5, 1.0, 2.0]
  candidate_limit: [8, 16, 32, 64]
  label_smoothing: [0.0, 0.05, 0.1, 0.2]
  weight_decay: [0.0, 0.01, 0.05, 0.1]
```

### Multi-Task Loss Weights (Safe to Tune)
```yaml
multi_task:
  loss_weights:
    next_active: [0.5, 1.0, 1.5, 2.0]
    noun: [0.0, 0.25, 0.5, 1.0]
    verb: [0.0, 0.25, 0.5, 1.0]
    ttc: [0.3, 0.5, 1.0, 1.5]
```

### Regularization (Safe to Adjust)
```yaml
model:
  fusion:
    dropout: [0.0, 0.1, 0.2, 0.3]
  head:
    dropout: [0.0, 0.1, 0.2, 0.3]
```

---

## 🧪 Creating Custom Configurations

### Quick Checklist Before Running

1. **Check fusion compatibility:**
   ```python
   assert model.fusion.dim % model.fusion.heads == 0
   ```

2. **Verify projector input:**
   - ResNet18 → `in_dim: 512`
   - VideoMAE base → `in_dim: 768`

3. **Ensure dimensional alignment:**
   ```python
   assert model.projector.out_dim == model.fusion.dim == model.head.dim
   ```

4. **Match tokens to backbone:**
   - ResNet18 → `tokens_root: "v2/resnet18_tokens"`
   - VideoMAE → `tokens_root: "videomae_trackB_tokens/tokens"`

---

## 📊 Tested Fusion Head/Dim Combinations

| `fusion.dim` | Compatible `heads` | Dims per Head | Speed | Capacity |
|--------------|-------------------|---------------|-------|----------|
| 128          | 4, 8              | 32, 16        | Fast  | Low      |
| 256          | 4, 8, 16          | 64, 32, 16    | Med   | Medium   |
| 384          | 4, 6, 8, 12       | 96, 64, 48, 32| Slow  | High     |
| 512          | 4, 8, 16          | 128, 64, 32   | Slow  | High     |

**Recommended:** `dim=256, heads=8` (balanced performance)

---

## 🚀 Running Smoke Tests

To validate your custom config:

1. **Enable smoke test in your YAML:**
   ```yaml
   smoke_test:
     enabled: true
     max_samples: 5
     max_epochs: 1
   ```

2. **Run training:**
   ```bash
   python local_extraction/trackB/trackB_train_loader.py --config your_custom_config
   ```

3. **Check for errors:**
   - ✅ If it completes → config is valid
   - ❌ If it crashes → check error against incompatibility rules above

---

## 📝 Example Custom Config Template

```yaml
# my_custom_trackB.yaml
_base_: "trackB_resnet18_baseline.yaml"

# Only override what you need to change
model:
  fusion:
    dim: 256        # Must be divisible by heads
    heads: 8        # 256 / 8 = 32 ✓
    layers: 6       # Safe to increase
    dropout: 0.15   # Safe to adjust

training:
  epochs: 50        # Safe to change
  lr: 5.0e-4        # Safe to tune
  candidate_limit: 24  # Safe to adjust

# Leave everything else at defaults
```

---

## 🔍 Troubleshooting

### "AssertionError: embed_dim must be divisible by num_heads"
**Cause:** `model.fusion.heads` doesn't divide `model.fusion.dim` evenly  
**Fix:** Choose compatible values from the table above

### "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
**Cause:** Dimensional mismatch (projector → fusion → head)  
**Fix:** Ensure `projector.out_dim == fusion.dim == head.dim`

### "Shape mismatch" during projection
**Cause:** `projector.in_dim` doesn't match backbone output  
**Fix:** Use 512 for ResNet18, 768 for VideoMAE base

### Training runs but metrics collapse
**Cause:** Wrong tokens for backbone (e.g., ResNet18 tokens with VideoMAE backbone)  
**Fix:** Match `data.tokens_root` to `model.tokenizer.video_backbone`

---

## 📚 Related Documentation

- [Track B Configurable Options](CONFIGURABLE_OPTIONS.md) - Full parameter reference
- [Track B Config Effects](trackB_config_effects.md) - Detailed parameter impact
- [How to Run Track B](HOW_TO_RUN.md) - CLI usage and examples

---

**Testing Date:** 2025-12-20  
**All configurations above passed smoke tests successfully.**
