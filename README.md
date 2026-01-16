# Ego4D-LiteSTA: Lightweight Short-Term Object Interaction Anticipation

A modular three-track pipeline for reproducible egocentric anticipation on consumer hardware. Predicts **what object** (bounding box), **which action** (verb/noun), and **when** (time-to-contact) before physical interaction occurs.


![big_picture](thesis/big_picture_pipeline_diagram.png)


## 🎯 Key Results

- **Track A**: 67.8% Recall@6 with lightweight YOLO detector (6 hours GPU training)
- **Track B**: 15.28% N-mAP, 4.81% Overall mAP (class-weighted, ~3 hours CPU training)
- **Track C**: 42% latency reduction (40.7ms → 23.6ms) with 99.9% accuracy retention

**Architecture Insights**:
- Exo-transfer (ImageNet ResNet18) outperforms ego-pretrained VideoMAE by 7.19%
- Class weighting prevents collapse: 123% noun diversity gain (44→98 classes)
- TTC prediction (43.3% < 100ms) transfers better than semantic classification (20.14% noun accuracy)

## 📋 Prerequisites

```bash
# System requirements
- Python 3.8+
- CPU: 16GB RAM (local training)
- GPU: Google Colab Pro (YOLO fine-tuning, VideoMAE pretraining)

# Install dependencies
pip install torch torchvision opencv-python pyyaml tqdm numpy pandas
pip install ultralytics  # YOLOv8
```

## 🚀 Quick Start (3-Step Pipeline)

### Step 1: Track A - Generate Candidate Proposals

```bash
cd local_extraction/trackA/trackA_stageA

# Fine-tune YOLOv8 detector (Google Colab, 6 hours)
python train_yolo_ego4d.py --config configs/yolo_finetune.yaml

# Generate proposals on decision frames (local, ~30 min)
python generate_proposals.py --checkpoint best.pt --k 6 --split val
```

**Output**: `manifests/candidates_k6_val.jsonl` (2,323 decision frames × 6 candidates/frame)

### Step 2: Track B - Train Fusion Head & Score Candidates

```bash
cd local_extraction/trackB

# Extract video tokens (local CPU, ~1.5 hours)
python extract_tokens_resnet18.py --manifest ../trackA/manifests/candidates_k6_val.jsonl

# Train lightweight fusion head (local CPU, ~3 hours)
python train_fusion_head.py --config configs/resnet18_weighted.yaml \
    --tokens v2/resnet18_tokens \
    --manifest ../trackA/manifests/candidates_k6_train.jsonl

# Evaluate on validation set
python evaluate.py --checkpoint trackB_best_mAP_0.3708.pt \
    --manifest ../trackA/manifests/candidates_k6_val.jsonl
```

**Output**: Top-5 mAP metrics + ranked predictions per decision frame

### Step 3: Track C - Efficiency Analysis (Training-Free)

```bash
cd local_extraction/trackC

# Apply rollout-guided token pruning at inference (no retraining)
python evaluate_pruning.py --checkpoint ../trackB/trackB_best_mAP_0.3708.pt \
    --prune_rates 0.0 0.1 0.3 0.5 \
    --measure_latency

# Generate Pareto frontier plot
python plot_pareto.py --metrics trackC_results.json
```

**Output**: Accuracy-latency trade-off curves (42% speedup @ 99.9% accuracy retention)

## 📂 Project Structure

```
local_extraction/
├── trackA/               # Proposal generation (YOLO fine-tuning)
│   ├── trackA_stageA/    # Stage A: Candidate detection
│   └── trackA_stageB/    # Stage B: Manifest construction
├── trackB/               # Fusion head training (ResNet18 + 4-layer transformer)
│   ├── train_fusion_head.py
│   ├── extract_tokens_resnet18.py
│   └── configs/          # YAML configs for reproducibility
├── trackC/               # Training-free efficiency analysis (RGTP pruning)
├── labels/               # Ground truth annotations (Ego4D-STA v2)
└── final_scripts/        # End-to-end execution scripts
```

## 🎨 Qualitative Results

### Success Cases (thesis/success_*.jpg)
![Success Example](thesis/success_01.jpg)

**When it works**:
- Unobstructed, well-lit objects (cups, bottles, pans)
- Predictable hand trajectories toward contact zones
- Clean foreground-background separation

### Failure Cases (thesis/failure_*.jpg)
![Failure Example](thesis/failure_01.jpg)

**Common failures**:
- **Confusion**: Visually similar objects (multiple cups, identical containers)
- **Occlusion**: Hands blocking target objects during motion
- **Tail classes**: Rare verb/noun combinations (e.g., "sponge" with 7 training examples)

## 📊 Reproducing Key Experiments

### Ablation 1: Class Weighting Impact

```bash
# Train WITHOUT class weighting (baseline)
python trackB/train_fusion_head.py --config configs/resnet18_unweighted.yaml

# Train WITH class weighting (α=0.5)
python trackB/train_fusion_head.py --config configs/resnet18_weighted.yaml

# Compare diversity metrics
python final_scripts/analyze_diversity.py --checkpoints unweighted.pt weighted.pt
```

**Expected**: Weighted checkpoint sacrifices 2.7% mAP but gains 123% noun diversity (44→98 unique classes)

### Ablation 2: ResNet18 vs VideoMAE

```bash
# Extract VideoMAE tokens (requires Colab GPU, ~7 hours pretraining)
python trackB/extract_tokens_videomae.py --manifest trackA/manifests/candidates_k6_train.jsonl

# Train with VideoMAE backbone
python trackB/train_fusion_head.py --config configs/videomae_weighted.yaml

# Compare with ResNet18 baseline
python final_scripts/compare_backbones.py --checkpoints resnet18.pt videomae.pt
```

**Expected**: ResNet18 outperforms VideoMAE by 7.19% overall (exo-transfer wins for spatial discrimination tasks)

## 🔧 Configuration System

All experiments use YAML configs for reproducibility:

```yaml
# Example: configs/resnet18_weighted.yaml
model:
  backbone: resnet18
  fusion_layers: 4
  token_dim: 512
  
training:
  epochs: 50
  learning_rate: 0.0001
  class_weights: true
  weight_alpha: 0.5
  
data:
  manifest: trackA/manifests/candidates_k6_train.jsonl
  tokens: v2/resnet18_tokens
  batch_size: 32
```

Run logs are auto-saved to `runs/` with timestamps for full traceability.

## 📈 Performance Benchmarks

| Method | N-mAP | N+V-mAP | Overall-mAP | Latency (ms) |
|--------|-------|---------|-------------|--------------|
| STAformer + AFF | 29.39% | 15.38% | 5.67% | ~200ms* |
| **Ego4D-LiteSTA** | 15.28% | 4.81% | 10.53%** | 23.6ms |

*Estimated from paper (DINOv2 + TimeSformer)  
**Unweighted checkpoint for fair comparison (weighted: 4.81% Overall-mAP, preferred in practice)

## 🐛 Troubleshooting

**Issue**: Low recall in Track A  
**Fix**: Increase K to 8-10 or retune YOLO confidence threshold

**Issue**: Class collapse (predicting only "hand", "cup")  
**Fix**: Enable class weighting with α=0.5 in config

**Issue**: OOM during token extraction  
**Fix**: Reduce batch size or use CPU-only mode for ResNet18 tokens

## 📖 Citation

```bibtex
@mastersthesis{ego4d_litesta_2026,
  title={Ego4D-LiteSTA: A Lightweight, Modular Pipeline for Reproducible Short-Term Object Interaction Anticipation},
  author={[Saeed Zohoorian]},
  year={2026},
  school={[Sapienza University]}
}
```

## 📜 License

MIT License - See LICENSE file for details. Built on Ego4D dataset (Meta AI, CC BY-NC 4.0).

---

**See `thesis/` folder for additional visualizations and analysis figures.**
