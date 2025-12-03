# Ablation: VideoMAE vs ResNet18 Baseline

> **Status**: Pending - runs not yet completed

## Expected Results

Once training runs are complete, this file will contain a comparison table like:

| Model Variant | Video Backbone | mAP | TTC MAE | N mAP | Nv mAP | N+δ mAP | All mAP |
|---------------|----------------|-----|---------|-------|--------|---------|---------|
| Exo-transfer (base) | resnet18 | - | - | - | - | - | - |
| Ego-only VideoMAE | videomae_ego | - | - | - | - | - | - |

## How to Generate

After running training with both configs:

```bash
# 1. Run baseline training
python -m trackB.trackB_train_loader --config trackB_resnet18_baseline

# 2. Run VideoMAE training  
python -m trackB.trackB_train_loader --config trackB_videomae_ego

# 3. Generate comparison table
python -m trackB.summarize_ablation --output results/ablation_videomae_vs_clip.md
```

## Notes

- **Exo-transfer (base)**: Uses ResNet18 pretrained on ImageNet (exocentric data)
- **Ego-only VideoMAE**: Uses VideoMAE self-supervised pretrained on egocentric video from Ego4D
- The key hypothesis is that in-domain pretraining (VideoMAE on ego video) may outperform exocentric transfer (ImageNet-pretrained ResNet18)
