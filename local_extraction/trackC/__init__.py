"""
Track C — Runtime-Guided Token Pruning (RGTP)

Training-free token pruning for inference efficiency. Evaluates Track B 
checkpoints with optional RGTP that prunes low-importance tokens.

Modules:
- trackC_pruning.py — Main evaluation + pruning harness
- trackC_compare_metrics.py — Compare Track B vs Track C metrics
- trackC_plots.py — Visualization of metrics over time
- trackC_tests.py — Unit tests

Supports both video backbones:
- resnet18: ImageNet-pretrained per-frame encoding
- videomae_ego: Ego4D-pretrained spatiotemporal encoding
"""
from pathlib import Path

__all__ = ['TRACK_C_DIR']

TRACK_C_DIR = Path(__file__).parent
