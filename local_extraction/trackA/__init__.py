"""
Track A — Object Detection and Candidate Generation

Stage A: Run YOLO/oracle detector on frames to generate candidate boxes.
Stage B: Crop candidates and prepare manifests for Track B training.

Modules:
- trackA_stageA/trackA_stageA.py — Detector (YOLO, oracle, grid modes)
- trackA_stageB/trackA_stageB.py — Cropping and manifest preparation
- trackA_plots.py — Visualization utilities
- trackA_tests.py — Unit tests
"""
from pathlib import Path

__all__ = ['TRACK_A_DIR']

TRACK_A_DIR = Path(__file__).parent
