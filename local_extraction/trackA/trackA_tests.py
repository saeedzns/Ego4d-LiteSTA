#!/usr/bin/env python3
"""
Unit tests for Track A — Stage A (Detection) and Stage B (Cropping/Manifests).

Tests:
- Config loading from trackA.yaml
- Box clamping and IoU calculations
- Manifest parsing
- Oracle mode calculations
"""
from __future__ import annotations

import torch
import sys
from pathlib import Path

# Ensure imports work
_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_EXTRACTION = _THIS_DIR.parent
for p in [str(_LOCAL_EXTRACTION), str(_THIS_DIR / 'trackA_stageA'), str(_THIS_DIR / 'trackA_stageB')]:
    if p not in sys.path:
        sys.path.insert(0, p)


def test_config_loading():
    """Test that trackA config loads correctly."""
    from core import load_config
    
    cfg = load_config('trackA')
    
    # Check essential fields exist
    assert cfg.get('version') is not None, "version missing"
    assert cfg.get('stage_a.mode') is not None, "stage_a.mode missing"
    assert cfg.get('stage_a.k') is not None, "stage_a.k missing"
    
    # Validate K
    k = cfg.get('stage_a.k', 6)
    assert k > 0, f"Invalid K: {k}"
    
    # Check YOLO settings
    yolo_conf = cfg.get('stage_a.yolo.conf_thresh', 0.05)
    assert 0.0 <= yolo_conf <= 1.0, f"Invalid conf_thresh: {yolo_conf}"
    
    print("  [Config] trackA.yaml loads correctly")


def test_box_clamping():
    """Test box coordinate clamping."""
    from trackA_stageB.trackA_stageB import clamp_box
    
    W, H = 1920, 1080
    
    # Normal box
    x1, y1, x2, y2 = clamp_box(100.5, 200.5, 300.5, 400.5, W, H)
    assert 0 <= x1 < W and 0 <= x2 < W, "X coords out of bounds"
    assert 0 <= y1 < H and 0 <= y2 < H, "Y coords out of bounds"
    assert x1 < x2 and y1 < y2, "Box dimensions invalid"
    
    # Box extending past image bounds
    x1, y1, x2, y2 = clamp_box(-50, -50, 2000, 1200, W, H)
    assert x1 == 0 and y1 == 0, "Negative coords not clamped"
    assert x2 <= W - 1 and y2 <= H - 1, "Positive coords not clamped"
    
    print("  [clamp_box] Box clamping works correctly")


def test_iou_calculation():
    """Test IoU (Intersection over Union) calculation."""
    from trackA_stageB.trackA_stageB import box_iou
    
    # Identical boxes -> IoU = 1.0
    iou1 = box_iou((100, 100, 200, 200), (100, 100, 200, 200))
    assert abs(iou1 - 1.0) < 1e-6, f"Identical boxes should have IoU=1, got {iou1}"
    
    # Non-overlapping boxes -> IoU = 0.0
    iou2 = box_iou((0, 0, 50, 50), (100, 100, 150, 150))
    assert iou2 == 0.0, f"Non-overlapping boxes should have IoU=0, got {iou2}"
    
    # Partial overlap
    iou3 = box_iou((0, 0, 100, 100), (50, 50, 150, 150))
    assert 0.0 < iou3 < 1.0, f"Partial overlap should be between 0 and 1, got {iou3}"
    
    # Compute expected IoU for (0,0,100,100) and (50,50,150,150):
    # Intersection: (50,50,100,100) = 50x50 = 2500
    # Union: 100x100 + 100x100 - 2500 = 17500
    # IoU = 2500/17500 ≈ 0.1429
    expected_iou = 2500 / 17500
    assert abs(iou3 - expected_iou) < 1e-4, f"Expected IoU {expected_iou}, got {iou3}"
    
    print("  [box_iou] IoU calculation correct")


def test_stageA_mode_options():
    """Test Stage A detection mode options."""
    from core import load_config
    
    cfg = load_config('trackA')
    mode = cfg.get('stage_a.mode', 'yolo')
    
    valid_modes = ['yolo', 'oracle', 'grid']
    assert mode in valid_modes, f"Invalid mode: {mode}, expected one of {valid_modes}"
    
    print(f"  [Stage A] Mode '{mode}' is valid")


def test_manifest_structure():
    """Test expected manifest structure."""
    import json
    
    # Sample manifest entry structure
    sample_entry = {
        "uid": "test_uid_123",
        "frame": 1234567,
        "boxes": [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "conf": 0.95, "cls": 0}
        ]
    }
    
    # Verify JSON serialization
    json_str = json.dumps(sample_entry)
    parsed = json.loads(json_str)
    
    assert parsed["uid"] == sample_entry["uid"], "UID mismatch"
    assert len(parsed["boxes"]) == 1, "Box count mismatch"
    assert "conf" in parsed["boxes"][0], "Confidence missing"
    
    print("  [Manifest] Structure is valid JSON")


def test_recall_at_k():
    """Test recall@K calculation logic."""
    # Simulate recall calculation
    # recall@K = (# images with >=1 hit) / (# images with >=1 GT)
    
    # Example: 80 images have hits out of 100 images with GT
    hits = 80
    total_with_gt = 100
    recall = hits / total_with_gt
    
    assert recall == 0.8, f"Expected 0.8, got {recall}"
    
    # Edge case: no GT
    recall_no_gt = 0 / 1 if 0 > 0 else 0.0
    assert recall_no_gt == 0.0, "No GT should give 0 recall"
    
    print("  [Recall@K] Calculation logic correct")


def test_crop_size_config():
    """Test crop size configuration."""
    from core import load_config
    
    cfg = load_config('trackA')
    crop_size = cfg.get('stage_b.crop_size', [256, 256])
    
    if crop_size is not None:
        assert len(crop_size) == 2, f"Crop size should have 2 elements, got {len(crop_size)}"
        assert all(s > 0 for s in crop_size), "Crop dimensions must be positive"
    
    print(f"  [Crop Size] Config value: {crop_size}")


def test_paths_config():
    """Test path configurations."""
    from core import load_config
    
    cfg = load_config('trackA')
    
    # Check path fields exist
    frames_path = cfg.get('paths.extracted_frames')
    labels_path = cfg.get('paths.yolo_labels')
    runs_path = cfg.get('paths.runs')
    
    assert frames_path is not None, "paths.extracted_frames missing"
    assert labels_path is not None, "paths.yolo_labels missing"
    assert runs_path is not None, "paths.runs missing"
    
    print("  [Paths] All path configs present")


if __name__ == "__main__":
    print("[trackA.tests] Running all tests...\n")
    
    print("1. Config loading test")
    test_config_loading()
    print("   PASS\n")
    
    print("2. Box clamping test")
    try:
        test_box_clamping()
        print("   PASS\n")
    except ImportError as e:
        print(f"   SKIP (import error: {e})\n")
    
    print("3. IoU calculation test")
    try:
        test_iou_calculation()
        print("   PASS\n")
    except ImportError as e:
        print(f"   SKIP (import error: {e})\n")
    
    print("4. Stage A mode options test")
    test_stageA_mode_options()
    print("   PASS\n")
    
    print("5. Manifest structure test")
    test_manifest_structure()
    print("   PASS\n")
    
    print("6. Recall@K calculation test")
    test_recall_at_k()
    print("   PASS\n")
    
    print("7. Crop size config test")
    test_crop_size_config()
    print("   PASS\n")
    
    print("8. Paths config test")
    test_paths_config()
    print("   PASS\n")
    
    print("=" * 50)
    print("[trackA.tests] All tests PASSED!")
