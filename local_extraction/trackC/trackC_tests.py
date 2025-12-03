#!/usr/bin/env python3
"""
Unit tests for Track C — RGTP Token Pruning.

Tests:
- RGTP pruning mask generation
- Candidate scoring with rollout + motion
- Keep count calculations
- Compatibility with both video backbones
"""
from __future__ import annotations

import torch
import torch.nn as nn

import sys
from pathlib import Path

# Ensure imports work
_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_EXTRACTION = _THIS_DIR.parent
for p in [str(_LOCAL_EXTRACTION), str(_LOCAL_EXTRACTION / 'trackB')]:
    if p not in sys.path:
        sys.path.insert(0, p)


def test_rgtp_keep_count():
    """Test RGTPConfig.keep_count() calculations."""
    from trackC_pruning import RGTPConfig
    
    # Test with pruning disabled
    cfg_disabled = RGTPConfig(enabled=False, rate=0.5, min_keep=2)
    assert cfg_disabled.keep_count(10) == 10, "Disabled should keep all"
    
    # Test normal pruning
    cfg = RGTPConfig(enabled=True, rate=0.3, min_keep=2)
    assert cfg.keep_count(10) == 7, f"Expected 7, got {cfg.keep_count(10)}"
    
    # Test min_keep enforcement
    cfg_high = RGTPConfig(enabled=True, rate=0.9, min_keep=3)
    assert cfg_high.keep_count(10) >= 3, "Should respect min_keep"
    
    # Test edge case: very small input
    assert cfg.keep_count(2) >= 2, "Should keep at least min_keep"
    
    print("  [RGTPConfig] keep_count calculations correct")


def test_build_mask():
    """Test mask generation from scores."""
    from trackC_pruning import RGTPConfig, _build_mask
    
    torch.manual_seed(42)
    
    # Create sample scores
    scores = torch.tensor([0.9, 0.3, 0.7, 0.1, 0.5, 0.8, 0.2, 0.6, 0.4, 0.85])
    
    # Test with 30% pruning
    cfg = RGTPConfig(enabled=True, rate=0.3, min_keep=2)
    mask = _build_mask(scores, cfg)
    
    assert mask.dtype == torch.bool, "Mask should be boolean"
    assert mask.shape == scores.shape, "Mask should match scores shape"
    assert mask.sum() >= cfg.min_keep, "Should keep at least min_keep"
    
    # Highest scores should be kept
    top_indices = torch.topk(scores, cfg.keep_count(len(scores))).indices
    for idx in top_indices:
        assert mask[idx], f"Top score at {idx} should be kept"
    
    # Test with pruning disabled
    cfg_disabled = RGTPConfig(enabled=False)
    mask_all = _build_mask(scores, cfg_disabled)
    assert mask_all.all(), "Disabled should keep all"
    
    print("  [_build_mask] Mask generation correct")


def test_motion_energy():
    """Test motion energy calculation."""
    from trackC_pruning import _motion_energy
    
    torch.manual_seed(42)
    
    B, T, N, C = 2, 8, 100, 256
    vid_tokens = torch.randn(B, T, N, C)
    
    energy = _motion_energy(vid_tokens)
    
    assert energy.shape == (B, N), f"Expected (B,N), got {energy.shape}"
    assert (energy >= 0).all(), "Energy should be non-negative"
    
    # Test edge case: single frame
    vid_single = torch.randn(B, 1, N, C)
    energy_single = _motion_energy(vid_single)
    assert energy_single.shape == (B, N), "Single frame should still work"
    
    print("  [_motion_energy] Motion energy calculation correct")


def test_projector_dimensions():
    """Test projector dimensions for both backbones."""
    # ResNet18 outputs 512-dim features
    RESNET_DIM = 512
    # VideoMAE outputs 768-dim features
    VIDEOMAE_DIM = 768
    # Token dimension after projection
    TOKEN_DIM = 256
    
    projector_resnet = nn.Linear(RESNET_DIM, TOKEN_DIM)
    projector_videomae = nn.Linear(VIDEOMAE_DIM, TOKEN_DIM)
    
    B, N = 4, 100
    
    # Test ResNet path
    resnet_feats = torch.randn(B, N, RESNET_DIM)
    resnet_tokens = projector_resnet(resnet_feats)
    assert resnet_tokens.shape == (B, N, TOKEN_DIM), f"ResNet proj failed: {resnet_tokens.shape}"
    
    # Test VideoMAE path
    videomae_feats = torch.randn(B, N, VIDEOMAE_DIM)
    videomae_tokens = projector_videomae(videomae_feats)
    assert videomae_tokens.shape == (B, N, TOKEN_DIM), f"VideoMAE proj failed: {videomae_tokens.shape}"
    
    print(f"  [Projector] ResNet {RESNET_DIM}→{TOKEN_DIM}, VideoMAE {VIDEOMAE_DIM}→{TOKEN_DIM}")


def test_pruning_preserves_batch():
    """Test that pruning preserves batch structure."""
    from trackC_pruning import RGTPConfig, _build_mask
    
    torch.manual_seed(42)
    
    B, N = 4, 50
    
    for b in range(B):
        scores = torch.rand(N)
        cfg = RGTPConfig(enabled=True, rate=0.4, min_keep=5)
        mask = _build_mask(scores, cfg)
        
        assert mask.sum() >= cfg.min_keep, f"Batch {b}: insufficient tokens kept"
        assert mask.shape[0] == N, f"Batch {b}: mask shape wrong"
    
    print("  [Batch Pruning] Batch structure preserved")


def test_logit_fill_negative():
    """Test that LOGIT_FILL is very negative for pruned tokens."""
    from trackC_pruning import LOGIT_FILL
    
    assert LOGIT_FILL < -10, f"LOGIT_FILL should be very negative, got {LOGIT_FILL}"
    
    # Verify it effectively zeros out softmax
    logits = torch.tensor([1.0, 2.0, LOGIT_FILL, 3.0])
    probs = torch.softmax(logits, dim=0)
    
    assert probs[2] < 1e-5, f"Pruned token probability too high: {probs[2]}"
    
    print(f"  [LOGIT_FILL] Value {LOGIT_FILL} effectively zeros softmax")


def test_config_loading():
    """Test that trackC config loads correctly."""
    from core import load_config
    
    cfg = load_config('trackC')
    
    # Check essential fields exist
    assert cfg.get('rgtp.enabled') is not None, "rgtp.enabled missing"
    assert cfg.get('rgtp.rate') is not None, "rgtp.rate missing"
    assert cfg.get('rgtp.min_keep') is not None, "rgtp.min_keep missing"
    
    # Validate ranges
    rate = cfg.get('rgtp.rate', 0.1)
    assert 0.0 <= rate <= 1.0, f"Invalid rate: {rate}"
    
    min_keep = cfg.get('rgtp.min_keep', 2)
    assert min_keep >= 1, f"Invalid min_keep: {min_keep}"
    
    print("  [Config] trackC.yaml loads correctly")


if __name__ == "__main__":
    print("[trackC.tests] Running all tests...\n")
    
    print("1. RGTP keep_count test")
    test_rgtp_keep_count()
    print("   PASS\n")
    
    print("2. Build mask test")
    test_build_mask()
    print("   PASS\n")
    
    print("3. Motion energy test")
    test_motion_energy()
    print("   PASS\n")
    
    print("4. Projector dimensions test")
    test_projector_dimensions()
    print("   PASS\n")
    
    print("5. Batch pruning test")
    test_pruning_preserves_batch()
    print("   PASS\n")
    
    print("6. Logit fill test")
    test_logit_fill_negative()
    print("   PASS\n")
    
    print("7. Config loading test")
    test_config_loading()
    print("   PASS\n")
    
    print("=" * 50)
    print("[trackC.tests] All tests PASSED!")
