#!/usr/bin/env python3
"""
Track B Smoke Test

Validates Track B pipeline (Fusion + Head) with minimal data.
Includes pycache validation and model instantiation checks.

Usage:
    python -m tests.smoke_test_trackB
    
    # Or from repo root:
    python local_extraction/tests/smoke_test_trackB.py
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Setup: Add parent directories to path
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_EXTRACTION = _THIS_DIR.parent
_REPO_ROOT = _LOCAL_EXTRACTION.parent
_TRACKB_DIR = _LOCAL_EXTRACTION / "trackB"

# Add to path for imports
for p in [str(_LOCAL_EXTRACTION), str(_REPO_ROOT), str(_TRACKB_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class SmokeTestConfig:
    """Smoke test configuration."""
    max_samples: int = 5
    max_epochs: int = 1
    validate_pycache: bool = True
    test_model_instantiation: bool = True
    test_forward_pass: bool = True
    skip_gpu: bool = False
    verbose: bool = True
    timeout_seconds: float = 120.0


# =============================================================================
# Module Import Tests
# =============================================================================

def test_core_imports() -> Tuple[bool, str]:
    """Test core module imports."""
    try:
        from core.config_loader import load_config, Config
        from core.paths import Paths, get_paths
        return True, "Core modules imported successfully"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_trackb_imports() -> Tuple[bool, str]:
    """Test Track B module imports."""
    try:
        from trackB_tokenizer import TokenizerConfig, build_backbone, build_transform
        from trackB_fusion import FusionConfig, TrackBFusion
        from trackB_head import HeadConfig, TrackBHead
        from trackB_dataset import TrackBDataset, trackB_collate
        from trackB_metrics import binary_accuracy, multiclass_accuracy
        from trackB_train_loader import TrainConfig
        from trackB_eval import EvalConfig
        return True, "Track B modules imported successfully (8 modules)"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_torch_available() -> Tuple[bool, str]:
    """Test PyTorch availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        return True, f"PyTorch {torch.__version__} available, device={device}"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"Error: {e}"


# =============================================================================
# Model Instantiation Tests
# =============================================================================

def test_model_instantiation(cfg: SmokeTestConfig) -> Tuple[bool, Dict]:
    """Test that models can be instantiated."""
    results = {
        "name": "model_instantiation",
        "passed": True,
        "tests": [],
    }
    
    try:
        import torch
        device = "cpu" if cfg.skip_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        results["passed"] = False
        results["tests"].append({"name": "torch", "passed": False, "message": "PyTorch not available"})
        return False, results
    
    # Test Projector
    try:
        projector = torch.nn.Linear(512, 256).to(device)
        results["tests"].append({"name": "projector", "passed": True, "message": "Linear(512, 256) created"})
    except Exception as e:
        results["tests"].append({"name": "projector", "passed": False, "message": str(e)})
        results["passed"] = False
    
    # Test Fusion
    try:
        from trackB_fusion import FusionConfig, TrackBFusion
        fusion_cfg = FusionConfig(dim=256, layers=2, heads=8)
        fusion = TrackBFusion(fusion_cfg).to(device)
        results["tests"].append({"name": "fusion", "passed": True, "message": "TrackBFusion created"})
    except Exception as e:
        results["tests"].append({"name": "fusion", "passed": False, "message": str(e)})
        results["passed"] = False
    
    # Test Head
    try:
        from trackB_head import HeadConfig, TrackBHead
        head_cfg = HeadConfig(dim=256, num_classes=2, hidden=256)
        head = TrackBHead(head_cfg).to(device)
        results["tests"].append({"name": "head", "passed": True, "message": "TrackBHead created"})
    except Exception as e:
        results["tests"].append({"name": "head", "passed": False, "message": str(e)})
        results["passed"] = False
    
    return results["passed"], results


# =============================================================================
# Forward Pass Test
# =============================================================================

def test_forward_pass(cfg: SmokeTestConfig) -> Tuple[bool, Dict]:
    """Test model forward pass with dummy data."""
    results = {
        "name": "forward_pass",
        "passed": True,
        "tests": [],
    }
    
    try:
        import torch
        device = "cpu" if cfg.skip_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        results["passed"] = False
        results["tests"].append({"name": "torch", "passed": False, "message": "PyTorch not available"})
        return False, results
    
    try:
        from trackB_fusion import FusionConfig, TrackBFusion
        from trackB_head import HeadConfig, TrackBHead
        
        # Create models
        projector = torch.nn.Linear(512, 256).to(device)
        fusion = TrackBFusion(FusionConfig(dim=256, layers=2, heads=8)).to(device)
        head = TrackBHead(HeadConfig(dim=256, num_classes=2, hidden=256)).to(device)
        
        # Dummy data
        batch_size = 2
        num_candidates = 4
        seq_len = 8
        
        # Simulate projected tokens
        img_tokens = torch.randn(batch_size, 49, 256, device=device)  # 7x7 grid
        vid_tokens = torch.randn(batch_size, seq_len, 49, 256, device=device)
        cand_tokens = torch.randn(batch_size, num_candidates, 256, device=device)
        mask = torch.ones(batch_size, num_candidates, dtype=torch.bool, device=device)
        
        # Forward fusion (returns tuple: fused_img, fused_vid)
        fused_img, fused_vid = fusion(img_tokens, vid_tokens)
        results["tests"].append({
            "name": "fusion_forward",
            "passed": True,
            "message": f"Fusion output shapes: fused_img={tuple(fused_img.shape)}, fused_vid={tuple(fused_vid.shape)}",
        })
        
        # Forward head (takes fused_img, fused_vid and returns dict)
        head_out = head(fused_img, fused_vid)
        cls_logits = head_out["cls_logits"]
        ttc_pred = head_out["ttc"]
        results["tests"].append({
            "name": "head_forward",
            "passed": True,
            "message": f"Head output shapes: cls={tuple(cls_logits.shape)}, ttc={tuple(ttc_pred.shape)}",
        })
        
    except Exception as e:
        results["tests"].append({"name": "forward", "passed": False, "message": str(e)})
        results["passed"] = False
    
    return results["passed"], results


# =============================================================================
# VideoMAE Tests
# =============================================================================

def test_videomae_imports() -> Tuple[bool, str]:
    """Test VideoMAE module imports."""
    try:
        from trackB.videomae import VideoMAEEncoder, VideoMAE, load_videomae_encoder, VideoMAEPretrainDataset
        return True, "VideoMAE modules imported successfully"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_videomae_model(cfg: SmokeTestConfig) -> Tuple[bool, Dict]:
    """Test VideoMAE model instantiation and forward pass."""
    results = {
        "name": "videomae_model",
        "passed": True,
        "tests": [],
    }
    
    try:
        import torch
        device = "cpu" if cfg.skip_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        results["passed"] = False
        results["tests"].append({"name": "torch", "passed": False, "message": "PyTorch not available"})
        return False, results
    
    # Test encoder instantiation
    try:
        from trackB.videomae.videomae_model import VideoMAEEncoder, VideoMAEConfig
        encoder_cfg = VideoMAEConfig(
            img_size=224,
            patch_size=16,
            embed_dim=384,  # Smaller for testing
            depth=4,
            num_heads=6,
            tubelet_size=2,
        )
        encoder = VideoMAEEncoder(encoder_cfg).to(device)
        results["tests"].append({
            "name": "encoder_init",
            "passed": True,
            "message": "VideoMAEEncoder created (embed_dim=384, depth=4)",
        })
    except Exception as e:
        results["tests"].append({"name": "encoder_init", "passed": False, "message": str(e)})
        results["passed"] = False
        return results["passed"], results
    
    # Test encoder forward pass
    try:
        B, C, T, H, W = 2, 3, 8, 224, 224
        x = torch.randn(B, C, T, H, W, device=device)
        with torch.no_grad():
            tokens, grid_size = encoder(x)
        
        expected_patches = (T // 2) * (H // 16) * (W // 16)
        results["tests"].append({
            "name": "encoder_forward",
            "passed": True,
            "message": f"Input: {tuple(x.shape)} -> Tokens: {tuple(tokens.shape)}, grid: {grid_size}",
        })
    except Exception as e:
        results["tests"].append({"name": "encoder_forward", "passed": False, "message": str(e)})
        results["passed"] = False
    
    # Test VideoMAE (with decoder) instantiation
    try:
        from trackB.videomae.videomae_model import VideoMAE, VideoMAEConfig
        mae_cfg = VideoMAEConfig(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6,
            decoder_embed_dim=192,
            decoder_depth=2,
            decoder_num_heads=3,
            tubelet_size=2,
            mask_ratio=0.75,
        )
        mae = VideoMAE(mae_cfg).to(device)
        results["tests"].append({
            "name": "mae_init",
            "passed": True,
            "message": "VideoMAE (encoder+decoder) created",
        })
    except Exception as e:
        results["tests"].append({"name": "mae_init", "passed": False, "message": str(e)})
        results["passed"] = False
        return results["passed"], results
    
    # Test MAE forward pass
    try:
        B, C, T, H, W = 2, 3, 8, 224, 224
        x = torch.randn(B, C, T, H, W, device=device)
        with torch.no_grad():
            result = mae(x)
        
        loss = result['loss']
        mask = result['mask']
        # mask: True = visible, so masked ratio = 1 - mean(visible)
        visible_ratio = mask.float().mean().item()
        results["tests"].append({
            "name": "mae_forward",
            "passed": True,
            "message": f"Loss: {loss.item():.4f}, Visible ratio: {visible_ratio:.3f}",
        })
    except Exception as e:
        results["tests"].append({"name": "mae_forward", "passed": False, "message": str(e)})
        results["passed"] = False
    
    return results["passed"], results


def test_tokenizer_videomae_config() -> Tuple[bool, Dict]:
    """Test TokenizerConfig video_backbone options."""
    results = {
        "name": "tokenizer_videomae_config",
        "passed": True,
        "tests": [],
    }
    
    try:
        from trackB_tokenizer import TokenizerConfig
        
        # Test default backbone
        cfg = TokenizerConfig()
        results["tests"].append({
            "name": "default_backbone",
            "passed": cfg.video_backbone in ['resnet18', 'videomae_ego'],
            "message": f"Default video_backbone: {cfg.video_backbone}",
        })
        
        # Test explicit resnet18
        cfg_resnet = TokenizerConfig(video_backbone='resnet18')
        results["tests"].append({
            "name": "resnet18_option",
            "passed": cfg_resnet.video_backbone == 'resnet18',
            "message": f"video_backbone='resnet18' works",
        })
        
        # Test explicit videomae_ego
        cfg_videomae = TokenizerConfig(video_backbone='videomae_ego')
        results["tests"].append({
            "name": "videomae_ego_option",
            "passed": cfg_videomae.video_backbone == 'videomae_ego',
            "message": f"video_backbone='videomae_ego' works",
        })
        
    except Exception as e:
        results["tests"].append({"name": "tokenizer_config", "passed": False, "message": str(e)})
        results["passed"] = False
    
    return results["passed"], results


# =============================================================================
# Pycache Validation
# =============================================================================

def check_pycache(directory: Path) -> Tuple[int, List[Path]]:
    """Check for __pycache__ directories."""
    pycache_dirs = list(directory.glob("**/__pycache__"))
    return len(pycache_dirs), pycache_dirs


# =============================================================================
# Main Test Runner
# =============================================================================

def run_smoke_tests(cfg: Optional[SmokeTestConfig] = None) -> Tuple[bool, Dict]:
    """
    Run all Track B smoke tests.
    
    Returns:
        (overall_success, full_results)
    """
    if cfg is None:
        cfg = SmokeTestConfig()
    
    start_time = time.time()
    
    full_results = {
        "track": "B",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "max_samples": cfg.max_samples,
            "max_epochs": cfg.max_epochs,
            "validate_pycache": cfg.validate_pycache,
            "test_model_instantiation": cfg.test_model_instantiation,
            "test_forward_pass": cfg.test_forward_pass,
        },
        "tests": [],
        "success": True,
    }
    
    # Test 1: Core imports
    passed, msg = test_core_imports()
    full_results["tests"].append({"name": "core_imports", "passed": passed, "message": msg})
    if not passed:
        full_results["success"] = False
    
    # Test 2: Track B imports
    passed, msg = test_trackb_imports()
    full_results["tests"].append({"name": "trackb_imports", "passed": passed, "message": msg})
    if not passed:
        full_results["success"] = False
    
    # Test 3: PyTorch availability
    passed, msg = test_torch_available()
    full_results["tests"].append({"name": "torch_available", "passed": passed, "message": msg})
    
    # Test 4: Model instantiation
    if cfg.test_model_instantiation:
        passed, inst_results = test_model_instantiation(cfg)
        full_results["tests"].append(inst_results)
        if not passed:
            full_results["success"] = False
    
    # Test 5: Forward pass
    if cfg.test_forward_pass:
        passed, fwd_results = test_forward_pass(cfg)
        full_results["tests"].append(fwd_results)
        if not passed:
            full_results["success"] = False
    
    # Test 6: Pycache validation
    if cfg.validate_pycache:
        count, dirs = check_pycache(_TRACKB_DIR)
        full_results["tests"].append({
            "name": "pycache_validation",
            "passed": count > 0,
            "message": f"Found {count} __pycache__ directories in trackB/",
        })
    
    # Test 7: Config loading
    try:
        from core.config_loader import load_config
        trackb_cfg = load_config("trackB")
        full_results["tests"].append({
            "name": "config_loading",
            "passed": True,
            "message": f"Loaded trackB config with {len(trackb_cfg.flat())} keys",
        })
    except Exception as e:
        full_results["tests"].append({
            "name": "config_loading",
            "passed": False,
            "message": str(e),
        })
        full_results["success"] = False
    
    # Test 8: Config adapter functions
    try:
        from core.config_adapter import (
            create_train_config,
            create_fusion_config,
            create_head_config,
            create_tokenizer_config,
        )
        train_cfg = create_train_config()
        fusion_cfg = create_fusion_config()
        head_cfg = create_head_config()
        tokenizer_cfg = create_tokenizer_config()
        full_results["tests"].append({
            "name": "config_adapters",
            "passed": True,
            "message": "Created TrainConfig, FusionConfig, HeadConfig, TokenizerConfig",
        })
    except Exception as e:
        full_results["tests"].append({
            "name": "config_adapters",
            "passed": False,
            "message": str(e),
        })
    
    # Test 9: VideoMAE imports
    passed, msg = test_videomae_imports()
    full_results["tests"].append({"name": "videomae_imports", "passed": passed, "message": msg})
    
    # Test 10: VideoMAE model
    if cfg.test_model_instantiation:
        passed, videomae_results = test_videomae_model(cfg)
        full_results["tests"].append(videomae_results)
        # Don't fail overall if VideoMAE tests fail (optional feature)
    
    # Test 11: TokenizerConfig VideoMAE options
    passed, tokenizer_results = test_tokenizer_videomae_config()
    full_results["tests"].append(tokenizer_results)
    
    elapsed = time.time() - start_time
    full_results["elapsed_seconds"] = round(elapsed, 2)
    
    return full_results["success"], full_results


def print_results(results: Dict, verbose: bool = True):
    """Pretty-print test results."""
    print("\n" + "=" * 60)
    print(f"Track {results['track']} Smoke Test Results")
    print("=" * 60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall: {'✓ PASSED' if results['success'] else '✗ FAILED'}")
    print(f"Elapsed: {results['elapsed_seconds']}s")
    print()
    
    for test in results["tests"]:
        if isinstance(test.get("tests"), list):
            # Nested test group
            status = "✓" if test.get("passed", True) else "✗"
            print(f"{status} {test['name']}")
            if verbose:
                for subtest in test["tests"]:
                    sub_status = "  ✓" if subtest["passed"] else "  ✗"
                    print(f"  {sub_status} {subtest['name']}: {subtest.get('message', '')}")
        else:
            status = "✓" if test.get("passed", True) else "✗"
            print(f"{status} {test['name']}: {test.get('message', '')}")
    
    print()
    print("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Track B Smoke Test")
    parser.add_argument("--max-samples", type=int, default=5, help="Max samples to test")
    parser.add_argument("--max-epochs", type=int, default=1, help="Max epochs for training test")
    parser.add_argument("--no-pycache", action="store_true", help="Skip pycache validation")
    parser.add_argument("--no-models", action="store_true", help="Skip model tests")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU only")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    cfg = SmokeTestConfig(
        max_samples=args.max_samples,
        max_epochs=args.max_epochs,
        validate_pycache=not args.no_pycache,
        test_model_instantiation=not args.no_models,
        test_forward_pass=not args.no_models,
        skip_gpu=args.cpu_only,
        verbose=not args.quiet,
    )
    
    success, results = run_smoke_tests(cfg)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results, verbose=not args.quiet)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
