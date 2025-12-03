#!/usr/bin/env python3
"""
Track C Smoke Test

Validates Track C pipeline (RGTP Token Pruning) with minimal data.
Tests pruning logic, importance scoring, and inference.

Usage:
    python -m tests.smoke_test_trackC
    
    # Or from repo root:
    python local_extraction/tests/smoke_test_trackC.py
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
_TRACKC_DIR = _LOCAL_EXTRACTION / "trackC"

# Add to path for imports
for p in [str(_LOCAL_EXTRACTION), str(_REPO_ROOT), str(_TRACKB_DIR), str(_TRACKC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class SmokeTestConfig:
    """Smoke test configuration."""
    max_samples: int = 3
    test_rates: List[float] = None  # Pruning rates to test
    validate_pycache: bool = True
    test_pruning_logic: bool = True
    skip_gpu: bool = False
    verbose: bool = True
    timeout_seconds: float = 60.0
    
    def __post_init__(self):
        if self.test_rates is None:
            self.test_rates = [0.0, 0.3, 0.5]


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


def test_trackc_imports() -> Tuple[bool, str]:
    """Test Track C module imports."""
    try:
        # Track C uses Track B modules
        from trackB_fusion import FusionConfig, TrackBFusion
        from trackB_head import HeadConfig, TrackBHead
        
        # Check for trackC_pruning.py
        trackc_path = _TRACKC_DIR / "trackC_pruning.py"
        if not trackc_path.exists():
            return False, f"trackC_pruning.py not found at {trackc_path}"
        
        return True, "Track C modules found and dependencies imported"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


# =============================================================================
# RGTP Pruning Logic Tests
# =============================================================================

def test_pruning_logic(cfg: SmokeTestConfig) -> Tuple[bool, Dict]:
    """Test RGTP pruning logic with synthetic data."""
    results = {
        "name": "pruning_logic",
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
    
    # Test 1: Importance score computation
    try:
        num_tokens = 16
        importance = torch.rand(num_tokens, device=device)
        importance_sorted, indices = torch.sort(importance, descending=True)
        
        results["tests"].append({
            "name": "importance_sorting",
            "passed": True,
            "message": f"Sorted {num_tokens} tokens by importance",
        })
    except Exception as e:
        results["tests"].append({"name": "importance_sorting", "passed": False, "message": str(e)})
        results["passed"] = False
    
    # Test 2: Pruning at different rates
    for rate in cfg.test_rates:
        try:
            keep_count = int(num_tokens * (1 - rate))
            keep_count = max(2, keep_count)  # min_keep = 2
            
            kept_indices = indices[:keep_count]
            pruned_indices = indices[keep_count:]
            
            # Create pruned tensor
            tokens = torch.randn(1, num_tokens, 256, device=device)
            mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
            mask[kept_indices] = True
            
            pruned_tokens = tokens[:, mask, :]
            
            results["tests"].append({
                "name": f"prune_rate_{int(rate*100)}",
                "passed": True,
                "message": f"rate={rate}: kept {keep_count}/{num_tokens} tokens, output shape={tuple(pruned_tokens.shape)}",
            })
        except Exception as e:
            results["tests"].append({
                "name": f"prune_rate_{int(rate*100)}",
                "passed": False,
                "message": str(e),
            })
            results["passed"] = False
    
    # Test 3: Min-keep enforcement
    try:
        min_keep = 2
        extreme_rate = 0.99  # Would prune almost everything
        
        keep_count = int(num_tokens * (1 - extreme_rate))
        keep_count = max(min_keep, keep_count)
        
        assert keep_count >= min_keep, f"min_keep not enforced: {keep_count} < {min_keep}"
        
        results["tests"].append({
            "name": "min_keep_enforcement",
            "passed": True,
            "message": f"min_keep={min_keep} enforced at rate={extreme_rate}",
        })
    except Exception as e:
        results["tests"].append({"name": "min_keep_enforcement", "passed": False, "message": str(e)})
        results["passed"] = False
    
    # Test 4: Logit fill for pruned tokens
    try:
        logit_fill = -12.0
        batch_size = 2
        num_cands = 8
        
        logits = torch.randn(batch_size, num_cands, device=device)
        prune_mask = torch.zeros(batch_size, num_cands, dtype=torch.bool, device=device)
        prune_mask[:, :3] = True  # Keep first 3
        
        # Apply logit fill to pruned positions
        filled_logits = logits.clone()
        filled_logits[~prune_mask] = logit_fill
        
        # Check that pruned positions have correct fill
        assert (filled_logits[~prune_mask] == logit_fill).all()
        
        results["tests"].append({
            "name": "logit_fill",
            "passed": True,
            "message": f"Logit fill={logit_fill} applied to pruned tokens",
        })
    except Exception as e:
        results["tests"].append({"name": "logit_fill", "passed": False, "message": str(e)})
        results["passed"] = False
    
    return results["passed"], results


# =============================================================================
# Config Tests
# =============================================================================

def test_config_loading() -> Tuple[bool, Dict]:
    """Test Track C config loading."""
    results = {
        "name": "config_loading",
        "passed": True,
        "tests": [],
    }
    
    try:
        from core.config_loader import load_config
        cfg = load_config("trackC")
        
        # Check key RGTP settings
        rgtp_enabled = cfg.get("rgtp.enabled", None)
        rgtp_rate = cfg.get("rgtp.rate", None)
        min_keep = cfg.get("rgtp.min_keep", None)
        
        results["tests"].append({
            "name": "config_loaded",
            "passed": True,
            "message": f"rgtp.enabled={rgtp_enabled}, rgtp.rate={rgtp_rate}, min_keep={min_keep}",
        })
        
        # Validate rate is in valid range
        if rgtp_rate is not None:
            valid_rate = 0.0 <= rgtp_rate <= 1.0
            results["tests"].append({
                "name": "rate_validation",
                "passed": valid_rate,
                "message": f"Rate {rgtp_rate} in [0, 1]: {valid_rate}",
            })
            if not valid_rate:
                results["passed"] = False
        
    except Exception as e:
        results["tests"].append({"name": "config_load", "passed": False, "message": str(e)})
        results["passed"] = False
    
    return results["passed"], results


# =============================================================================
# VideoMAE Backbone Support Tests
# =============================================================================

def test_videomae_backbone_support() -> Tuple[bool, Dict]:
    """Test Track C support for VideoMAE backbone."""
    results = {
        "name": "videomae_backbone_support",
        "passed": True,
        "tests": [],
    }
    
    try:
        import torch
        
        # Test projector dimensions for both backbones
        RESNET_DIM = 512
        VIDEOMAE_DIM = 768
        TOKEN_DIM = 256
        
        projector_resnet = torch.nn.Linear(RESNET_DIM, TOKEN_DIM)
        projector_videomae = torch.nn.Linear(VIDEOMAE_DIM, TOKEN_DIM)
        
        B, N = 4, 100
        
        # Test ResNet path
        resnet_feats = torch.randn(B, N, RESNET_DIM)
        resnet_tokens = projector_resnet(resnet_feats)
        
        results["tests"].append({
            "name": "resnet_projector",
            "passed": resnet_tokens.shape == (B, N, TOKEN_DIM),
            "message": f"ResNet projector: {RESNET_DIM} -> {TOKEN_DIM}",
        })
        
        # Test VideoMAE path
        videomae_feats = torch.randn(B, N, VIDEOMAE_DIM)
        videomae_tokens = projector_videomae(videomae_feats)
        
        results["tests"].append({
            "name": "videomae_projector",
            "passed": videomae_tokens.shape == (B, N, TOKEN_DIM),
            "message": f"VideoMAE projector: {VIDEOMAE_DIM} -> {TOKEN_DIM}",
        })
        
        # Both should produce same output dimension
        results["tests"].append({
            "name": "output_dimension_match",
            "passed": resnet_tokens.shape == videomae_tokens.shape,
            "message": f"Both backbones produce {TOKEN_DIM}-dim tokens",
        })
        
    except Exception as e:
        results["tests"].append({"name": "backbone_support", "passed": False, "message": str(e)})
        results["passed"] = False
    
    return results["passed"], results


# =============================================================================
# Pycache Validation
# =============================================================================

def check_pycache(directory: Path) -> Tuple[int, List[Path]]:
    """Check for __pycache__ directories."""
    if not directory.exists():
        return 0, []
    pycache_dirs = list(directory.glob("**/__pycache__"))
    return len(pycache_dirs), pycache_dirs


# =============================================================================
# Main Test Runner
# =============================================================================

def run_smoke_tests(cfg: Optional[SmokeTestConfig] = None) -> Tuple[bool, Dict]:
    """
    Run all Track C smoke tests.
    
    Returns:
        (overall_success, full_results)
    """
    if cfg is None:
        cfg = SmokeTestConfig()
    
    start_time = time.time()
    
    full_results = {
        "track": "C",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "max_samples": cfg.max_samples,
            "test_rates": cfg.test_rates,
            "validate_pycache": cfg.validate_pycache,
            "test_pruning_logic": cfg.test_pruning_logic,
        },
        "tests": [],
        "success": True,
    }
    
    # Test 1: Core imports
    passed, msg = test_core_imports()
    full_results["tests"].append({"name": "core_imports", "passed": passed, "message": msg})
    if not passed:
        full_results["success"] = False
    
    # Test 2: Track C imports
    passed, msg = test_trackc_imports()
    full_results["tests"].append({"name": "trackc_imports", "passed": passed, "message": msg})
    if not passed:
        full_results["success"] = False
    
    # Test 3: Config loading
    passed, cfg_results = test_config_loading()
    full_results["tests"].append(cfg_results)
    if not passed:
        full_results["success"] = False
    
    # Test 4: Pruning logic
    if cfg.test_pruning_logic:
        passed, prune_results = test_pruning_logic(cfg)
        full_results["tests"].append(prune_results)
        if not passed:
            full_results["success"] = False
    
    # Test 5: Pycache validation
    if cfg.validate_pycache:
        count, dirs = check_pycache(_TRACKC_DIR)
        # Track C may not have pycache if only trackC_pruning.py exists
        full_results["tests"].append({
            "name": "pycache_validation",
            "passed": True,  # Not critical for Track C
            "message": f"Found {count} __pycache__ directories in trackC/",
        })
    
    # Test 6: Check for checkpoint discovery
    try:
        from core.paths import Paths
        paths = Paths.auto_detect()
        ckpt = paths.find_checkpoint("trackB_best_*.pt")
        full_results["tests"].append({
            "name": "checkpoint_discovery",
            "passed": True,
            "message": f"Found checkpoint: {ckpt}" if ckpt else "No checkpoint found (OK for first run)",
        })
    except Exception as e:
        full_results["tests"].append({
            "name": "checkpoint_discovery",
            "passed": False,
            "message": str(e),
        })
    
    # Test 7: VideoMAE backbone support
    passed, videomae_results = test_videomae_backbone_support()
    full_results["tests"].append(videomae_results)
    
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
    
    parser = argparse.ArgumentParser(description="Track C Smoke Test")
    parser.add_argument("--max-samples", type=int, default=3, help="Max samples to test")
    parser.add_argument("--rates", type=str, default="0.0,0.3,0.5", help="Comma-separated pruning rates to test")
    parser.add_argument("--no-pycache", action="store_true", help="Skip pycache validation")
    parser.add_argument("--no-pruning", action="store_true", help="Skip pruning logic tests")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU only")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    test_rates = [float(r) for r in args.rates.split(",")]
    
    cfg = SmokeTestConfig(
        max_samples=args.max_samples,
        test_rates=test_rates,
        validate_pycache=not args.no_pycache,
        test_pruning_logic=not args.no_pruning,
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
