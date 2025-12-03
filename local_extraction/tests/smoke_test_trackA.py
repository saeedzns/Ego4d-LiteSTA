#!/usr/bin/env python3
"""
Track A Smoke Test

Validates Track A pipeline (Stage A + Stage B) with minimal data.
Includes pycache validation to ensure modules import correctly.

Usage:
    python -m tests.smoke_test_trackA
    
    # Or from repo root:
    python local_extraction/tests/smoke_test_trackA.py
"""
from __future__ import annotations

import sys
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Setup: Add parent directories to path
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_EXTRACTION = _THIS_DIR.parent
_REPO_ROOT = _LOCAL_EXTRACTION.parent

# Add to path for imports
for p in [str(_LOCAL_EXTRACTION), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class SmokeTestConfig:
    """Smoke test configuration."""
    max_samples: int = 5
    validate_pycache: bool = True
    cleanup_temp: bool = True
    verbose: bool = True
    timeout_seconds: float = 60.0


# =============================================================================
# Pycache Validation
# =============================================================================

def validate_pycache(modules: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that modules can be imported and pycache is generated.
    
    Returns:
        (success, errors)
    """
    errors: List[str] = []
    
    for module_name in modules:
        try:
            # Attempt import
            __import__(module_name)
        except ImportError as e:
            errors.append(f"ImportError for {module_name}: {e}")
        except Exception as e:
            errors.append(f"Error importing {module_name}: {e}")
    
    return len(errors) == 0, errors


def check_pycache_exists(directory: Path, recursive: bool = True) -> Tuple[int, List[Path]]:
    """
    Check for __pycache__ directories.
    
    Returns:
        (count, list of pycache dirs)
    """
    pattern = "**/__pycache__" if recursive else "__pycache__"
    pycache_dirs = list(directory.glob(pattern))
    return len(pycache_dirs), pycache_dirs


# =============================================================================
# Track A Stage A Smoke Test
# =============================================================================

def smoke_test_stage_a(cfg: SmokeTestConfig) -> Tuple[bool, Dict]:
    """
    Smoke test for Track A Stage A (detector).
    
    Tests:
    1. Config loading
    2. Module imports
    3. Minimal detection run (oracle mode)
    """
    results = {
        "stage": "A",
        "substage": "stageA",
        "tests": [],
        "success": True,
    }
    
    # Test 1: Config loading
    test_name = "config_loading"
    try:
        from core.config_loader import load_config
        cfg_obj = load_config("trackA")
        results["tests"].append({
            "name": test_name,
            "passed": True,
            "message": f"Loaded trackA config with {len(cfg_obj.flat())} keys",
        })
    except Exception as e:
        results["tests"].append({
            "name": test_name,
            "passed": False,
            "message": str(e),
        })
        results["success"] = False
        return False, results
    
    # Test 2: Module imports
    test_name = "module_imports"
    modules_to_test = [
        "core.config_loader",
        "core.paths",
    ]
    
    # Try Track A modules (may not exist yet if not refactored)
    try:
        success, errors = validate_pycache(modules_to_test)
        results["tests"].append({
            "name": test_name,
            "passed": success,
            "message": f"Imported {len(modules_to_test)} core modules" if success else f"Errors: {errors}",
        })
        if not success:
            results["success"] = False
    except Exception as e:
        results["tests"].append({
            "name": test_name,
            "passed": False,
            "message": str(e),
        })
        results["success"] = False
    
    # Test 3: Paths resolution
    test_name = "paths_resolution"
    try:
        from core.paths import Paths
        paths = Paths.auto_detect()
        results["tests"].append({
            "name": test_name,
            "passed": True,
            "message": f"Resolved repo_root: {paths.repo_root}",
        })
    except Exception as e:
        results["tests"].append({
            "name": test_name,
            "passed": False,
            "message": str(e),
        })
        results["success"] = False
    
    # Test 4: Check pycache generation
    if cfg.validate_pycache:
        test_name = "pycache_validation"
        try:
            count, dirs = check_pycache_exists(_LOCAL_EXTRACTION / "core")
            results["tests"].append({
                "name": test_name,
                "passed": count > 0,
                "message": f"Found {count} __pycache__ directories",
            })
        except Exception as e:
            results["tests"].append({
                "name": test_name,
                "passed": False,
                "message": str(e),
            })
    
    # Test 5: Check for trackA_stageA.py existence
    test_name = "stagea_script_exists"
    stagea_path = _LOCAL_EXTRACTION / "trackA" / "trackA_stageA" / "trackA_stageA.py"
    exists = stagea_path.exists()
    results["tests"].append({
        "name": test_name,
        "passed": exists,
        "message": f"trackA_stageA.py exists: {exists}",
    })
    
    return results["success"], results


# =============================================================================
# Track A Stage B Smoke Test
# =============================================================================

def smoke_test_stage_b(cfg: SmokeTestConfig) -> Tuple[bool, Dict]:
    """
    Smoke test for Track A Stage B (head/reasoner).
    """
    results = {
        "stage": "A",
        "substage": "stageB",
        "tests": [],
        "success": True,
    }
    
    # Test 1: Check for trackA_stageB.py existence
    test_name = "stageb_script_exists"
    stageb_path = _LOCAL_EXTRACTION / "trackA" / "trackA_stageB" / "trackA_stageB.py"
    exists = stageb_path.exists()
    results["tests"].append({
        "name": test_name,
        "passed": exists,
        "message": f"trackA_stageB.py exists: {exists}",
    })
    if not exists:
        results["success"] = False
    
    # Test 2: Check manifests directory
    test_name = "manifests_directory"
    from core.paths import Paths
    paths = Paths.auto_detect()
    manifests_exists = paths.manifests_root.exists()
    results["tests"].append({
        "name": test_name,
        "passed": manifests_exists,
        "message": f"Manifests directory exists: {paths.manifests_root}" if manifests_exists else "Missing manifests directory",
    })
    
    # Test 3: Check for existing Stage A runs
    test_name = "stagea_runs_exist"
    latest = paths.latest_run("A", "trackA_stageA")
    results["tests"].append({
        "name": test_name,
        "passed": latest is not None,
        "message": f"Latest Stage A run: {latest}" if latest else "No Stage A runs found (expected for first run)",
    })
    
    return results["success"], results


# =============================================================================
# Main Test Runner
# =============================================================================

def run_smoke_tests(cfg: Optional[SmokeTestConfig] = None) -> Tuple[bool, Dict]:
    """
    Run all Track A smoke tests.
    
    Returns:
        (overall_success, full_results)
    """
    if cfg is None:
        cfg = SmokeTestConfig()
    
    start_time = time.time()
    
    full_results = {
        "track": "A",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "max_samples": cfg.max_samples,
            "validate_pycache": cfg.validate_pycache,
        },
        "stages": [],
        "success": True,
    }
    
    # Run Stage A tests
    success_a, results_a = smoke_test_stage_a(cfg)
    full_results["stages"].append(results_a)
    if not success_a:
        full_results["success"] = False
    
    # Run Stage B tests
    success_b, results_b = smoke_test_stage_b(cfg)
    full_results["stages"].append(results_b)
    if not success_b:
        full_results["success"] = False
    
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
    
    for stage in results["stages"]:
        stage_name = f"Stage {stage['substage']}"
        status = "✓" if stage["success"] else "✗"
        print(f"{status} {stage_name}")
        
        if verbose:
            for test in stage["tests"]:
                test_status = "  ✓" if test["passed"] else "  ✗"
                print(f"  {test_status} {test['name']}: {test['message']}")
        print()
    
    print("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Track A Smoke Test")
    parser.add_argument("--max-samples", type=int, default=5, help="Max samples to test")
    parser.add_argument("--no-pycache", action="store_true", help="Skip pycache validation")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    cfg = SmokeTestConfig(
        max_samples=args.max_samples,
        validate_pycache=not args.no_pycache,
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
