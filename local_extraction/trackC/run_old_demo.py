#!/usr/bin/env python3
"""
Run demo training on both old and new trackB code to compare them.
Results for old code saved to local_extraction_old/trackB/old_runs/
Results for new code saved to local_extraction/runs/Track_B/demo_comparison/
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# ===== OLD CODE DEMO =====
def run_old_demo():
    """Run demo using old trackB code."""
    print("="*80)
    print("RUNNING OLD TRACKB DEMO")
    print("="*80)
    
    # Add old path to sys.path
    old_path = Path("local_extraction_old/trackB").resolve()
    sys.path.insert(0, str(old_path))
    sys.path.insert(0, str(old_path.parent))
    
    # Change working directory context
    original_cwd = os.getcwd()
    
    # Import old modules
    from trackB_train_loader import TrainConfig, main as old_main
    from trackB_train_loader import _config_to_dict
    
    # Override config for demo mode with SAME settings as new
    TrainConfig.mode = 'demo'
    TrainConfig.demo_steps = 20
    TrainConfig.epochs = 2
    TrainConfig.batch_size = 4
    TrainConfig.loss_w_next = 1.0  # Same as new
    TrainConfig.loss_w_noun = 1.0  # Same as new
    TrainConfig.loss_w_verb = 1.0  # Same as new
    TrainConfig.loss_w_ttc = 1.0   # Same as new
    TrainConfig.save_epoch_checkpoints = False
    TrainConfig.save_best_checkpoint = False
    
    # Store config for comparison
    cfg = TrainConfig()
    config_dict = _config_to_dict(cfg)
    
    # Create output directory
    output_dir = Path("local_extraction_old/trackB/old_runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save config
    config_path = output_dir / f"demo_config_{ts}.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"[OLD] Saved config to {config_path}")
    
    # Run demo and capture timing
    start_time = time.time()
    try:
        # We can't easily capture the main() output, so we'll run it
        # and check the loss values printed
        old_main()
        elapsed = time.time() - start_time
        result = {
            "status": "completed",
            "elapsed_seconds": elapsed,
            "config": config_dict,
            "timestamp": ts
        }
    except Exception as e:
        elapsed = time.time() - start_time
        result = {
            "status": "error",
            "error": str(e),
            "elapsed_seconds": elapsed,
            "config": config_dict,
            "timestamp": ts
        }
    
    # Save result
    result_path = output_dir / f"demo_result_{ts}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[OLD] Saved result to {result_path}")
    
    # Clean up sys.path
    sys.path.remove(str(old_path))
    sys.path.remove(str(old_path.parent))
    
    return result


if __name__ == '__main__':
    # Only run old demo for now
    print("Starting OLD trackB demo comparison...")
    old_result = run_old_demo()
    print("\n" + "="*80)
    print("OLD DEMO COMPLETE")
    print("="*80)
    print(f"Status: {old_result['status']}")
    print(f"Elapsed: {old_result['elapsed_seconds']:.2f}s")
