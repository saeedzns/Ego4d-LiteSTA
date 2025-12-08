#!/usr/bin/env python3
"""Quick comparison: Evaluate November vs December checkpoints with same eval code."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Use December's eval code for both
from trackB.trackB_eval import EvalConfig, evaluate

def main():
    # November checkpoint
    nov_ckpt = Path("local_extraction_old/runs/Track_B/checkpoints/trackB_best_1122_1735.pt")
    # December checkpoint  
    dec_ckpt = Path("local_extraction/runs/Track_B/checkpoints/trackB_best_mAP_0.3580_20251206_083053.pt")
    
    val_manifest = Path("local_extraction/runs/Track_A/trackA_stageB_20251117_184342/head_val.jsonl")
    
    print("="*80)
    print("Evaluating NOVEMBER checkpoint with current eval code")
    print("="*80)
    
    cfg_nov = EvalConfig()
    cfg_nov.checkpoint_path = nov_ckpt
    cfg_nov.val_manifest = val_manifest
    cfg_nov.save_overlays = False
    cfg_nov.use_hotspot_priors = False
    cfg_nov.use_clip_rerank = False
    
    metrics_nov = evaluate(cfg_nov)
    print(f"\nNov N_top5_mAP: {metrics_nov.get('N_top5_mAP', 'N/A')}")
    print(f"Nov All_top5_mAP: {metrics_nov.get('All_top5_mAP', 'N/A')}")
    
    print("\n" + "="*80)
    print("Evaluating DECEMBER checkpoint with current eval code")
    print("="*80)
    
    cfg_dec = EvalConfig()
    cfg_dec.checkpoint_path = dec_ckpt
    cfg_dec.val_manifest = val_manifest
    cfg_dec.save_overlays = False
    cfg_dec.use_hotspot_priors = False
    cfg_dec.use_clip_rerank = False
    
    metrics_dec = evaluate(cfg_dec)
    print(f"\nDec N_top5_mAP: {metrics_dec.get('N_top5_mAP', 'N/A')}")
    print(f"Dec All_top5_mAP: {metrics_dec.get('All_top5_mAP', 'N/A')}")

if __name__ == '__main__':
    main()
