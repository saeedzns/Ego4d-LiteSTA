"""
Compare training speed WITH vs WITHOUT pre-extracted tokens
"""
import time
import sys
sys.path.insert(0, r'd:\Thesis\Ego4d-LiteSTA\local_extraction')

import torch

def time_training(use_tokens: bool, steps: int = 10):
    """Time a short training run"""
    
    # Temporarily modify config
    from core import load_config
    cfg = load_config('trackB')
    
    if use_tokens:
        tokens_root = cfg.get('data.tokens_root')
        print(f"[TEST] Using pre-extracted tokens: {tokens_root}")
    else:
        # We need to reload with null tokens_root
        print("[TEST] Using on-the-fly ResNet extraction (NO tokens)")
    
    # Import training components
    from trackB.trackB_dataset import TrackBDataset, trackB_collate
    from trackB.trackB_fusion import FusionConfig, TrackBFusion
    from trackB.trackB_head import HeadConfig, TrackBHead
    from trackB.trackB_tokenizer import TokenizerConfig
    from pathlib import Path
    
    # Paths
    local_extraction = Path(r'd:\Thesis\Ego4d-LiteSTA\local_extraction')
    frames_root = local_extraction / 'v2' / 'extracted_frames'
    manifests_root = local_extraction / 'v2' / 'manifests'
    manifest = local_extraction / 'runs' / 'Track_A' / 'trackA_stageB_20251117_184342' / 'head_train.jsonl'
    
    tok_cfg = TokenizerConfig()
    
    # Build dataset
    tokens_path = local_extraction / 'v2' / 'resnet18_tokens' if use_tokens else None
    
    start_build = time.time()
    ds = TrackBDataset(
        manifest_path=manifest,
        frames_root=frames_root,
        manifests_root=manifests_root,
        tokenizer_cfg=tok_cfg,
        tokens_root=tokens_path,
        candidate_limit=16,
    )
    build_time = time.time() - start_build
    print(f"  Dataset build time: {build_time:.2f}s")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=trackB_collate, num_workers=0)
    
    # Iterate for `steps` batches
    start_iter = time.time()
    for i, batch in enumerate(loader):
        if i >= steps:
            break
        # Just load data, don't train
        _ = list(batch.keys())
    iter_time = time.time() - start_iter
    
    print(f"  Data loading time ({steps} batches): {iter_time:.2f}s")
    print(f"  Per-batch: {iter_time/steps*1000:.1f}ms")
    
    return build_time, iter_time

if __name__ == '__main__':
    print("="*60)
    print("SPEED COMPARISON: Pre-extracted Tokens vs On-the-fly")
    print("="*60)
    
    # Test WITH tokens (current config)
    print("\n[1] WITH pre-extracted tokens:")
    t1_build, t1_iter = time_training(use_tokens=True, steps=10)
    
    print("\nDone! To test WITHOUT tokens, set tokens_root: null in trackB.yaml")
    print("(Skipping no-tokens test to avoid slow extraction)")
