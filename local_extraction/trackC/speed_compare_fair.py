"""
Compare training speed WITH vs WITHOUT pre-extracted tokens (using only UIDs that have tokens)
"""
import time
import sys
sys.path.insert(0, r'd:\Thesis\Ego4d-LiteSTA\local_extraction')

import torch
from pathlib import Path
import json

def get_uids_with_tokens():
    """Get UIDs that have pre-extracted tokens"""
    tokens_dir = Path(r'd:\Thesis\Ego4d-LiteSTA\local_extraction\v2\resnet18_tokens')
    return set(d.name for d in tokens_dir.iterdir() if d.is_dir())

def create_filtered_manifest(uids_with_tokens):
    """Create a temp manifest with only UIDs that have tokens"""
    manifest = Path(r'd:\Thesis\Ego4d-LiteSTA\local_extraction\runs\Track_A\trackA_stageB_20251117_184342\head_train.jsonl')
    filtered = Path(r'd:\Thesis\Ego4d-LiteSTA\local_extraction\v2\temp_filtered_manifest.jsonl')
    
    count = 0
    with open(manifest) as fin, open(filtered, 'w') as fout:
        for line in fin:
            rec = json.loads(line.strip())
            uid = None
            if 'image_path' in rec:
                parts = Path(rec['image_path']).parts
                for i, p in enumerate(parts):
                    if p == 'extracted_frames' and i+1 < len(parts):
                        uid = parts[i+1]
                        break
            if uid and uid in uids_with_tokens:
                fout.write(line)
                count += 1
    
    print(f"Filtered manifest: {count} records (UIDs with tokens)")
    return filtered

def time_training(use_tokens: bool, manifest_path: Path, steps: int = 5):
    """Time a short training run"""
    from trackB.trackB_dataset import TrackBDataset, trackB_collate
    from trackB.trackB_tokenizer import TokenizerConfig
    from torch.utils.data import DataLoader
    
    local_extraction = Path(r'd:\Thesis\Ego4d-LiteSTA\local_extraction')
    frames_root = local_extraction / 'v2' / 'extracted_frames'
    manifests_root = local_extraction / 'v2' / 'manifests'
    
    tok_cfg = TokenizerConfig()
    tokens_path = local_extraction / 'v2' / 'resnet18_tokens' if use_tokens else None
    
    mode = "WITH tokens" if use_tokens else "WITHOUT tokens (on-the-fly ResNet)"
    print(f"\n[{mode}]")
    
    start_build = time.time()
    ds = TrackBDataset(
        manifest_path=manifest_path,
        frames_root=frames_root,
        manifests_root=manifests_root,
        tokenizer_cfg=tok_cfg,
        tokens_root=tokens_path,
        candidate_limit=16,
    )
    build_time = time.time() - start_build
    print(f"  Dataset build: {build_time:.2f}s")
    print(f"  Dataset size: {len(ds)} samples")
    
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=trackB_collate, num_workers=0)
    
    # Warm up
    print(f"  Warming up...")
    for i, batch in enumerate(loader):
        if i >= 1:
            break
    
    # Time actual loading
    print(f"  Timing {steps} batches...")
    start_iter = time.time()
    for i, batch in enumerate(loader):
        if i >= steps:
            break
    iter_time = time.time() - start_iter
    
    per_batch_ms = iter_time / steps * 1000
    print(f"  Data loading: {iter_time:.2f}s ({steps} batches)")
    print(f"  Per-batch: {per_batch_ms:.1f}ms")
    
    return per_batch_ms

if __name__ == '__main__':
    print("="*60)
    print("SPEED COMPARISON: Pre-extracted Tokens vs On-the-fly")
    print("(Using only UIDs that have pre-extracted tokens)")
    print("="*60)
    
    # Get UIDs with tokens and create filtered manifest
    uids_with_tokens = get_uids_with_tokens()
    print(f"\nFound {len(uids_with_tokens)} UIDs with pre-extracted tokens")
    
    filtered_manifest = create_filtered_manifest(uids_with_tokens)
    
    # Test WITH tokens
    time_with = time_training(use_tokens=True, manifest_path=filtered_manifest, steps=5)
    
    # Test WITHOUT tokens  
    time_without = time_training(use_tokens=False, manifest_path=filtered_manifest, steps=5)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  WITH tokens:    {time_with:.1f}ms per batch")
    print(f"  WITHOUT tokens: {time_without:.1f}ms per batch")
    print(f"  Speedup:        {time_without/time_with:.1f}x faster")
    
    # Cleanup
    filtered_manifest.unlink()
