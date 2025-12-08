#!/usr/bin/env python3
"""
Run demo training on NEW trackB code with same config as old.
Results saved to local_extraction/runs/Track_B/demo_comparison/
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

def run_new_demo():
    """Run demo using new trackB code."""
    print("="*80)
    print("RUNNING NEW TRACKB DEMO")
    print("="*80)
    
    # Add new path to sys.path
    new_path = Path("local_extraction/trackB").resolve()
    local_extraction = Path("local_extraction").resolve()
    sys.path.insert(0, str(new_path))
    sys.path.insert(0, str(local_extraction))
    
    # We need to run with specific config overrides
    # The new code uses YAML, so we'll import and override
    
    import torch
    torch.manual_seed(42)  # Set seed for reproducibility
    
    from trackB_train_loader import TrainConfig, _config_to_dict
    
    # Create output directory
    output_dir = Path("local_extraction/runs/Track_B/demo_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get config (it's loaded from YAML by default)
    cfg = TrainConfig()
    
    # Override to match old demo settings
    cfg.mode = 'demo'
    cfg.demo_steps = 20
    cfg.epochs = 2
    cfg.batch_size = 4
    cfg.loss_w_next = 1.0
    cfg.loss_w_noun = 1.0
    cfg.loss_w_verb = 1.0
    cfg.loss_w_ttc = 1.0
    cfg.save_epoch_checkpoints = False
    cfg.save_best_checkpoint = False
    
    config_dict = _config_to_dict(cfg)
    
    # Save config
    config_path = output_dir / f"demo_config_{ts}.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"[NEW] Saved config to {config_path}")
    
    # Print config for comparison
    print(f"[NEW] Config: mode={cfg.mode}, demo_steps={cfg.demo_steps}, batch_size={cfg.batch_size}")
    print(f"[NEW] Loss weights: next={cfg.loss_w_next}, noun={cfg.loss_w_noun}, verb={cfg.loss_w_verb}, ttc={cfg.loss_w_ttc}")
    
    # Now we need to run the training loop manually since main() has hardcoded paths
    # Import required modules
    from trackB_dataset import TrackBDataset, trackB_collate, latest_stageB_run, resolve_stageB_manifest
    from trackB_tokenizer import TokenizerConfig, roi_pool_tokens_mean
    from trackB_fusion import FusionConfig, TrackBFusion
    from trackB_head import HeadConfig, TrackBHead
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames_root = Path("local_extraction/v2/extracted_frames")
    
    # Get Stage B run
    trackA_runs_root = Path("local_extraction/runs/Track_A")
    stageB_run = latest_stageB_run(trackA_runs_root)
    stageB_train_manifest = resolve_stageB_manifest(stageB_run, 'head_train')
    
    print(f"[NEW] Using StageB run: {stageB_run}")
    
    tcfg = TokenizerConfig()
    
    # Dataset
    ds = TrackBDataset(
        frames_root=frames_root,
        manifests_root=stageB_run,
        manifest_path=stageB_train_manifest,
        tokenizer_cfg=tcfg,
        candidate_limit=cfg.candidate_limit,
        normalize_ttc=cfg.normalize_ttc,
    )
    
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=trackB_collate
    )
    
    # Models
    TOKEN_DIM = 256
    FUSION_LAYERS = 2
    
    projector = nn.Linear(512, TOKEN_DIM).to(device)
    fusion = TrackBFusion(FusionConfig(dim=TOKEN_DIM, layers=FUSION_LAYERS)).to(device)
    
    # Infer vocab from dataset
    used_noun_ids = set()
    used_verb_ids = set()
    for p in ds.parsed:
        for c in p.get('candidates', []):
            nid = c.get('noun_id', -1)
            vid = c.get('verb_id', -1)
            if nid >= 0:
                used_noun_ids.add(int(nid))
            if vid >= 0:
                used_verb_ids.add(int(vid))
    
    noun_id_list = sorted(used_noun_ids)
    verb_id_list = sorted(used_verb_ids)
    
    head = TrackBHead(HeadConfig(
        dim=TOKEN_DIM,
        hidden=256,
        num_classes=2,
        num_noun_classes=len(noun_id_list),
        num_verb_classes=len(verb_id_list),
        num_ttc_bins=4,
    )).to(device)
    
    # Optimizer and loss
    params = list(projector.parameters()) + list(fusion.parameters()) + list(head.parameters())
    opt = optim.AdamW(params, lr=cfg.lr)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    l1 = nn.SmoothL1Loss(reduction='none')
    
    # Demo training loop
    start_time = time.time()
    steps = cfg.demo_steps
    batch_iter = iter(loader)
    pbar = tqdm(range(steps), desc='[TrackB NEW demo] steps')
    
    losses = []
    for step in pbar:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)
        
        if not batch.get('valid'):
            continue
        
        samples = batch['samples']
        img_cands = []
        vid_cands = []
        labels_list = []
        ttc_list = []
        
        for s in samples:
            img_b = projector(s['img_tokens'].to(device)).unsqueeze(0)
            vid_b = projector(s['vid_tokens'].to(device)).unsqueeze(0)
            fused_img, fused_vid = fusion(img_b, vid_b)
            W, H = s['image_size']
            hw = s['hw']
            pooled_img = []
            pooled_vid = []
            for box in s['bboxes']:
                pooled_img.append(roi_pool_tokens_mean(hw, fused_img[0], box, (W, H)))
                pooled_vid.append(roi_pool_tokens_mean(hw, fused_vid[0], box, (W, H)))
            if not pooled_img:
                continue
            img_cands.append(torch.stack(pooled_img, dim=0))
            vid_cands.append(torch.stack(pooled_vid, dim=0))
            labels_list.append(s.get('is_positive', s['labels']).to(device))
            ttc_list.append(s.get('ttc_norm', s['ttc']).to(device))
        
        if not img_cands:
            continue
        
        # Pad and stack
        def pad_and_stack(cand_list, pad_value=0.0):
            B = len(cand_list)
            C = cand_list[0].shape[-1]
            Nc_max = max(x.shape[0] for x in cand_list)
            feats = cand_list[0].new_full((B, Nc_max, C), pad_value)
            mask = cand_list[0].new_zeros((B, Nc_max), dtype=torch.bool)
            for i, f in enumerate(cand_list):
                n = f.shape[0]
                feats[i, :n, :] = f
                mask[i, :n] = True
            return feats, mask
        
        img_pad, mask = pad_and_stack(img_cands)
        vid_pad, _ = pad_and_stack(vid_cands)
        labels_pad, _ = pad_and_stack([x.unsqueeze(-1).float() for x in labels_list])
        labels_pad = labels_pad.long().squeeze(-1)
        ttc_pad, _ = pad_and_stack([x.unsqueeze(-1) for x in ttc_list])
        
        out = head(img_pad.to(device), vid_pad.to(device))
        B, Nc, K = out['cls_logits'].shape
        ce_loss = ce(out['cls_logits'].reshape(B * Nc, K), labels_pad.reshape(B * Nc))
        ttc_loss_all = l1(out['ttc'], ttc_pad).squeeze(-1)
        ttc_loss = (ttc_loss_all * mask.to(device).float()).sum() / mask.to(device).float().sum().clamp_min(1.0)
        
        loss = cfg.loss_w_next * ce_loss + cfg.loss_w_ttc * 0.1 * ttc_loss
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'cls': f"{ce_loss.item():.4f}", 'ttc': f"{ttc_loss.item():.4f}"})
    
    elapsed = time.time() - start_time
    
    # Save checkpoint
    ckpt_path = output_dir / f"trackB_demo_{ts}.pt"
    torch.save({
        'projector': projector.state_dict(),
        'fusion': fusion.state_dict(),
        'head': head.state_dict(),
        'noun_id_list': noun_id_list,
        'verb_id_list': verb_id_list,
        'train_config': config_dict,
    }, ckpt_path)
    print(f"[NEW] Saved checkpoint to {ckpt_path}")
    
    result = {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "final_loss": losses[-1] if losses else None,
        "avg_loss": sum(losses) / len(losses) if losses else None,
        "config": config_dict,
        "timestamp": ts,
        "checkpoint": str(ckpt_path)
    }
    
    # Save result
    result_path = output_dir / f"demo_result_{ts}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[NEW] Saved result to {result_path}")
    
    return result


if __name__ == '__main__':
    print("Starting NEW trackB demo...")
    new_result = run_new_demo()
    print("\n" + "="*80)
    print("NEW DEMO COMPLETE")
    print("="*80)
    print(f"Status: {new_result['status']}")
    print(f"Elapsed: {new_result['elapsed_seconds']:.2f}s")
    print(f"Final Loss: {new_result['final_loss']:.4f}")
    print(f"Avg Loss: {new_result['avg_loss']:.4f}")
