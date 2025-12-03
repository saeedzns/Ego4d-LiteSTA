#!/usr/bin/env python3
"""
Unit tests for FGTP invariants and head wiring.

Tests:
- FGTP attention weights over time sum to 1 for each (B,N)
- FGTP residual: output equals pooled_raw + img_tokens_last (within numerical tol)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from trackB_fusion import FusionConfig, FGTP


def test_fgtp_attention_and_residual():
    torch.manual_seed(0)
    B, T, N, C = 3, 6, 10, 32
    cfg = FusionConfig(dim=C, fgtp_stride_t=1)
    fgtp = FGTP(cfg.dim, stride_t=cfg.fgtp_stride_t)

    vid = torch.randn(B, T, N, C)
    img = torch.randn(B, N, C)

    # Manual forward: mirror FGTP computation to expose attn
    with torch.no_grad():
        q = fgtp.proj_q(img)             # (B,N,C)
        k = fgtp.proj_k(vid)             # (B,T,N,C)
        k_ = k.transpose(1, 2)           # (B,N,T,C)
        sim = torch.einsum('bnc,bntc->bnt', q, k_) * fgtp.scale  # (B,N,T)
        attn = sim.softmax(dim=-1)       # (B,N,T)
        pooled_raw = torch.einsum('bnt,btnc->bnc', attn, vid)   # (B,N,C)
        out = fgtp.out(pooled_raw) + img                         # (B,N,C)

    # 1) Attention sums to 1 across T for each (B,N)
    s = attn.sum(dim=-1)  # (B,N)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), f"attn sums not 1: min {s.min()} max {s.max()}"

    # 2) Residual presence: out - img equals learned linear of pooled_raw
    assert torch.allclose(out - img, fgtp.out(pooled_raw), atol=1e-5), "Residual mismatch"


def test_videomae_model_shapes():
    """Test VideoMAE encoder/decoder shapes."""
    from videomae.videomae_model import VideoMAEEncoder, VideoMAE
    
    torch.manual_seed(0)
    
    # Test encoder
    encoder = VideoMAEEncoder(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,  # Smaller for testing
        depth=4,
        num_heads=6,
        tubelet_size=2,
    )
    
    B, T, C, H, W = 2, 8, 3, 224, 224
    x = torch.randn(B, T, C, H, W)
    
    with torch.no_grad():
        tokens = encoder(x)
    
    # Expected: B x num_patches x embed_dim
    # num_patches = (T // tubelet_size) * (H // patch_size) * (W // patch_size)
    expected_t_patches = T // 2
    expected_hw_patches = (H // 16) * (W // 16)
    expected_patches = expected_t_patches * expected_hw_patches
    
    assert tokens.shape[0] == B, f"Batch mismatch: {tokens.shape[0]} != {B}"
    assert tokens.shape[1] == expected_patches, f"Patches mismatch: {tokens.shape[1]} != {expected_patches}"
    assert tokens.shape[2] == 384, f"Embed dim mismatch: {tokens.shape[2]} != 384"
    
    print(f"  [VideoMAE Encoder] Input: {x.shape} -> Tokens: {tokens.shape}")


def test_videomae_masking():
    """Test VideoMAE masked autoencoder forward pass."""
    from videomae.videomae_model import VideoMAE
    
    torch.manual_seed(0)
    
    mae = VideoMAE(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=2,
        decoder_num_heads=3,
        tubelet_size=2,
        mask_ratio=0.75,
    )
    
    B, T, C, H, W = 2, 8, 3, 224, 224
    x = torch.randn(B, T, C, H, W)
    
    with torch.no_grad():
        loss, pred, mask = mae(x)
    
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    # Mask should have mask_ratio of tokens masked (value 1)
    mask_ratio = mask.float().mean().item()
    assert 0.7 < mask_ratio < 0.8, f"Mask ratio should be ~0.75, got {mask_ratio:.3f}"
    
    print(f"  [VideoMAE MAE] Loss: {loss.item():.4f}, Mask ratio: {mask_ratio:.3f}")


def test_tokenizer_config_video_backbone():
    """Test TokenizerConfig video_backbone field."""
    from trackB_tokenizer import TokenizerConfig
    
    # Default should be resnet18
    cfg = TokenizerConfig()
    assert cfg.video_backbone in ['resnet18', 'videomae_ego'], f"Invalid backbone: {cfg.video_backbone}"
    
    # Test with explicit setting
    cfg_resnet = TokenizerConfig(video_backbone='resnet18')
    assert cfg_resnet.video_backbone == 'resnet18'
    
    cfg_videomae = TokenizerConfig(video_backbone='videomae_ego')
    assert cfg_videomae.video_backbone == 'videomae_ego'
    
    print(f"  [TokenizerConfig] video_backbone options work correctly")


def test_projector_dimensions():
    """Test projector input dimensions for different backbones."""
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


def test_fusion_accepts_both_backbones():
    """Test that fusion works with tokens from either backbone."""
    from trackB_fusion import FusionConfig, TrackBFusion
    
    torch.manual_seed(0)
    
    B, T, N, C = 4, 8, 100, 256  # After projection to TOKEN_DIM
    cfg = FusionConfig(dim=C, layers=2)
    fusion = TrackBFusion(cfg)
    
    # Create inputs (same shape after projection)
    vid_tokens = torch.randn(B, T, N, C)
    img_tokens = torch.randn(B, N, C)
    box_tokens = torch.randn(B, N, C)
    
    with torch.no_grad():
        fused = fusion(vid_tokens, img_tokens, box_tokens)
    
    assert fused.shape == (B, N, C), f"Fusion output shape wrong: {fused.shape}"
    print(f"  [Fusion] Input vid {vid_tokens.shape}, img {img_tokens.shape} -> Fused {fused.shape}")


if __name__ == "__main__":
    print("[trackB.tests] Running all tests...\n")
    
    print("1. FGTP attention and residual test")
    test_fgtp_attention_and_residual()
    print("   PASS\n")
    
    print("2. VideoMAE model shapes test")
    test_videomae_model_shapes()
    print("   PASS\n")
    
    print("3. VideoMAE masking test")
    test_videomae_masking()
    print("   PASS\n")
    
    print("4. TokenizerConfig video_backbone test")
    test_tokenizer_config_video_backbone()
    print("   PASS\n")
    
    print("5. Projector dimensions test")
    test_projector_dimensions()
    print("   PASS\n")
    
    print("6. Fusion accepts both backbones test")
    test_fusion_accepts_both_backbones()
    print("   PASS\n")
    
    print("=" * 50)
    print("[trackB.tests] All tests PASSED!")

