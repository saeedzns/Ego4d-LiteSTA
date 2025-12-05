#!/usr/bin/env python3
"""
VideoMAE Model: Video Masked Autoencoder for Egocentric Pretraining

This implements a minimal VideoMAE architecture:
- Encoder: 3D patch embedding + Transformer encoder (only visible patches)
- Decoder: Lightweight Transformer decoder for reconstruction

References:
- VideoMAE: https://arxiv.org/abs/2203.12602
- MAE: https://arxiv.org/abs/2111.06377

The encoder is the reusable component for Track B downstream tasks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VideoMAEConfig:
    """Configuration for VideoMAE model."""
    
    # Input
    img_size: int = 224
    in_channels: int = 3
    
    # Patch embedding
    patch_size: int = 16          # spatial patch size
    tubelet_size: int = 2         # temporal patch size (how many frames per tube)
    
    # Encoder (ViT-Base by default)
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    
    # Decoder (lightweight)
    decoder_embed_dim: int = 384
    decoder_depth: int = 4
    decoder_num_heads: int = 6
    
    # Masking
    mask_ratio: float = 0.9       # VideoMAE uses very high masking (90%)
    
    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Normalization
    norm_layer: str = "layernorm"
    
    @property
    def num_frames(self) -> int:
        """Number of frames expected (T must be divisible by tubelet_size)."""
        return 16  # Default, overridden during forward
    
    @property
    def patches_per_frame(self) -> int:
        """Number of spatial patches per frame."""
        return (self.img_size // self.patch_size) ** 2


# =============================================================================
# Patch Embedding (3D Tubelet)
# =============================================================================

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding using temporal tubelets.
    
    Converts video (B, C, T, H, W) -> sequence of tokens (B, N, D)
    where N = (T/tubelet_size) * (H/patch_size) * (W/patch_size)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        
        self.grid_size = img_size // patch_size
        
        # 3D convolution for tubelet embedding
        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            patches: (B, N, D) where N = T' * H' * W'
            grid_size: (T', H', W')
        """
        B, C, T, H, W = x.shape
        
        # Apply 3D conv
        x = self.proj(x)  # (B, D, T', H', W')
        
        T_out, H_out, W_out = x.shape[2:]
        
        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        return x, (T_out, H_out, W_out)


# =============================================================================
# Transformer Components
# =============================================================================

class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP block."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# VideoMAE Encoder
# =============================================================================

class VideoMAEEncoder(nn.Module):
    """
    VideoMAE Encoder.
    
    Takes visible patches only (after masking) and encodes them.
    This is the reusable component for downstream tasks.
    """
    
    def __init__(self, cfg: VideoMAEConfig):
        super().__init__()
        self.cfg = cfg
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            tubelet_size=cfg.tubelet_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )
        
        # Positional embedding (for all patches, including masked)
        # Will be indexed by visible patches during forward
        self.num_patches = None  # Computed dynamically
        self.pos_embed = None    # Registered during first forward or init
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                drop=cfg.dropout,
                attn_drop=cfg.attention_dropout,
            )
            for _ in range(cfg.depth)
        ])
        
        self.norm = nn.LayerNorm(cfg.embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_recursive)
    
    def _init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def _get_pos_embed(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Get or create positional embeddings."""
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches + 1:
            # Create sinusoidal positional embeddings
            pos_embed = torch.zeros(1, num_patches + 1, self.cfg.embed_dim, device=device)
            pos_embed = nn.Parameter(pos_embed, requires_grad=False)
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embed = pos_embed.to(device)
        return self.pos_embed
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            x: (B, C, T, H, W) video tensor
            mask: Optional (B, N) boolean mask where True = keep, False = mask out
        
        Returns:
            encoded: (B, N_visible + 1, D) encoded visible tokens + CLS
            grid_size: (T', H', W') patch grid dimensions
        """
        B = x.shape[0]
        
        # Patch embedding
        x, grid_size = self.patch_embed(x)  # (B, N, D)
        N = x.shape[1]
        
        # Get positional embeddings
        pos_embed = self._get_pos_embed(N, x.device)
        
        # Add positional embedding (excluding CLS position)
        x = x + pos_embed[:, 1:, :]
        
        # Apply mask if provided (select visible patches only)
        if mask is not None:
            # mask: (B, N) boolean, True = visible
            # Select visible patches
            visible_indices = mask.nonzero(as_tuple=False)
            
            # Gather visible patches
            x_visible = []
            for b in range(B):
                batch_mask = mask[b]
                x_visible.append(x[b][batch_mask])
            
            # Pad to same length (if needed)
            max_visible = max(xv.shape[0] for xv in x_visible)
            x_padded = torch.zeros(B, max_visible, x.shape[-1], device=x.device, dtype=x.dtype)
            for b, xv in enumerate(x_visible):
                x_padded[b, :xv.shape[0]] = xv
            x = x_padded
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + pos_embed[:, :1, :]
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x, grid_size
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video (no masking, full forward).
        
        Args:
            x: (B, C, T, H, W) video tensor
        
        Returns:
            features: (B, N + 1, D) all tokens including CLS
        """
        encoded, _ = self.forward(x, mask=None)
        return encoded
    
    def get_intermediate_features(self, x: torch.Tensor, layers: list = None) -> Dict[str, torch.Tensor]:
        """
        Get features from intermediate layers.
        
        Args:
            x: (B, C, T, H, W) video tensor
            layers: List of layer indices to extract from (default: all)
        
        Returns:
            dict mapping layer_idx -> features
        """
        if layers is None:
            layers = list(range(len(self.blocks)))
        
        B = x.shape[0]
        x, grid_size = self.patch_embed(x)
        N = x.shape[1]
        
        pos_embed = self._get_pos_embed(N, x.device)
        x = x + pos_embed[:, 1:, :]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + pos_embed[:, :1, :]
        x = torch.cat([cls_tokens, x], dim=1)
        
        features = {}
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in layers:
                features[idx] = self.norm(x.clone())
        
        return features


# =============================================================================
# VideoMAE Decoder
# =============================================================================

class VideoMAEDecoder(nn.Module):
    """
    Lightweight decoder for reconstruction.
    Only used during pretraining, not needed for downstream tasks.
    """
    
    def __init__(self, cfg: VideoMAEConfig):
        super().__init__()
        self.cfg = cfg
        
        # Project encoder output to decoder dim
        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.decoder_embed_dim,
                num_heads=cfg.decoder_num_heads,
                mlp_ratio=cfg.mlp_ratio,
                drop=cfg.dropout,
                attn_drop=cfg.attention_dropout,
            )
            for _ in range(cfg.decoder_depth)
        ])
        
        self.norm = nn.LayerNorm(cfg.decoder_embed_dim)
        
        # Reconstruction head: predict pixel values for each patch
        patch_dim = cfg.tubelet_size * cfg.patch_size * cfg.patch_size * cfg.in_channels
        self.pred = nn.Linear(cfg.decoder_embed_dim, patch_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights_recursive)
    
    def _init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        num_patches: int,
    ) -> torch.Tensor:
        """
        Reconstruct masked patches.
        
        Args:
            encoded: (B, N_visible + 1, D) encoder output (with CLS)
            mask: (B, N) boolean mask where True = visible
            num_patches: total number of patches N
        
        Returns:
            pred: (B, N, patch_dim) reconstructed patches
        """
        B = encoded.shape[0]
        device = encoded.device
        
        # Project to decoder dimension
        x = self.decoder_embed(encoded)  # (B, N_visible + 1, D_dec)
        
        # Separate CLS and visible tokens
        cls_token = x[:, :1, :]
        visible_tokens = x[:, 1:, :]
        
        # Create full sequence with mask tokens
        full_tokens = self.mask_token.expand(B, num_patches, -1).clone()
        
        # Fill in visible tokens
        for b in range(B):
            visible_idx = mask[b].nonzero(as_tuple=True)[0]
            full_tokens[b, visible_idx] = visible_tokens[b, :len(visible_idx)]
        
        # Add CLS token back
        x = torch.cat([cls_token, full_tokens], dim=1)
        
        # Decoder blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Predict patches (exclude CLS)
        pred = self.pred(x[:, 1:, :])  # (B, N, patch_dim)
        
        return pred


# =============================================================================
# Full VideoMAE Model
# =============================================================================

class VideoMAE(nn.Module):
    """
    Full VideoMAE model for self-supervised pretraining.
    
    Combines encoder and decoder with random masking.
    """
    
    def __init__(self, cfg: VideoMAEConfig):
        super().__init__()
        self.cfg = cfg
        
        self.encoder = VideoMAEEncoder(cfg)
        self.decoder = VideoMAEDecoder(cfg)
    
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generate random mask.
        
        Args:
            x: (B, N, D) patch sequence
            mask_ratio: fraction to mask (0.9 = 90% masked)
        
        Returns:
            mask: (B, N) boolean where True = visible (keep)
        """
        B, N, _ = x.shape
        num_keep = int(N * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise to get indices
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Create mask: True for kept patches
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        for b in range(B):
            mask[b, ids_shuffle[b, :num_keep]] = True
        
        return mask
    
    def patchify(self, video: torch.Tensor) -> torch.Tensor:
        """
        Convert video to patches for loss computation.
        
        Args:
            video: (B, C, T, H, W)
        
        Returns:
            patches: (B, N, patch_dim)
        """
        B, C, T, H, W = video.shape
        p = self.cfg.patch_size
        t = self.cfg.tubelet_size
        
        T_out = T // t
        H_out = H // p
        W_out = W // p
        
        # Reshape to patches
        video = video.reshape(B, C, T_out, t, H_out, p, W_out, p)
        video = video.permute(0, 2, 4, 6, 3, 5, 7, 1)  # (B, T', H', W', t, p, p, C)
        patches = video.reshape(B, T_out * H_out * W_out, t * p * p * C)
        
        return patches
    
    def forward(
        self,
        video: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with masking and reconstruction.
        
        Args:
            video: (B, C, T, H, W) input video
            mask_ratio: override default mask ratio
        
        Returns:
            dict with 'loss', 'pred', 'target', 'mask'
        """
        if mask_ratio is None:
            mask_ratio = self.cfg.mask_ratio
        
        B = video.shape[0]
        
        # Get patches for loss computation
        target = self.patchify(video)  # (B, N, patch_dim)
        N = target.shape[1]
        
        # Create random mask
        mask = self.random_masking(target, mask_ratio)  # (B, N) True = visible
        
        # Encode visible patches
        encoded, grid_size = self.encoder(video, mask)
        
        # Decode and reconstruct
        pred = self.decoder(encoded, mask, N)  # (B, N, patch_dim)
        
        # Compute loss only on masked patches
        masked_indices = ~mask  # True = masked (need to reconstruct)
        
        loss = F.mse_loss(
            pred[masked_indices],
            target[masked_indices],
            reduction='mean'
        )
        
        return {
            'loss': loss,
            'pred': pred,
            'target': target,
            'mask': mask,
        }
    
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video without masking (for downstream tasks).
        
        Args:
            video: (B, C, T, H, W)
        
        Returns:
            features: (B, N + 1, D) encoded tokens including CLS
        """
        return self.encoder.forward_features(video)


# =============================================================================
# Utilities
# =============================================================================

def load_videomae_encoder(
    checkpoint_path: str,
    cfg: Optional[VideoMAEConfig] = None,
    device: str = "cpu",
) -> VideoMAEEncoder:
    """
    Load pretrained VideoMAE encoder from checkpoint.
    
    Args:
        checkpoint_path: path to checkpoint file
        cfg: config (if None, loaded from checkpoint)
        device: device to load to
    
    Returns:
        VideoMAEEncoder with loaded weights
        
    Supports multiple checkpoint formats:
    1. Our custom format: {'encoder_state_dict': ...}
    2. Full model: {'state_dict': ...} with encoder.* prefix
    3. HuggingFace format: {'model': ...} with videomae.* prefix
    4. Raw state dict
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if cfg is None:
        if 'config' in ckpt:
            cfg_dict = ckpt['config']
            # Handle HuggingFace config format
            if isinstance(cfg_dict, dict) and 'hidden_size' in cfg_dict:
                # Map HuggingFace keys to our config
                cfg = VideoMAEConfig(
                    embed_dim=cfg_dict.get('hidden_size', 768),
                    depth=cfg_dict.get('num_hidden_layers', 12),
                    num_heads=cfg_dict.get('num_attention_heads', 12),
                    patch_size=cfg_dict.get('patch_size', 16),
                    tubelet_size=cfg_dict.get('tubelet_size', 2),
                    img_size=cfg_dict.get('image_size', 224),
                )
            else:
                try:
                    cfg = VideoMAEConfig(**cfg_dict)
                except TypeError:
                    cfg = VideoMAEConfig()  # Use defaults
        else:
            cfg = VideoMAEConfig()  # Use defaults
    
    # Create encoder
    encoder = VideoMAEEncoder(cfg)
    
    # Load weights - support multiple formats
    if 'encoder_state_dict' in ckpt:
        # Our custom encoder-only format
        encoder.load_state_dict(ckpt['encoder_state_dict'])
    elif 'model' in ckpt:
        # HuggingFace VideoMAEForPreTraining format
        # Keys are like: videomae.embeddings.*, videomae.encoder.*
        state_dict = ckpt['model']
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('videomae.'):
                # Remove 'videomae.' prefix
                new_key = k[len('videomae.'):]
                encoder_state_dict[new_key] = v
        if encoder_state_dict:
            encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"[VideoMAE] Loaded HuggingFace encoder weights ({len(encoder_state_dict)} keys)")
        else:
            print("[VideoMAE] Warning: No encoder weights found in HuggingFace checkpoint")
    elif 'state_dict' in ckpt:
        # Full model checkpoint with encoder.* prefix
        state_dict = {
            k.replace('encoder.', ''): v 
            for k, v in ckpt['state_dict'].items() 
            if k.startswith('encoder.')
        }
        encoder.load_state_dict(state_dict)
    else:
        # Assume it's just encoder state dict
        encoder.load_state_dict(ckpt)
    
    encoder.eval()
    return encoder


def create_videomae(
    variant: str = "base",
    **kwargs
) -> VideoMAE:
    """
    Create VideoMAE model with predefined configurations.
    
    Args:
        variant: "tiny", "small", "base", "large"
        **kwargs: override config values
    
    Returns:
        VideoMAE model
    """
    configs = {
        "tiny": dict(embed_dim=192, depth=6, num_heads=3, decoder_embed_dim=96, decoder_depth=2),
        "small": dict(embed_dim=384, depth=8, num_heads=6, decoder_embed_dim=192, decoder_depth=3),
        "base": dict(embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=384, decoder_depth=4),
        "large": dict(embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8),
    }
    
    cfg_dict = configs.get(variant, configs["base"])
    cfg_dict.update(kwargs)
    
    cfg = VideoMAEConfig(**cfg_dict)
    return VideoMAE(cfg)


# =============================================================================
# Demo
# =============================================================================

def _demo():
    """Test VideoMAE model."""
    print("=" * 60)
    print("VideoMAE Model Demo")
    print("=" * 60)
    
    # Create model
    cfg = VideoMAEConfig(
        img_size=224,
        patch_size=16,
        tubelet_size=2,
        embed_dim=384,  # Small for demo
        depth=4,
        num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=2,
    )
    
    model = VideoMAE(cfg)
    
    # Count parameters
    n_params_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_params_decoder = sum(p.numel() for p in model.decoder.parameters())
    n_params_total = sum(p.numel() for p in model.parameters())
    
    print(f"Encoder params: {n_params_encoder / 1e6:.2f}M")
    print(f"Decoder params: {n_params_decoder / 1e6:.2f}M")
    print(f"Total params: {n_params_total / 1e6:.2f}M")
    
    # Test forward
    B, C, T, H, W = 2, 3, 16, 224, 224
    video = torch.randn(B, C, T, H, W)
    
    print(f"\nInput shape: {video.shape}")
    
    # Forward with masking (pretraining)
    output = model(video)
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Pred shape: {output['pred'].shape}")
    print(f"Target shape: {output['target'].shape}")
    print(f"Visible patches: {output['mask'].sum().item()} / {output['mask'].numel()}")
    
    # Encode without masking (downstream)
    features = model.encode(video)
    print(f"\nEncoded features shape: {features.shape}")
    
    # Test encoder only
    encoder_out, grid_size = model.encoder(video)
    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"Grid size: {grid_size}")
    
    print("\n[Demo] Success!")


if __name__ == "__main__":
    _demo()
