#!/usr/bin/env python3
"""
Track B — Lightweight Fusion (STAformer‑Lite spirit)

Modules provided:
  1) Frame‑Guided Temporal Pooling (FGTP): projects short‑clip tokens onto the last‑frame grid.
  2) Dual Image↔Video Cross‑Attention: 2–4 lightweight layers @ 256‑dim that let appearance sharpen motion and vice versa.

Outcome goal (per thesis plan): +1–2 mAP on N+V and N+δ with small FLOPs/VRAM increase.

This file is a minimal, shape‑clean PyTorch scaffold. It expects tokens from backbones and returns fused tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------- Config ----------------------------

@dataclass
class FusionConfig:
    dim: int = 256            # token feature dim
    heads: int = 8            # attention heads
    layers: int = 2           # number of Dual Cross‑Attention layers (2–4 recommended)
    dropout: float = 0.1
    fgtp_stride_t: int = 2    # temporal stride for FGTP (downsample T by stride)
    ff_mult: int = 4          # feedforward expansion ratio


# -------------------- Frame‑Guided Temporal Pooling --------------------

class FGTP(nn.Module):
    """Frame‑Guided Temporal Pooling (FGTP).

    Inputs:
      - vid_tokens: (B, T, N, C) video tokens over time (T frames; N spatial tokens per frame; dim C)
      - img_tokens_last: (B, N, C) tokens of the last frame (guidance)

    Output:
      - pooled: (B, N, C) temporally pooled tokens aligned to last frame

    Mechanism:
      - Optionally subsample time by stride (fgtp_stride_t)
      - Compute per‑spatial similarity between last‑frame queries and each time slice keys
      - Softmax over T to get weights, then weighted sum over T
      - Residual add with last‑frame tokens
    """
    def __init__(self, dim: int, stride_t: int = 2):
        super().__init__()
        self.stride_t = max(1, int(stride_t))
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.out = nn.Linear(dim, dim, bias=True)

    def forward(self, vid_tokens: torch.Tensor, img_tokens_last: torch.Tensor) -> torch.Tensor:
        B, T, N, C = vid_tokens.shape
        if self.stride_t > 1 and T > 1:
            vid_tokens = vid_tokens[:, :: self.stride_t]
            T = vid_tokens.size(1)
        q = self.proj_q(img_tokens_last)               # (B, N, C)
        k = self.proj_k(vid_tokens)                    # (B, T, N, C)
        k_ = k.transpose(1, 2)                         # (B, N, T, C)
        # Similarity per spatial token across time; einsum performs dot over C
        sim = torch.einsum('bnc,bntc->bnt', q, k_) * self.scale  # (B, N, T)
        attn = sim.softmax(dim=-1)                     # (B, N, T)
        pooled = torch.einsum('bnt,btnc->bnc', attn, vid_tokens)  # (B, N, C)
        pooled = self.out(pooled)
        return pooled + img_tokens_last


# ----------------------- Dual Cross‑Attention Block ----------------------

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DualCrossAttention(nn.Module):
    """Symmetric cross‑attention between image and pooled video tokens.

    Inputs:
      - img: (B, N_img, C)
      - vid: (B, N_vid, C)
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.ln_img_q = nn.LayerNorm(dim)
        self.ln_vid_kv = nn.LayerNorm(dim)
        self.ln_vid_q = nn.LayerNorm(dim)
        self.ln_img_kv = nn.LayerNorm(dim)

        self.attn_img_to_vid = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_vid_to_img = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.ff_img = MLP(dim, mlp_ratio=ff_mult, dropout=dropout)
        self.ff_vid = MLP(dim, mlp_ratio=ff_mult, dropout=dropout)

    def forward(self, img: torch.Tensor, vid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Image queries video
        iq = self.ln_img_q(img)
        vk = self.ln_vid_kv(vid)
        img_x, _ = self.attn_img_to_vid(iq, vk, vk, need_weights=False)
        img = img + self.drop(img_x)
        img = img + self.ff_img(self.ln_img_q(img))

        # Video queries image
        vq = self.ln_vid_q(vid)
        ik = self.ln_img_kv(img)
        vid_x, _ = self.attn_vid_to_img(vq, ik, ik, need_weights=False)
        vid = vid + self.drop(vid_x)
        vid = vid + self.ff_vid(self.ln_vid_q(vid))
        return img, vid


# ----------------------------- Fusion Model -----------------------------

class TrackBFusion(nn.Module):
    """FGTP + stacked Dual Cross‑Attention blocks.

    Forward contract:
      - img_tokens: (B, N, C)     last‑frame tokens
      - vid_tokens: (B, T, N, C)  short‑clip tokens
    Returns:
      - fused_img: (B, N, C)
      - fused_vid: (B, N, C)  pooled and cross‑attended
    """
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.fgtp = FGTP(cfg.dim, stride_t=cfg.fgtp_stride_t)
        self.blocks = nn.ModuleList([
            DualCrossAttention(cfg.dim, heads=cfg.heads, dropout=cfg.dropout, ff_mult=cfg.ff_mult)
            for _ in range(cfg.layers)
        ])
        self.norm_img = nn.LayerNorm(cfg.dim)
        self.norm_vid = nn.LayerNorm(cfg.dim)

    def forward(self, img_tokens: torch.Tensor, vid_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_vid = self.fgtp(vid_tokens, img_tokens)
        img, vid = img_tokens, pooled_vid
        for blk in self.blocks:
            img, vid = blk(img, vid)
        return self.norm_img(img), self.norm_vid(vid)


# ------------------------------- Smoke Test ------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, T, N, C = 2, 8, 14*14, 256
    cfg = FusionConfig(dim=C, heads=8, layers=2, dropout=0.1, fgtp_stride_t=2)
    img_tokens = torch.randn(B, N, C)
    vid_tokens = torch.randn(B, T, N, C)
    model = TrackBFusion(cfg)
    fused_img, fused_vid = model(img_tokens, vid_tokens)
    assert fused_img.shape == (B, N, C)
    assert fused_vid.shape == (B, N, C)
    print("[trackB] Smoke test OK:", {'fused_img': tuple(fused_img.shape), 'fused_vid': tuple(fused_vid.shape)})


if __name__ == "__main__":
    _smoke_test()
