#!/usr/bin/env python3
"""
Track B Head: small placeholder head that maps fused tokens to logits and TTC.

Contract:
- Inputs: fused_img, fused_vid as (B,N,C)
- Outputs: dict with
    - cls_logits: (B,N,num_classes)
    - ttc: (B,N,1)  # regressed time-to-contact (or any scalar)

It uses a light MLP on the concatenation [fused_img || fused_vid].
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class HeadConfig:
    dim: int
    num_classes: int = 2
    hidden: int = 256
    dropout: float = 0.1
    # Optional multi-task heads (set >0 to enable)
    num_noun_classes: int = 0
    num_verb_classes: int = 0
    num_ttc_bins: int = 0
    # TTC mode: 'reg' (default) or 'bin' (classification) or 'both'
    ttc_mode: str = "reg"


class TrackBHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        in_dim = cfg.dim * 2  # concat of image and video fused tokens
        self.cfg = cfg
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.cls_head = nn.Linear(cfg.hidden, cfg.num_classes)
        # Continuous TTC regression head (always available for backward-compat)
        self.ttc_head = nn.Linear(cfg.hidden, 1)
        # Optional multi-task heads
        if cfg.num_noun_classes and cfg.num_noun_classes > 0:
            self.noun_head = nn.Linear(cfg.hidden, cfg.num_noun_classes)
        if cfg.num_verb_classes and cfg.num_verb_classes > 0:
            self.verb_head = nn.Linear(cfg.hidden, cfg.num_verb_classes)
        if cfg.num_ttc_bins and cfg.num_ttc_bins > 0:
            self.ttc_bin_head = nn.Linear(cfg.hidden, cfg.num_ttc_bins)

    def forward(self, fused_img: torch.Tensor, fused_vid: torch.Tensor):
        # fused_img, fused_vid: (B,N,C)
        x = torch.cat([fused_img, fused_vid], dim=-1)
        h = self.backbone(x)
        out = {
            "cls_logits": self.cls_head(h),  # (B,N,K_next)
            "ttc": self.ttc_head(h),         # (B,N,1) continuous TTC
        }
        if hasattr(self, "noun_head"):
            out["noun_logits"] = self.noun_head(h)   # (B,N,num_noun_classes)
        if hasattr(self, "verb_head"):
            out["verb_logits"] = self.verb_head(h)   # (B,N,num_verb_classes)
        if hasattr(self, "ttc_bin_head"):
            out["ttc_bin_logits"] = self.ttc_bin_head(h)  # (B,N,num_ttc_bins)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, C = 2, 49, 256
    from trackB_fusion import FusionConfig, TrackBFusion
    fcfg = FusionConfig(dim=C)
    fusion = TrackBFusion(fcfg)
    img = torch.randn(B, N, C)
    vid = torch.randn(B, 8, N, C)
    fused_img, fused_vid = fusion(img, vid)
    head = TrackBHead(HeadConfig(dim=C))
    out = head(fused_img, fused_vid)
    print({k: tuple(v.shape) for k, v in out.items()})
