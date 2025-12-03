#!/usr/bin/env python3
"""Track B Tokenizer: build grid tokens for last frame and a short clip using a lightweight backbone.

Inputs (existing assets):
    - Frames: local_extraction/v2/extracted_frames/<uid>/<frame:07d>.jpg
    - (Optional) Clips: H:\\My Drive\\ego4d_data\\v2\\clips_540\\clips (not required; we use frames window)

Outputs (in-memory):
    - img_tokens: (N, C)  last-frame grid tokens (flattened Hf*Wf)
    - vid_tokens: (T, N, C)  stacked per-frame grid tokens over a temporal window ending at t

Notes:
    - Uses torchvision ResNet18 backbone by default; returns layer4 feature map (C=512, grid ~7x7 at 224).
    - ROI pooling is provided as simple grid-average inside the projected ROI; avoids compiled ops.

Configuration loaded from configs/trackB.yaml
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm


# ================== YAML CONFIG LOADING ==================
def _load_yaml_config():
    """Load YAML config for Track B."""
    _THIS_DIR = Path(__file__).resolve().parent
    _LOCAL_EXTRACTION = _THIS_DIR.parent
    for p in [str(_LOCAL_EXTRACTION), str(_LOCAL_EXTRACTION.parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from core import load_config
        return load_config('trackB')
    except Exception:
        return None

_cfg = _load_yaml_config()
# =========================================================


# ---------------------------- Config ----------------------------

@dataclass
class TokenizerConfig:
    """Tokenizer configuration. Defaults loaded from YAML if available.
    
    video_backbone options:
      - "resnet18": Current exocentric/web-pretrained 2D backbone (baseline)
      - "videomae_ego": In-domain egocentric VideoMAE encoder (requires pretrained weights)
    """
    # Video backbone selection
    video_backbone: str = field(default_factory=lambda: _cfg.get('model.tokenizer.video_backbone', 'resnet18') if _cfg else 'resnet18')
    
    # Image settings
    img_size: int = field(default_factory=lambda: _cfg.get('model.tokenizer.img_size', 224) if _cfg else 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    device: str = field(default_factory=lambda: _cfg.get('runtime.device', 'auto') if _cfg else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_half: bool = field(default_factory=lambda: _cfg.get('runtime.use_half', False) if _cfg else False)
    time_len: int = field(default_factory=lambda: _cfg.get('model.tokenizer.time_len', 8) if _cfg else 8)
    time_stride: int = field(default_factory=lambda: _cfg.get('model.tokenizer.time_stride', 2) if _cfg else 2)
    
    # VideoMAE-specific settings
    videomae_weights_path: Optional[str] = field(default_factory=lambda: _cfg.get('model.tokenizer.videomae.weights_path', None) if _cfg else None)
    videomae_patch_size: int = field(default_factory=lambda: _cfg.get('model.tokenizer.videomae.patch_size', 16) if _cfg else 16)
    videomae_tubelet_size: int = field(default_factory=lambda: _cfg.get('model.tokenizer.videomae.tubelet_size', 2) if _cfg else 2)
    videomae_embed_dim: int = field(default_factory=lambda: _cfg.get('model.tokenizer.videomae.embed_dim', 768) if _cfg else 768)
    videomae_depth: int = field(default_factory=lambda: _cfg.get('model.tokenizer.videomae.depth', 12) if _cfg else 12)
    videomae_num_heads: int = field(default_factory=lambda: _cfg.get('model.tokenizer.videomae.num_heads', 12) if _cfg else 12)
    videomae_freeze_encoder: bool = field(default_factory=lambda: _cfg.get('model.tokenizer.videomae.freeze_encoder', True) if _cfg else True)
    
    def __post_init__(self):
        # Handle 'auto' device
        if self.device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate video_backbone
        valid_backbones = ("resnet18", "videomae_ego")
        if self.video_backbone not in valid_backbones:
            raise ValueError(f"video_backbone must be one of {valid_backbones}, got '{self.video_backbone}'")


# ------------------------- Backbone builder -------------------------

class ResNet18Backbone(nn.Module):
    """ResNet18 up to layer4 returning a feature map (B, C, Hf, Wf)."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        try:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            m = models.resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        # Freeze by default for feature extraction
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B,C,Hf,Wf)


def build_backbone(device: str) -> nn.Module:
    model = ResNet18Backbone(pretrained=True).to(device).eval()
    return model


def build_transform(cfg: TokenizerConfig) -> T.Compose:
    return T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize(mean=cfg.mean, std=cfg.std),
    ])


# ------------------------ Frame utilities ------------------------

def list_uid_frames(uid_dir: Path) -> List[Path]:
    return sorted([p for p in uid_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])


def sample_window_ending_at(frame_path: Path, frames_root: Path, Tlen: int, stride: int) -> List[Path]:
    """Return a list of frame paths of length <= Tlen ending at frame_path, stepping back by stride."""
    uid_dir = frame_path.parent
    all_frames = list_uid_frames(uid_dir)
    if not all_frames:
        return []
    try:
        idx = all_frames.index(frame_path)
    except ValueError:
        # fall back by matching stem
        stems = [p.stem for p in all_frames]
        if frame_path.stem in stems:
            idx = stems.index(frame_path.stem)
        else:
            return []
    sel = []
    i = idx
    while i >= 0 and len(sel) < Tlen:
        sel.append(all_frames[i])
        i -= stride
    sel.reverse()
    return sel


# ------------------------ Token extraction ------------------------

def image_grid_tokens(img_path: Path, backbone: nn.Module, tfm: T.Compose, cfg: TokenizerConfig) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Return (tokens, (Hf,Wf)). tokens: (N, C) float32 on cpu.
    """
    from PIL import Image
    with Image.open(str(img_path)) as im:
        im = im.convert("RGB")
        x = tfm(im).unsqueeze(0).to(cfg.device)
    if cfg.use_half and cfg.device.startswith("cuda"):
        x = x.half()
        backbone = backbone.half()
    with torch.no_grad():
        feat = backbone(x)  # (1,C,Hf,Wf)
    _, C, Hf, Wf = feat.shape
    tok = feat.permute(0, 2, 3, 1).reshape(-1, C).float().cpu()  # (N,C)
    return tok, (Hf, Wf)


def video_grid_tokens(window_paths: List[Path], backbone: nn.Module, tfm: T.Compose, cfg: TokenizerConfig) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Return (vid_tokens, (Hf,Wf)) with shape (T, N, C)."""
    toks: List[torch.Tensor] = []
    Hf = Wf = None
    for p in tqdm(window_paths, desc="[trackB.tokenizer] frames", leave=False):
        t, (hf, wf) = image_grid_tokens(p, backbone, tfm, cfg)
        if Hf is None:
            Hf, Wf = hf, wf
        toks.append(t)
    if not toks:
        return torch.empty(0), (0, 0)
    Tlen = len(toks)
    N, C = toks[0].shape
    vtok = torch.stack(toks, dim=0)  # (T,N,C)
    return vtok, (Hf, Wf)  # type: ignore


# ----------------------------- ROI utils -----------------------------

def roi_pool_tokens_mean(tokens_hw: Tuple[int, int], tokens: torch.Tensor, box_xyxy: Tuple[float, float, float, float], img_wh: Tuple[int, int]) -> torch.Tensor:
    """Average tokens inside ROI projected to the token grid.

    tokens_hw: (Hf,Wf); tokens: (N,C) flattened; box in pixels; img_wh: (W,H)
    Returns (C,) vector. If ROI projects to empty set, returns mean over all tokens.
    """
    Hf, Wf = tokens_hw
    N, C = tokens.shape
    W, H = img_wh
    if Hf <= 0 or Wf <= 0 or W <= 0 or H <= 0 or N != Hf * Wf:
        return tokens.mean(dim=0)
    x1, y1, x2, y2 = box_xyxy
    gx1 = max(0, min(Wf - 1, int(math.floor(x1 / max(1e-6, W) * Wf))))
    gy1 = max(0, min(Hf - 1, int(math.floor(y1 / max(1e-6, H) * Hf))))
    gx2 = max(0, min(Wf - 1, int(math.ceil(x2 / max(1e-6, W) * Wf) - 1)))
    gy2 = max(0, min(Hf - 1, int(math.ceil(y2 / max(1e-6, H) * Hf) - 1)))
    if gx2 < gx1 or gy2 < gy1:
        return tokens.mean(dim=0)
    # gather indices
    idxs = []
    for yy in range(gy1, gy2 + 1):
        base = yy * Wf
        for xx in range(gx1, gx2 + 1):
            idxs.append(base + xx)
    sel = tokens.index_select(0, torch.tensor(idxs))
    return sel.mean(dim=0)


# ========================= VideoMAE Encoder Support =========================

# Global cache for VideoMAE encoder (lazy loading)
_videomae_encoder_cache: Optional[nn.Module] = None


def load_videomae_encoder(cfg: TokenizerConfig) -> nn.Module:
    """Load pretrained VideoMAE encoder from checkpoint.
    
    Returns:
        VideoMAE encoder module (frozen if specified in config)
    """
    global _videomae_encoder_cache
    
    if _videomae_encoder_cache is not None:
        return _videomae_encoder_cache
    
    # Import VideoMAE module
    try:
        from .videomae import load_videomae_encoder as _load_encoder, VideoMAEConfig
    except ImportError:
        from trackB.videomae import load_videomae_encoder as _load_encoder, VideoMAEConfig
    
    # Create config
    vmae_cfg = VideoMAEConfig(
        img_size=cfg.img_size,
        patch_size=cfg.videomae_patch_size,
        tubelet_size=cfg.videomae_tubelet_size,
        embed_dim=cfg.videomae_embed_dim,
        depth=cfg.videomae_depth,
        num_heads=cfg.videomae_num_heads,
    )
    
    # Load weights
    weights_path = cfg.videomae_weights_path
    if weights_path is None:
        # Default path
        weights_path = str(Path(__file__).parent.parent / "runs" / "VideoMAE" / "videomae_ego_encoder.pt")
    
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(
            f"VideoMAE encoder weights not found at: {weights_path}\n"
            f"Please run VideoMAE pretraining first:\n"
            f"  python -m trackB.videomae.videomae_pretrain --epochs 100"
        )
    
    encoder = _load_encoder(str(weights_path), cfg=vmae_cfg, device=cfg.device)
    
    # Freeze if specified
    if cfg.videomae_freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()
    
    encoder = encoder.to(cfg.device)
    _videomae_encoder_cache = encoder
    
    print(f"[trackB.tokenizer] Loaded VideoMAE encoder from {weights_path}")
    
    return encoder


def video_grid_tokens_videomae(
    window_paths: List[Path], 
    cfg: TokenizerConfig,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Extract video tokens using VideoMAE encoder.
    
    Args:
        window_paths: List of frame paths (T frames)
        cfg: Tokenizer configuration
    
    Returns:
        vid_tokens: (T', N, C) spatiotemporal tokens
            T' = T / tubelet_size
            N = (H / patch_size) * (W / patch_size)
            C = embed_dim
        grid_size: (T', H', W') patch grid dimensions
    """
    from PIL import Image
    
    # Load encoder
    encoder = load_videomae_encoder(cfg)
    
    # Build transform
    tfm = build_transform(cfg)
    
    # Load frames and stack
    frames = []
    for p in window_paths:
        with Image.open(str(p)) as im:
            im = im.convert("RGB")
            frame_tensor = tfm(im)
            frames.append(frame_tensor)
    
    # Pad if needed to match expected frame count
    T = len(frames)
    expected_T = cfg.time_len
    
    if T < expected_T:
        # Pad by repeating last frame
        frames = frames + [frames[-1]] * (expected_T - T)
    elif T > expected_T:
        # Take last T frames
        frames = frames[-expected_T:]
    
    # Stack: (T, C, H, W)
    video = torch.stack(frames, dim=0)
    
    # Convert to (C, T, H, W) for VideoMAE
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    video = video.to(cfg.device)
    
    if cfg.use_half and cfg.device.startswith("cuda"):
        video = video.half()
    
    # Encode
    with torch.no_grad():
        encoded, grid_size = encoder(video)  # (1, N+1, D), (T', H', W')
    
    # Remove CLS token
    tokens = encoded[:, 1:, :]  # (1, N, D)
    
    # Reshape to (T', H'*W', D) format compatible with fusion
    T_out, H_out, W_out = grid_size
    N_spatial = H_out * W_out
    
    tokens = tokens.reshape(1, T_out, N_spatial, -1)  # (1, T', H'*W', D)
    tokens = tokens.squeeze(0)  # (T', N, D)
    tokens = tokens.float().cpu()
    
    return tokens, grid_size


def image_grid_tokens_videomae(
    img_path: Path,
    cfg: TokenizerConfig,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Extract image tokens using VideoMAE encoder (single frame, replicated).
    
    For VideoMAE, we need T frames. We replicate the single image to match.
    This is useful for the last-frame encoding in Track B.
    
    Args:
        img_path: Path to image
        cfg: Tokenizer configuration
    
    Returns:
        img_tokens: (N, C) flattened spatial tokens
        grid_hw: (H', W') spatial grid dimensions
    """
    from PIL import Image
    
    # Load encoder
    encoder = load_videomae_encoder(cfg)
    
    # Build transform
    tfm = build_transform(cfg)
    
    # Load image
    with Image.open(str(img_path)) as im:
        im = im.convert("RGB")
        frame_tensor = tfm(im)
    
    # Replicate to T frames
    T = cfg.time_len
    video = frame_tensor.unsqueeze(0).repeat(T, 1, 1, 1)  # (T, C, H, W)
    
    # Convert to (C, T, H, W) for VideoMAE
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    video = video.to(cfg.device)
    
    if cfg.use_half and cfg.device.startswith("cuda"):
        video = video.half()
    
    # Encode
    with torch.no_grad():
        encoded, grid_size = encoder(video)  # (1, N+1, D), (T', H', W')
    
    # Remove CLS token
    tokens = encoded[:, 1:, :]  # (1, N, D)
    
    # For single image, take tokens from the last temporal position
    T_out, H_out, W_out = grid_size
    N_spatial = H_out * W_out
    
    # Reshape to (T', H'*W', D) and take last temporal slice
    tokens = tokens.reshape(1, T_out, N_spatial, -1)  # (1, T', N, D)
    tokens = tokens[:, -1, :, :]  # (1, N, D) - last temporal position
    tokens = tokens.squeeze(0)  # (N, D)
    tokens = tokens.float().cpu()
    
    return tokens, (H_out, W_out)


# ========================= Unified Token Extraction =========================

def get_tokens(
    frame_path: Path,
    frames_root: Path,
    cfg: TokenizerConfig,
    backbone: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Unified token extraction based on video_backbone config.
    
    Args:
        frame_path: Path to the last frame
        frames_root: Root directory for frames
        cfg: Tokenizer configuration
        backbone: Optional pre-built ResNet backbone (for resnet18 mode)
    
    Returns:
        img_tokens: (N, C) last-frame tokens
        vid_tokens: (T, N, C) video tokens
        grid_hw: (H', W') spatial grid dimensions
    """
    if cfg.video_backbone == "videomae_ego":
        # Use VideoMAE encoder
        window = sample_window_ending_at(frame_path, frames_root, cfg.time_len, cfg.time_stride)
        
        if not window:
            # Fallback: just use the single frame
            window = [frame_path]
        
        vid_tokens, grid_size = video_grid_tokens_videomae(window, cfg)
        img_tokens, grid_hw = image_grid_tokens_videomae(frame_path, cfg)
        
        return img_tokens, vid_tokens, grid_hw
    
    else:
        # Use ResNet18 (default baseline)
        if backbone is None:
            backbone = build_backbone(cfg.device)
        
        tfm = build_transform(cfg)
        
        img_tokens, grid_hw = image_grid_tokens(frame_path, backbone, tfm, cfg)
        
        window = sample_window_ending_at(frame_path, frames_root, cfg.time_len, cfg.time_stride)
        vid_tokens, _ = video_grid_tokens(window, backbone, tfm, cfg)
        
        return img_tokens, vid_tokens, grid_hw


# ------------------------------ Demo ------------------------------

def _demo():
    """Test tokenizer with both backbones."""
    print("=" * 60)
    print("Track B Tokenizer Demo")
    print("=" * 60)
    
    cfg = TokenizerConfig()
    print(f"Video backbone: {cfg.video_backbone}")
    print(f"Device: {cfg.device}")
    
    frames_root = Path("local_extraction") / "v2" / "extracted_frames"
    
    # Pick first uid and its last frame
    uids = [d for d in frames_root.iterdir() if d.is_dir()]
    if not uids:
        print("[trackB.tokenizer] No UIDs under:", frames_root)
        return
    
    uid_dir = uids[0]
    frames = list_uid_frames(uid_dir)
    if not frames:
        print("[trackB.tokenizer] No frames in:", uid_dir)
        return
    
    last = frames[-1]
    print(f"\nTest frame: {last}")
    
    # Test with current backbone setting
    print(f"\n--- Testing with backbone={cfg.video_backbone} ---")
    
    if cfg.video_backbone == "resnet18":
        backbone = build_backbone(cfg.device)
        tfm = build_transform(cfg)
        img_tok, hw = image_grid_tokens(last, backbone, tfm, cfg)
        window = sample_window_ending_at(last, frames_root, cfg.time_len, cfg.time_stride)
        vid_tok, _ = video_grid_tokens(window, backbone, tfm, cfg)
    else:
        img_tok, vid_tok, hw = get_tokens(last, frames_root, cfg)
    
    print(f"  img_tokens: {tuple(img_tok.shape)}")
    print(f"  vid_tokens: {tuple(vid_tok.shape)}")
    print(f"  grid: {hw}")
    
    # Test unified interface
    print(f"\n--- Testing unified get_tokens() ---")
    img_tok2, vid_tok2, hw2 = get_tokens(last, frames_root, cfg)
    print(f"  img_tokens: {tuple(img_tok2.shape)}")
    print(f"  vid_tokens: {tuple(vid_tok2.shape)}")
    print(f"  grid: {hw2}")
    
    print("\n[Demo] Success!")


if __name__ == "__main__":
    _demo()
