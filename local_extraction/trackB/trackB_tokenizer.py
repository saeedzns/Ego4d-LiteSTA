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
# Shared config - can be set by trackB_train_loader.py to use CLI-specified config
_cfg = None

def _load_yaml_config(config_name: str = 'trackB'):
    """Load YAML config for Track B."""
    _THIS_DIR = Path(__file__).resolve().parent
    _LOCAL_EXTRACTION = _THIS_DIR.parent
    for p in [str(_LOCAL_EXTRACTION), str(_LOCAL_EXTRACTION.parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from core import load_config
        return load_config(config_name)
    except Exception:
        return None

def set_tokenizer_config(cfg):
    """Set the shared config from external caller (e.g., trackB_train_loader)."""
    global _cfg
    _cfg = cfg

def get_tokenizer_config():
    """Get the current config, loading default if not set."""
    global _cfg
    if _cfg is None:
        _cfg = _load_yaml_config('trackB')
    return _cfg
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
    video_backbone: str = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.video_backbone', 'resnet18') if get_tokenizer_config() else 'resnet18')
    
    # Image settings
    img_size: int = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.img_size', 224) if get_tokenizer_config() else 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    device: str = field(default_factory=lambda: get_tokenizer_config().get('runtime.device', 'auto') if get_tokenizer_config() else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_half: bool = field(default_factory=lambda: get_tokenizer_config().get('runtime.use_half', False) if get_tokenizer_config() else False)
    time_len: int = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.time_len', 8) if get_tokenizer_config() else 8)
    time_stride: int = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.time_stride', 2) if get_tokenizer_config() else 2)
    
    # VideoMAE-specific settings
    videomae_weights_path: Optional[str] = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.videomae.weights_path', None) if get_tokenizer_config() else None)
    videomae_patch_size: int = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.videomae.patch_size', 16) if get_tokenizer_config() else 16)
    videomae_tubelet_size: int = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.videomae.tubelet_size', 2) if get_tokenizer_config() else 2)
    videomae_embed_dim: int = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.videomae.embed_dim', 768) if get_tokenizer_config() else 768)
    videomae_depth: int = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.videomae.depth', 12) if get_tokenizer_config() else 12)
    videomae_num_heads: int = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.videomae.num_heads', 12) if get_tokenizer_config() else 12)
    videomae_freeze_encoder: bool = field(default_factory=lambda: get_tokenizer_config().get('model.tokenizer.videomae.freeze_encoder', True) if get_tokenizer_config() else True)
    
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


class VideoMAEBackboneWrapper(nn.Module):
    """Wrapper for VideoMAE encoder to match ResNet interface.
    
    VideoMAE outputs (B, N+1, D) where N = T'*H'*W' patches + 1 CLS token.
    For compatibility with existing code, we reshape to (B, C, Hf, Wf) format
    by taking only spatial tokens from the last temporal slice.
    """
    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        # Import with fallback for notebook vs module execution
        try:
            from .videomae.videomae_model import VideoMAEEncoder, VideoMAEConfig
        except ImportError:
            from trackB.videomae.videomae_model import VideoMAEEncoder, VideoMAEConfig
        
        # Build VideoMAE config from TokenizerConfig
        vmae_cfg = VideoMAEConfig(
            img_size=cfg.img_size,
            patch_size=cfg.videomae_patch_size,
            tubelet_size=cfg.videomae_tubelet_size,
            embed_dim=cfg.videomae_embed_dim,
            depth=cfg.videomae_depth,
            num_heads=cfg.videomae_num_heads,
        )
        self.encoder = VideoMAEEncoder(vmae_cfg)
        self.cfg = cfg
        self.vmae_cfg = vmae_cfg
        
        # Load pretrained weights if available
        if cfg.videomae_weights_path:
            self._load_weights(cfg.videomae_weights_path)
        
        # Optionally freeze encoder
        if cfg.videomae_freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
    
    def _load_weights(self, weights_path: str):
        """Load pretrained VideoMAE encoder weights."""
        import os
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'encoder' in state:
                self.encoder.load_state_dict(state['encoder'], strict=False)
            elif 'model' in state:
                # Filter encoder keys
                encoder_state = {k.replace('encoder.', ''): v 
                                for k, v in state['model'].items() 
                                if k.startswith('encoder.')}
                self.encoder.load_state_dict(encoder_state, strict=False)
            else:
                self.encoder.load_state_dict(state, strict=False)
            print(f"[VideoMAE] Loaded weights from {weights_path}")
        else:
            print(f"[VideoMAE] Warning: weights not found at {weights_path}, using random init")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) single frame OR (B, C, T, H, W) video
        
        Returns:
            features: (B, C, Hf, Wf) feature map compatible with ResNet interface
        """
        # Handle single frame input by adding temporal dimension
        if x.dim() == 4:
            # (B, C, H, W) -> (B, C, T, H, W) with T frames by repeating
            x = x.unsqueeze(2).repeat(1, 1, self.vmae_cfg.tubelet_size, 1, 1)
        
        # Forward through encoder (no masking for feature extraction)
        encoded, grid_size = self.encoder(x, mask=None)  # (B, N+1, D)
        
        T_patches, H_patches, W_patches = grid_size
        
        # Remove CLS token
        tokens = encoded[:, 1:, :]  # (B, T'*H'*W', D)
        
        # Reshape to (B, T', H', W', D)
        B, _, D = tokens.shape
        tokens = tokens.view(B, T_patches, H_patches, W_patches, D)
        
        # Take last temporal slice to get spatial feature map (like ResNet)
        spatial = tokens[:, -1, :, :, :]  # (B, H', W', D)
        
        # Permute to (B, D, H', W') to match ResNet output format
        features = spatial.permute(0, 3, 1, 2).contiguous()  # (B, D, Hf, Wf)
        
        return features


def build_backbone(cfg: TokenizerConfig) -> nn.Module:
    """Build video backbone based on config.
    
    Args:
        cfg: TokenizerConfig with video_backbone set to:
            - "resnet18": ImageNet-pretrained ResNet18 (512-dim, 7x7 grid)
            - "videomae_ego": Egocentric VideoMAE encoder (768-dim, 14x14 grid)
    
    Returns:
        Backbone module that outputs (B, C, Hf, Wf) feature maps
    """
    device = cfg.device
    
    if cfg.video_backbone == "videomae_ego":
        print(f"[Tokenizer] Building VideoMAE backbone (embed_dim={cfg.videomae_embed_dim})")
        try:
            model = VideoMAEBackboneWrapper(cfg)
            model = model.to(device)
            model = model.eval()
        except Exception as e:
            print(f"[Tokenizer] ⚠️ VideoMAE build failed: {e}")
            print(f"[Tokenizer] Falling back to ResNet18...")
            model = ResNet18Backbone(pretrained=True).to(device).eval()
    else:
        # Default: ResNet18
        print(f"[Tokenizer] Building ResNet18 backbone")
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
    for p in window_paths:  # Removed tqdm - causes overhead and noisy output
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
    sel = tokens.index_select(0, torch.tensor(idxs, device=tokens.device))
    return sel.mean(dim=0)


# ========================= VideoMAE Encoder Support =========================

# Global cache for VideoMAE encoder (lazy loading)
_videomae_encoder_cache: Optional[nn.Module] = None


def _load_videomae_encoder_cached(cfg: TokenizerConfig) -> nn.Module:
    """Load pretrained VideoMAE encoder from checkpoint (with caching).
    
    Supports both our custom format and HuggingFace format checkpoints.
    
    Returns:
        VideoMAE encoder module (frozen if specified in config)
    """
    global _videomae_encoder_cache
    
    if _videomae_encoder_cache is not None:
        return _videomae_encoder_cache
    
    # Load weights
    weights_path = cfg.videomae_weights_path
    if weights_path is None:
        # Default path
        weights_path = str(Path(__file__).parent.parent / "runs" / "VideoMAE" / "videomae_ego_encoder.pt")
    
    weights_path = Path(weights_path)
    
    # Handle relative paths: resolve from local_extraction directory
    if not weights_path.is_absolute():
        local_extraction_dir = Path(__file__).resolve().parent.parent
        weights_path = local_extraction_dir / weights_path
        # Remove duplicate 'local_extraction' if path starts with it
        if str(weights_path).replace("\\", "/").count("local_extraction") > 1:
            # Path like .../local_extraction/local_extraction/... - fix it
            parts = weights_path.parts
            # Find last occurrence of 'local_extraction' and take from there
            last_idx = len(parts) - 1 - parts[::-1].index("local_extraction")
            weights_path = Path(*parts[:last_idx]) / Path(*parts[last_idx+1:])
    
    if not weights_path.exists():
        raise FileNotFoundError(
            f"VideoMAE encoder weights not found at: {weights_path}\n"
            f"Please run VideoMAE pretraining first:\n"
            f"  python -m trackB.videomae.videomae_pretrain --epochs 100"
        )
    
    # Load checkpoint to determine format
    ckpt = torch.load(str(weights_path), map_location=cfg.device)
    
    # Detect HuggingFace format: must have 'model' with 'videomae.*' keys AND 'config' with 'hidden_size'
    # This distinguishes from our custom format which may also have 'model' key
    is_huggingface = (
        'model' in ckpt 
        and 'config' in ckpt 
        and isinstance(ckpt.get('config'), dict)
        and 'hidden_size' in ckpt.get('config', {})
        and any(k.startswith('videomae.') for k in ckpt.get('model', {}).keys())
    )
    
    if is_huggingface:
        # Use HuggingFace VideoMAEModel - only if transformers is available
        try:
            from transformers import VideoMAEModel, VideoMAEConfig as HFVideoMAEConfig
            
            hf_config = HFVideoMAEConfig(**ckpt['config'])
            encoder = VideoMAEModel(hf_config)
            
            # Load encoder weights
            encoder_weights = {k[len('videomae.'):]: v for k, v in ckpt['model'].items() if k.startswith('videomae.')}
            encoder.load_state_dict(encoder_weights, strict=False)
            
            print(f"[trackB.tokenizer] Loaded HuggingFace VideoMAE encoder from {weights_path}")
            print(f"  Hidden size: {hf_config.hidden_size}, Layers: {hf_config.num_hidden_layers}")
        except ImportError:
            print("[trackB.tokenizer] HuggingFace transformers not installed, using custom loader...")
            is_huggingface = False  # Fall through to custom loader
    
    if not is_huggingface:
        # Use our custom format
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
        
        encoder = _load_encoder(str(weights_path), cfg=vmae_cfg, device=cfg.device)
        print(f"[trackB.tokenizer] Loaded custom VideoMAE encoder from {weights_path}")
    
    # Freeze if specified
    if cfg.videomae_freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()
    
    encoder = encoder.to(cfg.device)
    _videomae_encoder_cache = encoder
    
    return encoder


def video_grid_tokens_videomae(
    window_paths: List[Path], 
    cfg: TokenizerConfig,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Extract video tokens using VideoMAE encoder.
    
    Supports both our custom encoder and HuggingFace VideoMAEModel.
    
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
    
    # Load encoder (cached)
    encoder = _load_videomae_encoder_cached(cfg)
    
    # Check if it's HuggingFace model
    is_huggingface = hasattr(encoder, 'config') and hasattr(encoder.config, 'hidden_size')
    
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
    
    # Convert to (C, T, H, W) for VideoMAE -> then (B, C, T, H, W)
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    video = video.to(cfg.device)
    
    if cfg.use_half and cfg.device.startswith("cuda"):
        video = video.half()
    
    # Encode
    with torch.no_grad():
        if is_huggingface:
            # HuggingFace VideoMAEModel expects (B, T, C, H, W) or (B, C, T, H, W)
            # Actually it expects (B, num_frames, num_channels, height, width)
            video_hf = video.permute(0, 2, 1, 3, 4)  # (1, T, C, H, W)
            outputs = encoder(pixel_values=video_hf, return_dict=True)
            encoded = outputs.last_hidden_state  # (B, N+1, D) or (B, N, D)
            
            # Calculate grid size from config
            patch_size = encoder.config.patch_size
            tubelet_size = encoder.config.tubelet_size
            num_frames = encoder.config.num_frames
            image_size = encoder.config.image_size
            
            T_out = num_frames // tubelet_size
            H_out = image_size // patch_size
            W_out = image_size // patch_size
            grid_size = (T_out, H_out, W_out)
        else:
            # Our custom encoder
            encoded, grid_size = encoder(video)  # (1, N+1, D), (T', H', W')
    
    # Remove CLS token if present (HuggingFace has CLS at position 0)
    if is_huggingface:
        # HuggingFace: first token is CLS (if use_mean_pooling=False)
        # Check shape to determine
        expected_patches = grid_size[0] * grid_size[1] * grid_size[2]
        if encoded.shape[1] == expected_patches + 1:
            tokens = encoded[:, 1:, :]  # Remove CLS
        else:
            tokens = encoded
    else:
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
    
    Supports both our custom encoder and HuggingFace VideoMAEModel.
    
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
    
    # Load encoder (cached)
    encoder = _load_videomae_encoder_cached(cfg)
    
    # Check if it's HuggingFace model
    is_huggingface = hasattr(encoder, 'config') and hasattr(encoder.config, 'hidden_size')
    
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
        if is_huggingface:
            # HuggingFace expects (B, T, C, H, W)
            video_hf = video.permute(0, 2, 1, 3, 4)  # (1, T, C, H, W)
            outputs = encoder(pixel_values=video_hf, return_dict=True)
            encoded = outputs.last_hidden_state
            
            # Calculate grid size from config
            patch_size = encoder.config.patch_size
            tubelet_size = encoder.config.tubelet_size
            num_frames = encoder.config.num_frames
            image_size = encoder.config.image_size
            
            T_out = num_frames // tubelet_size
            H_out = image_size // patch_size
            W_out = image_size // patch_size
            grid_size = (T_out, H_out, W_out)
        else:
            encoded, grid_size = encoder(video)  # (1, N+1, D), (T', H', W')
    
    # Remove CLS token if present
    if is_huggingface:
        expected_patches = grid_size[0] * grid_size[1] * grid_size[2]
        if encoded.shape[1] == expected_patches + 1:
            tokens = encoded[:, 1:, :]
        else:
            tokens = encoded
    else:
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

# Cache transform to avoid recreating on every call
_transform_cache: Optional[T.Compose] = None
_transform_cache_key: Optional[Tuple[int, Tuple, Tuple]] = None

def _get_cached_transform(cfg: TokenizerConfig) -> T.Compose:
    """Get or create cached transform."""
    global _transform_cache, _transform_cache_key
    key = (cfg.img_size, cfg.mean, cfg.std)
    if _transform_cache is None or _transform_cache_key != key:
        _transform_cache = build_transform(cfg)
        _transform_cache_key = key
    return _transform_cache


# ========================= Pre-extracted Token Loading =========================

def load_preextracted_tokens(
    frame_path: Path,
    frames_root: Path,
    tokens_root: Path,
    time_len: int,
    time_stride: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]:
    """
    Load pre-extracted ResNet18 tokens from .pt files.
    
    Args:
        frame_path: Path to the last frame (used to find corresponding .pt file)
        frames_root: Root directory for frames (to determine UID)
        tokens_root: Root directory for pre-extracted tokens (e.g., v2/resnet18_tokens)
        time_len: Number of frames in temporal window
        time_stride: Stride between frames
    
    Returns:
        (img_tokens, vid_tokens, grid_hw) or None if tokens not found
    """
    # Determine UID from frame path
    uid = frame_path.parent.name
    tokens_uid_dir = tokens_root / uid
    
    if not tokens_uid_dir.exists():
        return None
    
    # Load image tokens for the last frame
    img_pt_path = tokens_uid_dir / f"{frame_path.stem}.pt"
    if not img_pt_path.exists():
        return None
    
    try:
        img_data = torch.load(img_pt_path, map_location='cpu')
        img_tokens = img_data['tokens']  # (N, 512)
        grid_hw = img_data['hw']  # (Hf, Wf)
    except Exception:
        return None
    
    # Get window frames for video tokens
    window = sample_window_ending_at(frame_path, frames_root, time_len, time_stride)
    
    if not window:
        # Fallback: replicate image tokens
        vid_tokens = img_tokens.unsqueeze(0).repeat(time_len, 1, 1)
        return img_tokens, vid_tokens, grid_hw
    
    # Load tokens for each frame in window
    vid_tokens_list = []
    for wf in window:
        wf_pt_path = tokens_uid_dir / f"{wf.stem}.pt"
        if wf_pt_path.exists():
            try:
                wf_data = torch.load(wf_pt_path, map_location='cpu')
                vid_tokens_list.append(wf_data['tokens'])
            except Exception:
                # Fallback to image tokens if loading fails
                vid_tokens_list.append(img_tokens)
        else:
            # Missing frame token - use image tokens as fallback
            vid_tokens_list.append(img_tokens)
    
    vid_tokens = torch.stack(vid_tokens_list, dim=0)  # (T, N, 512)
    
    return img_tokens, vid_tokens, grid_hw


def get_tokens(
    frame_path: Path,
    frames_root: Path,
    cfg: TokenizerConfig,
    backbone: Optional[nn.Module] = None,
    transform: Optional[T.Compose] = None,
    tokens_root: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Unified token extraction based on video_backbone config.
    
    Supports loading pre-extracted tokens from .pt files for faster training.
    
    Args:
        frame_path: Path to the last frame
        frames_root: Root directory for frames
        cfg: Tokenizer configuration
        backbone: Optional pre-built ResNet backbone (for resnet18 mode)
        transform: Optional pre-built transform (for resnet18 mode, avoids recreation)
        tokens_root: Optional path to pre-extracted tokens directory (e.g., v2/resnet18_tokens)
    
    Returns:
        img_tokens: (N, C) last-frame tokens
        vid_tokens: (T, N, C) video tokens
        grid_hw: (H', W') spatial grid dimensions
    """
    # Try loading pre-extracted tokens first (ResNet18 only for now)
    if tokens_root is not None and cfg.video_backbone == "resnet18":
        preextracted = load_preextracted_tokens(
            frame_path, frames_root, tokens_root, cfg.time_len, cfg.time_stride
        )
        if preextracted is not None:
            return preextracted
    
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
            backbone = build_backbone(cfg)
        
        # Use provided transform, or get from cache
        tfm = transform if transform is not None else _get_cached_transform(cfg)
        
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
        backbone = build_backbone(cfg)
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
