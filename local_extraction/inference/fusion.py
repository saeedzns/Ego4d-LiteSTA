"""Production Track B projector and temporal fusion."""

from __future__ import annotations

from dataclasses import dataclass
import re
import math
from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactError, TrackBFeatureContract, load_track_b_feature_contract
from .config import InferenceConfig
from .tokenizer import ExtractedTokens


class FusionError(RuntimeError):
    """Raised when Track B projector/fusion setup or execution fails."""


@dataclass(frozen=True)
class FusedTokens:
    """Unbatched fused image/video token grids returned by TrackBFusion."""

    fused_image_tokens: torch.Tensor
    fused_video_tokens: torch.Tensor


class TrackBFeatureFusion:
    """Load and run the trained Track B projector and TrackBFusion modules."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        try:
            self.contract = load_track_b_feature_contract(config.track_b_feature_contract_path)
        except ArtifactError as exc:
            raise FusionError(str(exc)) from exc
        if config.track_b_checkpoint.name != self.contract.checkpoint:
            raise FusionError(
                "Track B checkpoint does not match feature contract: "
                f"configured={config.track_b_checkpoint.name}, contract={self.contract.checkpoint}"
            )
        if config.temporal_window_length != self.contract.temporal_window:
            raise FusionError(
                "Temporal window does not match feature contract: "
                f"configured={config.temporal_window_length}, contract={self.contract.temporal_window}"
            )
        if config.temporal_stride != self.contract.temporal_stride:
            raise FusionError(
                "Temporal stride does not match feature contract: "
                f"configured={config.temporal_stride}, contract={self.contract.temporal_stride}"
            )
        checkpoint = Path(config.track_b_checkpoint)
        if not checkpoint.exists():
            raise FusionError(f"Track B checkpoint does not exist: {checkpoint}")
        if not checkpoint.is_file():
            raise FusionError(f"Track B checkpoint is not a file: {checkpoint}")

        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self._load_models(payload)
        except FusionError:
            raise
        except Exception as exc:
            raise FusionError(f"Could not load Track B projector/fusion: {exc}") from exc

    def _load_models(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise FusionError("Track B checkpoint must contain a mapping")
        projector_state = payload.get("projector")
        fusion_state = payload.get("fusion")
        if not isinstance(projector_state, dict):
            raise FusionError("Track B checkpoint is missing projector state")
        if not isinstance(fusion_state, dict):
            raise FusionError("Track B checkpoint is missing fusion state")

        projector_weight = projector_state.get("weight")
        if projector_weight is None or len(projector_weight.shape) != 2:
            raise FusionError("Track B projector state must contain a 2D weight")
        projector_out, projector_in = map(int, projector_weight.shape)
        if projector_in != self.contract.token_dim:
            raise FusionError(
                f"Projector input dimension {projector_in} does not match feature contract "
                f"token_dim {self.contract.token_dim}"
            )
        if projector_out <= 0:
            raise FusionError("Track B projector output dimension must be positive")

        train_config = payload.get("train_config")
        if not isinstance(train_config, dict):
            raise FusionError("Track B checkpoint is missing train_config metadata")

        required_metadata = (
            "fusion_heads", "fusion_dropout", "fusion_ff_mult", "fgtp_stride_t",
        )
        missing_metadata = [key for key in required_metadata if key not in train_config]
        if missing_metadata:
            raise FusionError(f"Track B checkpoint is missing fusion metadata: {missing_metadata}")
        heads = train_config["fusion_heads"]
        dropout = train_config["fusion_dropout"]
        ff_mult = train_config["fusion_ff_mult"]
        fgtp_stride = train_config["fgtp_stride_t"]
        if isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0:
            raise FusionError("fusion_heads must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise FusionError("fusion_dropout must be numeric")
        if not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1:
            raise FusionError("fusion_dropout must be finite and in [0,1)")
        if isinstance(ff_mult, bool) or not isinstance(ff_mult, int) or ff_mult <= 0:
            raise FusionError("fusion_ff_mult must be a positive integer")
        if isinstance(fgtp_stride, bool) or not isinstance(fgtp_stride, int) or fgtp_stride <= 0:
            raise FusionError("fgtp_stride_t must be a positive integer")
        if projector_out % heads != 0:
            raise FusionError("fusion_heads must divide the projector output dimension")
        projector_in_metadata = train_config.get("projector_in_dim")
        if projector_in_metadata is not None and (
            isinstance(projector_in_metadata, bool)
            or not isinstance(projector_in_metadata, int)
            or projector_in_metadata != projector_in
        ):
            raise FusionError("projector_in_dim metadata does not match projector state")
        block_indices = {
            int(match.group(1))
            for key in fusion_state
            if (match := re.match(r"blocks\.(\d+)\.", key))
        }
        if not block_indices:
            raise FusionError("Track B fusion state contains no attention blocks")
        layers = max(block_indices) + 1
        if block_indices != set(range(layers)):
            raise FusionError("Track B fusion state has non-contiguous attention blocks")

        try:
            from local_extraction.trackB.trackB_fusion import FusionConfig, TrackBFusion

            self.projector = torch.nn.Linear(projector_in, projector_out)
            self.fusion = TrackBFusion(
                FusionConfig(
                    dim=projector_out,
                    heads=heads,
                    layers=layers,
                    dropout=float(dropout),
                    fgtp_stride_t=fgtp_stride,
                    ff_mult=ff_mult,
                )
            )
            self.projector.load_state_dict(projector_state, strict=True)
            self.fusion.load_state_dict(fusion_state, strict=True)
        except FusionError:
            raise
        except Exception as exc:
            raise FusionError(f"Could not reconstruct Track B projector/fusion: {exc}") from exc

        self.projector.to(self.config.device).eval()
        self.fusion.to(self.config.device).eval()
        self.projector_dim = projector_out
        self.fusion_config = self.fusion.cfg

    def fuse(self, extracted: ExtractedTokens) -> FusedTokens:
        """Project and fuse one tokenizer result using thesis batch semantics."""
        image_tokens = extracted.image_tokens
        video_tokens = extracted.video_tokens
        expected_grid = self.contract.grid_hw
        expected_shape = (self.contract.token_count, self.contract.token_dim)
        if extracted.grid_hw != expected_grid:
            raise FusionError(
                f"Feature grid {extracted.grid_hw} does not match contract grid {expected_grid}"
            )
        if not isinstance(image_tokens, torch.Tensor) or image_tokens.ndim != 2:
            raise FusionError("Image tokens must have shape (N,C)")
        if not isinstance(video_tokens, torch.Tensor) or video_tokens.ndim != 3:
            raise FusionError("Video tokens must have shape (T,N,C)")
        if image_tokens.dtype != torch.float32 or video_tokens.dtype != torch.float32:
            raise FusionError("Fusion input tokens must have dtype torch.float32")
        if tuple(image_tokens.shape) != expected_shape:
            raise FusionError(
                f"Image token shape {tuple(image_tokens.shape)} does not match expected {expected_shape}"
            )
        if (
            tuple(video_tokens.shape[1:]) != expected_shape
            or not 1 <= video_tokens.shape[0] <= self.contract.temporal_window
        ):
            raise FusionError(
                f"Video token shape {tuple(video_tokens.shape)} must use "
                f"1 <= T <= {self.contract.temporal_window} and "
                f"(N,C)=({expected_shape[0]},{expected_shape[1]})"
            )
        try:
            with torch.inference_mode():
                image_batch = self.projector(image_tokens.unsqueeze(0).to(self.config.device))
                video_batch = self.projector(video_tokens.unsqueeze(0).to(self.config.device))
                fused_image, fused_video = self.fusion(image_batch, video_batch)
        except Exception as exc:
            raise FusionError(f"Track B projector/fusion forward failed: {exc}") from exc

        expected_output = (self.contract.token_count, self.projector_dim)
        if tuple(fused_image.shape) != (1, *expected_output) or tuple(fused_video.shape) != (1, *expected_output):
            raise FusionError(
                "Unexpected Track B fusion output shapes: "
                f"image={tuple(fused_image.shape)}, video={tuple(fused_video.shape)}"
            )
        return FusedTokens(
            fused_image_tokens=fused_image[0].float().cpu(),
            fused_video_tokens=fused_video[0].float().cpu(),
        )


__all__ = ["FusedTokens", "FusionError", "TrackBFeatureFusion"]
