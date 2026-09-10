"""Production ResNet18 preprocessing and temporal token extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from .config import InferenceConfig
from .artifacts import ArtifactError, TrackBFeatureContract, load_track_b_feature_contract
from .inputs import ResolvedFrameInput


class TokenizerError(RuntimeError):
    """Raised when tokenizer setup, image loading, or feature extraction fails."""


@dataclass(frozen=True)
class ExtractedTokens:
    """CPU float32 spatial tokens for one frame and its temporal context."""

    image_tokens: torch.Tensor
    video_tokens: torch.Tensor
    grid_hw: tuple[int, int]


def _default_image_loader(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise TokenizerError(f"Could not load image '{path}': {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_resnet18(weights_path: Path, contract: TrackBFeatureContract) -> torch.nn.Module:
    try:
        if not weights_path.exists():
            raise TokenizerError(f"ResNet18 weight file does not exist: {weights_path}")
        if not weights_path.is_file():
            raise TokenizerError(f"ResNet18 weight path is not a file: {weights_path}")
        actual_hash = _sha256(weights_path)
        if actual_hash != contract.backbone_weights_sha256:
            raise TokenizerError(
                "ResNet18 weight SHA-256 mismatch: "
                f"expected {contract.backbone_weights_sha256}, got {actual_hash}"
            )
        backbone = models.resnet18(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TokenizerError(f"ResNet18 weight file must contain a state dict: {weights_path}")
        backbone.load_state_dict(state)
        return _ResNet18Layer4(backbone)
    except Exception as exc:
        raise TokenizerError(f"Could not construct ResNet18 backbone: {exc}") from exc


class _ResNet18Layer4(torch.nn.Module):
    """ResNet18 stem through layer4, matching the thesis backbone wrapper."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        return self.layer4(outputs)


class TrackBTokenizer:
    """Load one ResNet18 backbone and extract thesis-compatible tokens."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        backbone: torch.nn.Module | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        image_loader: Callable[[Path], Image.Image] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        try:
            self.contract = load_track_b_feature_contract(config.track_b_feature_contract_path)
        except ArtifactError as exc:
            raise TokenizerError(str(exc)) from exc
        if config.track_b_checkpoint.name != self.contract.checkpoint:
            raise TokenizerError(
                "Track B checkpoint does not match feature contract: "
                f"configured={config.track_b_checkpoint.name}, contract={self.contract.checkpoint}"
            )
        if config.temporal_window_length != self.contract.temporal_window:
            raise TokenizerError(
                "Temporal window does not match feature contract: "
                f"configured={config.temporal_window_length}, contract={self.contract.temporal_window}"
            )
        if config.temporal_stride != self.contract.temporal_stride:
            raise TokenizerError(
                "Temporal stride does not match feature contract: "
                f"configured={config.temporal_stride}, contract={self.contract.temporal_stride}"
            )
        if backbone is None:
            if config.tokenizer_weights_path is None:
                raise TokenizerError(
                    "An explicit ResNet18 weight file is required; set "
                    "InferenceConfig.tokenizer_weights_path to the provisioned "
                    "IMAGENET1K_V1 artifact."
                )
            if self.contract.backbone != "resnet18":
                raise TokenizerError(f"Unsupported feature-contract backbone: {self.contract.backbone}")
        image_size = self.contract.image_size
        mean = self.contract.normalization_mean
        std = self.contract.normalization_std
        self.image_loader = image_loader or _default_image_loader
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        try:
            self.backbone = backbone or _build_resnet18(config.tokenizer_weights_path, self.contract)
            self.backbone.to(self.device)
            self.backbone.eval()
        except TokenizerError:
            raise
        except Exception as exc:
            raise TokenizerError(f"Could not initialize ResNet18 tokenizer: {exc}") from exc

    def _frame_tokens(self, path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        try:
            image = self.image_loader(path)
            inputs = self.transform(image)
            if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3:
                raise TokenizerError(
                    f"Preprocessing for '{path}' must return a 3D tensor (C,H,W)"
                )
            inputs = inputs.unsqueeze(0).to(self.device)
            with torch.inference_mode():
                features = self.backbone(inputs)
        except TokenizerError:
            raise
        except Exception as exc:
            raise TokenizerError(f"Could not extract tokens from image '{path}': {exc}") from exc

        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise TokenizerError(
                f"Backbone output for '{path}' must be a 4D tensor (B,C,H,W)"
            )
        if features.shape[0] != 1 or features.shape[1] != 512:
            raise TokenizerError(
                f"Backbone output for '{path}' has unexpected shape {tuple(features.shape)}; "
                "expected (1, 512, H, W)"
            )
        height, width = int(features.shape[2]), int(features.shape[3])
        if height <= 0 or width <= 0:
            raise TokenizerError(f"Backbone output for '{path}' has an empty spatial grid")
        tokens = features.permute(0, 2, 3, 1).reshape(-1, 512).float().cpu()
        if self.contract is not None:
            if (height, width) != self.contract.grid_hw or tuple(tokens.shape) != (
                self.contract.token_count, self.contract.token_dim
            ):
                raise TokenizerError(
                    f"Feature-contract mismatch for '{path}': grid={(height, width)}, "
                    f"tokens={tuple(tokens.shape)}, expected grid={self.contract.grid_hw}, "
                    f"tokens=({self.contract.token_count}, {self.contract.token_dim})"
                )
        return tokens, (height, width)

    def extract(self, resolved_input: ResolvedFrameInput) -> ExtractedTokens:
        """Extract target-frame and chronological context tokens."""
        if not resolved_input.context_frames:
            raise TokenizerError("Resolved input contains no context frames")
        if resolved_input.context_frames[-1] != resolved_input.target_frame:
            raise TokenizerError("Resolved context must end with the target frame")

        image_tokens, grid_hw = self._frame_tokens(resolved_input.target_frame)
        temporal_tokens = []
        for frame in resolved_input.context_frames:
            tokens, frame_grid = self._frame_tokens(frame)
            if frame_grid != grid_hw or tokens.shape != image_tokens.shape:
                raise TokenizerError(
                    f"Inconsistent backbone output shape for context frame '{frame}'"
                )
            temporal_tokens.append(tokens)
        return ExtractedTokens(
            image_tokens=image_tokens,
            video_tokens=torch.stack(temporal_tokens, dim=0),
            grid_hw=grid_hw,
        )


__all__ = ["ExtractedTokens", "TokenizerError", "TrackBTokenizer"]