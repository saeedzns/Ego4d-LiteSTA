"""Local end-to-end short-term anticipation inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from PIL import Image

from .config import InferenceConfig
from .detector import DetectionCandidate, TrackADetector
from .fusion import FusedTokens, TrackBFeatureFusion
from .head import RawCandidateOutput, TrackBInferenceHead
from .inputs import ResolvedFrameInput, resolve_frame_input
from .predictions import PredictionAssembler, STAPrediction
from .tokenizer import ExtractedTokens, TrackBTokenizer


class PipelineError(RuntimeError):
    """Raised when the local inference pipeline cannot complete."""


class DetectorLike(Protocol):
    def detect(self, target_frame: str | Path) -> tuple[DetectionCandidate, ...]:
        ...


class TokenizerLike(Protocol):
    def extract(self, resolved_input: ResolvedFrameInput) -> ExtractedTokens:
        ...


class FusionLike(Protocol):
    def fuse(self, extracted: ExtractedTokens) -> FusedTokens:
        ...


class HeadLike(Protocol):
    def predict(
        self,
        candidates: tuple[DetectionCandidate, ...],
        fused_tokens: FusedTokens,
        image_size: tuple[int, int],
    ) -> tuple[RawCandidateOutput, ...]:
        ...


class AssemblerLike(Protocol):
    def assemble(self, raw_outputs: tuple[RawCandidateOutput, ...]) -> tuple[STAPrediction, ...]:
        ...


@dataclass(frozen=True)
class PipelineResult:
    """Artifacts and ranked predictions for one target frame."""

    resolved_input: ResolvedFrameInput
    image_size: tuple[int, int]
    candidates: tuple[DetectionCandidate, ...]
    predictions: tuple[STAPrediction, ...]


def _read_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise PipelineError(f"Could not read target-frame image size: {path}") from exc
    if height <= 0 or width <= 0:
        raise PipelineError(f"Target-frame image size must be positive: {path}")
    return (height, width)


class ShortTermAnticipationPredictor:
    """Run the local production inference stack for one target frame."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        detector: DetectorLike | None = None,
        tokenizer: TokenizerLike | None = None,
        fusion: FusionLike | None = None,
        head: HeadLike | None = None,
        assembler: AssemblerLike | None = None,
        frame_resolver: Callable[[str | Path, InferenceConfig], ResolvedFrameInput] = resolve_frame_input,
        image_size_reader: Callable[[Path], tuple[int, int]] = _read_image_size,
    ) -> None:
        self.config = config
        self.detector = detector if detector is not None else TrackADetector(config)
        self.tokenizer = tokenizer if tokenizer is not None else TrackBTokenizer(config)
        self.fusion = fusion if fusion is not None else TrackBFeatureFusion(config)
        self.head = head if head is not None else TrackBInferenceHead(config)
        self.assembler = assembler if assembler is not None else PredictionAssembler(config)
        self.frame_resolver = frame_resolver
        self.image_size_reader = image_size_reader

    def predict(self, target_frame: str | Path) -> PipelineResult:
        """Return ranked STA predictions for one target frame."""
        try:
            resolved = self.frame_resolver(target_frame, self.config)
            image_size = self.image_size_reader(resolved.target_frame)
            candidates = self.detector.detect(resolved.target_frame)
            if not candidates:
                return PipelineResult(
                    resolved_input=resolved,
                    image_size=image_size,
                    candidates=(),
                    predictions=(),
                )
            extracted = self.tokenizer.extract(resolved)
            fused = self.fusion.fuse(extracted)
            raw_outputs = self.head.predict(candidates, fused, image_size)
            predictions = self.assembler.assemble(raw_outputs)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"Short-term anticipation prediction failed: {exc}") from exc
        return PipelineResult(
            resolved_input=resolved,
            image_size=image_size,
            candidates=candidates,
            predictions=predictions,
        )


__all__ = [
    "PipelineError",
    "PipelineResult",
    "ShortTermAnticipationPredictor",
]
