"""Final Track B prediction assembly from raw candidate head outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import torch
import torch.nn.functional as F

from .artifacts import ArtifactError, TTCStats, load_ttc_stats
from .config import InferenceConfig
from .detector import DetectionCandidate
from .head import RawCandidateOutput


class PredictionAssemblyError(RuntimeError):
    """Raised when raw head outputs cannot be assembled into predictions."""


@dataclass(frozen=True)
class STAPrediction:
    """One decoded short-term anticipation prediction."""

    candidate: DetectionCandidate
    candidate_index: int
    score: float
    detector_confidence: float
    next_active_probability: float
    predicted_class: int
    noun_id: int
    noun_local_index: int
    noun_probability: float
    verb_id: int
    verb_local_index: int
    verb_probability: float
    ttc_seconds: float
    ttc_raw: float
    ttc_bin: int


def _ttc_to_bin(ttc_seconds: float, thresholds: tuple[float, float, float] = (0.5, 1.0, 2.0)) -> int:
    if ttc_seconds < thresholds[0]:
        return 0
    if ttc_seconds < thresholds[1]:
        return 1
    if ttc_seconds < thresholds[2]:
        return 2
    return 3


def _load_id_maps(checkpoint: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    path = Path(checkpoint)
    if not path.exists():
        raise PredictionAssemblyError(f"Track B checkpoint does not exist: {path}")
    if not path.is_file():
        raise PredictionAssemblyError(f"Track B checkpoint is not a file: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PredictionAssemblyError(f"Could not load Track B checkpoint mappings: {exc}") from exc
    if not isinstance(payload, dict):
        raise PredictionAssemblyError("Track B checkpoint must contain a mapping")
    noun_ids = payload.get("noun_id_list")
    verb_ids = payload.get("verb_id_list")
    if not isinstance(noun_ids, (list, tuple)) or not noun_ids:
        raise PredictionAssemblyError("Track B checkpoint is missing noun_id_list")
    if not isinstance(verb_ids, (list, tuple)) or not verb_ids:
        raise PredictionAssemblyError("Track B checkpoint is missing verb_id_list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in noun_ids):
        raise PredictionAssemblyError("Track B noun_id_list must contain integer IDs")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in verb_ids):
        raise PredictionAssemblyError("Track B verb_id_list must contain integer IDs")
    return tuple(int(value) for value in noun_ids), tuple(int(value) for value in verb_ids)


class PredictionAssembler:
    """Decode raw Track B candidate outputs and rank final STA predictions."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        noun_id_list: tuple[int, ...] | None = None,
        verb_id_list: tuple[int, ...] | None = None,
        ttc_stats: TTCStats | None = None,
    ) -> None:
        self.config = config
        if noun_id_list is None or verb_id_list is None:
            loaded_nouns, loaded_verbs = _load_id_maps(config.track_b_checkpoint)
            noun_id_list = loaded_nouns if noun_id_list is None else noun_id_list
            verb_id_list = loaded_verbs if verb_id_list is None else verb_id_list
        if not noun_id_list or not verb_id_list:
            raise PredictionAssemblyError("Noun and verb ID lists must be non-empty")
        self.noun_id_list = tuple(noun_id_list)
        self.verb_id_list = tuple(verb_id_list)
        try:
            self.ttc_stats = ttc_stats if ttc_stats is not None else load_ttc_stats(config.ttc_stats_path)
        except ArtifactError as exc:
            raise PredictionAssemblyError(str(exc)) from exc

    def assemble(
        self,
        raw_outputs: tuple[RawCandidateOutput, ...] | list[RawCandidateOutput],
    ) -> tuple[STAPrediction, ...]:
        """Return final predictions sorted by next-active probability descending."""
        if not raw_outputs:
            return ()

        predictions = []
        for index, raw in enumerate(raw_outputs):
            predictions.append(self._decode_one(raw, index))
        return tuple(sorted(predictions, key=lambda item: item.score, reverse=True))

    def _decode_one(self, raw: RawCandidateOutput, index: int) -> STAPrediction:
        if not isinstance(raw, RawCandidateOutput):
            raise PredictionAssemblyError("Raw outputs must be RawCandidateOutput instances")
        if not isinstance(raw.candidate, DetectionCandidate):
            raise PredictionAssemblyError("Raw output candidate must be a DetectionCandidate")

        cls_logits = _one_dim_float_tensor(raw.cls_logits, "cls_logits")
        noun_logits = _one_dim_float_tensor(raw.noun_logits, "noun_logits")
        verb_logits = _one_dim_float_tensor(raw.verb_logits, "verb_logits")
        ttc_raw_tensor = _one_dim_float_tensor(raw.ttc_raw, "ttc_raw")
        if cls_logits.numel() < 1:
            raise PredictionAssemblyError("cls_logits must contain at least one class")
        if noun_logits.numel() != len(self.noun_id_list):
            raise PredictionAssemblyError("noun_logits width does not match noun ID list")
        if verb_logits.numel() != len(self.verb_id_list):
            raise PredictionAssemblyError("verb_logits width does not match verb ID list")
        if ttc_raw_tensor.numel() != 1:
            raise PredictionAssemblyError("ttc_raw must contain exactly one value")

        cls_probs = F.softmax(cls_logits, dim=0)
        predicted_class = int(torch.argmax(cls_probs).item())
        next_active_index = 1 if cls_probs.numel() > 1 else 0
        next_active_probability = float(cls_probs[next_active_index].item())

        noun_probs = F.softmax(noun_logits, dim=0)
        noun_local = int(torch.argmax(noun_probs).item())
        verb_probs = F.softmax(verb_logits, dim=0)
        verb_local = int(torch.argmax(verb_probs).item())

        ttc_raw = float(ttc_raw_tensor[0].item())
        ttc_seconds = ttc_raw * self.ttc_stats.std + self.ttc_stats.mean
        if not math.isfinite(ttc_seconds):
            raise PredictionAssemblyError("Decoded TTC seconds must be finite")
        if raw.ttc_bin_logits is not None:
            ttc_bin_logits = _one_dim_float_tensor(raw.ttc_bin_logits, "ttc_bin_logits")
            if ttc_bin_logits.numel() < 1:
                raise PredictionAssemblyError("ttc_bin_logits must contain at least one bin")
            ttc_bin = int(torch.argmax(ttc_bin_logits).item())
        else:
            ttc_bin = _ttc_to_bin(ttc_seconds)

        return STAPrediction(
            candidate=raw.candidate,
            candidate_index=index,
            score=next_active_probability,
            detector_confidence=float(raw.candidate.confidence),
            next_active_probability=next_active_probability,
            predicted_class=predicted_class,
            noun_id=int(self.noun_id_list[noun_local]),
            noun_local_index=noun_local,
            noun_probability=float(noun_probs[noun_local].item()),
            verb_id=int(self.verb_id_list[verb_local]),
            verb_local_index=verb_local,
            verb_probability=float(verb_probs[verb_local].item()),
            ttc_seconds=float(ttc_seconds),
            ttc_raw=ttc_raw,
            ttc_bin=ttc_bin,
        )


def _one_dim_float_tensor(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PredictionAssemblyError(f"{name} must be a tensor")
    if value.ndim != 1:
        raise PredictionAssemblyError(f"{name} must be one-dimensional")
    if value.dtype != torch.float32:
        raise PredictionAssemblyError(f"{name} must have dtype torch.float32")
    if value.device.type != "cpu":
        raise PredictionAssemblyError(f"{name} must be on CPU")
    if not torch.isfinite(value).all():
        raise PredictionAssemblyError(f"{name} must contain finite values")
    return value


__all__ = ["PredictionAssembler", "PredictionAssemblyError", "STAPrediction"]
