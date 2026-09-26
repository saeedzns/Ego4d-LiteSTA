"""CPU-side validation and loading for production inference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

class ArtifactError(RuntimeError):
    """Raised when a required inference artifact is missing or inconsistent."""


@dataclass(frozen=True)
class TTCStats:
    """Training-time TTC normalization statistics.

    These are persisted rather than recomputed at inference time because
    recomputing from a validation or runtime manifest can shift every decoded
    TTC value away from the scale used to train the checkpoint.
    """

    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    source_manifest: str
    normalization: str


@dataclass(frozen=True)
class TrackBArtifactMetadata:
    """Validated metadata exposed by a Track B checkpoint."""

    checkpoint_path: Path
    checkpoint_keys: tuple[str, ...]
    noun_id_list: tuple[int, ...]
    verb_id_list: tuple[int, ...]
    num_noun_classes: int
    num_verb_classes: int
    has_projector_state: bool
    has_fusion_state: bool
    has_head_state: bool
    train_config: Mapping[str, Any]
    architecture_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TrackBFeatureContract:
    """Checkpoint-specific contract for raw Track B visual features."""

    checkpoint: str
    backbone: str
    backbone_weights: str
    backbone_weights_filename: str
    backbone_weights_sha256: str
    image_size: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    grid_hw: tuple[int, int]
    token_count: int
    token_dim: int
    temporal_window: int
    temporal_stride: int
    token_dtype: str
    token_device: str
    source_cache: str
    cache_generation_timestamp: str
    provenance_note: str


def _require_file(path: Path, description: str) -> Path:
    path = Path(path).expanduser()
    if not path.exists():
        raise ArtifactError(f"{description} is missing: {path}")
    if not path.is_file():
        raise ArtifactError(f"{description} is not a file: {path}")
    return path


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"TTC stats field '{field_name}' must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactError(f"TTC stats field '{field_name}' must be finite")
    return result


def load_ttc_stats(path: Path) -> TTCStats:
    """Load and validate persisted training TTC statistics."""
    path = _require_file(Path(path), "TTC statistics file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Could not read TTC statistics file '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError(f"TTC statistics file must contain a JSON object: {path}")

    required = ("count", "mean", "std", "min", "max", "source_manifest", "normalization")
    missing = [name for name in required if name not in data]
    if missing:
        raise ArtifactError(f"TTC statistics file is missing fields {missing}: {path}")

    count = data["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ArtifactError(f"TTC stats field 'count' must be a positive integer: {path}")
    mean = _number(data["mean"], "mean")
    std = _number(data["std"], "std")
    if std <= 0:
        raise ArtifactError(f"TTC stats field 'std' must be greater than zero: {path}")
    minimum = _number(data["min"], "min")
    maximum = _number(data["max"], "max")
    if minimum > maximum:
        raise ArtifactError(f"TTC stats fields 'min' and 'max' are inconsistent: {path}")
    if not isinstance(data["source_manifest"], str) or not data["source_manifest"].strip():
        raise ArtifactError(f"TTC stats field 'source_manifest' must be a non-empty string: {path}")
    if not isinstance(data["normalization"], str) or not data["normalization"].strip():
        raise ArtifactError(f"TTC stats field 'normalization' must be a non-empty string: {path}")

    return TTCStats(
        count=count,
        mean=mean,
        std=std,
        minimum=minimum,
        maximum=maximum,
        source_manifest=data["source_manifest"],
        normalization=data["normalization"],
    )


def _contract_float_vector(data: Mapping[str, Any], name: str) -> tuple[float, float, float]:
    values = data.get(name)
    if not isinstance(values, list) or len(values) != 3:
        raise ArtifactError(f"Feature contract field '{name}' must contain three values")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"Feature contract field '{name}' must contain numeric values") from exc
    if not all(math.isfinite(value) for value in result):
        raise ArtifactError(f"Feature contract field '{name}' must contain finite values")
    if name == "normalization_std" and any(value <= 0 for value in result):
        raise ArtifactError("Feature contract normalization_std values must be greater than zero")
    return result  # type: ignore[return-value]


def load_track_b_feature_contract(path: Path) -> TrackBFeatureContract:
    """Load and validate the proven checkpoint-specific feature contract."""
    path = _require_file(Path(path), "Track B feature contract")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Could not read Track B feature contract '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError(f"Track B feature contract must contain a JSON object: {path}")

    required_strings = (
        "checkpoint", "backbone", "backbone_weights", "backbone_weights_filename",
        "backbone_weights_sha256", "token_dtype", "token_device", "source_cache",
        "cache_generation_timestamp",
    )
    for name in required_strings:
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise ArtifactError(f"Feature contract field '{name}' must be a non-empty string")
    sha256 = data["backbone_weights_sha256"]
    if re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
        raise ArtifactError("Feature contract SHA-256 must be 64 hexadecimal characters")
    if data["token_dtype"] != "float32":
        raise ArtifactError("Feature contract token_dtype must be 'float32'")
    if data["token_device"] != "cpu":
        raise ArtifactError("Feature contract token_device must be 'cpu'")

    positive_fields = ("image_size", "token_count", "token_dim", "temporal_window", "temporal_stride")
    for name in positive_fields:
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ArtifactError(f"Feature contract field '{name}' must be a positive integer")
    grid = data.get("grid_hw")
    if not isinstance(grid, list) or len(grid) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid
    ):
        raise ArtifactError("Feature contract field 'grid_hw' must contain two positive integers")
    if data["token_count"] != grid[0] * grid[1]:
        raise ArtifactError("Feature contract token_count must equal grid_hw product")

    return TrackBFeatureContract(
        checkpoint=data["checkpoint"], backbone=data["backbone"],
        backbone_weights=data["backbone_weights"],
        backbone_weights_filename=data["backbone_weights_filename"],
        backbone_weights_sha256=sha256.lower(), image_size=data["image_size"],
        normalization_mean=_contract_float_vector(data, "normalization_mean"),
        normalization_std=_contract_float_vector(data, "normalization_std"),
        grid_hw=(grid[0], grid[1]), token_count=data["token_count"],
        token_dim=data["token_dim"], temporal_window=data["temporal_window"],
        temporal_stride=data["temporal_stride"], token_dtype=data["token_dtype"],
        token_device=data["token_device"], source_cache=data["source_cache"],
        cache_generation_timestamp=data["cache_generation_timestamp"],
        provenance_note=str(data.get("provenance_note", "")),
    )


def load_taxonomy(path: Path) -> Any:
    """Load the taxonomy JSON so missing or malformed labels fail early."""
    path = _require_file(Path(path), "taxonomy annotation file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Could not read taxonomy annotation file '{path}': {exc}") from exc


def _class_count(state: Mapping[str, Any], key: str, label: str) -> int:
    weight = state.get(key)
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
        raise ArtifactError(f"Track B checkpoint has no valid {label} state: expected '{key}'")
    return int(weight.shape[0])


def _id_list(payload: Mapping[str, Any], name: str) -> tuple[int, ...]:
    values = payload.get(name)
    if not isinstance(values, (list, tuple)) or not values:
        raise ArtifactError(f"Track B checkpoint is missing a non-empty '{name}'")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ArtifactError(f"Track B checkpoint '{name}' must contain integer IDs")
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ArtifactError(f"Track B checkpoint '{name}' contains duplicate IDs")
    return result


def load_track_b_artifacts(path: Path) -> TrackBArtifactMetadata:
    """Load a Track B checkpoint on CPU and validate its inspection metadata."""
    import torch

    path = _require_file(Path(path), "Track B checkpoint")
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ArtifactError(f"Could not load Track B checkpoint '{path}': {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ArtifactError(f"Track B checkpoint must contain a mapping: {path}")

    required_states = ("projector", "fusion", "head")
    missing_states = [name for name in required_states if not isinstance(checkpoint.get(name), dict)]
    if missing_states:
        raise ArtifactError(f"Track B checkpoint is missing state dictionaries {missing_states}: {path}")

    noun_id_list = _id_list(checkpoint, "noun_id_list")
    verb_id_list = _id_list(checkpoint, "verb_id_list")
    head_state = checkpoint["head"]
    noun_count = _class_count(head_state, "noun_head.weight", "noun")
    verb_count = _class_count(head_state, "verb_head.weight", "verb")
    if noun_count != len(noun_id_list):
        raise ArtifactError(
            f"Noun mapping length ({len(noun_id_list)}) does not match checkpoint head ({noun_count})"
        )
    if verb_count != len(verb_id_list):
        raise ArtifactError(
            f"Verb mapping length ({len(verb_id_list)}) does not match checkpoint head ({verb_count})"
        )

    train_config = checkpoint.get("train_config", {})
    if not isinstance(train_config, dict):
        raise ArtifactError(f"Track B checkpoint 'train_config' must be a mapping: {path}")
    architecture_metadata = {
        "config_name": checkpoint.get("config_name"),
        "train_config": train_config,
        "yaml_config_flat": checkpoint.get("yaml_config_flat"),
    }
    return TrackBArtifactMetadata(
        checkpoint_path=path.resolve(),
        checkpoint_keys=tuple(sorted(checkpoint.keys())),
        noun_id_list=noun_id_list,
        verb_id_list=verb_id_list,
        num_noun_classes=noun_count,
        num_verb_classes=verb_count,
        has_projector_state=True,
        has_fusion_state=True,
        has_head_state=True,
        train_config=train_config,
        architecture_metadata=architecture_metadata,
    )
