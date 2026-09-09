"""Explicit configuration for the future Ego4D-LiteSTA predictor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: Path, repo_root: Path) -> Path:
    """Resolve relative artifact paths against the repository, never the CWD."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


@dataclass(frozen=True)
class InferenceConfig:
    """Paths and runtime settings shared by future inference components.

    Paths are anchored to the repository containing this package by default so
    artifact loading does not depend on the process working directory.
    """

    repo_root: Path = field(default_factory=lambda: _REPO_ROOT)
    track_a_checkpoint: Path = Path(
        "local_extraction/toolkit_yolo/runs/"
        "sta_yolov8s_singlecls_20251113_002330/weights/best.pt"
    )
    track_b_checkpoint: Path = Path(
        "local_extraction/runs/Track_B/checkpoints/"
        "trackB_best_mAP_0.3708_20251225_224220.pt"
    )
    taxonomy_path: Path = Path(
        "local_extraction/v2/org_annotations/fho_sta_val_height-540.json"
    )
    ttc_stats_path: Path = Path(
        "local_extraction/inference/resources/trackB_ttc_stats.json"
    )
    device: str = "cpu"
    temporal_window_length: int = 16
    temporal_stride: int = 2
    candidate_limit: int = 16

    def __post_init__(self) -> None:
        root = Path(self.repo_root).expanduser().resolve()
        object.__setattr__(self, "repo_root", root)
        for name in (
            "track_a_checkpoint",
            "track_b_checkpoint",
            "taxonomy_path",
            "ttc_stats_path",
        ):
            value = _resolve_path(getattr(self, name), root).resolve()
            object.__setattr__(self, name, value)

        if self.temporal_window_length <= 0:
            raise ValueError("temporal_window_length must be greater than zero")
        if self.temporal_stride <= 0:
            raise ValueError("temporal_stride must be greater than zero")
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than zero")
        if not self.device:
            raise ValueError("device must not be empty")
