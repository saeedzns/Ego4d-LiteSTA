"""Validation and temporal frame resolution for production inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import InferenceConfig


class InferenceInputError(ValueError):
    """Raised when a prediction frame cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedFrameInput:
    """A target frame and the chronological context used by Track B."""

    target_frame: Path
    uid: str
    frame_directory: Path
    context_frames: tuple[Path, ...]


def _list_uid_frames(frame_directory: Path) -> list[Path]:
    """Mirror Track B's `.jpg` filtering and lexicographic Path sorting."""
    return sorted(
        path
        for path in frame_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )


def resolve_frame_input(
    target_frame: str | Path,
    config: InferenceConfig,
) -> ResolvedFrameInput:
    """Validate a frame path and resolve its Track B temporal context.

    The sequence semantics intentionally match ``sample_window_ending_at``:
    selection walks backward through the sorted available-frame list by list
    position, includes the target frame, and reverses the result to
    chronological order. Missing historical frames are not padded.
    """
    target = Path(target_frame).expanduser()
    if not target.exists():
        raise InferenceInputError(f"Target frame does not exist: {target}")
    if not target.is_file():
        raise InferenceInputError(f"Target frame is not a file: {target}")
    if target.suffix.lower() != ".jpg":
        raise InferenceInputError(f"Unsupported frame extension: {target}")

    target = target.resolve()
    frame_directory = target.parent
    try:
        frames = _list_uid_frames(frame_directory)
    except OSError as exc:
        raise InferenceInputError(
            f"Could not enumerate frame directory '{frame_directory}': {exc}"
        ) from exc
    if not frames:
        raise InferenceInputError(
            f"Frame directory contains no valid .jpg frames: {frame_directory}"
        )

    frames = [frame.resolve() for frame in frames]
    try:
        target_index = frames.index(target)
    except ValueError as exc:
        raise InferenceInputError(
            f"Target frame could not be located in frame sequence: {target}"
        ) from exc

    selected: list[Path] = []
    index = target_index
    while index >= 0 and len(selected) < config.temporal_window_length:
        selected.append(frames[index])
        index -= config.temporal_stride
    selected.reverse()

    return ResolvedFrameInput(
        target_frame=target,
        uid=frame_directory.name,
        frame_directory=frame_directory,
        context_frames=tuple(selected),
    )
