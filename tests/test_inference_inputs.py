from pathlib import Path
import subprocess
import sys

import pytest

from local_extraction.inference import (
    InferenceConfig,
    InferenceInputError,
    resolve_frame_input,
)
from local_extraction.trackB.trackB_tokenizer import sample_window_ending_at


def _make_frames(directory: Path, count: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(count):
        frame = directory / f"{index:07d}.jpg"
        frame.touch()
        frames.append(frame)
    return frames


def test_resolves_chronological_window_with_target_and_stride(tmp_path: Path) -> None:
    frame_directory = tmp_path / "uid-normal"
    frames = _make_frames(frame_directory, 10)
    config = InferenceConfig(temporal_window_length=4, temporal_stride=2)

    resolved = resolve_frame_input(frames[7], config)

    assert resolved.uid == "uid-normal"
    assert resolved.frame_directory == frame_directory.resolve()
    assert resolved.target_frame == frames[7].resolve()
    assert resolved.context_frames == tuple(
        frame.resolve() for frame in (frames[1], frames[3], frames[5], frames[7])
    )
    assert resolved.context_frames[-1] == resolved.target_frame


def test_matches_existing_thesis_temporal_helper(tmp_path: Path) -> None:
    frame_directory = tmp_path / "uid-equivalence"
    frames = _make_frames(frame_directory, 12)
    config = InferenceConfig(temporal_window_length=5, temporal_stride=3)

    resolved = resolve_frame_input(frames[10], config)
    expected = sample_window_ending_at(
        frames[10], tmp_path, config.temporal_window_length, config.temporal_stride
    )

    assert resolved.context_frames == tuple(frame.resolve() for frame in expected)


def test_first_frame_returns_only_available_history(tmp_path: Path) -> None:
    frames = _make_frames(tmp_path / "uid-first", 5)
    config = InferenceConfig(temporal_window_length=16, temporal_stride=2)

    resolved = resolve_frame_input(frames[0], config)

    assert resolved.context_frames == (frames[0].resolve(),)


def test_short_history_is_not_padded(tmp_path: Path) -> None:
    frames = _make_frames(tmp_path / "uid-short", 3)
    config = InferenceConfig(temporal_window_length=16, temporal_stride=2)

    resolved = resolve_frame_input(frames[2], config)

    assert resolved.context_frames == (frames[0].resolve(), frames[2].resolve())
    assert len(resolved.context_frames) < config.temporal_window_length


def test_irregular_filenames_follow_sorted_available_files(tmp_path: Path) -> None:
    frame_directory = tmp_path / "uid-gapped"
    frame_directory.mkdir()
    frames = [
        frame_directory / "0000002.jpg",
        frame_directory / "0000010.jpg",
        frame_directory / "0000100.jpg",
    ]
    for frame in frames:
        frame.touch()
    config = InferenceConfig(temporal_window_length=2, temporal_stride=1)

    resolved = resolve_frame_input(frames[2], config)

    assert resolved.context_frames == (frames[1].resolve(), frames[2].resolve())


def test_missing_target_frame_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(InferenceInputError, match="Target frame does not exist"):
        resolve_frame_input(tmp_path / "uid-missing" / "0000001.jpg", InferenceConfig())


def test_target_directory_is_rejected(tmp_path: Path) -> None:
    target_directory = tmp_path / "uid-directory"
    target_directory.mkdir()

    with pytest.raises(InferenceInputError, match="Target frame is not a file"):
        resolve_frame_input(target_directory, InferenceConfig())


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "uid-unsupported" / "0000001.png"
    target.parent.mkdir()
    target.touch()

    with pytest.raises(InferenceInputError, match="Unsupported frame extension"):
        resolve_frame_input(target, InferenceConfig())


def test_resolution_is_deterministic(tmp_path: Path) -> None:
    frames = _make_frames(tmp_path / "uid-repeat", 10)
    config = InferenceConfig(temporal_window_length=6, temporal_stride=2)

    first = resolve_frame_input(frames[8], config)
    second = resolve_frame_input(frames[8], config)

    assert first == second


def test_package_input_import_is_lightweight() -> None:
    script = """
import sys
from local_extraction.inference import InferenceConfig, resolve_frame_input
assert InferenceConfig is not None
assert resolve_frame_input is not None
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
