from pathlib import Path

import pytest

from local_extraction.inference import InferenceConfig
from local_extraction.inference.detector import (
    DetectionCandidate,
    DetectorError,
    TrackADetector,
)


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index]
        return FakeTensor(value) if isinstance(value, list) else value

    def tolist(self):
        self.calls.append("tolist")
        return self.values

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        return self

    def detach(self):
        self.calls.append("detach")
        return self


class FakeBoxes:
    def __init__(self, coordinates, confidences, classes):
        self.xyxy = FakeTensor(coordinates)
        self.conf = FakeTensor(confidences)
        self.cls = FakeTensor(classes)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.predict_calls = []
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return self.results


def _config(tmp_path: Path, **overrides) -> InferenceConfig:
    checkpoint = tmp_path / "track_a.pt"
    checkpoint.touch()
    values = {
        "repo_root": tmp_path,
        "track_a_checkpoint": checkpoint,
        **overrides,
    }
    return InferenceConfig(**values)


def _frame(tmp_path: Path) -> Path:
    frame = tmp_path / "uid" / "0000001.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.touch()
    return frame


def _model() -> FakeModel:
    return FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    [[1, 2, 10, 20], [3, 4, 30, 40]],
                    [0.2, 0.9],
                    [0, 0],
                )
            )
        ]
    )


def test_detection_candidate_validates_and_is_immutable() -> None:
    candidate = DetectionCandidate((1.0, 2.0, 3.0, 4.0), 0.5, 0)

    assert candidate.box_xyxy == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(DetectorError, match="exactly four"):
        DetectionCandidate((1.0, 2.0, 3.0), 0.5)
    with pytest.raises(DetectorError, match="finite"):
        DetectionCandidate((1.0, 2.0, float("nan"), 4.0), 0.5)
    with pytest.raises(DetectorError, match="between zero and one"):
        DetectionCandidate((1.0, 2.0, 3.0, 4.0), 1.5)


def test_detector_uses_configured_checkpoint_and_loads_once(tmp_path: Path) -> None:
    model = _model()
    loaded_paths = []

    def loader(path: str):
        loaded_paths.append(path)
        return model

    config = _config(tmp_path)
    detector = TrackADetector(config, model_loader=loader)

    assert loaded_paths == [str(config.track_a_checkpoint)]
    assert model.devices == ["cpu"]
    detector.detect(_frame(tmp_path))
    detector.detect(_frame(tmp_path))
    assert len(loaded_paths) == 1
    assert len(model.predict_calls) == 2


def test_detect_preserves_coordinates_confidence_and_verified_settings(tmp_path: Path) -> None:
    model = _model()
    detector = TrackADetector(_config(tmp_path), model=model)

    candidates = detector.detect(_frame(tmp_path))

    assert candidates == (
        DetectionCandidate((3.0, 4.0, 30.0, 40.0), 0.9, 0),
        DetectionCandidate((1.0, 2.0, 10.0, 20.0), 0.2, 0),
    )
    assert model.predict_calls[0] == {
        "source": str(_frame(tmp_path)),
        "imgsz": 960,
        "conf": 0.05,
        "iou": 0.45,
        "verbose": False,
    }


def test_top_k_is_stable_for_equal_confidence(tmp_path: Path) -> None:
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    [[1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]],
                    [0.8, 0.9, 0.9],
                    [0, 1, 2],
                )
            )
        ]
    )
    detector = TrackADetector(_config(tmp_path, track_a_top_k=2), model=model)

    candidates = detector.detect(_frame(tmp_path))

    assert [candidate.class_id for candidate in candidates] == [1, 2]
    assert [candidate.confidence for candidate in candidates] == [0.9, 0.9]


def test_empty_detection_result_is_valid(tmp_path: Path) -> None:
    detector = TrackADetector(
        _config(tmp_path), model=FakeModel([FakeResult(None)])
    )

    assert detector.detect(_frame(tmp_path)) == ()


def test_empty_result_collection_is_valid(tmp_path: Path) -> None:
    detector = TrackADetector(_config(tmp_path), model=FakeModel([]))

    assert detector.detect(_frame(tmp_path)) == ()


def test_malformed_detector_output_raises_clear_error(tmp_path: Path) -> None:
    malformed = FakeModel([FakeResult(type("Boxes", (), {"xyxy": FakeTensor([[1, 2, 3, 4]])})())])
    detector = TrackADetector(_config(tmp_path), model=malformed)

    with pytest.raises(DetectorError, match="missing xyxy, conf, or cls"):
        detector.detect(_frame(tmp_path))


def test_missing_checkpoint_raises_clear_error(tmp_path: Path) -> None:
    config = InferenceConfig(
        repo_root=tmp_path,
        track_a_checkpoint=tmp_path / "missing.pt",
    )

    with pytest.raises(DetectorError, match="checkpoint does not exist"):
        TrackADetector(config, model=_model())


def test_checkpoint_directory_raises_clear_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / "track_a.pt"
    checkpoint.mkdir()
    config = InferenceConfig(repo_root=tmp_path, track_a_checkpoint=checkpoint)

    with pytest.raises(DetectorError, match="checkpoint is not a file"):
        TrackADetector(config, model=_model())


def test_model_loader_failure_is_wrapped_and_chained(tmp_path: Path) -> None:
    original = RuntimeError("loader failed")

    def loader(path: str):
        raise original

    with pytest.raises(DetectorError, match="Could not load Track A YOLO checkpoint") as error:
        TrackADetector(_config(tmp_path), model_loader=loader)

    assert error.value.__cause__ is original


def test_mismatched_detector_output_lengths_raise_clear_error(tmp_path: Path) -> None:
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    [[1, 2, 10, 20], [3, 4, 30, 40]],
                    [0.9],
                    [0, 1],
                )
            )
        ]
    )
    detector = TrackADetector(_config(tmp_path), model=model)

    with pytest.raises(DetectorError, match="output lengths are inconsistent: xyxy=2, conf=1, cls=2"):
        detector.detect(_frame(tmp_path))


def test_detector_uses_explicit_cpu_safe_tensor_conversion(tmp_path: Path) -> None:
    model = _model()
    detector = TrackADetector(_config(tmp_path), model=model)

    detector.detect(_frame(tmp_path))

    boxes = model.results[0].boxes
    assert boxes.xyxy.calls == ["detach", "cpu", "tolist"]
    assert boxes.conf.calls == ["detach", "cpu", "tolist"]
    assert boxes.cls.calls == ["detach", "cpu", "tolist"]


def test_missing_or_invalid_input_frame_raises_clear_error(tmp_path: Path) -> None:
    detector = TrackADetector(_config(tmp_path), model=_model())

    with pytest.raises(DetectorError, match="does not exist"):
        detector.detect(tmp_path / "missing.jpg")
    directory = tmp_path / "directory.jpg"
    directory.mkdir()
    with pytest.raises(DetectorError, match="not a file"):
        detector.detect(directory)
    png = tmp_path / "uid" / "0000002.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    png.touch()
    with pytest.raises(DetectorError, match="Unsupported detector input frame extension"):
        detector.detect(png)


def test_postprocessing_matches_track_a_helper(monkeypatch, tmp_path: Path) -> None:
    from local_extraction.trackA.trackA_stageA import trackA_stageA

    coordinates = [[1, 2, 10, 20], [3, 4, 30, 40], [5, 6, 50, 60]]
    confidences = [0.2, 0.9, 0.8]
    classes = [0, 1, 2]
    model = FakeModel([FakeResult(FakeBoxes(coordinates, confidences, classes))])
    frame = _frame(tmp_path)
    monkeypatch.setattr(trackA_stageA, "K", 2)

    thesis = trackA_stageA.yolo_detect_one(frame, model)
    production = TrackADetector(_config(tmp_path, track_a_top_k=2), model=model).detect(frame)

    assert [(item["x1"], item["y1"], item["x2"], item["y2"], item["conf"], item["cls"]) for item in thesis] == [
        (candidate.box_xyxy[0], candidate.box_xyxy[1], candidate.box_xyxy[2], candidate.box_xyxy[3], candidate.confidence, candidate.class_id)
        for candidate in production
    ]


def test_package_root_import_remains_lightweight() -> None:
    import subprocess
    import sys

    script = """
import sys
from local_extraction.inference import InferenceConfig, resolve_frame_input
assert InferenceConfig is not None
assert resolve_frame_input is not None
assert 'torch' not in sys.modules
assert 'ultralytics' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
