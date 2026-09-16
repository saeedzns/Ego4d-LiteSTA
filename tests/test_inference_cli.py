import json
from pathlib import Path

import pytest

from local_extraction.inference import InferenceConfig, ResolvedFrameInput
from local_extraction.inference.cli import InferenceCLIError, main, result_to_dict, run
from local_extraction.inference.detector import DetectionCandidate
from local_extraction.inference.pipeline import PipelineResult
from local_extraction.inference.predictions import STAPrediction


def _result(tmp_path: Path) -> PipelineResult:
    frame = tmp_path / "uid" / "0000001.jpg"
    candidate = DetectionCandidate((1.0, 2.0, 3.0, 4.0), 0.75, 9)
    prediction = STAPrediction(
        candidate=candidate,
        candidate_index=0,
        score=0.8,
        detector_confidence=0.75,
        next_active_probability=0.8,
        predicted_class=1,
        noun_id=10,
        noun_local_index=2,
        noun_probability=0.6,
        verb_id=20,
        verb_local_index=1,
        verb_probability=0.7,
        ttc_seconds=1.25,
        ttc_raw=2.0,
        ttc_bin=2,
    )
    return PipelineResult(
        resolved_input=ResolvedFrameInput(
            target_frame=frame,
            uid="uid",
            frame_directory=frame.parent,
            context_frames=(frame,),
        ),
        image_size=(480, 640),
        candidates=(candidate,),
        predictions=(prediction,),
    )


class FakePredictor:
    def __init__(self, config: InferenceConfig, result: PipelineResult) -> None:
        self.config = config
        self.result = result
        self.calls = []

    def predict(self, target_frame: Path) -> PipelineResult:
        self.calls.append(target_frame)
        return self.result


def test_result_to_dict_uses_stable_json_ready_schema(tmp_path: Path) -> None:
    payload = result_to_dict(_result(tmp_path))

    assert payload["target_frame"].endswith("0000001.jpg")
    assert payload["uid"] == "uid"
    assert payload["image_size"] == {"height": 480, "width": 640}
    assert payload["candidate_count"] == 1
    assert payload["prediction_count"] == 1
    assert payload["predictions"] == [
        {
            "candidate_index": 0,
            "box_xyxy": [1.0, 2.0, 3.0, 4.0],
            "detector_confidence": 0.75,
            "detector_class_id": 9,
            "score": 0.8,
            "next_active_probability": 0.8,
            "predicted_class": 1,
            "noun_id": 10,
            "noun_local_index": 2,
            "noun_probability": 0.6,
            "verb_id": 20,
            "verb_local_index": 1,
            "verb_probability": 0.7,
            "ttc_seconds": 1.25,
            "ttc_raw": 2.0,
            "ttc_bin": 2,
        }
    ]


def test_run_builds_config_and_invokes_injected_predictor(tmp_path: Path) -> None:
    result = _result(tmp_path)
    instances = []

    def factory(config: InferenceConfig) -> FakePredictor:
        predictor = FakePredictor(config, result)
        instances.append(predictor)
        return predictor

    args = type(
        "Args",
        (),
        {
            "device": "cpu",
            "tokenizer_weights": tmp_path / "resnet18.pth",
            "target_frame": result.resolved_input.target_frame,
        },
    )()

    payload = run(args, predictor_factory=factory)

    assert payload["prediction_count"] == 1
    assert instances[0].config.device == "cpu"
    assert instances[0].config.tokenizer_weights_path == (tmp_path / "resnet18.pth").resolve()
    assert instances[0].calls == [result.resolved_input.target_frame]


def test_main_writes_json_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = _result(tmp_path)

    def factory(config: InferenceConfig) -> FakePredictor:
        return FakePredictor(config, result)

    code = main([str(result.resolved_input.target_frame)], predictor_factory=factory)

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["prediction_count"] == 1


def test_main_writes_json_to_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = _result(tmp_path)
    output = tmp_path / "nested" / "prediction.json"

    def factory(config: InferenceConfig) -> FakePredictor:
        return FakePredictor(config, result)

    code = main([str(result.resolved_input.target_frame), "--output", str(output)], predictor_factory=factory)

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert json.loads(output.read_text(encoding="utf-8"))["target_frame"].endswith("0000001.jpg")


def test_run_wraps_predictor_failure(tmp_path: Path) -> None:
    original = RuntimeError("boom")

    class BrokenPredictor:
        def __init__(self, config: InferenceConfig) -> None:
            pass

        def predict(self, target_frame: Path) -> PipelineResult:
            raise original

    args = type(
        "Args",
        (),
        {"device": "cpu", "tokenizer_weights": None, "target_frame": tmp_path / "frame.jpg"},
    )()

    with pytest.raises(InferenceCLIError, match="Local inference failed") as error:
        run(args, predictor_factory=BrokenPredictor)

    assert error.value.__cause__ is original


def test_package_root_import_stays_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "local_extraction.inference" or name.startswith("local_extraction.inference."):
            monkeypatch.delitem(sys.modules, name, raising=False)
        if name in {"torch", "torchvision", "ultralytics"}:
            monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("local_extraction.inference")

    assert module.InferenceConfig
    assert module.resolve_frame_input
    assert "torch" not in sys.modules
    assert "torchvision" not in sys.modules
    assert "ultralytics" not in sys.modules
