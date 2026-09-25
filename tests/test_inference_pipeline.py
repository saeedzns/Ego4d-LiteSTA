from pathlib import Path

import pytest
import torch
from PIL import Image

from local_extraction.inference import InferenceConfig, ResolvedFrameInput
from local_extraction.inference.detector import DetectionCandidate
from local_extraction.inference.fusion import FusedTokens
from local_extraction.inference.head import RawCandidateOutput
from local_extraction.inference.pipeline import (
    PipelineError,
    ShortTermAnticipationPredictor,
)
from local_extraction.inference.predictions import STAPrediction
from local_extraction.inference.tokenizer import ExtractedTokens


class FakeDetector:
    def __init__(self, candidates: tuple[DetectionCandidate, ...]) -> None:
        self.candidates = candidates
        self.calls = []

    def detect(self, target_frame):
        self.calls.append(target_frame)
        return self.candidates


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []
        self.output = ExtractedTokens(
            image_tokens=torch.ones(49, 512),
            video_tokens=torch.ones(2, 49, 512),
            grid_hw=(7, 7),
        )

    def extract(self, resolved_input):
        self.calls.append(resolved_input)
        return self.output


class FakeFusion:
    def __init__(self) -> None:
        self.calls = []
        self.output = FusedTokens(
            fused_image_tokens=torch.ones(49, 256),
            fused_video_tokens=torch.ones(49, 256) * 2,
        )

    def fuse(self, extracted):
        self.calls.append(extracted)
        return self.output


class FakeHead:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, candidates, fused_tokens, image_size):
        self.calls.append((candidates, fused_tokens, image_size))
        return tuple(
            RawCandidateOutput(
                candidate=candidate,
                cls_logits=torch.tensor([0.0, float(index + 1)], dtype=torch.float32),
                noun_logits=torch.tensor([1.0], dtype=torch.float32),
                verb_logits=torch.tensor([1.0], dtype=torch.float32),
                ttc_raw=torch.tensor([0.0], dtype=torch.float32),
                ttc_bin_logits=None,
            )
            for index, candidate in enumerate(candidates)
        )


class FakeAssembler:
    def __init__(self) -> None:
        self.calls = []

    def assemble(self, raw_outputs):
        self.calls.append(raw_outputs)
        return tuple(
            STAPrediction(
                candidate=raw.candidate,
                candidate_index=index,
                score=float(index),
                detector_confidence=float(raw.candidate.confidence),
                next_active_probability=float(index),
                predicted_class=1,
                noun_id=10,
                noun_local_index=0,
                noun_probability=1.0,
                verb_id=20,
                verb_local_index=0,
                verb_probability=1.0,
                ttc_seconds=0.3,
                ttc_raw=0.0,
                ttc_bin=0,
            )
            for index, raw in enumerate(raw_outputs)
        )


def _frame(tmp_path: Path) -> Path:
    path = tmp_path / "uid" / "0000001.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (20, 30, 40)).save(path)
    return path


def _resolved(path: Path) -> ResolvedFrameInput:
    return ResolvedFrameInput(
        target_frame=path,
        uid=path.parent.name,
        frame_directory=path.parent,
        context_frames=(path,),
    )


def _predictor(
    tmp_path: Path,
    *,
    candidates: tuple[DetectionCandidate, ...],
    resolver=None,
    image_size_reader=None,
):
    detector = FakeDetector(candidates)
    tokenizer = FakeTokenizer()
    fusion = FakeFusion()
    head = FakeHead()
    assembler = FakeAssembler()
    predictor = ShortTermAnticipationPredictor(
        InferenceConfig(repo_root=tmp_path),
        detector=detector,
        tokenizer=tokenizer,
        fusion=fusion,
        head=head,
        assembler=assembler,
        frame_resolver=resolver or (lambda target, config: _resolved(Path(target))),
        image_size_reader=image_size_reader or (lambda path: (480, 640)),
    )
    return predictor, detector, tokenizer, fusion, head, assembler


def test_pipeline_wires_components_in_order_and_returns_ranked_predictions(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    candidates = (
        DetectionCandidate((0.0, 0.0, 10.0, 10.0), 0.5, 1),
        DetectionCandidate((1.0, 1.0, 20.0, 20.0), 0.9, 2),
    )
    predictor, detector, tokenizer, fusion, head, assembler = _predictor(tmp_path, candidates=candidates)

    result = predictor.predict(frame)

    assert result.resolved_input == _resolved(frame)
    assert result.image_size == (480, 640)
    assert result.candidates == candidates
    assert tuple(prediction.candidate for prediction in result.predictions) == candidates
    assert detector.calls == [frame]
    assert tokenizer.calls == [result.resolved_input]
    assert fusion.calls == [tokenizer.output]
    assert head.calls == [(candidates, fusion.output, (480, 640))]
    assert len(assembler.calls) == 1
    assert tuple(raw.candidate for raw in assembler.calls[0]) == candidates


def test_pipeline_uses_default_image_size_reader(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    candidates = (DetectionCandidate((0.0, 0.0, 10.0, 10.0), 0.5, 1),)
    head = FakeHead()

    predictor = ShortTermAnticipationPredictor(
        InferenceConfig(repo_root=tmp_path),
        detector=FakeDetector(candidates),
        tokenizer=FakeTokenizer(),
        fusion=FakeFusion(),
        head=head,
        assembler=FakeAssembler(),
        frame_resolver=lambda target, config: _resolved(Path(target)),
    )

    result = predictor.predict(frame)

    assert result.image_size == (480, 640)
    assert head.calls[-1][2] == (480, 640)


def test_pipeline_short_circuits_when_detector_returns_no_candidates(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    predictor, detector, tokenizer, fusion, head, assembler = _predictor(tmp_path, candidates=())

    result = predictor.predict(frame)

    assert result.candidates == ()
    assert result.predictions == ()
    assert detector.calls == [frame]
    assert tokenizer.calls == []
    assert fusion.calls == []
    assert head.calls == []
    assert assembler.calls == []


def test_pipeline_wraps_component_failure_with_chained_error(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    original = RuntimeError("detector failed")

    class BrokenDetector:
        def detect(self, target_frame):
            raise original

    predictor = ShortTermAnticipationPredictor(
        InferenceConfig(repo_root=tmp_path),
        detector=BrokenDetector(),
        tokenizer=FakeTokenizer(),
        fusion=FakeFusion(),
        head=FakeHead(),
        assembler=FakeAssembler(),
        frame_resolver=lambda target, config: _resolved(Path(target)),
    )

    with pytest.raises(PipelineError, match="prediction failed") as error:
        predictor.predict(frame)

    assert error.value.__cause__ is original


def test_pipeline_package_root_import_stays_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
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
