from dataclasses import replace
import importlib
import math
import sys
from pathlib import Path

import pytest
import torch

from local_extraction.inference import InferenceConfig
from local_extraction.inference.artifacts import TTCStats
from local_extraction.inference.detector import DetectionCandidate
from local_extraction.inference.head import RawCandidateOutput
from local_extraction.inference.predictions import (
    PredictionAssembler,
    PredictionAssemblyError,
)


def _stats() -> TTCStats:
    return TTCStats(
        count=10,
        mean=0.3,
        std=0.4,
        minimum=0.0,
        maximum=5.0,
        source_manifest="train.jsonl",
        normalization="(ttc - mean) / std",
    )


def _assembler(**kwargs) -> PredictionAssembler:
    return PredictionAssembler(
        InferenceConfig(),
        noun_id_list=(10, 20, 30),
        verb_id_list=(4, 5),
        ttc_stats=_stats(),
        **kwargs,
    )


def _raw(
    candidate: DetectionCandidate,
    *,
    cls_logits: tuple[float, ...] = (0.0, 2.0),
    noun_logits: tuple[float, ...] = (0.0, 5.0, 1.0),
    verb_logits: tuple[float, ...] = (3.0, 0.0),
    ttc_raw: float = 2.0,
    ttc_bin_logits: tuple[float, ...] | None = None,
) -> RawCandidateOutput:
    return RawCandidateOutput(
        candidate=candidate,
        cls_logits=torch.tensor(cls_logits, dtype=torch.float32),
        noun_logits=torch.tensor(noun_logits, dtype=torch.float32),
        verb_logits=torch.tensor(verb_logits, dtype=torch.float32),
        ttc_raw=torch.tensor([ttc_raw], dtype=torch.float32),
        ttc_bin_logits=torch.tensor(ttc_bin_logits, dtype=torch.float32) if ttc_bin_logits is not None else None,
    )


def test_assemble_decodes_ids_probabilities_and_ttc_seconds() -> None:
    candidate = DetectionCandidate((1.0, 2.0, 3.0, 4.0), 0.75, 8)
    prediction = _assembler().assemble((_raw(candidate),))[0]

    assert prediction.candidate is candidate
    assert prediction.candidate_index == 0
    assert prediction.predicted_class == 1
    assert prediction.next_active_probability == pytest.approx(float(torch.softmax(torch.tensor([0.0, 2.0]), 0)[1]))
    assert prediction.score == prediction.next_active_probability
    assert prediction.detector_confidence == 0.75
    assert prediction.noun_local_index == 1
    assert prediction.noun_id == 20
    assert prediction.verb_local_index == 0
    assert prediction.verb_id == 4
    assert prediction.ttc_raw == 2.0
    assert prediction.ttc_seconds == pytest.approx(1.1)
    assert prediction.ttc_bin == 2


def test_assemble_preserves_candidate_index_and_sorts_by_next_active_probability() -> None:
    low = _raw(DetectionCandidate((0.0, 0.0, 1.0, 1.0), 0.1), cls_logits=(2.0, 0.0))
    high = _raw(DetectionCandidate((1.0, 1.0, 2.0, 2.0), 0.2), cls_logits=(0.0, 4.0))
    middle = _raw(DetectionCandidate((2.0, 2.0, 3.0, 3.0), 0.3), cls_logits=(0.0, 1.0))

    predictions = _assembler().assemble((low, high, middle))

    assert [prediction.candidate for prediction in predictions] == [high.candidate, middle.candidate, low.candidate]
    assert [prediction.candidate_index for prediction in predictions] == [1, 2, 0]


def test_assemble_uses_single_class_probability_when_only_one_cls_logit_exists() -> None:
    prediction = _assembler().assemble((_raw(DetectionCandidate((0.0, 0.0, 1.0, 1.0), 0.5), cls_logits=(3.0,)),))[0]

    assert prediction.predicted_class == 0
    assert prediction.next_active_probability == 1.0


def test_ttc_bin_logits_override_threshold_bin() -> None:
    prediction = _assembler().assemble((
        _raw(DetectionCandidate((0.0, 0.0, 1.0, 1.0), 0.5), ttc_raw=0.0, ttc_bin_logits=(0.0, 1.0, 5.0, 2.0)),
    ))[0]

    assert prediction.ttc_seconds == pytest.approx(0.3)
    assert prediction.ttc_bin == 2


def test_zero_raw_outputs_returns_empty() -> None:
    assert _assembler().assemble(()) == ()


def test_malformed_raw_output_is_rejected() -> None:
    with pytest.raises(PredictionAssemblyError, match="RawCandidateOutput"):
        _assembler().assemble(("bad",))  # type: ignore[arg-type]


def test_mismatched_noun_width_is_rejected() -> None:
    raw = _raw(DetectionCandidate((0.0, 0.0, 1.0, 1.0), 0.5), noun_logits=(1.0, 2.0))

    with pytest.raises(PredictionAssemblyError, match="noun_logits width"):
        _assembler().assemble((raw,))


def test_nonfinite_logits_are_rejected() -> None:
    raw = _raw(DetectionCandidate((0.0, 0.0, 1.0, 1.0), 0.5))
    raw = replace(raw, cls_logits=torch.tensor([0.0, float("nan")], dtype=torch.float32))

    with pytest.raises(PredictionAssemblyError, match="finite"):
        _assembler().assemble((raw,))


@pytest.mark.requires_model_artifacts
def test_real_checkpoint_id_maps_and_ttc_stats_load_when_available() -> None:
    config = InferenceConfig()
    assembler = PredictionAssembler(config)

    assert len(assembler.noun_id_list) == 118
    assert len(assembler.verb_id_list) == 64
    assert math.isclose(assembler.ttc_stats.mean, 0.308969769291965)


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    config = InferenceConfig(track_b_checkpoint=tmp_path / "missing.pt")

    with pytest.raises(PredictionAssemblyError, match="checkpoint does not exist"):
        PredictionAssembler(config, ttc_stats=_stats())


def test_package_root_import_stays_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
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
