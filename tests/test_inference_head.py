import importlib
import sys
from pathlib import Path

import pytest
import torch

from local_extraction.inference import InferenceConfig
from local_extraction.inference.detector import DetectionCandidate, DetectorError
from local_extraction.inference.fusion import FusedTokens
from local_extraction.inference.head import HeadInferenceError, TrackBInferenceHead


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "local_extraction/runs/Track_B/checkpoints/trackB_best_mAP_0.3708_20251225_224220.pt"
CONTRACT = REPO_ROOT / "local_extraction/inference/resources/trackB_feature_contract.json"
IMAGE_SIZE = (480, 640)
IMAGE_WH = (640, 480)


@pytest.fixture(scope="module")
def checkpoint_payload() -> dict:
    return torch.load(CHECKPOINT, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def head() -> TrackBInferenceHead:
    return TrackBInferenceHead(InferenceConfig(device="cpu"))


def _config_with_checkpoint(path: Path) -> InferenceConfig:
    return InferenceConfig(track_b_checkpoint=path, track_b_feature_contract_path=CONTRACT, device="cpu")


def _fused(seed: int = 77) -> FusedTokens:
    generator = torch.Generator().manual_seed(seed)
    return FusedTokens(torch.randn(49, 256, generator=generator), torch.randn(49, 256, generator=generator))


def _cell_tokens() -> torch.Tensor:
    return torch.arange(49, dtype=torch.float32).unsqueeze(1).repeat(1, 256)


def _candidates() -> tuple[DetectionCandidate, ...]:
    return (
        DetectionCandidate((0.0, 0.0, 320.0, 240.0), 0.9, 10),
        DetectionCandidate((320.0, 240.0, 640.0, 480.0), 0.8, 11),
        DetectionCandidate((-20.0, -10.0, 100.0, 90.0), 0.7, 12),
    )


def _thesis_pool(tokens: torch.Tensor, box: tuple[float, float, float, float]) -> torch.Tensor:
    from local_extraction.trackB.trackB_tokenizer import roi_pool_tokens_mean

    return roi_pool_tokens_mean((7, 7), tokens, box, IMAGE_WH)


def _copy_checkpoint(tmp_path: Path, payload: dict, name: str = CHECKPOINT.name) -> Path:
    checkpoint = tmp_path / name
    torch.save(payload, checkpoint)
    return checkpoint


def _direct_thesis_output(payload: dict, fused: FusedTokens, candidates: tuple[DetectionCandidate, ...]) -> dict:
    from local_extraction.trackB.trackB_head import HeadConfig, TrackBHead
    from local_extraction.trackB.trackB_tokenizer import roi_pool_tokens_mean

    state = payload["head"]
    hidden = int(state["backbone.1.weight"].shape[0])
    fused_dim = int(state["backbone.1.weight"].shape[1]) // 2
    num_ttc_bins = int(state["ttc_bin_head.weight"].shape[0]) if "ttc_bin_head.weight" in state else 0
    thesis = TrackBHead(HeadConfig(
        dim=fused_dim,
        num_classes=int(state["cls_head.weight"].shape[0]),
        hidden=hidden,
        dropout=float(payload["train_config"]["head_dropout"]),
        num_noun_classes=int(state["noun_head.weight"].shape[0]),
        num_verb_classes=int(state["verb_head.weight"].shape[0]),
        num_ttc_bins=num_ttc_bins,
        ttc_mode="bin" if num_ttc_bins else "reg",
    ))
    thesis.load_state_dict(state, strict=True)
    thesis.eval()
    image_pooled = torch.stack([
        roi_pool_tokens_mean((7, 7), fused.fused_image_tokens, candidate.box_xyxy, IMAGE_WH)
        for candidate in candidates
    ]).unsqueeze(0)
    video_pooled = torch.stack([
        roi_pool_tokens_mean((7, 7), fused.fused_video_tokens, candidate.box_xyxy, IMAGE_WH)
        for candidate in candidates
    ]).unsqueeze(0)
    with torch.inference_mode():
        return thesis(image_pooled, video_pooled)


def test_real_head_architecture_and_mode(head: TrackBInferenceHead, checkpoint_payload: dict) -> None:
    state = checkpoint_payload["head"]
    assert head.head.training is False
    assert next(head.head.parameters()).device.type == "cpu"
    assert head.head_input_dim == int(state["backbone.1.weight"].shape[1])
    assert head.fused_dim == head.head_input_dim // 2 == 256
    assert head.num_classes == int(state["cls_head.weight"].shape[0]) == 2
    assert head.num_noun_classes == len(checkpoint_payload["noun_id_list"]) == 118
    assert head.num_verb_classes == len(checkpoint_payload["verb_id_list"]) == 64
    assert head.num_ttc_bins == 0


def test_zero_candidates_returns_empty(head: TrackBInferenceHead) -> None:
    assert head.predict((), _fused(), IMAGE_SIZE) == ()


def test_raw_outputs_preserve_candidate_order_and_shapes(head: TrackBInferenceHead) -> None:
    candidates = _candidates()
    outputs = head.predict(candidates, _fused(), IMAGE_SIZE)

    assert tuple(output.candidate for output in outputs) == candidates
    assert all(output.cls_logits.shape == (head.num_classes,) for output in outputs)
    assert all(output.noun_logits.shape == (head.num_noun_classes,) for output in outputs)
    assert all(output.verb_logits.shape == (head.num_verb_classes,) for output in outputs)
    assert all(output.ttc_raw.shape == (1,) for output in outputs)
    assert all(output.ttc_bin_logits is None for output in outputs)
    assert not hasattr(outputs[0], "pooled_image")
    assert not hasattr(outputs[0], "pooled_video")


@pytest.mark.parametrize(
    ("box", "cells"),
    [
        ((0.0, 0.0, 640.0, 480.0), list(range(49))),
        ((0.0, 0.0, 640.0 / 7, 480.0 / 7), [0]),
        ((640.0 * 6 / 7, 480.0 * 6 / 7, 640.0, 480.0), [48]),
        ((0.0, 0.0, 640.0 * 2 / 7, 480.0 * 2 / 7), [0, 1, 7, 8]),
        ((-10.0, -10.0, 640.0 / 7, 480.0 / 7), [0]),
        ((700.0, 520.0, 800.0, 600.0), [48]),
        ((10.0, 20.0, 10.0, 70.0), [0, 7]),
        ((20.0, 20.0, 80.0, 20.0), [0]),
    ],
)
def test_roi_edge_boxes_match_analytical_thesis_mean(box, cells) -> None:
    expected = torch.full((256,), float(torch.tensor(cells, dtype=torch.float32).mean()))

    assert torch.equal(_thesis_pool(_cell_tokens(), box), expected)


def test_reversed_box_uses_thesis_empty_roi_mean() -> None:
    tokens = _cell_tokens()

    assert torch.equal(_thesis_pool(tokens, (400.0, 300.0, 100.0, 100.0)), tokens.mean(dim=0))


def test_production_outputs_match_thesis_roi_and_head_for_all_candidates(
    head: TrackBInferenceHead,
    checkpoint_payload: dict,
) -> None:
    fused = _fused()
    candidates = _candidates()
    expected = _direct_thesis_output(checkpoint_payload, fused, candidates)
    actual = head.predict(candidates, fused, IMAGE_SIZE)

    for index, output in enumerate(actual):
        assert torch.equal(output.cls_logits, expected["cls_logits"][0, index])
        assert torch.equal(output.noun_logits, expected["noun_logits"][0, index])
        assert torch.equal(output.verb_logits, expected["verb_logits"][0, index])
        assert torch.equal(output.ttc_raw, expected["ttc"][0, index])
        assert output.ttc_bin_logits is None


def test_synthetic_milestone6_path_uses_real_head_without_decoding_or_ranking(head: TrackBInferenceHead) -> None:
    candidates = _candidates()
    outputs = head.predict(candidates, FusedTokens(_cell_tokens(), _cell_tokens() + 1000.0), IMAGE_SIZE)

    assert len(outputs) == len(candidates)
    assert tuple(output.candidate for output in outputs) == candidates
    assert all(output.cls_logits.dtype == torch.float32 and output.cls_logits.device.type == "cpu" for output in outputs)
    assert all(not hasattr(output, "noun") and not hasattr(output, "verb") for output in outputs)
    assert all(not hasattr(output, "score") and not hasattr(output, "ttc_seconds") for output in outputs)


@pytest.mark.parametrize("image_size", [(0, 640), (480, 0), (-1, 640)])
def test_invalid_image_size_is_rejected(head: TrackBInferenceHead, image_size) -> None:
    with pytest.raises(HeadInferenceError, match="must be positive"):
        head.predict(_candidates(), _fused(), image_size)


def test_wrong_fused_shape_is_rejected(head: TrackBInferenceHead) -> None:
    bad = FusedTokens(torch.randn(64, 256), torch.randn(64, 256))

    with pytest.raises(HeadInferenceError, match=r"must be \(49, 256\)"):
        head.predict(_candidates(), bad, IMAGE_SIZE)


def test_wrong_fused_dtype_is_rejected(head: TrackBInferenceHead) -> None:
    bad = FusedTokens(torch.randn(49, 256).double(), torch.randn(49, 256))

    with pytest.raises(HeadInferenceError, match="dtype torch.float32"):
        head.predict(_candidates(), bad, IMAGE_SIZE)


def test_malformed_candidate_is_rejected(head: TrackBInferenceHead) -> None:
    with pytest.raises(HeadInferenceError, match="DetectionCandidate"):
        head.predict(("bad",), _fused(), IMAGE_SIZE)  # type: ignore[arg-type]


def test_detection_candidate_rejects_nonfinite_box_before_head() -> None:
    with pytest.raises(DetectorError, match="finite"):
        DetectionCandidate((0.0, 0.0, float("nan"), 1.0), 0.1)


def test_head_forward_is_deterministic_and_without_gradients(head: TrackBInferenceHead) -> None:
    fused = _fused()
    first = head.predict(_candidates(), fused, IMAGE_SIZE)
    second = head.predict(_candidates(), fused, IMAGE_SIZE)

    for left, right in zip(first, second):
        assert torch.equal(left.cls_logits, right.cls_logits)
        assert torch.equal(left.noun_logits, right.noun_logits)
        assert torch.equal(left.verb_logits, right.verb_logits)
        assert torch.equal(left.ttc_raw, right.ttc_raw)
        assert left.cls_logits.requires_grad is False


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(HeadInferenceError, match="checkpoint does not exist"):
        TrackBInferenceHead(_config_with_checkpoint(tmp_path / CHECKPOINT.name))


def test_checkpoint_directory_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / CHECKPOINT.name
    checkpoint.mkdir()

    with pytest.raises(HeadInferenceError, match="checkpoint is not a file"):
        TrackBInferenceHead(_config_with_checkpoint(checkpoint))


def test_missing_head_state_is_rejected(tmp_path: Path, checkpoint_payload: dict) -> None:
    payload = dict(checkpoint_payload)
    payload.pop("head")

    with pytest.raises(HeadInferenceError, match="missing head state"):
        TrackBInferenceHead(_config_with_checkpoint(_copy_checkpoint(tmp_path, payload)))


def test_missing_train_metadata_is_rejected(tmp_path: Path, checkpoint_payload: dict) -> None:
    payload = dict(checkpoint_payload)
    payload["train_config"] = {}

    with pytest.raises(HeadInferenceError, match="missing head metadata"):
        TrackBInferenceHead(_config_with_checkpoint(_copy_checkpoint(tmp_path, payload)))


def test_malformed_head_tensor_dimensions_are_rejected(tmp_path: Path, checkpoint_payload: dict) -> None:
    payload = dict(checkpoint_payload)
    payload["head"] = dict(checkpoint_payload["head"])
    payload["head"]["backbone.1.weight"] = torch.randn(256)

    with pytest.raises(HeadInferenceError, match="malformed weight dimensions"):
        TrackBInferenceHead(_config_with_checkpoint(_copy_checkpoint(tmp_path, payload)))


def test_noun_id_output_mismatch_is_rejected(tmp_path: Path, checkpoint_payload: dict) -> None:
    payload = dict(checkpoint_payload)
    payload["noun_id_list"] = list(checkpoint_payload["noun_id_list"])[:-1]

    with pytest.raises(HeadInferenceError, match="Noun mapping length"):
        TrackBInferenceHead(_config_with_checkpoint(_copy_checkpoint(tmp_path, payload)))


def test_verb_id_output_mismatch_is_rejected(tmp_path: Path, checkpoint_payload: dict) -> None:
    payload = dict(checkpoint_payload)
    payload["verb_id_list"] = list(checkpoint_payload["verb_id_list"])[:-1]

    with pytest.raises(HeadInferenceError, match="Verb mapping length"):
        TrackBInferenceHead(_config_with_checkpoint(_copy_checkpoint(tmp_path, payload)))


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
