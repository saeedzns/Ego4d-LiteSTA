from pathlib import Path

import pytest
import torch

from local_extraction.inference import InferenceConfig
from local_extraction.inference.fusion import FusionError, TrackBFeatureFusion
from local_extraction.inference.tokenizer import ExtractedTokens


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "local_extraction/runs/Track_B/checkpoints/trackB_best_mAP_0.3708_20251225_224220.pt"
CONTRACT = REPO_ROOT / "local_extraction/inference/resources/trackB_feature_contract.json"


@pytest.fixture(scope="module")
def fusion() -> TrackBFeatureFusion:
    return TrackBFeatureFusion(InferenceConfig(device="cpu"))


def _tokens(time_length: int = 16, feature_dim: int = 512) -> ExtractedTokens:
    torch.manual_seed(1234 + time_length)
    image = torch.randn(49, feature_dim)
    video = torch.randn(time_length, 49, feature_dim)
    return ExtractedTokens(image_tokens=image, video_tokens=video, grid_hw=(7, 7))


def _config_with_paths(checkpoint: Path) -> InferenceConfig:
    return InferenceConfig(
        track_b_checkpoint=checkpoint,
        track_b_feature_contract_path=CONTRACT,
        device="cpu",
    )


def _write_modified_checkpoint(tmp_path: Path, **train_changes) -> Path:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    payload["train_config"] = dict(payload["train_config"])
    payload["train_config"].update(train_changes)
    checkpoint = tmp_path / CHECKPOINT.name
    torch.save(payload, checkpoint)
    return checkpoint


@pytest.mark.requires_model_artifacts
def test_real_checkpoint_reconstructs_expected_projector_and_fusion(fusion: TrackBFeatureFusion) -> None:
    assert tuple(fusion.projector.weight.shape) == (256, 512)
    assert fusion.fusion.cfg.dim == 256
    assert fusion.fusion.cfg.heads == 8
    assert fusion.fusion.cfg.layers == 4
    assert fusion.fusion.cfg.dropout == pytest.approx(0.2)
    assert fusion.fusion.cfg.ff_mult == 4
    assert fusion.fusion.cfg.fgtp_stride_t == 2


@pytest.mark.requires_model_artifacts
def test_models_are_loaded_once_on_configured_device_and_eval(fusion: TrackBFeatureFusion) -> None:
    assert fusion.projector.training is False
    assert fusion.fusion.training is False
    assert next(fusion.projector.parameters()).device.type == "cpu"
    assert next(fusion.fusion.parameters()).device.type == "cpu"
    assert fusion.projector is fusion.projector
    assert fusion.fusion is fusion.fusion


@pytest.mark.requires_model_artifacts
def test_valid_contract_tokens_return_expected_shapes(fusion: TrackBFeatureFusion) -> None:
    result = fusion.fuse(_tokens())

    assert result.fused_image_tokens.shape == (49, 256)
    assert result.fused_video_tokens.shape == (49, 256)
    assert result.fused_image_tokens.dtype == torch.float32
    assert result.fused_video_tokens.dtype == torch.float32
    assert result.fused_image_tokens.device.type == "cpu"
    assert result.fused_video_tokens.device.type == "cpu"


@pytest.mark.parametrize("time_length", [1, 2, 5, 16])
@pytest.mark.requires_model_artifacts
def test_short_temporal_context_is_supported(fusion: TrackBFeatureFusion, time_length: int) -> None:
    result = fusion.fuse(_tokens(time_length))

    assert result.fused_image_tokens.shape == (49, 256)
    assert result.fused_video_tokens.shape == (49, 256)


@pytest.mark.requires_model_artifacts
def test_repeated_forward_is_deterministic(fusion: TrackBFeatureFusion) -> None:
    tokens = _tokens()
    first = fusion.fuse(tokens)
    second = fusion.fuse(tokens)

    assert torch.equal(first.fused_image_tokens, second.fused_image_tokens)
    assert torch.equal(first.fused_video_tokens, second.fused_video_tokens)


@pytest.mark.requires_model_artifacts
def test_forward_runs_without_gradients(fusion: TrackBFeatureFusion) -> None:
    result = fusion.fuse(_tokens(1))

    assert result.fused_image_tokens.requires_grad is False
    assert result.fused_video_tokens.requires_grad is False


@pytest.mark.requires_model_artifacts
def test_wrong_feature_dimension_is_rejected(fusion: TrackBFeatureFusion) -> None:
    with pytest.raises(FusionError, match="does not match expected"):
        fusion.fuse(_tokens(feature_dim=256))


@pytest.mark.requires_model_artifacts
def test_wrong_grid_and_token_count_are_rejected(fusion: TrackBFeatureFusion) -> None:
    bad = ExtractedTokens(
        image_tokens=torch.randn(64, 512),
        video_tokens=torch.randn(1, 64, 512),
        grid_hw=(8, 8),
    )

    with pytest.raises(FusionError, match="does not match contract grid"):
        fusion.fuse(bad)


@pytest.mark.requires_model_artifacts
def test_empty_or_malformed_temporal_inputs_are_rejected(fusion: TrackBFeatureFusion) -> None:
    empty = ExtractedTokens(torch.randn(49, 512), torch.empty(0, 49, 512), (7, 7))
    malformed = ExtractedTokens(torch.randn(49, 512), torch.randn(49, 512), (7, 7))

    with pytest.raises(FusionError, match="Video token shape"):
        fusion.fuse(empty)
    with pytest.raises(FusionError, match="Video tokens must have shape"):
        fusion.fuse(malformed)


@pytest.mark.requires_model_artifacts
def test_too_long_temporal_context_is_rejected(fusion: TrackBFeatureFusion) -> None:
    with pytest.raises(FusionError, match="1 <= T <= 16"):
        fusion.fuse(_tokens(17))


@pytest.mark.requires_model_artifacts
def test_non_float32_inputs_are_rejected(fusion: TrackBFeatureFusion) -> None:
    tokens = _tokens()
    tokens = ExtractedTokens(tokens.image_tokens.double(), tokens.video_tokens, tokens.grid_hw)

    with pytest.raises(FusionError, match="torch.float32"):
        fusion.fuse(tokens)


@pytest.mark.requires_model_artifacts
def test_thesis_reconstruction_is_numerically_equivalent(fusion: TrackBFeatureFusion) -> None:
    from local_extraction.trackB.trackB_fusion import FusionConfig, TrackBFusion

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    train = payload["train_config"]
    projector = torch.nn.Linear(512, 256)
    projector.load_state_dict(payload["projector"], strict=True)
    thesis_fusion = TrackBFusion(
        FusionConfig(
            dim=256,
            heads=int(train["fusion_heads"]),
            layers=4,
            dropout=float(train["fusion_dropout"]),
            fgtp_stride_t=int(train["fgtp_stride_t"]),
            ff_mult=int(train["fusion_ff_mult"]),
        )
    )
    thesis_fusion.load_state_dict(payload["fusion"], strict=True)
    projector.eval()
    thesis_fusion.eval()
    tokens = _tokens()
    with torch.inference_mode():
        image_batch = projector(tokens.image_tokens.unsqueeze(0))
        video_batch = projector(tokens.video_tokens.unsqueeze(0))
        expected_image, expected_video = thesis_fusion(image_batch, video_batch)
    actual = fusion.fuse(tokens)

    assert torch.equal(actual.fused_image_tokens, expected_image[0])
    assert torch.equal(actual.fused_video_tokens, expected_video[0])


def test_missing_projector_state_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / CHECKPOINT.name
    torch.save({"fusion": {}}, checkpoint)

    with pytest.raises(FusionError, match="missing projector state"):
        TrackBFeatureFusion(_config_with_paths(checkpoint))


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    config = _config_with_paths(tmp_path / CHECKPOINT.name)

    with pytest.raises(FusionError, match="checkpoint does not exist"):
        TrackBFeatureFusion(config)


def test_checkpoint_directory_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / CHECKPOINT.name
    checkpoint.mkdir()

    with pytest.raises(FusionError, match="checkpoint is not a file"):
        TrackBFeatureFusion(_config_with_paths(checkpoint))


def test_missing_fusion_state_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / CHECKPOINT.name
    torch.save({"projector": {}}, checkpoint)

    with pytest.raises(FusionError, match="missing fusion state"):
        TrackBFeatureFusion(_config_with_paths(checkpoint))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("fusion_heads", None, "missing fusion metadata"),
        ("fusion_dropout", None, "missing fusion metadata"),
        ("fusion_ff_mult", None, "missing fusion metadata"),
        ("fgtp_stride_t", None, "missing fusion metadata"),
        ("fusion_heads", 0, "fusion_heads must be a positive integer"),
        ("fusion_heads", 3, "fusion_heads must divide"),
        ("fusion_dropout", float("nan"), "fusion_dropout must be finite"),
        ("fusion_dropout", 1.0, "fusion_dropout must be finite"),
        ("fusion_ff_mult", 0, "fusion_ff_mult must be a positive integer"),
        ("fgtp_stride_t", 0, "fgtp_stride_t must be a positive integer"),
    ],
)
@pytest.mark.requires_model_artifacts
def test_fusion_metadata_is_required_and_validated(tmp_path: Path, field, value, message) -> None:
    changes = {field: value}
    if value is None:
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        payload["train_config"] = dict(payload["train_config"])
        payload["train_config"].pop(field)
        checkpoint = tmp_path / CHECKPOINT.name
        torch.save(payload, checkpoint)
    else:
        checkpoint = _write_modified_checkpoint(tmp_path, **changes)

    with pytest.raises(FusionError, match=message):
        TrackBFeatureFusion(_config_with_paths(checkpoint))


def test_package_root_remains_lightweight() -> None:
    import subprocess
    import sys

    script = """
import sys
from local_extraction.inference import InferenceConfig, resolve_frame_input
assert InferenceConfig is not None
assert resolve_frame_input is not None
assert 'torch' not in sys.modules
assert 'torchvision' not in sys.modules
assert 'ultralytics' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr or result.stdout
