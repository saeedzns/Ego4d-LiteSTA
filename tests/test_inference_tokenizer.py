from pathlib import Path

import pytest
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

from local_extraction.inference import InferenceConfig, ResolvedFrameInput
from local_extraction.inference.tokenizer import (
    TokenizerError,
    TrackBTokenizer,
)


class FakeBackbone(torch.nn.Module):
    def __init__(self, channels: int = 512, grid: tuple[int, int] = (7, 7)):
        super().__init__()
        self.channels = channels
        self.grid = grid
        self.calls = []
        self.devices = []

    def to(self, device):
        self.devices.append(str(device))
        return super().to(device)

    def forward(self, inputs):
        self.calls.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "training": self.training,
                "value": float(inputs.mean().item()),
            }
        )
        value = inputs.mean().reshape(1, 1, 1, 1)
        return value.expand(inputs.shape[0], self.channels, *self.grid)


class BadBackbone(torch.nn.Module):
    def forward(self, inputs):
        return torch.zeros((1, 512, 2))


def _frame(directory: Path, name: str, value: int = 128) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new("RGB", (4, 4), (value, value, value)).save(path)
    return path


def _resolved(frames: tuple[Path, ...]) -> ResolvedFrameInput:
    return ResolvedFrameInput(
        target_frame=frames[-1],
        uid=frames[-1].parent.name,
        frame_directory=frames[-1].parent,
        context_frames=frames,
    )


def _tokenizer(backbone=None, transform=None, image_loader=None, **kwargs):
    config = InferenceConfig(device="cpu", **kwargs)
    return TrackBTokenizer(
        config,
        backbone=backbone or FakeBackbone(),
        transform=transform,
        image_loader=image_loader,
    )


def test_preprocessing_matches_resize_tensor_and_imagenet_normalization(tmp_path: Path) -> None:
    path = tmp_path / "red.jpg"
    Image.new("RGB", (2, 3), (255, 0, 0)).save(path, format="PNG")
    backbone = FakeBackbone()
    tokenizer = _tokenizer(backbone=backbone)

    tokenizer.extract(_resolved((path,)))

    first_value = backbone.calls[0]["value"]
    expected = ((1.0 - 0.485) / 0.229 - 0.456 / 0.224 - 0.406 / 0.225) / 3.0
    assert first_value == pytest.approx(expected, abs=1e-3)


def test_production_transform_matches_thesis_transform_independently() -> None:
    from local_extraction.trackB.trackB_tokenizer import TokenizerConfig, build_transform

    image = Image.new("RGB", (13, 7), (37, 101, 203))
    config = InferenceConfig(device="cpu")
    production = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    thesis = build_transform(TokenizerConfig(
        device="cpu", img_size=224,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ))

    production_tensor = production(image)
    thesis_tensor = thesis(image)

    assert production_tensor.shape == thesis_tensor.shape == (3, 224, 224)
    assert production_tensor.dtype == thesis_tensor.dtype == torch.float32
    assert torch.equal(production_tensor, thesis_tensor)


def test_real_resnet18_weights_none_shape_matches_feature_contract() -> None:
    model = models.resnet18(weights=None)
    inputs = torch.zeros((1, 3, 224, 224))
    with torch.inference_mode():
        outputs = model.maxpool(model.relu(model.bn1(model.conv1(inputs))))
        outputs = model.layer1(outputs)
        outputs = model.layer2(outputs)
        outputs = model.layer3(outputs)
        outputs = model.layer4(outputs)

    assert outputs.shape == (1, 512, 7, 7)
    tokens = outputs.permute(0, 2, 3, 1).reshape(-1, 512)
    assert tokens.shape == (49, 512)


def test_feature_flatten_order_matches_thesis() -> None:
    features = torch.tensor([
        [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
         [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]]
    ])

    tokens = features.permute(0, 2, 3, 1).reshape(-1, 2)

    assert tokens.tolist() == [
        [0.0, 10.0], [1.0, 11.0], [2.0, 12.0],
        [3.0, 13.0], [4.0, 14.0], [5.0, 15.0],
    ]


def test_default_image_loader_converts_rgba_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "rgba.jpg"
    Image.new("RGBA", (4, 4), (255, 0, 0, 100)).save(path, format="PNG")
    loaded_modes = []

    def loader(value: Path):
        with Image.open(value) as image:
            converted = image.convert("RGB")
            loaded_modes.append(converted.mode)
            return converted

    tokenizer = _tokenizer(image_loader=loader)
    tokenizer.extract(_resolved((path,)))

    assert loaded_modes == ["RGB", "RGB"]


def test_backbone_is_eval_device_bound_and_reused(tmp_path: Path) -> None:
    directory = tmp_path / "uid"
    frames = tuple(_frame(directory, f"{index:07d}.jpg", 40 + index) for index in range(3))
    backbone = FakeBackbone()
    tokenizer = _tokenizer(backbone=backbone)

    first = tokenizer.extract(_resolved(frames))
    second = tokenizer.extract(_resolved(frames))

    assert tokenizer.backbone is backbone
    assert backbone.devices == ["cpu"]
    assert backbone.training is False
    assert len(backbone.calls) == 8
    assert torch.equal(first.video_tokens, second.video_tokens)


def test_tokens_have_expected_shapes_and_temporal_order(tmp_path: Path) -> None:
    directory = tmp_path / "uid"
    frames = tuple(_frame(directory, f"{index:07d}.jpg", 20 + index * 50) for index in range(3))
    backbone = FakeBackbone()
    tokenizer = _tokenizer(backbone=backbone)

    output = tokenizer.extract(_resolved(frames))

    assert output.image_tokens.shape == (49, 512)
    assert output.video_tokens.shape == (3, 49, 512)
    assert output.grid_hw == (7, 7)
    assert output.image_tokens.dtype == torch.float32
    assert output.video_tokens.dtype == torch.float32
    assert output.image_tokens.equal(output.video_tokens[-1])
    assert output.video_tokens[0, 0, 0] < output.video_tokens[1, 0, 0] < output.video_tokens[2, 0, 0]


def test_short_context_is_accepted_without_padding(tmp_path: Path) -> None:
    directory = tmp_path / "uid"
    frames = (_frame(directory, "0000000.jpg", 10), _frame(directory, "0000002.jpg", 20))
    output = _tokenizer(backbone=FakeBackbone()).extract(_resolved(frames))

    assert output.video_tokens.shape[0] == 2


def test_unreadable_image_raises_tokenizer_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not an image")
    tokenizer = _tokenizer()

    with pytest.raises(TokenizerError, match="Could not load image"):
        tokenizer.extract(_resolved((path,)))


def test_malformed_backbone_output_raises_tokenizer_error(tmp_path: Path) -> None:
    path = _frame(tmp_path, "0000000.jpg")
    tokenizer = _tokenizer(backbone=BadBackbone())

    with pytest.raises(TokenizerError, match="must be a 4D tensor"):
        tokenizer.extract(_resolved((path,)))


def test_unexpected_channel_count_raises_tokenizer_error(tmp_path: Path) -> None:
    path = _frame(tmp_path, "0000000.jpg")
    tokenizer = _tokenizer(backbone=FakeBackbone(channels=256))

    with pytest.raises(TokenizerError, match=r"expected \(1, 512, H, W\)"):
        tokenizer.extract(_resolved((path,)))


def test_context_must_end_with_target(tmp_path: Path) -> None:
    directory = tmp_path / "uid"
    first = _frame(directory, "0000000.jpg")
    target = _frame(directory, "0000001.jpg")
    invalid = ResolvedFrameInput(target, "uid", directory, (target, first))

    with pytest.raises(TokenizerError, match="must end with the target"):
        _tokenizer().extract(invalid)


def test_real_tokenizer_requires_explicit_weight_path() -> None:
    config = InferenceConfig(tokenizer_weights_path=None)

    with pytest.raises(TokenizerError, match="explicit ResNet18 weight file is required"):
        TrackBTokenizer(config)


def test_injected_backbone_rejects_contract_mismatch(tmp_path: Path) -> None:
    path = _frame(tmp_path, "0000000.jpg")
    tokenizer = _tokenizer(backbone=FakeBackbone(grid=(8, 8)))

    with pytest.raises(TokenizerError, match="Feature-contract mismatch"):
        tokenizer.extract(_resolved((path,)))


def test_checkpoint_contract_identity_is_required(tmp_path: Path) -> None:
    config = InferenceConfig(
        repo_root=tmp_path,
        track_b_checkpoint=tmp_path / "different.pt",
        track_b_feature_contract_path=Path(__file__).resolve().parents[1]
        / "local_extraction/inference/resources/trackB_feature_contract.json",
    )

    with pytest.raises(TokenizerError, match="does not match feature contract"):
        TrackBTokenizer(config, backbone=FakeBackbone())


def test_temporal_window_must_match_contract() -> None:
    with pytest.raises(TokenizerError, match="Temporal window does not match"):
        TrackBTokenizer(InferenceConfig(temporal_window_length=8), backbone=FakeBackbone())


def test_temporal_stride_must_match_contract() -> None:
    with pytest.raises(TokenizerError, match="Temporal stride does not match"):
        TrackBTokenizer(InferenceConfig(temporal_stride=1), backbone=FakeBackbone())


def test_weight_directory_is_rejected(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pth"
    weights.mkdir()
    with pytest.raises(TokenizerError, match="is not a file"):
        TrackBTokenizer(InferenceConfig(tokenizer_weights_path=weights))


def test_real_tokenizer_rejects_weight_hash_mismatch(tmp_path: Path) -> None:
    weights = tmp_path / "wrong.pth"
    weights.write_bytes(b"not the ImageNet weights")
    config = InferenceConfig(tokenizer_weights_path=weights)

    with pytest.raises(TokenizerError, match="SHA-256 mismatch"):
        TrackBTokenizer(config)


def test_token_values_match_thesis_tokenizer_helpers(tmp_path: Path) -> None:
    from local_extraction.trackB.trackB_tokenizer import (
        TokenizerConfig,
        image_grid_tokens,
        video_grid_tokens,
    )

    directory = tmp_path / "uid"
    frames = tuple(_frame(directory, f"{index:07d}.jpg", 30 + index * 20) for index in range(2))
    backbone = FakeBackbone()
    tokenizer = _tokenizer(backbone=backbone)
    resolved = _resolved(frames)
    thesis_config = TokenizerConfig(device="cpu", img_size=224)

    thesis_image, thesis_hw = image_grid_tokens(
        resolved.target_frame, backbone, tokenizer.transform, thesis_config
    )
    thesis_video, _ = video_grid_tokens(
        list(resolved.context_frames), backbone, tokenizer.transform, thesis_config
    )
    production = tokenizer.extract(resolved)

    assert production.grid_hw == thesis_hw
    assert torch.equal(production.image_tokens, thesis_image)
    assert torch.equal(production.video_tokens, thesis_video)


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
