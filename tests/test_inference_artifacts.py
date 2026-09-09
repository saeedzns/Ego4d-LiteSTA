from pathlib import Path

import pytest

from local_extraction.inference.artifacts import (
    ArtifactError,
    load_track_b_artifacts,
    load_ttc_stats,
)
from local_extraction.inference.config import InferenceConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MEAN = 0.308969769291965
EXPECTED_STD = 0.4092708906167367


def test_default_config_resolves_repository_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = InferenceConfig()

    assert config.repo_root == REPO_ROOT.resolve()
    assert config.track_a_checkpoint == (
        REPO_ROOT
        / "local_extraction/toolkit_yolo/runs/"
        "sta_yolov8s_singlecls_20251113_002330/weights/best.pt"
    ).resolve()
    assert config.track_b_checkpoint == (
        REPO_ROOT
        / "local_extraction/runs/Track_B/checkpoints/"
        "trackB_best_mAP_0.3708_20251225_224220.pt"
    ).resolve()
    assert config.taxonomy_path == (
        REPO_ROOT / "local_extraction/v2/org_annotations/fho_sta_val_height-540.json"
    ).resolve()
    assert config.ttc_stats_path == (
        REPO_ROOT / "local_extraction/inference/resources/trackB_ttc_stats.json"
    ).resolve()
    assert config.temporal_window_length == 16
    assert config.temporal_stride == 2
    assert config.candidate_limit == 16


def test_training_ttc_stats_load() -> None:
    stats = load_ttc_stats(InferenceConfig().ttc_stats_path)

    assert stats.count == 3352
    assert stats.mean == EXPECTED_MEAN
    assert stats.std == EXPECTED_STD
    assert stats.source_manifest.endswith("head_train.jsonl")
    assert stats.normalization == "(ttc - mean) / std"


def test_invalid_ttc_std_raises_clear_error(tmp_path: Path) -> None:
    stats_path = tmp_path / "invalid.json"
    stats_path.write_text(
        '{"count": 1, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 1.0, '
        '"source_manifest": "train.jsonl", "normalization": "(ttc - mean) / std"}',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="std.*greater than zero"):
        load_ttc_stats(stats_path)


def test_missing_ttc_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="TTC statistics file is missing"):
        load_ttc_stats(tmp_path / "missing.json")


def test_real_track_b_checkpoint_exposes_consistent_mappings() -> None:
    checkpoint = InferenceConfig().track_b_checkpoint
    if not checkpoint.exists():
        pytest.skip("Track B checkpoint is unavailable in this test environment")

    metadata = load_track_b_artifacts(checkpoint)

    assert metadata.num_noun_classes == 118
    assert metadata.num_verb_classes == 64
    assert len(metadata.noun_id_list) == metadata.num_noun_classes
    assert len(metadata.verb_id_list) == metadata.num_verb_classes
    assert len(set(metadata.noun_id_list)) == len(metadata.noun_id_list)
    assert len(set(metadata.verb_id_list)) == len(metadata.verb_id_list)
    assert metadata.has_projector_state
    assert metadata.has_fusion_state
    assert metadata.has_head_state
    assert metadata.train_config["video_backbone"] == "resnet18"
    assert metadata.train_config["projector_in_dim"] == 512


def test_missing_taxonomy_file_is_reported(tmp_path: Path) -> None:
    from local_extraction.inference.artifacts import load_taxonomy

    with pytest.raises(ArtifactError, match="taxonomy annotation file is missing"):
        load_taxonomy(tmp_path / "missing.json")
