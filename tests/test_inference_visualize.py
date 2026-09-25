import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from local_extraction.inference.visualize import (
    VisualizationError,
    build_parser,
    calculate_zoom_crop,
    format_prediction_label,
    main,
    place_label,
    render_prediction_result,
    select_predictions,
    visualize_prediction_json,
)


def _frame(path: Path, color=(30, 40, 50)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (240, 180), color).save(path)
    return path


def _taxonomy(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "noun_categories": [{"id": 16, "name": "container_(box,_jar)"}],
                "verb_categories": [{"id": 62, "name": "take_(pick,_grab)"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _prediction(box, candidate_index=0, active=0.8) -> dict:
    return {
        "candidate_index": candidate_index,
        "box_xyxy": list(box),
        "detector_confidence": 0.75,
        "score": active,
        "next_active_probability": active,
        "noun_id": 16,
        "noun_probability": 0.6,
        "verb_id": 62,
        "verb_probability": 0.7,
        "ttc_seconds": 0.25,
    }


def _payload(frame: Path, predictions=None) -> dict:
    predictions = [_prediction((20, 30, 100, 120))] if predictions is None else predictions
    return {
        "target_frame": str(frame),
        "uid": "uid",
        "context_frames": [str(frame)],
        "image_size": {"height": 180, "width": 240},
        "candidate_count": len(predictions),
        "prediction_count": len(predictions),
        "predictions": predictions,
    }


def test_successful_rendering_resolves_taxonomy_names_and_preserves_payload(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    payload = _payload(frame)
    original = copy.deepcopy(payload)
    output = tmp_path / "nested" / "visualization.png"

    result = render_prediction_result(payload, taxonomy, output)

    assert result == output
    assert output.is_file()
    assert payload == original
    with Image.open(output) as rendered:
        assert rendered.size == (240, 180)
        assert rendered.getpixel((20, 30)) != (30, 40, 50)


def test_taxonomy_name_resolution_is_used_in_rendered_label(tmp_path: Path, monkeypatch) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    captured = []
    original = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).ImageDraw.multiline_text

    def record(self, xy, text, *args, **kwargs):
        captured.append(text)
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr("PIL.ImageDraw.ImageDraw.multiline_text", record)
    render_prediction_result(_payload(frame), taxonomy, tmp_path / "out.png")

    assert any("container - take" in text for text in captured)


def test_top_k_draws_only_requested_predictions(tmp_path: Path) -> None:
    background = (30, 40, 50)
    frame = _frame(tmp_path / "frame.png", background)
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    predictions = [
        _prediction((10, 20, 70, 80), candidate_index=0, active=0.9),
        _prediction((150, 100, 220, 160), candidate_index=1, active=0.8),
    ]
    output = tmp_path / "top1.png"

    render_prediction_result(_payload(frame, predictions), taxonomy, output, top_k=1)

    with Image.open(output).convert("RGB") as rendered:
        assert rendered.getpixel((10, 20)) != background
        assert rendered.getpixel((150, 100)) == background


def test_explicit_target_frame_overrides_json_path(tmp_path: Path) -> None:
    override = _frame(tmp_path / "host" / "frame.jpg", (90, 20, 10))
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    payload = _payload(Path("/data/frames/uid/frame.jpg"), [])
    prediction_json = tmp_path / "prediction.json"
    prediction_json.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "override.png"

    visualize_prediction_json(prediction_json, taxonomy, output, target_frame=override)

    assert output.is_file()


def test_zero_predictions_produces_annotated_output(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    output = tmp_path / "empty.jpg"

    render_prediction_result(_payload(frame, []), taxonomy, output)

    assert output.is_file()
    with Image.open(output) as rendered:
        assert rendered.size == (240, 180)


def test_invalid_prediction_box_has_clear_error(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    payload = _payload(frame, [_prediction((100, 20, 10, 80))])

    with pytest.raises(VisualizationError, match="malformed box_xyxy"):
        render_prediction_result(payload, taxonomy, tmp_path / "invalid.png")


def test_cli_creates_output_parent_and_accepts_missing_detector_class_id(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.jpg")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    prediction_json = tmp_path / "prediction.json"
    prediction_json.write_text(json.dumps(_payload(frame)), encoding="utf-8")
    output = tmp_path / "deep" / "path" / "result.jpg"

    code = main(
        [
            "--prediction-json", str(prediction_json),
            "--taxonomy", str(taxonomy),
            "--output", str(output),
            "--top-k", "1",
        ]
    )

    assert code == 0
    assert output.is_file()


def test_presentation_defaults_to_top_one_and_explicit_top_two(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.png")
    predictions = [
        _prediction((10, 20, 70, 80), candidate_index=0),
        _prediction((150, 100, 220, 160), candidate_index=1),
    ]
    payload = _payload(frame, predictions)

    assert len(select_predictions(payload, style="presentation")) == 1
    assert len(select_predictions(payload, style="presentation", top_k=2)) == 2
    assert len(select_predictions(payload, style="debug")) == 2

    default_output = tmp_path / "presentation-default.png"
    top_two_output = tmp_path / "presentation-top-two.png"
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    render_prediction_result(payload, taxonomy, default_output, style="presentation")
    render_prediction_result(payload, taxonomy, top_two_output, style="presentation", top_k=2)
    with Image.open(default_output).convert("RGB") as default_image:
        assert default_image.getpixel((150, 100)) == (30, 40, 50)
    with Image.open(top_two_output).convert("RGB") as top_two_image:
        assert top_two_image.getpixel((150, 100)) != (30, 40, 50)


def test_presentation_overlay_renders_successfully(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.png")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    output = tmp_path / "presentation.png"

    render_prediction_result(
        _payload(frame), taxonomy, output, style="presentation", layout="overlay"
    )

    assert output.is_file()
    with Image.open(output) as rendered:
        assert rendered.size == (240, 180)


def test_presentation_label_is_ascii_safe_and_uses_clear_wording() -> None:
    label = format_prediction_label(
        _prediction((20, 30, 100, 120)), 1,
        {16: "container"}, {62: "shake"}, style="presentation",
    )

    assert label.startswith("#1  container - shake\n")
    assert "Interaction score: 0.800" in label
    assert "—" not in label


def test_side_by_side_has_equal_panels_and_small_header(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.png")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    output = tmp_path / "side.png"

    render_prediction_result(
        _payload(frame), taxonomy, output, style="presentation", layout="side-by-side"
    )

    with Image.open(output) as rendered:
        assert rendered.width == 480
        assert 180 < rendered.height <= 232


def test_zoom_crop_is_clamped_and_contains_box() -> None:
    crop = calculate_zoom_crop((0, 2, 35, 50), (240, 180))

    assert 0 <= crop[0] < crop[2] <= 240
    assert 0 <= crop[1] < crop[3] <= 180
    assert crop[0] <= 0 < 35 <= crop[2]
    assert crop[1] <= 2 < 50 <= crop[3]


def test_zoom_renders_valid_transformed_top_box(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.png")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    output = tmp_path / "zoom.png"
    payload = _payload(frame, [_prediction((0, 2, 35, 50))])

    render_prediction_result(payload, taxonomy, output, style="presentation", layout="zoom")

    with Image.open(output) as rendered:
        assert rendered.width > 0
        assert rendered.height > 0
        # Rank-one box begins at a valid transformed crop coordinate.
        assert rendered.getpixel((0, 4)) != (30, 40, 50)


@pytest.mark.parametrize("layout", ["overlay", "side-by-side", "zoom"])
def test_zero_predictions_work_in_every_layout(tmp_path: Path, layout: str) -> None:
    frame = _frame(tmp_path / f"{layout}.png")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    output = tmp_path / f"empty-{layout}.png"

    render_prediction_result(
        _payload(frame, []), taxonomy, output, style="presentation", layout=layout
    )

    assert output.is_file()


@pytest.mark.parametrize(
    "option,value", [("--style", "invalid"), ("--layout", "invalid")]
)
def test_invalid_cli_choices_are_rejected(option: str, value: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--prediction-json", "prediction.json", "--taxonomy", "taxonomy.json",
            "--output", "output.png", option, value,
        ])


def test_rendering_is_pixel_deterministic(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.png")
    taxonomy = _taxonomy(tmp_path / "taxonomy.json")
    first, second = tmp_path / "first.png", tmp_path / "second.png"

    render_prediction_result(_payload(frame), taxonomy, first, style="presentation")
    render_prediction_result(_payload(frame), taxonomy, second, style="presentation")

    with Image.open(first) as image_a, Image.open(second) as image_b:
        assert list(image_a.getdata()) == list(image_b.getdata())


def test_label_placement_stays_inside_image_and_avoids_available_collision() -> None:
    first = place_label((30, 40, 80, 90), (50, 20), (120, 100), [])
    second = place_label((35, 42, 85, 92), (50, 20), (120, 100), [first])

    for rectangle in (first, second):
        assert 0 <= rectangle[0] < rectangle[2] <= 120
        assert 0 <= rectangle[1] < rectangle[3] <= 100
    assert not (
        first[0] < second[2] and first[2] > second[0]
        and first[1] < second[3] and first[3] > second[1]
    )


def test_top_k_must_be_positive(tmp_path: Path) -> None:
    frame = _frame(tmp_path / "frame.png")
    with pytest.raises(VisualizationError, match="greater than or equal to 1"):
        select_predictions(_payload(frame), top_k=0)
