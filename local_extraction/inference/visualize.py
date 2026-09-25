"""Render serialized production STA predictions over their target frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .artifacts import ArtifactError, load_taxonomy

Style = Literal["debug", "presentation"]
Layout = Literal["overlay", "side-by-side", "zoom"]
Box = tuple[int, int, int, int]


class VisualizationError(RuntimeError):
    """Raised when a production prediction result cannot be visualized."""


_COLORS = ((46, 204, 113), (52, 152, 219), (241, 196, 15),
           (231, 76, 60), (155, 89, 182), (26, 188, 156))


def load_prediction_json(path: Path) -> dict[str, Any]:
    """Load a prediction JSON object without changing it."""
    path = Path(path).expanduser()
    if not path.exists():
        raise VisualizationError(f"Prediction JSON is missing: {path}")
    if not path.is_file():
        raise VisualizationError(f"Prediction JSON is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualizationError(f"Could not read prediction JSON '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise VisualizationError(f"Prediction JSON must contain an object: {path}")
    return value


def load_taxonomy_names(taxonomy_path: Path) -> tuple[dict[int, str], dict[int, str]]:
    """Load concise noun and verb display names with the project's taxonomy loader."""
    try:
        taxonomy = load_taxonomy(Path(taxonomy_path))
    except ArtifactError as exc:
        raise VisualizationError(str(exc)) from exc
    if not isinstance(taxonomy, dict):
        raise VisualizationError("Taxonomy annotation file must contain a JSON object")

    def category_map(field: str) -> dict[int, str]:
        categories = taxonomy.get(field)
        if not isinstance(categories, list):
            raise VisualizationError(f"Taxonomy is missing the '{field}' list")
        result: dict[int, str] = {}
        for item in categories:
            if not isinstance(item, dict) or "id" not in item or "name" not in item:
                raise VisualizationError(f"Taxonomy '{field}' entries must contain id and name")
            try:
                category_id = int(item["id"])
            except (TypeError, ValueError) as exc:
                raise VisualizationError(f"Taxonomy '{field}' contains a non-integer id") from exc
            raw_name = str(item["name"]).strip()
            if not raw_name:
                raise VisualizationError(f"Taxonomy '{field}' contains an empty name")
            result[category_id] = raw_name.split("_(", 1)[0].replace("_", " ")
        return result

    return category_map("noun_categories"), category_map("verb_categories")


def _number(prediction: Mapping[str, Any], field: str, index: int) -> float:
    value = prediction.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisualizationError(f"Prediction {index} field '{field}' must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise VisualizationError(f"Prediction {index} field '{field}' must be finite")
    return result


def _integer(prediction: Mapping[str, Any], field: str, index: int) -> int:
    value = prediction.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisualizationError(f"Prediction {index} field '{field}' must be an integer")
    return int(value)


def _box(prediction: Mapping[str, Any], index: int) -> tuple[float, float, float, float]:
    value = prediction.get("box_xyxy")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise VisualizationError(f"Prediction {index} field 'box_xyxy' must contain four coordinates")
    coordinates = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise VisualizationError(f"Prediction {index} field 'box_xyxy' must be numeric")
        coordinate = float(coordinate)
        if not math.isfinite(coordinate):
            raise VisualizationError(f"Prediction {index} field 'box_xyxy' must be finite")
        coordinates.append(coordinate)
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        raise VisualizationError(
            f"Prediction {index} has malformed box_xyxy: expected x2 > x1 and y2 > y1"
        )
    return x1, y1, x2, y2


def select_predictions(
    payload: Mapping[str, Any], *, style: Style = "debug", top_k: int | None = None
) -> list[dict[str, Any]]:
    """Validate and select predictions, preserving their serialized rank order."""
    if style not in {"debug", "presentation"}:
        raise VisualizationError(f"Unknown visualization style: {style}")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise VisualizationError("Prediction result field 'predictions' must be a list")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise VisualizationError("top_k must be an integer greater than or equal to 1")
    limit = top_k if top_k is not None else (1 if style == "presentation" else None)
    selected = predictions[:limit]
    numeric_fields = ("detector_confidence", "score", "next_active_probability",
                      "noun_probability", "verb_probability", "ttc_seconds")
    result = []
    for index, prediction in enumerate(selected):
        if not isinstance(prediction, dict):
            raise VisualizationError(f"Prediction {index} must be a JSON object")
        result.append({
            "candidate_index": _integer(prediction, "candidate_index", index),
            "box_xyxy": _box(prediction, index),
            "noun_id": _integer(prediction, "noun_id", index),
            "verb_id": _integer(prediction, "verb_id", index),
            **{field: _number(prediction, field, index) for field in numeric_fields},
        })
    return result


def format_prediction_label(
    prediction: Mapping[str, Any], rank: int, noun_names: Mapping[int, str],
    verb_names: Mapping[int, str], *, style: Style,
) -> str:
    """Build either the detailed engineering label or compact presentation label."""
    noun_id, verb_id = int(prediction["noun_id"]), int(prediction["verb_id"])
    noun = noun_names.get(noun_id, f"noun {noun_id}")
    verb = verb_names.get(verb_id, f"verb {verb_id}")
    if style == "presentation":
        return (f"#{rank}  {noun} - {verb}\n"
                f"Interaction score: {prediction['next_active_probability']:.3f}\n"
                f"Detector: {prediction['detector_confidence']:.3f}\n"
                f"TTC: {prediction['ttc_seconds']:.3f} s")
    return (f"#{rank} {noun} - {verb}\n"
            f"active: {prediction['next_active_probability']:.3f}  "
            f"det: {prediction['detector_confidence']:.3f}\n"
            f"noun: {prediction['noun_probability']:.3f}  "
            f"verb: {prediction['verb_probability']:.3f}\n"
            f"TTC: {prediction['ttc_seconds']:.3f} s")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _intersection_area(a: Box, b: Box) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def place_label(
    box: Box, label_size: tuple[int, int], image_size: tuple[int, int], occupied: Sequence[Box]
) -> Box:
    """Place a label deterministically, preferring the first collision-free position."""
    x1, y1, x2, y2 = box
    label_width, label_height = min(label_size[0], image_size[0]), min(label_size[1], image_size[1])
    max_x, max_y = max(0, image_size[0] - label_width), max(0, image_size[1] - label_height)
    candidates = ((x1, y1 - label_height - 4), (x1, y2 + 4), (x1 + 4, y1 + 4),
                  (x2 + 4, y1), (x1 - label_width - 4, y1),
                  (x2 - label_width, y1 - label_height - 4), (x2 - label_width, y2 + 4))
    rectangles: list[Box] = []
    for x, y in candidates:
        x, y = min(max(0, int(x)), max_x), min(max(0, int(y)), max_y)
        rectangle = (x, y, x + label_width, y + label_height)
        rectangles.append(rectangle)
        if all(_intersection_area(rectangle, previous) == 0 for previous in occupied):
            return rectangle
    return min(rectangles, key=lambda rect: sum(_intersection_area(rect, old) for old in occupied))


def _clamp_box(box: Sequence[float], image_size: tuple[int, int], index: int) -> Box:
    width, height = image_size
    clamped = (max(0, min(width - 1, int(round(box[0])))),
               max(0, min(height - 1, int(round(box[1])))),
               max(0, min(width - 1, int(round(box[2])))),
               max(0, min(height - 1, int(round(box[3])))))
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise VisualizationError(f"Prediction {index} box_xyxy lies outside the target frame")
    return clamped


def render_annotated_image(
    source: Image.Image, predictions: Sequence[Mapping[str, Any]],
    noun_names: Mapping[int, str], verb_names: Mapping[int, str], *, style: Style = "debug",
) -> Image.Image:
    """Return an annotated RGB copy of an image."""
    image = source.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    presentation = style == "presentation"
    font_size = max(18, min(36, min(image.size) // 25)) if presentation else 10
    font = _font(font_size) if presentation else ImageFont.load_default()
    spacing, padding = (max(4, font_size // 5), max(8, font_size // 3)) if presentation else (2, 5)
    box_width = max(5, min(image.size) // 180) if presentation else 3
    occupied: list[Box] = []
    if not predictions:
        label = "No predictions"
        text_box = draw.textbbox((0, 0), label, font=font)
        rect = (10, 10, 10 + text_box[2] - text_box[0] + 2 * padding,
                10 + text_box[3] - text_box[1] + 2 * padding)
        draw.rectangle(rect, fill=(0, 0, 0), outline=(255, 255, 255), width=2)
        draw.text((rect[0] + padding, rect[1] + padding), label, fill=(255, 255, 255), font=font)
        return image
    boxes = [_clamp_box(prediction["box_xyxy"], image.size, index)
             for index, prediction in enumerate(predictions)]
    for index, box in enumerate(boxes):
        draw.rectangle(box, outline=_COLORS[index % len(_COLORS)], width=box_width)
    for index, (prediction, box) in enumerate(zip(predictions, boxes)):
        label = format_prediction_label(prediction, index + 1, noun_names, verb_names, style=style)
        text_box = draw.multiline_textbbox((0, 0), label, font=font, spacing=spacing)
        label_size = (text_box[2] - text_box[0] + 2 * padding,
                      text_box[3] - text_box[1] + 2 * padding)
        rect = place_label(box, label_size, image.size, occupied)
        occupied.append(rect)
        color = _COLORS[index % len(_COLORS)]
        draw.rectangle(rect, fill=(0, 0, 0), outline=color, width=2)
        draw.multiline_text((rect[0] + padding, rect[1] + padding), label,
                            fill=color, font=font, spacing=spacing)
    return image


def compose_side_by_side(original: Image.Image, annotated: Image.Image, *, style: Style) -> Image.Image:
    """Compose equal-sized original and annotated panels with compact headings."""
    if original.size != annotated.size:
        raise VisualizationError("Side-by-side panels must have equal dimensions")
    width, height = original.size
    header_height = max(28, min(52, height // 12))
    canvas = Image.new("RGB", (width * 2, height + header_height), (18, 18, 18))
    canvas.paste(original.convert("RGB"), (0, header_height))
    canvas.paste(annotated.convert("RGB"), (width, header_height))
    draw = ImageDraw.Draw(canvas)
    font = (_font(max(18, min(28, header_height * 3 // 5)))
            if style == "presentation" else ImageFont.load_default())
    y = max(2, (header_height - getattr(font, "size", 10)) // 2)
    draw.text((12, y), "Original", fill=(255, 255, 255), font=font)
    draw.text((width + 12, y), "Prediction", fill=(255, 255, 255), font=font)
    return canvas


def calculate_zoom_crop(box: Sequence[float], image_size: tuple[int, int]) -> Box:
    """Calculate a padded, boundary-clamped crop around the top-ranked box."""
    width, height = image_size
    x1, y1, x2, y2 = box
    crop_width = min(width, max((x2 - x1) * 3.0, width * 0.35))
    crop_height = min(height, max((y2 - y1) * 3.0, height * 0.35))
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(0.0, center_x - crop_width / 2), width - crop_width)
    top = min(max(0.0, center_y - crop_height / 2), height - crop_height)
    return (int(math.floor(left)), int(math.floor(top)),
            int(math.ceil(left + crop_width)), int(math.ceil(top + crop_height)))


def render_zoom_image(
    source: Image.Image, prediction: Mapping[str, Any], noun_names: Mapping[int, str],
    verb_names: Mapping[int, str], *, style: Style = "presentation",
) -> Image.Image:
    """Crop around rank one, transform its box, and render the zoom annotation."""
    crop = calculate_zoom_crop(prediction["box_xyxy"], source.size)
    zoomed = source.convert("RGB").crop(crop)
    scale = 2 if max(zoomed.size) < 900 else 1
    if scale > 1:
        zoomed = zoomed.resize((zoomed.width * scale, zoomed.height * scale), Image.Resampling.LANCZOS)
    transformed = dict(prediction)
    x1, y1, x2, y2 = prediction["box_xyxy"]
    transformed["box_xyxy"] = ((x1 - crop[0]) * scale, (y1 - crop[1]) * scale,
                                (x2 - crop[0]) * scale, (y2 - crop[1]) * scale)
    return render_annotated_image(zoomed, [transformed], noun_names, verb_names, style=style)


def _open_frame(payload: Mapping[str, Any], target_frame: Path | None) -> Image.Image:
    frame_value = target_frame if target_frame is not None else payload.get("target_frame")
    if not isinstance(frame_value, (str, Path)) or not str(frame_value):
        raise VisualizationError("Prediction result has no usable target_frame")
    frame_path = Path(frame_value).expanduser()
    if not frame_path.is_file():
        raise VisualizationError(f"Target frame is missing or not a file: {frame_path}")
    try:
        with Image.open(frame_path) as source:
            return source.convert("RGB")
    except Exception as exc:
        raise VisualizationError(f"Could not open target frame '{frame_path}': {exc}") from exc


def _save_image(image: Image.Image, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser()
    suffix = output_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise VisualizationError("Output path must end in .jpg, .jpeg, or .png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if suffix in {".jpg", ".jpeg"}:
            image.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=False)
        else:
            image.save(output_path, format="PNG", compress_level=6, optimize=False)
    except OSError as exc:
        raise VisualizationError(f"Could not save visualization '{output_path}': {exc}") from exc
    return output_path


def render_prediction_result(
    payload: Mapping[str, Any], taxonomy_path: Path, output_path: Path, *,
    target_frame: Path | None = None, top_k: int | None = None,
    style: Style = "debug", layout: Layout = "overlay",
) -> Path:
    """Render a production inference result without modifying the input mapping."""
    if not isinstance(payload, Mapping):
        raise VisualizationError("Prediction result must be a mapping")
    if layout not in {"overlay", "side-by-side", "zoom"}:
        raise VisualizationError(f"Unknown visualization layout: {layout}")
    predictions = select_predictions(payload, style=style, top_k=top_k)
    noun_names, verb_names = load_taxonomy_names(Path(taxonomy_path))
    original = _open_frame(payload, target_frame)
    annotated = render_annotated_image(original, predictions, noun_names, verb_names, style=style)
    if layout == "side-by-side":
        result = compose_side_by_side(original, annotated, style=style)
    elif layout == "zoom" and predictions:
        result = render_zoom_image(original, predictions[0], noun_names, verb_names, style=style)
    else:
        result = annotated
    return _save_image(result, output_path)


def visualize_prediction_json(
    prediction_json: Path, taxonomy_path: Path, output_path: Path, *,
    target_frame: Path | None = None, top_k: int | None = None,
    style: Style = "debug", layout: Layout = "overlay",
) -> Path:
    """Load a serialized production result and render it to an image."""
    return render_prediction_result(load_prediction_json(Path(prediction_json)), taxonomy_path, output_path,
                                    target_frame=target_frame, top_k=top_k,
                                    style=style, layout=layout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize an Ego4D-LiteSTA production prediction JSON.")
    parser.add_argument("--prediction-json", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-frame", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--style", choices=("debug", "presentation"), default="debug")
    parser.add_argument("--layout", choices=("overlay", "side-by-side", "zoom"), default="overlay")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    visualize_prediction_json(args.prediction_json, args.taxonomy, args.output,
                              target_frame=args.target_frame, top_k=args.top_k,
                              style=args.style, layout=args.layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VisualizationError", "build_parser", "calculate_zoom_crop", "compose_side_by_side",
           "format_prediction_label", "load_prediction_json", "load_taxonomy_names", "main",
           "place_label", "render_annotated_image", "render_prediction_result", "render_zoom_image",
           "select_predictions", "visualize_prediction_json"]
