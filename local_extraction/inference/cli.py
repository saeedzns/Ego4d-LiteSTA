"""Command-line entry point for local short-term anticipation inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import InferenceConfig


class InferenceCLIError(RuntimeError):
    """Raised when the local inference CLI cannot complete."""


def _prediction_to_dict(prediction: Any) -> dict[str, Any]:
    candidate = prediction.candidate
    return {
        "candidate_index": int(prediction.candidate_index),
        "box_xyxy": [float(value) for value in candidate.box_xyxy],
        "detector_confidence": float(prediction.detector_confidence),
        "detector_class_id": candidate.class_id,
        "score": float(prediction.score),
        "next_active_probability": float(prediction.next_active_probability),
        "predicted_class": int(prediction.predicted_class),
        "noun_id": int(prediction.noun_id),
        "noun_local_index": int(prediction.noun_local_index),
        "noun_probability": float(prediction.noun_probability),
        "verb_id": int(prediction.verb_id),
        "verb_local_index": int(prediction.verb_local_index),
        "verb_probability": float(prediction.verb_probability),
        "ttc_seconds": float(prediction.ttc_seconds),
        "ttc_raw": float(prediction.ttc_raw),
        "ttc_bin": int(prediction.ttc_bin),
    }


def result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a pipeline result to a stable JSON-ready dictionary."""
    resolved = result.resolved_input
    return {
        "target_frame": str(resolved.target_frame),
        "uid": resolved.uid,
        "frame_directory": str(resolved.frame_directory),
        "context_frames": [str(path) for path in resolved.context_frames],
        "image_size": {
            "height": int(result.image_size[0]),
            "width": int(result.image_size[1]),
        },
        "candidate_count": len(result.candidates),
        "prediction_count": len(result.predictions),
        "predictions": [_prediction_to_dict(prediction) for prediction in result.predictions],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local Ego4D short-term anticipation inference for one target frame."
    )
    parser.add_argument("target_frame", type=Path, help="Target .jpg frame path.")
    parser.add_argument(
        "--tokenizer-weights",
        type=Path,
        default=None,
        help="Explicit ResNet18 IMAGENET1K_V1 weight artifact for token extraction.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for model execution.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    return parser


def run(
    args: argparse.Namespace,
    *,
    predictor_factory: Callable[[InferenceConfig], Any] | None = None,
) -> dict[str, Any]:
    config = InferenceConfig(
        device=args.device,
        tokenizer_weights_path=args.tokenizer_weights,
    )
    if predictor_factory is None:
        from .pipeline import ShortTermAnticipationPredictor

        predictor_factory = ShortTermAnticipationPredictor
    try:
        predictor = predictor_factory(config)
        result = predictor.predict(args.target_frame)
    except Exception as exc:
        raise InferenceCLIError(f"Local inference failed: {exc}") from exc
    return result_to_dict(result)


def main(
    argv: Sequence[str] | None = None,
    *,
    predictor_factory: Callable[[InferenceConfig], Any] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run(args, predictor_factory=predictor_factory)
    text = json.dumps(payload, indent=args.indent, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["InferenceCLIError", "build_parser", "main", "result_to_dict", "run"]
