"""Production-facing Track A detector adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

from .config import InferenceConfig


class DetectorError(RuntimeError):
    """Raised when detector setup, input validation, or output decoding fails."""


@dataclass(frozen=True)
class DetectionCandidate:
    """One detector candidate in pixel-space ``xyxy`` format."""

    box_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int | None = None

    def __post_init__(self) -> None:
        if len(self.box_xyxy) != 4:
            raise DetectorError("Detection box must contain exactly four coordinates")
        if not all(math.isfinite(float(value)) for value in self.box_xyxy):
            raise DetectorError("Detection box coordinates must be finite")
        x1, y1, x2, y2 = (float(value) for value in self.box_xyxy)
        if x2 < x1 or y2 < y1:
            raise DetectorError("Detection box coordinates must be ordered as x1,y1,x2,y2")
        if not math.isfinite(float(self.confidence)):
            raise DetectorError("Detection confidence must be finite")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise DetectorError("Detection confidence must be between zero and one")
        if self.class_id is not None and isinstance(self.class_id, bool):
            raise DetectorError("Detection class ID must be an integer or None")
        if self.class_id is not None and not isinstance(self.class_id, int):
            raise DetectorError("Detection class ID must be an integer or None")


def _to_cpu_list(value: Any) -> Any:
    """Convert tensor-like detector output to CPU-backed Python values."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


class TrackADetector:
    """Load Track A YOLO once and decode one-frame detections."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        model: Any | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        checkpoint = config.track_a_checkpoint
        if not checkpoint.exists():
            raise DetectorError(f"Track A checkpoint does not exist: {checkpoint}")
        if not checkpoint.is_file():
            raise DetectorError(f"Track A checkpoint is not a file: {checkpoint}")

        if model is not None and model_loader is not None:
            raise DetectorError("Provide either model or model_loader, not both")
        try:
            self._model = model if model is not None else self._load_model(model_loader)
            self._move_model_to_device()
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(
                f"Could not load Track A YOLO checkpoint '{checkpoint}': {exc}"
            ) from exc

    def _load_model(self, model_loader: Callable[[str], Any] | None) -> Any:
        if model_loader is not None:
            return model_loader(str(self.config.track_a_checkpoint))
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise DetectorError(
                "Ultralytics YOLO is unavailable; install ultralytics to load Track A"
            ) from exc
        return YOLO(str(self.config.track_a_checkpoint))

    def _move_model_to_device(self) -> None:
        move = getattr(self._model, "to", None)
        if callable(move) and self.config.device != "auto":
            move(self.config.device)

    def detect(self, target_frame: str | Path) -> tuple[DetectionCandidate, ...]:
        """Run one-frame YOLO detection and return stable top-K candidates."""
        frame = Path(target_frame).expanduser()
        if not frame.exists():
            raise DetectorError(f"Detection input frame does not exist: {frame}")
        if not frame.is_file():
            raise DetectorError(f"Detection input frame is not a file: {frame}")
        if frame.suffix.lower() != ".jpg":
            raise DetectorError(f"Unsupported detector input frame extension: {frame}")

        try:
            results = self._model.predict(
                source=str(frame),
                imgsz=self.config.track_a_imgsz,
                conf=self.config.track_a_confidence,
                iou=self.config.track_a_iou,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"Track A YOLO inference failed for '{frame}': {exc}") from exc

        if results is None:
            raise DetectorError("Track A YOLO returned no result collection")
        try:
            decoded: list[DetectionCandidate] = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                xyxy = getattr(boxes, "xyxy", None)
                confidences = getattr(boxes, "conf", None)
                classes = getattr(boxes, "cls", None)
                if xyxy is None or confidences is None or classes is None:
                    raise DetectorError(
                        "Track A YOLO result is missing xyxy, conf, or cls boxes"
                    )
                xyxy = _to_cpu_list(xyxy)
                confidences = _to_cpu_list(confidences)
                classes = _to_cpu_list(classes)
                if len(xyxy) != len(confidences) or len(xyxy) != len(classes):
                    raise DetectorError(
                        "Track A YOLO output lengths are inconsistent: "
                        f"xyxy={len(xyxy)}, conf={len(confidences)}, cls={len(classes)}"
                    )
                count = len(xyxy)
                for index in range(count):
                    coordinates = tuple(float(value) for value in xyxy[index])
                    decoded.append(
                        DetectionCandidate(
                            box_xyxy=coordinates,  # type: ignore[arg-type]
                            confidence=float(_to_cpu_list(confidences[index])),
                            class_id=int(_to_cpu_list(classes[index])),
                        )
                    )
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"Malformed Track A YOLO result: {exc}") from exc

        decoded.sort(key=lambda candidate: candidate.confidence, reverse=True)
        return tuple(decoded[: self.config.track_a_top_k])


__all__ = ["DetectionCandidate", "DetectorError", "TrackADetector"]
