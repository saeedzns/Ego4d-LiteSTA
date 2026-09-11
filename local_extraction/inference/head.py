"""Production candidate ROI pooling and raw Track B head inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import torch

from .artifacts import ArtifactError, load_track_b_feature_contract
from .config import InferenceConfig
from .detector import DetectionCandidate
from .fusion import FusedTokens


class HeadInferenceError(RuntimeError):
    """Raised when ROI/head setup, inputs, or raw inference is invalid."""


@dataclass(frozen=True)
class RawCandidateOutput:
    """Raw head outputs for one detector candidate, without final decoding."""

    candidate: DetectionCandidate
    cls_logits: torch.Tensor
    noun_logits: torch.Tensor | None
    verb_logits: torch.Tensor | None
    ttc_raw: torch.Tensor
    ttc_bin_logits: torch.Tensor | None


class TrackBInferenceHead:
    """Pool fused token grids per candidate and run the trained TrackBHead."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        try:
            self.contract = load_track_b_feature_contract(config.track_b_feature_contract_path)
        except ArtifactError as exc:
            raise HeadInferenceError(str(exc)) from exc
        if config.track_b_checkpoint.name != self.contract.checkpoint:
            raise HeadInferenceError(
                "Track B checkpoint does not match feature contract: "
                f"configured={config.track_b_checkpoint.name}, contract={self.contract.checkpoint}"
            )
        checkpoint = Path(config.track_b_checkpoint)
        if not checkpoint.exists():
            raise HeadInferenceError(f"Track B checkpoint does not exist: {checkpoint}")
        if not checkpoint.is_file():
            raise HeadInferenceError(f"Track B checkpoint is not a file: {checkpoint}")
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self._load_head(payload)
        except HeadInferenceError:
            raise
        except Exception as exc:
            raise HeadInferenceError(f"Could not load Track B head: {exc}") from exc

    def _load_head(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise HeadInferenceError("Track B checkpoint must contain a mapping")
        state = payload.get("head")
        if not isinstance(state, dict):
            raise HeadInferenceError("Track B checkpoint is missing head state")
        train_config = payload.get("train_config")
        if not isinstance(train_config, dict):
            raise HeadInferenceError("Track B checkpoint is missing train_config metadata")
        required = ("head_dropout", "use_multi_task_labels")
        missing = [key for key in required if key not in train_config]
        if missing:
            raise HeadInferenceError(f"Track B checkpoint is missing head metadata: {missing}")
        dropout = train_config["head_dropout"]
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise HeadInferenceError("head_dropout must be numeric")
        if not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1:
            raise HeadInferenceError("head_dropout must be finite and in [0,1)")

        cls_weight = state.get("cls_head.weight")
        backbone_weight = state.get("backbone.1.weight")
        ttc_weight = state.get("ttc_head.weight")
        if cls_weight is None or backbone_weight is None or ttc_weight is None:
            raise HeadInferenceError("Track B head state is missing required output/input weights")
        if len(cls_weight.shape) != 2 or len(backbone_weight.shape) != 2 or len(ttc_weight.shape) != 2:
            raise HeadInferenceError("Track B head state has malformed weight dimensions")
        hidden_dim = int(backbone_weight.shape[0])
        head_input_dim = int(backbone_weight.shape[1])
        if head_input_dim <= 0 or head_input_dim % 2:
            raise HeadInferenceError(f"Track B head input width must be a positive even value, got {head_input_dim}")
        fused_dim = head_input_dim // 2
        projector_state = payload.get("projector")
        projector_weight = projector_state.get("weight") if isinstance(projector_state, dict) else None
        if projector_weight is not None and (
            len(projector_weight.shape) != 2 or int(projector_weight.shape[0]) != fused_dim
        ):
            raise HeadInferenceError("Track B projector output width does not match head fused dimension")
        num_classes = int(cls_weight.shape[0])
        if int(ttc_weight.shape[0]) != 1:
            raise HeadInferenceError("Track B TTC head must output one raw regression value")
        if int(ttc_weight.shape[1]) != hidden_dim or int(cls_weight.shape[1]) != hidden_dim:
            raise HeadInferenceError("Track B output weights do not match hidden dimension")
        noun_weight = state.get("noun_head.weight")
        verb_weight = state.get("verb_head.weight")
        noun_ids = payload.get("noun_id_list") or []
        verb_ids = payload.get("verb_id_list") or []
        if noun_weight is None or verb_weight is None:
            raise HeadInferenceError("Track B head is missing noun or verb output state")
        if len(noun_weight.shape) != 2 or len(verb_weight.shape) != 2:
            raise HeadInferenceError("Track B noun/verb head state has malformed dimensions")
        if int(noun_weight.shape[1]) != hidden_dim or int(verb_weight.shape[1]) != hidden_dim:
            raise HeadInferenceError("Track B noun/verb weights do not match hidden dimension")
        ttc_bin_weight = state.get("ttc_bin_head.weight")
        if ttc_bin_weight is not None:
            if len(ttc_bin_weight.shape) != 2 or int(ttc_bin_weight.shape[1]) != hidden_dim:
                raise HeadInferenceError("Track B TTC-bin head state has malformed dimensions")
        if len(noun_ids) != int(noun_weight.shape[0]):
            raise HeadInferenceError("Noun mapping length does not match noun head output")
        if len(verb_ids) != int(verb_weight.shape[0]):
            raise HeadInferenceError("Verb mapping length does not match verb head output")
        if not bool(train_config["use_multi_task_labels"]):
            raise HeadInferenceError("Track B checkpoint does not enable multi-task head outputs")

        try:
            from local_extraction.trackB.trackB_head import HeadConfig, TrackBHead

            self.head = TrackBHead(
                HeadConfig(
                    dim=fused_dim,
                    num_classes=num_classes,
                    hidden=hidden_dim,
                    dropout=float(dropout),
                    num_noun_classes=int(noun_weight.shape[0]),
                    num_verb_classes=int(verb_weight.shape[0]),
                    num_ttc_bins=int(ttc_bin_weight.shape[0]) if ttc_bin_weight is not None else 0,
                    ttc_mode="bin" if ttc_bin_weight is not None else "reg",
                )
            )
            self.head.load_state_dict(state, strict=True)
        except HeadInferenceError:
            raise
        except Exception as exc:
            raise HeadInferenceError(f"Could not reconstruct Track B head: {exc}") from exc
        self.head.to(self.config.device).eval()
        self.head_input_dim = head_input_dim
        self.num_classes = num_classes
        self.num_noun_classes = int(noun_weight.shape[0])
        self.num_verb_classes = int(verb_weight.shape[0])
        self.num_ttc_bins = int(ttc_bin_weight.shape[0]) if ttc_bin_weight is not None else 0
        self.fused_dim = fused_dim

    @staticmethod
    def _pool(tokens_hw: tuple[int, int], tokens: torch.Tensor, box: tuple[float, float, float, float], image_wh: tuple[int, int]) -> torch.Tensor:
        from local_extraction.trackB.trackB_tokenizer import roi_pool_tokens_mean
        return roi_pool_tokens_mean(tokens_hw, tokens, box, image_wh)

    def predict(
        self,
        candidates: tuple[DetectionCandidate, ...] | list[DetectionCandidate],
        fused_tokens: FusedTokens,
        image_size: tuple[int, int],
    ) -> tuple[RawCandidateOutput, ...]:
        """Return raw head outputs in the incoming candidate order.

        ``image_size`` is ``(height, width)``; detector boxes remain pixel-space
        ``xyxy`` coordinates and are passed to the thesis ROI helper as ``(W,H)``.
        """
        height, width = image_size
        if height <= 0 or width <= 0:
            raise HeadInferenceError("Image height and width must be positive")
        image_tokens = fused_tokens.fused_image_tokens
        video_tokens = fused_tokens.fused_video_tokens
        if image_tokens.dtype != torch.float32 or video_tokens.dtype != torch.float32:
            raise HeadInferenceError("Fused tokens must have dtype torch.float32")
        expected = (self.contract.token_count, self.fused_dim)
        if tuple(image_tokens.shape) != expected or tuple(video_tokens.shape) != expected:
            raise HeadInferenceError(f"Fused token shapes must be {expected}")
        if image_tokens.device.type != "cpu" or video_tokens.device.type != "cpu":
            raise HeadInferenceError("Fused tokens must be on CPU")
        if image_tokens.shape != video_tokens.shape:
            raise HeadInferenceError("Fused image/video token shapes must match")
        if self.contract.grid_hw != (7, 7):
            raise HeadInferenceError(f"Unsupported feature grid: {self.contract.grid_hw}")
        if not candidates:
            return ()

        boxes = []
        for candidate in candidates:
            if not isinstance(candidate, DetectionCandidate):
                raise HeadInferenceError("Candidates must be DetectionCandidate instances")
            if not all(math.isfinite(float(value)) for value in candidate.box_xyxy):
                raise HeadInferenceError("Candidate box coordinates must be finite")
            boxes.append(candidate.box_xyxy)
        image_pooled = torch.stack([
            self._pool(self.contract.grid_hw, image_tokens, box, (width, height))
            for box in boxes
        ])
        video_pooled = torch.stack([
            self._pool(self.contract.grid_hw, video_tokens, box, (width, height))
            for box in boxes
        ])
        if image_pooled.shape[1] + video_pooled.shape[1] != self.head_input_dim:
            raise HeadInferenceError("Candidate feature width does not match TrackBHead input")
        try:
            with torch.inference_mode():
                output = self.head(
                    image_pooled.unsqueeze(0).to(self.config.device),
                    video_pooled.unsqueeze(0).to(self.config.device),
                )
        except Exception as exc:
            raise HeadInferenceError(f"Track B head forward failed: {exc}") from exc

        required = ("cls_logits", "noun_logits", "verb_logits", "ttc")
        if any(key not in output for key in required):
            raise HeadInferenceError("Track B head output is missing required raw fields")
        count = len(candidates)
        expected_shapes = {
            "cls_logits": (1, count, self.num_classes),
            "noun_logits": (1, count, self.num_noun_classes),
            "verb_logits": (1, count, self.num_verb_classes),
            "ttc": (1, count, 1),
        }
        for key, shape in expected_shapes.items():
            if tuple(output[key].shape) != shape:
                raise HeadInferenceError(f"Unexpected Track B head output shape for {key}")
        bin_output = output.get("ttc_bin_logits")
        if self.num_ttc_bins == 0 and bin_output is not None:
            raise HeadInferenceError("Track B head returned unexpected TTC-bin output")
        if self.num_ttc_bins > 0 and bin_output is None:
            raise HeadInferenceError("Track B head did not return TTC-bin output")
        if bin_output is not None and tuple(bin_output.shape) != (1, count, self.num_ttc_bins):
            raise HeadInferenceError("Unexpected Track B TTC-bin output shape")

        results = []
        for index, candidate in enumerate(candidates):
            results.append(RawCandidateOutput(
                candidate=candidate,
                cls_logits=output["cls_logits"][0, index].float().cpu(),
                noun_logits=output["noun_logits"][0, index].float().cpu(),
                verb_logits=output["verb_logits"][0, index].float().cpu(),
                ttc_raw=output["ttc"][0, index].float().cpu(),
                ttc_bin_logits=bin_output[0, index].float().cpu() if bin_output is not None else None,
            ))
        return tuple(results)


__all__ = ["HeadInferenceError", "RawCandidateOutput", "TrackBInferenceHead"]
