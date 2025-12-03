"""Track B Metrics Utilities

Provides lightweight metric helpers used in evaluation and training:
- binary_accuracy / multiclass_accuracy
- binary_average_precision (small safe implementation)
- per_class_ap (one-vs-rest AP per class)
- ttc_mae (optionally denormalized given mean/std)

All functions accept PyTorch tensors and operate on CPU for simplicity.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """logits: (M,2) or (M,) probability-of-positive if already sigmoid.
       labels: (M,) int {0,1}
    """
    if logits.ndim == 2 and logits.shape[1] == 2:
        probs_pos = torch.softmax(logits, dim=1)[:, 1]
    else:
        probs_pos = logits.float()
    preds = (probs_pos >= 0.5).long()
    labels = labels.long()
    if labels.numel() == 0:
        return 0.0
    return (preds == labels).float().mean().item()


def multiclass_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """logits: (M,K); labels: (M,) in [0,K-1]."""
    if logits.numel() == 0:
        return 0.0
    preds = torch.argmax(logits, dim=1)
    labels = labels.long()
    return (preds == labels).float().mean().item()


def _ranked_positive_fraction(sorted_labels: torch.Tensor) -> torch.Tensor:
    # cumulative TP / index for precision curve points
    cum_pos = torch.cumsum(sorted_labels == 1, dim=0).float()
    denom = torch.arange(1, sorted_labels.numel() + 1, device=sorted_labels.device).float()
    return torch.where(denom > 0, cum_pos / denom, torch.zeros_like(denom))


def binary_average_precision(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Small AP implementation for binary classification.
    logits: (M,2) or raw score for positive class; labels: (M,) in {0,1}.
    """
    if labels.numel() == 0:
        return 0.0
    if logits.ndim == 2 and logits.shape[1] == 2:
        scores = torch.softmax(logits, dim=1)[:, 1]
    else:
        scores = logits.float()
    labels = labels.long()
    # Sort descending by score
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    prec_at_k = _ranked_positive_fraction(sorted_labels)
    total_pos = (labels == 1).sum().item()
    if total_pos == 0:
        return 0.0
    # AP = sum_{k where label_k==1} precision_at_k / total_pos
    ap = (prec_at_k[sorted_labels == 1].sum() / total_pos).item()
    return ap


def per_class_ap(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Dict[int, float]:
    """Compute one-vs-rest AP per class.
    logits: (M,K); labels: (M,)
    Returns dict {class_id: ap}.
    """
    out: Dict[int, float] = {}
    if logits.numel() == 0:
        return {i: 0.0 for i in range(num_classes)}
    probs = torch.softmax(logits, dim=1)
    for c in range(num_classes):
        # treat class c as positive
        scores = probs[:, c]
        bin_labels = (labels == c).long()
        out[c] = binary_average_precision(scores, bin_labels)
    return out


def ttc_mae(pred_ttc: torch.Tensor, gt_ttc: torch.Tensor) -> float:
    """MAE over TTC predictions (already denormalized)."""
    if pred_ttc.numel() == 0:
        return 0.0
    return torch.abs(pred_ttc - gt_ttc).mean().item()


def denorm_ttc(ttc_norm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return ttc_norm * std + mean

__all__ = [
    'binary_accuracy',
    'multiclass_accuracy',
    'binary_average_precision',
    'per_class_ap',
    'ttc_mae',
    'denorm_ttc',
]
