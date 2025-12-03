#!/usr/bin/env python3
"""Sweep K script for Track A Stage A

Purpose:
  Run YOLO inference over a selection of frames and compute detection Recall@K and Positive Ratio
  for multiple K values (e.g., 6, 8, 10). Results are aggregated and logged.

Definitions (assuming single-image object detection with YOLO-style labels):
  Ground-truth boxes: From YOLO label files (class cx cy w h) normalized (0-1).
  Predicted boxes: YOLO model outputs (class, confidence, xyxyn normalized coords).
  Matching rule: A prediction is a True Positive (TP) for a GT box if:
      - Class matches
      - IoU >= IOU_THRESHOLD
      - Each GT box matched to at most one prediction (greedy by descending conf)
  Recall@K (image-level): TP / GT_count (only if GT_count > 0, else ignored for macro recall)
  Positive Ratio@K (image-level): TP / K (always defined; if fewer than K preds, we use min(K, #preds))

Aggregate metrics reported:
  - micro_recall@K: total_TP / total_GT (across all images with GT)
  - macro_recall@K: average of per-image recalls (images with GT only)
  - micro_positive_ratio@K: total_TP / (sum over images of K_used)
  - macro_positive_ratio@K: average of per-image positive ratios (all processed images)

Config notes:
  To keep runtime manageable you can cap MAX_IMAGES or enable LAST_FRAME_ONLY.
  Set DRY_RUN=True to only print configuration without running inference.

Output:
    local_extraction/runs/Track_A/KSweep/StageA_KSweep_<timestamp>/
      metrics.csv  (one row per K)
      summary.json (agg info + per-K metrics repeated)
      samples.jsonl (optional small sample of per-image metrics for diagnostics)
    recall_vs_k.png, positive_ratio_vs_k.png (plots for quick visualization)

Requires: ultralytics (for YOLO), torch, numpy
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import random

# ================== CONFIG (edit here) ==================
VERSION = "v2"
SPACE = "clips"  # subfolder under yolo_labels_540 for GT ("clips" or "videos")
FRAMES_ROOT = Path("local_extraction") / VERSION / "extracted_frames"
LABELS_ROOT = Path("local_extraction") / VERSION / "yolo_labels_540"
K_VALUES = [4, 6, 8, 10, 12, 15]
IOU_THRESHOLD = 0.50
# Matching behavior
CLASS_MATCH_REQUIRED = False   # if False -> IoU-only matching (ignore classes)
ENABLE_CLASS_MAPPING = False  # if True -> remap predicted classes using CLASS_MAPPING before matching
# Map predicted model class IDs -> ground-truth class IDs. Leave empty by default.
# Example (COCO bottle=39 -> nouns bottle=0): {39: 0}
CLASS_MAPPING: Dict[int, int] = {}

# Progress bar for sweep
PRINT_PROGRESS = True
F_BETA = 2.0  # F-beta tradeoff (precision proxy = positive ratio)
LAST_FRAME_ONLY = True   # True: use only last frame per UID; False: all frames
MAX_IMAGES = None          # cap number of images processed (None for all)
DEMO_MODE = False         # if True use DEMO_N images regardless of MAX_IMAGES
DEMO_N = 50
YOLO_WEIGHTS = "D:\\Thesis\\Ego4d-LiteSTA\\local_extraction\\toolkit_yolo\\runs\\sta_yolov8s_singlecls_20251113_002330\\weights\\best.pt"  # initial weights (must exist or be resolvable)
YOLO_IMGSZ = 960
YOLO_CONF = 0.05
YOLO_IOU = 0.45
DRY_RUN = False           # print config and exit
SAVE_PER_IMAGE_SAMPLE = True  # write limited per-image diagnostics (first S images)
SAMPLE_LIMIT = 100
OUTPUT_BASE = Path("local_extraction") / "runs" / "Track_A" / "KSweep"

# --- New qualitative & diagnostic configs ---
ENABLE_PRED_STATS = True              # collect per-image raw prediction counts and confidence stats
ENABLE_QUALITATIVE = True             # generate qualitative overlays
QUAL_TOP_K = max(K_VALUES)            # K used for qualitative TP/FP/FN visualization
QUAL_SAMPLES = 20                     # number of frames to sample for overlays
QUAL_RANDOM_SEED = 42                 # seed for reproducible sampling
SMALL_AREA_THRESH = 0.005             # normalized area threshold to flag small-object misses
QUAL_OUT_DIR_NAME = "qualitative"     # subdir inside run dir
CONF_HIST_BINS = 40                   # bins for confidence histogram
# =======================================================

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_BASE / f"StageA_KSweep_{STAMP}"

# --------------- Utilities ---------------

def list_images(root: Path) -> List[Path]:
    imgs: List[Path] = []
    for uid_dir in sorted(root.iterdir()):
        if not uid_dir.is_dir():
            continue
        frames = sorted([p for p in uid_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])
        if not frames:
            continue
        if LAST_FRAME_ONLY:
            imgs.append(frames[-1])
        else:
            imgs.extend(frames)
    return imgs


def load_gt_boxes(image_path: Path, labels_root: Path, space: str) -> List[Dict]:
    """Load YOLO ground truth boxes normalized (xyxy in 0-1 coords)."""
    uid = image_path.parent.name
    try:
        frame_idx = int(image_path.stem)
    except Exception:
        return []
    lbl_path = labels_root / space / uid / f"{frame_idx:07d}.txt"
    if not lbl_path.exists():
        return []
    out: List[Dict] = []
    for ln in lbl_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
        except Exception:
            continue
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        # clip to [0,1]
        x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def iou(b1: Dict, b2: Dict) -> float:
    xa = max(b1["x1"], b2["x1"])
    ya = max(b1["y1"], b2["y1"])
    xb = min(b1["x2"], b2["x2"])
    yb = min(b1["y2"], b2["y2"])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0.0:
        return 0.0
    a = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    b = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    union = a + b - inter
    return inter / union if union > 0.0 else 0.0


_YOLO = None

def ensure_model():
    global _YOLO
    if _YOLO is not None:
        return _YOLO
    from ultralytics import YOLO  # type: ignore
    _YOLO = YOLO(YOLO_WEIGHTS)
    return _YOLO


def yolo_predict_normalized(image_path: Path) -> List[Dict]:
    model = ensure_model()
    res = model.predict(source=str(image_path), imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
    out: List[Dict] = []
    for r in res:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        b = r.boxes
        xyxyn = b.xyxyn.cpu().numpy() if hasattr(b, "xyxyn") else []
        confs = b.conf.cpu().numpy() if hasattr(b, "conf") else []
        clss = b.cls.cpu().numpy() if hasattr(b, "cls") else []
        n = min(len(xyxyn), len(confs), len(clss))
        for i in range(n):
            x1, y1, x2, y2 = map(float, xyxyn[i].tolist())
            pred_cls = int(clss[i])
            if ENABLE_CLASS_MAPPING:
                # If mapping enabled and class not mapped, mark as unmapped (-1) so it won't match when class is required
                pred_cls = CLASS_MAPPING.get(pred_cls, -1)
            out.append({"cls": pred_cls, "conf": float(confs[i]), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    # Sort by confidence desc
    out.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
    return out


def match_top_k(preds: List[Dict], gts: List[Dict], K: int, iou_thr: float) -> Tuple[int, int]:
    """Return (tp, gt_count) for top-K predictions using greedy matching."""
    if not gts:
        return 0, 0
    used = [False] * len(gts)
    tp = 0
    top = preds[:K]
    for p in top:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            if CLASS_MATCH_REQUIRED and (g["cls"] != p["cls"]):
                continue
            ii = iou(p, g)
            if ii >= iou_thr and ii > best_iou:
                best_iou = ii
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    return tp, len(gts)


def sweep(images: List[Path]) -> Dict:
    metrics_rows = []
    per_image_samples = []
    # Diagnostics: prediction counts and confidence collection
    per_image_pred_counts: List[int] = []
    all_confidences: List[float] = []
    total_gt_per_k = {K: 0 for K in K_VALUES}
    total_tp_per_k = {K: 0 for K in K_VALUES}
    macro_recall_sums = {K: 0.0 for K in K_VALUES}  # sum of per-image recall (images with GT)
    macro_recall_counts = {K: 0 for K in K_VALUES}
    macro_posratio_sums = {K: 0.0 for K in K_VALUES}  # all images
    macro_posratio_counts = {K: 0 for K in K_VALUES}
    total_K_used = {K: 0 for K in K_VALUES}

    # Optional progress bar
    pbar = None
    HAS_TQDM = False
    if PRINT_PROGRESS:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(images), unit="img", desc="SweepK", dynamic_ncols=True)
            HAS_TQDM = True
        except Exception:
            HAS_TQDM = False

    for idx, img in enumerate(images, 1):
        gts = load_gt_boxes(img, LABELS_ROOT, SPACE)
        preds = yolo_predict_normalized(img)
        if ENABLE_PRED_STATS:
            per_image_pred_counts.append(len(preds))
            all_confidences.extend([p.get("conf", 0.0) for p in preds])
        # diagnostic sample list
        if SAVE_PER_IMAGE_SAMPLE and idx <= SAMPLE_LIMIT:
            per_image_samples.append({"image": str(img), "gt": len(gts), "preds": len(preds)})
        for K in K_VALUES:
            tp, gt_count = match_top_k(preds, gts, K, IOU_THRESHOLD)
            if gt_count > 0:
                macro_recall_sums[K] += tp / gt_count
                macro_recall_counts[K] += 1
                total_gt_per_k[K] += gt_count
            total_tp_per_k[K] += tp
            # positive ratio uses K or number of preds if fewer
            used_k = min(K, len(preds))
            pos_ratio = tp / used_k if used_k > 0 else 0.0
            macro_posratio_sums[K] += pos_ratio
            macro_posratio_counts[K] += 1
            total_K_used[K] += used_k

        if HAS_TQDM and pbar is not None:
            pbar.set_postfix(gt=len(gts), preds=len(preds))
            pbar.update(1)

    prev_micro = None
    def _fbeta(p: float, r: float, beta: float) -> float:
        if p <= 0.0 and r <= 0.0:
            return 0.0
        b2 = beta * beta
        denom = b2 * p + r
        return ((1.0 + b2) * p * r / denom) if denom > 0.0 else 0.0

    for K in K_VALUES:
        micro_recall = (total_tp_per_k[K] / total_gt_per_k[K]) if total_gt_per_k[K] > 0 else 0.0
        macro_recall = (macro_recall_sums[K] / macro_recall_counts[K]) if macro_recall_counts[K] > 0 else 0.0
        micro_posratio = (total_tp_per_k[K] / total_K_used[K]) if total_K_used[K] > 0 else 0.0
        macro_posratio = (macro_posratio_sums[K] / macro_posratio_counts[K]) if macro_posratio_counts[K] > 0 else 0.0

        micro_fbeta = _fbeta(micro_posratio, micro_recall, F_BETA)
        macro_fbeta = _fbeta(macro_posratio, macro_recall, F_BETA)
        delta_micro = (micro_recall - prev_micro) if prev_micro is not None else 0.0
        prev_micro = micro_recall

        metrics_rows.append({
            "K": K,
            "images_with_gt": macro_recall_counts[K],
            "total_gt": total_gt_per_k[K],
            "total_tp": total_tp_per_k[K],
            "micro_recall": round(micro_recall, 6),
            "macro_recall": round(macro_recall, 6),
            "micro_positive_ratio": round(micro_posratio, 6),
            "macro_positive_ratio": round(macro_posratio, 6),
            "micro_fbeta": round(micro_fbeta, 6),
            "macro_fbeta": round(macro_fbeta, 6),
            "delta_micro_recall": round(delta_micro, 6),
        })
    if PRINT_PROGRESS and HAS_TQDM and pbar is not None:
        pbar.close()
    return {
        "rows": metrics_rows,
        "samples": per_image_samples,
        "pred_counts": per_image_pred_counts,
        "confidences": all_confidences,
    }


def main() -> int:
    print("[sweepK] Configuration:")
    for k,v in sorted(globals().items()):
        if k.isupper():
            print(f"  {k}: {v}")
    if DRY_RUN:
        print("[sweepK] DRY_RUN=True -> exiting before inference.")
        return 0
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    images = list_images(FRAMES_ROOT)
    if not images:
        print("[sweepK] No images found under:", FRAMES_ROOT)
        return 2
    if MAX_IMAGES is not None:
        images = images[:int(MAX_IMAGES)]
    if DEMO_MODE:
        images = images[:int(DEMO_N)]
    print(f"[sweepK] Total images selected: {len(images)}")
    t0 = time.time()
    # Select qualitative sample set (indices) BEFORE sweep for reproducibility
    if ENABLE_QUALITATIVE:
        random.seed(QUAL_RANDOM_SEED)
        if len(images) <= QUAL_SAMPLES:
            qual_indices = list(range(len(images)))
        else:
            qual_indices = sorted(random.sample(range(len(images)), QUAL_SAMPLES))
    else:
        qual_indices = []

    data = sweep(images)
    t1 = time.time()
    # Write CSV
    csv_path = RUN_DIR / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as cf:
        cf.write("K,images_with_gt,total_gt,total_tp,micro_recall,macro_recall,micro_positive_ratio,macro_positive_ratio,micro_fbeta,macro_fbeta,delta_micro_recall\n")
        for row in data["rows"]:
            cf.write(
                f"{row['K']},{row['images_with_gt']},{row['total_gt']},{row['total_tp']},{row['micro_recall']},{row['macro_recall']},{row['micro_positive_ratio']},{row['macro_positive_ratio']},{row['micro_fbeta']},{row['macro_fbeta']},{row['delta_micro_recall']}\n"
            )
    # Summary JSON (augment with prediction stats if enabled)
    if ENABLE_PRED_STATS:
        pred_counts = data.get("pred_counts", [])
        confs = data.get("confidences", [])
        import statistics
        pred_stats = {
            "num_images": len(pred_counts),
            "total_predictions": sum(pred_counts),
            "mean_preds_per_image": round(statistics.mean(pred_counts), 4) if pred_counts else 0.0,
            "median_preds_per_image": round(statistics.median(pred_counts), 4) if pred_counts else 0.0,
            "p95_preds_per_image": sorted(pred_counts)[int(0.95 * (len(pred_counts)-1))] if pred_counts else 0,
        }
        conf_stats = {
            "num_confidences": len(confs),
            "mean_conf": round(statistics.mean(confs), 4) if confs else 0.0,
            "median_conf": round(statistics.median(confs), 4) if confs else 0.0,
            "p90_conf": sorted(confs)[int(0.90 * (len(confs)-1))] if confs else 0.0,
            "p95_conf": sorted(confs)[int(0.95 * (len(confs)-1))] if confs else 0.0,
            "max_conf": round(max(confs),4) if confs else 0.0,
        }
    else:
        pred_stats = {}
        conf_stats = {}
    summary = {
        "version": VERSION,
        "space": SPACE,
        "frames_root": str(FRAMES_ROOT),
        "labels_root": str(LABELS_ROOT),
        "k_values": K_VALUES,
        "iou_threshold": IOU_THRESHOLD,
        "last_frame_only": LAST_FRAME_ONLY,
        "max_images": MAX_IMAGES,
        "demo_mode": DEMO_MODE,
        "yolo_weights": YOLO_WEIGHTS,
        "yolo_imgsz": YOLO_IMGSZ,
        "yolo_conf": YOLO_CONF,
        "yolo_iou": YOLO_IOU,
        "duration_sec": round(t1 - t0, 2),
        "metrics": data["rows"],
        "run_dir": str(RUN_DIR),
        "prediction_stats": pred_stats,
        "confidence_stats": conf_stats,
        "qualitative_sample_indices": qual_indices,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if SAVE_PER_IMAGE_SAMPLE and data["samples"]:
        samples_path = RUN_DIR / "samples.jsonl"
        with open(samples_path, "w", encoding="utf-8") as sf:
            for s in data["samples"]:
                sf.write(json.dumps(s) + "\n")
    # Plots
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import Rectangle  # type: ignore
        import numpy as np  # type: ignore
        Ks = [int(r["K"]) for r in data["rows"]]
        micro_rec = [float(r["micro_recall"]) for r in data["rows"]]
        macro_rec = [float(r["macro_recall"]) for r in data["rows"]]
        micro_pr = [float(r["micro_positive_ratio"]) for r in data["rows"]]
        macro_pr = [float(r["macro_positive_ratio"]) for r in data["rows"]]

        # Recall plot
        plt.figure(figsize=(6,4))
        plt.plot(Ks, micro_rec, marker='o', label='micro_recall')
        plt.plot(Ks, macro_rec, marker='o', label='macro_recall')
        plt.xlabel('K')
        plt.ylabel('Recall')
        plt.title('Recall vs K')
        plt.grid(True, alpha=0.3)
        plt.legend()
        recall_plot = RUN_DIR / 'recall_vs_k.png'
        plt.tight_layout(); plt.savefig(recall_plot); plt.close()

        # Positive ratio plot
        plt.figure(figsize=(6,4))
        plt.plot(Ks, micro_pr, marker='o', label='micro_positive_ratio')
        plt.plot(Ks, macro_pr, marker='o', label='macro_positive_ratio')
        plt.xlabel('K')
        plt.ylabel('Positive Ratio (Precision proxy)')
        plt.title('Positive Ratio vs K')
        plt.grid(True, alpha=0.3)
        plt.legend()
        pr_plot = RUN_DIR / 'positive_ratio_vs_k.png'
        plt.tight_layout(); plt.savefig(pr_plot); plt.close()

        # Confidence histogram and preds-per-image histogram
        if ENABLE_PRED_STATS and data.get("confidences") is not None:
            confs = data.get("confidences", [])
            if len(confs) > 0:
                plt.figure(figsize=(6,4))
                plt.hist(confs, bins=CONF_HIST_BINS, range=(0.0, 1.0), color='#4e79a7', alpha=0.8)
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.title('Prediction Confidence Histogram')
                plt.grid(True, alpha=0.3)
                plt.tight_layout(); plt.savefig(RUN_DIR / 'confidence_hist.png'); plt.close()
        if ENABLE_PRED_STATS and data.get("pred_counts") is not None:
            counts = data.get("pred_counts", [])
            if len(counts) > 0:
                plt.figure(figsize=(6,4))
                plt.hist(counts, bins=30, color='#f28e2b', alpha=0.8)
                plt.xlabel('# predictions per image (pre-K)')
                plt.ylabel('Image count')
                plt.title('Predictions per Image')
                plt.grid(True, alpha=0.3)
                plt.tight_layout(); plt.savefig(RUN_DIR / 'preds_per_image_hist.png'); plt.close()

        # Qualitative overlays
        if ENABLE_QUALITATIVE and qual_indices:
            qual_dir = RUN_DIR / QUAL_OUT_DIR_NAME
            qual_dir.mkdir(parents=True, exist_ok=True)
            for qi in qual_indices:
                if qi >= len(images):
                    continue
                img_path = images[qi]
                try:
                    import cv2  # type: ignore
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        continue
                    h, w = img_bgr.shape[:2]
                    gts = load_gt_boxes(img_path, LABELS_ROOT, SPACE)
                    preds_full = yolo_predict_normalized(img_path)
                    top_preds = preds_full[:QUAL_TOP_K]
                    # Determine TP/FP and collect FN
                    matched_gt = [False]*len(gts)
                    for p in top_preds:
                        best = -1; best_iou = 0.0
                        for gi, g in enumerate(gts):
                            if matched_gt[gi]:
                                continue
                            if CLASS_MATCH_REQUIRED and g['cls'] != p['cls']:
                                continue
                            ii = iou(p,g)
                            if ii >= IOU_THRESHOLD and ii > best_iou:
                                best_iou = ii; best = gi
                        if best >= 0:
                            matched_gt[best] = True
                            p['_match'] = True
                        else:
                            p['_match'] = False
                    # Draw
                    canvas = img_bgr.copy()
                    def to_px(box):
                        return int(box['x1']*w), int(box['y1']*h), int((box['x2']-box['x1'])*w), int((box['y2']-box['y1'])*h)
                    # GT boxes: green if matched later else red stroke (or yellow if small and missed)
                    for gi,g in enumerate(gts):
                        x,y,ww,hh = to_px(g)
                        area_norm = (g['x2']-g['x1'])*(g['y2']-g['y1'])
                        missed = not matched_gt[gi]
                        small = area_norm < SMALL_AREA_THRESH
                        if missed and small:
                            color = (0,255,255)  # yellow for small missed
                        elif missed:
                            color = (0,0,255)    # red for missed
                        else:
                            color = (0,255,0)    # green for matched
                        cv2.rectangle(canvas, (x,y), (x+ww, y+hh), color, 2)
                    # Predictions: TP blue, FP magenta
                    for p in top_preds:
                        x,y,ww,hh = to_px(p)
                        if p.get('_match'):
                            color = (255,0,0)   # blue-ish (BGR)
                        else:
                            color = (255,0,255) # magenta
                        cv2.rectangle(canvas,(x,y),(x+ww,y+hh),color,1)
                        cv2.putText(canvas, f"{p.get('conf',0):.2f}", (x,y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    # Legend bar
                    legend = [
                        ('GT matched','green'),('GT missed','red'),('GT small+miss','yellow'),('Pred TP','blue'),('Pred FP','magenta')
                    ]
                    # Convert color names to BGR
                    name_to_bgr = {'green':(0,255,0),'red':(0,0,255),'yellow':(0,255,255),'blue':(255,0,0),'magenta':(255,0,255)}
                    y0 = 15
                    for txt,colname in legend:
                        col = name_to_bgr[colname]
                        cv2.putText(canvas, txt, (5,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
                        y0 += 15
                    out_name = qual_dir / f"qual_{qi:04d}.jpg"
                    cv2.imwrite(str(out_name), canvas)
                except Exception as e:
                    print(f"[sweepK] Qualitative sample failed for {img_path}: {e}")
    except Exception as e:
        print(f"[sweepK] Plotting skipped: {e}")

    print("[sweepK] Done. Outputs:")
    print("         ", csv_path)
    print("         ", RUN_DIR / "summary.json")
    if (RUN_DIR / 'recall_vs_k.png').exists():
        print("         ", RUN_DIR / 'recall_vs_k.png')
    if (RUN_DIR / 'positive_ratio_vs_k.png').exists():
        print("         ", RUN_DIR / 'positive_ratio_vs_k.png')
    if (RUN_DIR / 'confidence_hist.png').exists():
        print("         ", RUN_DIR / 'confidence_hist.png')
    if (RUN_DIR / 'preds_per_image_hist.png').exists():
        print("         ", RUN_DIR / 'preds_per_image_hist.png')
    if (RUN_DIR / QUAL_OUT_DIR_NAME).exists():
        print("         ", RUN_DIR / QUAL_OUT_DIR_NAME)
    # Write per-image prediction counts CSV
    if ENABLE_PRED_STATS and data.get("pred_counts"):
        with open(RUN_DIR / "preds_per_image.csv", "w", encoding="utf-8") as pf:
            pf.write("image_index,pred_count\n")
            for i,c in enumerate(data.get("pred_counts", [])):
                pf.write(f"{i},{c}\n")
        print("         ", RUN_DIR / "preds_per_image.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
