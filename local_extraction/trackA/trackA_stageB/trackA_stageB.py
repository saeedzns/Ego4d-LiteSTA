#!/usr/bin/env python3
"""
Track A - Stage B (Head/Reasoner) — local-only prep runner

This script prepares Stage-B inputs from Stage-A candidates:
    - Reads Stage-A candidates.jsonl (uid, frame, boxes)
    - Loads the corresponding frame images
    - Crops ROIs for each candidate box (optionally resizes)
    - Writes crops to a run folder and emits a manifest (CSV + JSONL)

Extras:
    - If EVAL_WITH_LABELS=True, reads local YOLO labels to compute recall@K
        recall@K = (# images with >=1 hit) / (# images with >=1 GT)
        Also logs mean best IoU and a coarse histogram.
    - If head manifests are present, writes consolidated head_train.jsonl/head_val.jsonl
        merging candidate proposals with verb/noun/ttc, and an is_positive flag.

Configuration is loaded from configs/trackA.yaml
"""

from __future__ import annotations

import json
import math
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ================== YAML CONFIG LOADING ==================
_THIS_DIR = Path(__file__).resolve().parent
_TRACK_A_DIR = _THIS_DIR.parent
_LOCAL_EXTRACTION = _TRACK_A_DIR.parent
_REPO_ROOT = _LOCAL_EXTRACTION.parent

for p in [str(_LOCAL_EXTRACTION), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core import load_config

# Load configuration from YAML
_cfg = load_config('trackA')

# ================== CONFIG VALUES (from YAML) ==================
VERSION = _cfg.get('version', 'v2')

# Where frames live (input)
FRAMES_ROOT = Path(_cfg.get('paths.extracted_frames', f'local_extraction/{VERSION}/extracted_frames'))

# Where labels live for optional evaluation
LABELS_ROOT = Path(_cfg.get('paths.yolo_labels', f'local_extraction/{VERSION}/yolo_labels_540'))
LABEL_SPACE = _cfg.get('stage_a.oracle.label_space', 'clips')
EVAL_WITH_LABELS = _cfg.get('stage_b.eval_with_labels', True)
IOU_THRESH = _cfg.get('stage_b.iou_thresh', 0.5)

# Candidates source (input) - null means auto-detect
_cand_src = _cfg.get('stage_b.candidates_source', None)
CANDIDATES_JSONL: Optional[Path] = Path(_cand_src) if _cand_src else None

# Cropping/resizing
KEEP_TOP_N = _cfg.get('stage_b.keep_top_n', None)
_crop_size = _cfg.get('stage_b.crop_size', [256, 256])
CROP_SIZE = tuple(_crop_size) if _crop_size else None

# Limits/progress
MAX_IMAGES = _cfg.get('runtime.max_images', None)
DEMO_MODE = _cfg.get('demo.enabled', False)
DEMO_N = _cfg.get('demo.max_samples', 100)
PRINT_PROGRESS = _cfg.get('runtime.print_progress', True)

# Run folder
FILE_PREFIX = _cfg.get('stage_b.output.run_prefix', Path(__file__).stem)
RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = Path(_cfg.get('paths.runs', 'local_extraction/runs')) / "Track_A" / f"{FILE_PREFIX}_{RUN_STAMP}"

# Head manifests (optional semantic labels for head training)
MANIFESTS_DIR = Path(_cfg.get('paths.manifests', f'local_extraction/{VERSION}/manifests'))
USE_CLIP_MANIFEST = _cfg.get('stage_b.manifests.use_clip_manifest', True)
HEAD_TRAIN_MANIFEST = Path(_cfg.get('stage_b.manifests.train', 
    str(MANIFESTS_DIR / ("head_train_clip.json" if USE_CLIP_MANIFEST else "head_train_video.json"))))
HEAD_VAL_MANIFEST = Path(_cfg.get('stage_b.manifests.val',
    str(MANIFESTS_DIR / ("head_val_clip.json" if USE_CLIP_MANIFEST else "head_val_video.json"))))

# Consolidated head outputs (written in RUN_DIR)
WRITE_HEAD_TRAIN_VAL = _cfg.get('stage_b.output.write_head_train_val', True)
HEAD_TRAIN_OUT = RUN_DIR / "head_train.jsonl"
HEAD_VAL_OUT = RUN_DIR / "head_val.jsonl"

# Also include semantic fields in per-crop manifest.jsonl rows when available
WRITE_SEMANTICS_IN_MANIFEST = _cfg.get('stage_b.output.write_semantics_in_manifest', True)

# Optionally also write recall metrics back into Stage-A summary.json
WRITE_BACK_TO_STAGEA_SUMMARY = _cfg.get('stage_b.output.write_back_to_stagea_summary', True)
# =========================================================


# -------- Environment overrides (optional, used by smoke tests) --------
_env_max = os.environ.get("STAGEB_MAX_IMAGES")
if _env_max:
    try:
        MAX_IMAGES = int(_env_max)
    except Exception:
        pass


def clamp_box(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(W - 1, int(math.floor(x1))))
    y1 = max(0, min(H - 1, int(math.floor(y1))))
    x2 = max(0, min(W - 1, int(math.ceil(x2))))
    y2 = max(0, min(H - 1, int(math.ceil(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(str(path)) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        pass
    try:
        import imageio.v2 as imageio  # type: ignore
        im = imageio.imread(str(path))
        if hasattr(im, "shape") and len(im.shape) >= 2:
            h, w = int(im.shape[0]), int(im.shape[1])
            return w, h
    except Exception:
        pass
    return 0, 0


def iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = areaA + areaB - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def read_yolo_labels(uid: str, frame: int) -> List[Tuple[int, float, float, float, float]]:
    txt = LABELS_ROOT / LABEL_SPACE / uid / f"{frame:07d}.txt"
    if not txt.exists():
        return []
    lines: List[Tuple[int, float, float, float, float]] = []
    try:
        for ln in txt.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            if len(p) != 5:
                continue
            cls = int(float(p[0])); cx = float(p[1]); cy = float(p[2]); ww = float(p[3]); hh = float(p[4])
            lines.append((cls, cx, cy, ww, hh))
    except Exception:
        pass
    return lines


def load_head_manifest(path: Path) -> Dict[Tuple[str, int], Dict]:
    """Load head manifest JSON (list of dicts) and index by (uid, frame).
    Each item is expected to have keys: image, gt_box, verb_id, noun_id, ttc.
    We parse the image path: .../<uid>/<frame:07d>.jpg
    """
    out: Dict[Tuple[str, int], Dict] = {}
    if not path.exists():
        return out
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            return out
        for item in data:
            try:
                img = str(item.get("image", ""))
                p = Path(img)
                uid = p.parent.name
                frame = int(Path(img).stem)
                gt = {
                    "gt_box": item.get("gt_box"),
                    "verb_id": item.get("verb_id"),
                    "noun_id": item.get("noun_id"),
                    "ttc": item.get("ttc"),
                    "image": img,
                }
                out[(uid, frame)] = gt
            except Exception:
                continue
    except Exception:
        # malformed or empty file -> ignore
        pass
    return out


def latest_candidates() -> Optional[Path]:
    root = Path("local_extraction") / "runs" / "Track_A"
    if not root.exists():
        return None
    best: Optional[Path] = None
    best_mtime = -1.0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        cand = d / "candidates.jsonl"
        if cand.exists():
            mt = cand.stat().st_mtime
            if mt > best_mtime:
                best_mtime, best = mt, cand
    return best


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    crops_dir = RUN_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Resolve candidates path
    cand_path = CANDIDATES_JSONL or latest_candidates()
    if not cand_path or not Path(cand_path).exists():
        print("[stageB] No candidates.jsonl found. Set CANDIDATES_JSONL or run Stage A first.")
        return 2
    print("[stageB] Using candidates:", cand_path)

    # Count images (rough)
    try:
        total_images = sum(1 for _ in Path(cand_path).read_text().splitlines())
    except Exception:
        total_images = None  # unknown

    # Load head manifests if present
    head_train_map = load_head_manifest(HEAD_TRAIN_MANIFEST)
    head_val_map = load_head_manifest(HEAD_VAL_MANIFEST)
    if head_train_map or head_val_map:
        print(f"[stageB] Loaded head manifests: train={len(head_train_map)}, val={len(head_val_map)}")

    # Consolidated head outputs
    ht = open(HEAD_TRAIN_OUT, "w", encoding="utf-8") if WRITE_HEAD_TRAIN_VAL else None
    hv = open(HEAD_VAL_OUT, "w", encoding="utf-8") if WRITE_HEAD_TRAIN_VAL else None
    n_head_train_rows = 0
    n_head_val_rows = 0

    # Progress bar
    pbar = None
    HAS_TQDM = False
    if PRINT_PROGRESS:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total_images if total_images else None, unit='img', desc='StageB', dynamic_ncols=True)
            HAS_TQDM = True
        except Exception:
            HAS_TQDM = False

    # Outputs
    manifest_csv = RUN_DIR / "manifest.csv"
    manifest_jsonl = RUN_DIR / "manifest.jsonl"

    # Aggregators for recall@K and IoU stats (based on YOLO labels)
    per_image_stats: Dict[Tuple[str, int], Dict[str, float | int | bool]] = {}

    n_images = 0
    n_crops = 0
    t0 = time.time()
    with open(manifest_csv, "w", encoding="utf-8") as fcsv, open(manifest_jsonl, "w", encoding="utf-8") as fjl:
        fcsv.write("uid,frame,roi_idx,crop_path,x1,y1,x2,y2,conf,cls,hit\n")
        for li, ln in enumerate(Path(cand_path).read_text().splitlines(), 1):
            if MAX_IMAGES is not None and n_images >= int(MAX_IMAGES):
                break
            if DEMO_MODE and n_images >= int(DEMO_N):
                break
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            uid = str(rec.get("uid"))
            frame = int(rec.get("frame", -1))
            boxes = rec.get("boxes") or []
            img_path = FRAMES_ROOT / uid / f"{frame:07d}.jpg"
            if not img_path.exists():
                if PRINT_PROGRESS and not HAS_TQDM:
                    print(f"\n[stageB] Missing image: {img_path}")
                if HAS_TQDM and pbar is not None:
                    pbar.set_postfix(missing=str(img_path.name))
                continue

            # Load image
            try:
                from PIL import Image  # type: ignore
                im = Image.open(str(img_path)).convert("RGB")
                W, H = im.size
            except Exception:
                W, H = image_size(img_path)
                im = None
                if W <= 0 or H <= 0:
                    continue

            # Optional ground-truth labels (YOLO txt) for recall@K
            gt_pix: List[Tuple[int, int, int, int]] = []
            if EVAL_WITH_LABELS:
                labs = read_yolo_labels(uid, frame)
                for (_cls, cx, cy, ww, hh) in labs:
                    px, py = cx * W, cy * H
                    pw, ph = ww * W, hh * H
                    gx1, gy1 = px - pw / 2.0, py - ph / 2.0
                    gx2, gy2 = px + pw / 2.0, py + ph / 2.0
                    gt_pix.append((int(gx1), int(gy1), int(gx2), int(gy2)))
                per_image_stats[(uid, frame)] = {
                    "has_gt": bool(gt_pix),
                    "best_iou": 0.0,
                    "has_hit": False,
                }

            kept = 0
            for bi, b in enumerate(boxes):
                if KEEP_TOP_N is not None and kept >= int(KEEP_TOP_N):
                    break
                x1 = float(b.get("x1", 0)); y1 = float(b.get("y1", 0))
                x2 = float(b.get("x2", 0)); y2 = float(b.get("y2", 0))
                conf = b.get("conf"); cls = b.get("cls")

                cx1, cy1, cx2, cy2 = clamp_box(x1, y1, x2, y2, W, H)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue

                # Eval hit for this ROI and update per-image stats
                hit = False
                if EVAL_WITH_LABELS and gt_pix:
                    best = 0.0
                    for g in gt_pix:
                        ii = iou((cx1, cy1, cx2, cy2), g)
                        if ii >= IOU_THRESH:
                            hit = True
                        if ii > best:
                            best = ii
                    st = per_image_stats.get((uid, frame))
                    if st is not None:
                        if best > float(st["best_iou"]):
                            st["best_iou"] = float(best)
                        if hit:
                            st["has_hit"] = True

                # Crop and save
                crop = None
                crop_rel = Path("crops") / uid / f"{frame:07d}_{kept:02d}.jpg"
                crop_abs = RUN_DIR / crop_rel
                crop_abs.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if im is None:
                        from PIL import Image  # type: ignore
                        im = Image.open(str(img_path)).convert("RGB")
                    roi = im.crop((cx1, cy1, cx2, cy2))
                    if CROP_SIZE is not None:
                        roi = roi.resize(CROP_SIZE, Image.BILINEAR)
                    roi.save(str(crop_abs), quality=90)
                    crop = str(crop_rel)
                except Exception:
                    crop = None

                kept += 1
                n_crops += 1

                # If head manifests are present, attach semantics and write consolidated rows
                sem_verb = None
                sem_noun = None
                sem_ttc = None
                sem_is_pos = None
                sem_iou = None
                if WRITE_HEAD_TRAIN_VAL and (head_train_map or head_val_map):
                    key = (uid, frame)
                    gt_entry = head_train_map.get(key) or head_val_map.get(key)
                    if gt_entry and isinstance(gt_entry.get("gt_box"), (list, tuple)) and len(gt_entry.get("gt_box")) == 4:
                        gx1, gy1, gx2, gy2 = map(float, gt_entry["gt_box"])
                        pos_iou = iou((cx1, cy1, cx2, cy2), (gx1, gy1, gx2, gy2))
                        is_pos = pos_iou >= IOU_THRESH
                        # capture semantics for manifest row
                        sem_verb = gt_entry.get("verb_id")
                        sem_noun = gt_entry.get("noun_id")
                        sem_ttc = gt_entry.get("ttc")
                        sem_is_pos = bool(is_pos)
                        sem_iou = float(pos_iou)
                        out_line = {
                            "image_path": str(img_path),
                            "candidate_box": [cx1, cy1, cx2, cy2],
                            "candidate_conf": None if conf is None else float(conf),
                            "candidate_cls": None if cls is None else int(cls),
                            "verb_id": gt_entry.get("verb_id"),
                            "noun_id": gt_entry.get("noun_id"),
                            "ttc": gt_entry.get("ttc"),
                            "is_positive": bool(is_pos),
                            "iou": float(pos_iou),
                        }
                        if key in head_train_map and ht is not None:
                            ht.write(json.dumps(out_line) + "\n")
                            n_head_train_rows += 1
                        elif key in head_val_map and hv is not None:
                            hv.write(json.dumps(out_line) + "\n")
                            n_head_val_rows += 1

                # Write manifest rows
                row = {
                    "uid": uid,
                    "frame": frame,
                    "roi_idx": kept - 1,
                    "crop_path": crop,
                    "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2,
                    "conf": conf, "cls": cls,
                    "hit": bool(hit),
                }
                if WRITE_SEMANTICS_IN_MANIFEST and sem_is_pos is not None:
                    row.update({
                        "verb_id": sem_verb,
                        "noun_id": sem_noun,
                        "ttc": sem_ttc,
                        "is_positive": sem_is_pos,
                        "iou": sem_iou,
                    })
                fjl.write(json.dumps(row) + "\n")
                fcsv.write(f"{uid},{frame},{kept - 1},{crop},{cx1},{cy1},{cx2},{cy2},{conf},{cls},{int(hit)}\n")

            n_images += 1
            if PRINT_PROGRESS:
                if HAS_TQDM and pbar is not None:
                    pbar.set_postfix(crops=n_crops)
                    pbar.update(1)
                else:
                    if n_images % 20 == 0:
                        print(f"[stageB] {n_images} images | crops={n_crops}", end="\r", flush=True)

    if PRINT_PROGRESS and not HAS_TQDM:
        print()
    if PRINT_PROGRESS and HAS_TQDM and pbar is not None:
        pbar.close()

    # Compute recall@K and IoU distribution (based on YOLO labels)
    recall_metrics = None
    if EVAL_WITH_LABELS and per_image_stats:
        imgs_with_gt = sum(1 for v in per_image_stats.values() if v.get("has_gt"))
        imgs_with_hit = sum(1 for v in per_image_stats.values() if v.get("has_gt") and v.get("has_hit"))
        recall_at_k = (imgs_with_hit / imgs_with_gt) if imgs_with_gt > 0 else 0.0
        best_ious = [float(v.get("best_iou", 0.0)) for v in per_image_stats.values() if v.get("has_gt")]
        mean_best_iou = (sum(best_ious) / len(best_ious)) if best_ious else 0.0
        bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
        hist = {f"{bins[i]}-{bins[i+1]}": 0 for i in range(len(bins)-1)}
        for x in best_ious:
            for i in range(len(bins)-1):
                if bins[i] <= x < bins[i+1]:
                    hist[f"{bins[i]}-{bins[i+1]}"] += 1
                    break
        recall_metrics = {
            "images_with_gt": imgs_with_gt,
            "images_with_hit": imgs_with_hit,
            "recall_at_K": round(recall_at_k, 6),
            "mean_best_iou": round(mean_best_iou, 6),
            "best_iou_hist": hist,
        }

    # Summary
    summary = {
        "images": n_images,
        "crops": n_crops,
        "frames_root": str(FRAMES_ROOT),
        "labels_root": str(LABELS_ROOT),
        "candidates": str(cand_path),
        "run_dir": str(RUN_DIR),
        "file_prefix": FILE_PREFIX,
        "run_stamp": RUN_STAMP,
        "eval_with_labels": bool(EVAL_WITH_LABELS),
        "iou_thresh": IOU_THRESH,
        "config_name": "trackA",
        "yaml_config_flat": _cfg.flat(),
        "head_manifests": {
            "train_path": str(HEAD_TRAIN_MANIFEST),
            "val_path": str(HEAD_VAL_MANIFEST),
            "train_entries": len(head_train_map),
            "val_entries": len(head_val_map),
            "train_out": str(HEAD_TRAIN_OUT) if WRITE_HEAD_TRAIN_VAL else None,
            "val_out": str(HEAD_VAL_OUT) if WRITE_HEAD_TRAIN_VAL else None,
            "head_train_rows_written": n_head_train_rows,
            "head_val_rows_written": n_head_val_rows,
        },
    }
    # Try to propagate Stage A detector config by reading sibling summary.json next to candidates
    try:
        sa_sum_path = Path(str(cand_path)).parent / "summary.json"
        if sa_sum_path.exists():
            sa = json.loads(sa_sum_path.read_text())
            detector_info = {}
            if sa.get("mode") == "yolo" and isinstance(sa.get("yolo_config"), dict):
                detector_info = {
                    "mode": sa.get("mode"),
                    "K": sa.get("K"),
                    "yolo_config": sa.get("yolo_config"),
                }
            else:
                detector_info = {
                    "mode": sa.get("mode"),
                    "K": sa.get("K"),
                }
            summary["detector"] = detector_info
    except Exception:
        pass
    if recall_metrics is not None:
        summary["recall_metrics"] = recall_metrics

    # Save full resolved config snapshot alongside summary for provenance
    try:
        (RUN_DIR / "resolved_config.json").write_text(
            json.dumps(
                {
                    'config_name': 'trackA',
                    'config_resolved': _cfg.to_dict(),
                    'config_flat': _cfg.flat(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Optionally write recall metrics back into Stage-A summary.json
    try:
        if WRITE_BACK_TO_STAGEA_SUMMARY and recall_metrics is not None:
            sa_sum_path = Path(str(cand_path)).parent / "summary.json"
            if sa_sum_path.exists():
                sa = json.loads(sa_sum_path.read_text())
                sa.setdefault("stageB_metrics", {})
                sa["stageB_metrics"].update(recall_metrics)
                sa_sum_path.write_text(json.dumps(sa, indent=2), encoding="utf-8")
                # Use ASCII arrow for Windows console compatibility
                print("[stageB] Wrote recall metrics into Stage-A summary ->", sa_sum_path)
    except Exception as e:
        print("[stageB] Could not update Stage-A summary:", e)

    # ASCII arrow for Windows console compatibility
    print("[stageB] Done. Outputs ->", RUN_DIR)
    
    # Log run using RunLogger
    try:
        from core import RunLogger
        run_logger = RunLogger(track='trackA', stage='stageB', run_dir=RUN_DIR)
        run_logger.log_config(summary)
        
        metrics = {
            'images': n_images,
            'crops': n_crops,
            'head_train_rows': n_head_train_rows,
            'head_val_rows': n_head_val_rows,
        }
        if recall_metrics:
            metrics.update(recall_metrics)
        run_logger.log_metrics(metrics)
        
        artifacts = [str(RUN_DIR / "summary.json"), str(RUN_DIR / "resolved_config.json")]
        if WRITE_HEAD_TRAIN_VAL:
            artifacts.extend([str(HEAD_TRAIN_OUT), str(HEAD_VAL_OUT)])
        run_logger.log_artifacts(artifacts)
        run_logger.log_end(success=True)
        run_logger.print_summary()
    except Exception as e:
        print(f"[stageB] WARNING: failed to write run log: {e}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

