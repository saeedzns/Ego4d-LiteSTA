#!/usr/bin/env python3
"""
Starter: quick STA sanity — local-only.

What it does (no CLI needed; edit toggles below):
1) Loads STA annotations from Drive, filters to val split, and picks one sample
   (by clip or video). Visualizes the last frame with GT boxes overlaid and
   saves to local_extraction/runs/<RUN_NAME>/.
2) Prepares a tiny subset (val) and runs a quick evaluation:
   - If EVAL_OFFICIAL_SCRIPT points to an evaluator, calls it on the subset
     with perfect predictions (copies GT) to verify the setup.
   - Otherwise does a built-in mini-eval (IoU stats) and saves results.

All inputs come from local or mounted paths. No downloads or extraction here.
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import subprocess


# ================== TOGGLES (edit here) ==================
RUN_NAME = "starter_check"
RUNS_DIR = Path("local_extraction") / "runs" / RUN_NAME

# Drive inputs (read-only)
EGO4D_ROOT = os.environ.get("EGO4D_ROOT", "H:/My Drive/ego4d_data")
VERSION = "v2"
USE_HEIGHT_540 = True  # match 540p scaled annotations
SPLIT = "val"  # use 'val' split to visualize/evaluate

# Local frames root (written by your extractor)
FRAMES_ROOT = Path("local_extraction") / VERSION / "extracted_frames"

# Choose space: 'clips' or 'videos'
SPACE = "clips"

# Sample choice for visualization
SAMPLE_SELECT = "auto"  # 'auto' picks a random UID from val; or set to 'by_uid'
SAMPLE_UID = ""         # if SAMPLE_SELECT == 'by_uid', put a clip_uid or video_uid here

# Evaluation subset size (keep small)
EVAL_SUBSET_N = 50

# If you have the official evaluator, set its path here (a .py script)
EVAL_OFFICIAL_SCRIPT = None  # e.g., Path("tools")/"sta_eval.py"
EVAL_OFFICIAL_EXTRA_ARGS = []  # list of extra CLI flags for the official eval

# =========================================================


def detect_drive_root() -> str:
    env = os.environ.get("EGO4D_ROOT")
    if env and os.path.isdir(env):
        return env
    for p in (
        "H:/My Drive/ego4d_data",
        "D:/My Drive/ego4d_data",
        "/mnt/h/My Drive/ego4d_data",
        "/mnt/d/My Drive/ego4d_data",
        "/content/drive/MyDrive/ego4d_data",
        "/drive/MyDrive/ego4d_data",
    ):
        try:
            if os.path.isdir(p):
                return p
        except Exception:
            pass
    return EGO4D_ROOT


def _iter_list(obj):
    if isinstance(obj, dict):
        if "annotations" in obj and isinstance(obj["annotations"], list):
            return obj["annotations"]
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
    if isinstance(obj, list):
        return obj
    return []


def _as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _load_sta(ego4d_root: str, version: str, use_540: bool) -> List[dict]:
    ann_dir = os.path.join(ego4d_root, version, "annotations_540ss" if use_540 else "annotations")
    files = [
        ("fho_sta_train_height-540.json" if use_540 else "fho_sta_train.json", "train"),
        ("fho_sta_val_height-540.json" if use_540 else "fho_sta_val.json", "val"),
    ]
    out = []
    for name, split in files:
        path = os.path.join(ann_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            raw = json.load(f)
        for rec in _iter_list(raw):
            vuid = rec.get("video_uid"); cuid = rec.get("clip_uid")
            vfr = _as_int(rec.get("frame")); cfr = _as_int(rec.get("clip_frame"))
            for obj in rec.get("objects", []) or []:
                box = obj.get("box") or obj.get("bbox") or obj.get("gt_box")
                if not (isinstance(box, (list, tuple)) and len(box) == 4):
                    continue
                out.append({
                    "split": split,
                    "video_uid": vuid,
                    "clip_uid": cuid,
                    "video_frame": vfr,
                    "clip_frame": cfr,
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "noun_id": _as_int(obj.get("noun_category_id", obj.get("noun_id", 0))) or 0,
                })
    return out


def read_image_size(img_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(str(img_path)) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        pass
    try:
        import imageio.v2 as imageio  # type: ignore
        im = imageio.imread(str(img_path))
        if hasattr(im, "shape") and len(im.shape) >= 2:
            h, w = int(im.shape[0]), int(im.shape[1])
            return w, h
    except Exception:
        pass
    # Minimal JPEG/PNG parser fallback
    try:
        with open(img_path, "rb") as f:
            head = f.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                f.seek(16); w = int.from_bytes(f.read(4), "big"); h = int.from_bytes(f.read(4), "big"); return w, h
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    if not b:
                        break
                    while b != b"\xff":
                        b = f.read(1)
                        if not b:
                            return None, None
                    while True:
                        m = f.read(1)
                        if m != b"\xff":
                            break
                    if not m:
                        break
                    marker = m[0]
                    if marker in (0xD8, 0xD9):
                        continue
                    seglen_b = f.read(2)
                    if len(seglen_b) != 2:
                        break
                    seglen = int.from_bytes(seglen_b, "big")
                    if seglen < 2:
                        return None, None
                    if (0xC0 <= marker <= 0xC3) or (0xC5 <= marker <= 0xC7) or (0xC9 <= marker <= 0xCB) or (0xCD <= marker <= 0xCF):
                        f.read(1); h = int.from_bytes(f.read(2), "big"); w = int.from_bytes(f.read(2), "big"); return w, h
                    f.seek(seglen - 2, 1)
    except Exception:
        pass
    return None, None


def visualize_last_frame_val(records: List[dict]) -> Optional[Path]:
    # Filter to val split and chosen space
    val = [r for r in records if (r.get("split") or "").lower() == SPLIT.lower()]
    if SPACE == "clips":
        by_id: Dict[str, List[dict]] = {}
        for r in val:
            cuid = r.get("clip_uid")
            if not cuid:
                continue
            by_id.setdefault(str(cuid), []).append(r)
    else:
        by_id = {}
        for r in val:
            vuid = r.get("video_uid")
            if not vuid:
                continue
            by_id.setdefault(str(vuid), []).append(r)

    if not by_id:
        print("[viz] No val records found for chosen space.")
        return None

    # Choose UID
    if SAMPLE_SELECT == "by_uid" and SAMPLE_UID:
        uid = SAMPLE_UID
        group = by_id.get(uid, [])
        if not group:
            print(f"[viz] Provided UID not found: {uid}")
            return None
    else:
        uid, group = random.choice(list(by_id.items()))

    # Find last frame for this UID
    frame_key = "clip_frame" if SPACE == "clips" else "video_frame"
    frames = [r.get(frame_key) for r in group if r.get(frame_key) is not None]
    if not frames:
        print("[viz] No frames in group.")
        return None
    last_fr = max(int(f) for f in frames)

    # Boxes at last frame
    boxes = [r["box"] for r in group if int(r.get(frame_key, -1)) == last_fr]
    img_path = FRAMES_ROOT / uid / f"{int(last_fr):07d}.jpg"
    if not img_path.exists():
        print(f"[viz] Image not found: {img_path}")
        return None
    W, H = read_image_size(img_path)
    print(f"[viz] UID={uid} frame={last_fr} size={W}x{H} boxes={len(boxes)} → {img_path}")

    # Draw
    out_dir = RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uid}_{last_fr:07d}_viz.jpg"
    try:
        from PIL import Image, ImageDraw  # type: ignore
        im = Image.open(str(img_path)).convert("RGB")
        draw = ImageDraw.Draw(im)
        for (x1, y1, x2, y2) in boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        im.save(str(out_path), quality=90)
        print(f"[viz] Saved → {out_path}")
        return out_path
    except Exception as e:
        print(f"[viz] Pillow not available or failed ({e}); skipping drawing.")
        return None


def build_subset(records: List[dict], n: int) -> Tuple[List[dict], List[dict]]:
    # Create a tiny subset of val records by image key (uid,frame)
    val = [r for r in records if (r.get("split") or "").lower() == SPLIT.lower()]
    key_name = "clip_frame" if SPACE == "clips" else "video_frame"
    id_name = "clip_uid" if SPACE == "clips" else "video_uid"
    # Group by image
    by_img: Dict[Tuple[str, int], List[dict]] = {}
    for r in val:
        uid = r.get(id_name); fr = r.get(key_name)
        if uid and fr is not None:
            by_img.setdefault((str(uid), int(fr)), []).append(r)
    imgs = list(by_img.keys())
    random.shuffle(imgs)
    imgs = imgs[: max(1, int(n))]
    gt = []
    for (uid, fr) in imgs:
        for r in by_img[(uid, fr)]:
            gt.append({
                "uid": uid,
                "frame": int(fr),
                "box": r["box"],
                "cls": int(r.get("noun_id", 0)),
            })
    # For setup verification, build predictions identical to GT
    preds = [dict(x) for x in gt]
    return gt, preds


def run_official_eval(eval_script: Path, gt_json: Path, pred_json: Path, out_json: Path) -> bool:
    try:
        args = ["python", str(eval_script), "--groundtruth", str(gt_json), "--predictions", str(pred_json), "--output", str(out_json)]
        if EVAL_OFFICIAL_EXTRA_ARGS:
            args.extend(list(map(str, EVAL_OFFICIAL_EXTRA_ARGS)))
        print("[eval] Running:", " ".join(args))
        subprocess.run(args, check=True)
        print(f"[eval] Done. Output → {out_json}")
        return True
    except Exception as e:
        print(f"[eval] Failed to run official evaluator: {e}")
        return False


def mini_eval_iou(gt: List[dict], preds: List[dict]) -> Dict[str, float]:
    # Compute simple mean IoU with 1:1 matching per image by greedy pairing (cls-agnostic)
    from math import isfinite
    # Group by (uid, frame)
    def grp(xs):
        d: Dict[Tuple[str, int], List[dict]] = {}
        for r in xs:
            d.setdefault((r["uid"], int(r["frame"])), []).append(r)
        return d
    g = grp(gt); p = grp(preds)

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return (inter / denom) if denom > 0 else 0.0

    ious: List[float] = []
    for key in g.keys():
        G = list(g.get(key, []))
        P = list(p.get(key, []))
        used = [False] * len(P)
        for r in G:
            best = 0.0; best_j = -1
            for j, q in enumerate(P):
                if used[j]:
                    continue
                i = iou(r["box"], q["box"])
                if i > best:
                    best, best_j = i, j
            if best_j >= 0:
                used[best_j] = True
                ious.append(best)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return {"mean_iou": mean_iou, "pairs": len(ious)}


def main() -> int:
    runs = RUNS_DIR
    runs.mkdir(parents=True, exist_ok=True)
    ego4d_root = detect_drive_root()
    print(f"[cfg] EGO4D_ROOT={ego4d_root}")
    print(f"[cfg] FRAMES_ROOT={FRAMES_ROOT}")
    print(f"[cfg] RUNS_DIR={RUNS_DIR}")

    # 1) Load annotations and visualize a sample last frame
    recs = _load_sta(ego4d_root, VERSION, USE_HEIGHT_540)
    if not recs:
        print("[err] No STA annotations loaded. Check paths.")
        return 2
    visualize_last_frame_val(recs)

    # 2) Build tiny subset and evaluate
    gt, preds = build_subset(recs, EVAL_SUBSET_N)
    gt_path = RUNS_DIR / "gt_subset.json"
    pred_path = RUNS_DIR / "pred_subset.json"
    json.dump(gt, open(gt_path, "w"))
    json.dump(preds, open(pred_path, "w"))
    print(f"[subset] Wrote GT ({len(gt)}) → {gt_path}")
    print(f"[subset] Wrote PRED ({len(preds)}) → {pred_path}")

    if EVAL_OFFICIAL_SCRIPT:
        out_json = RUNS_DIR / "official_eval_result.json"
        ok = run_official_eval(Path(EVAL_OFFICIAL_SCRIPT), gt_path, pred_path, out_json)
        if not ok:
            print("[eval] Falling back to mini-eval.")
            stats = mini_eval_iou(gt, preds)
            json.dump(stats, open(RUNS_DIR / "mini_eval.json", "w"), indent=2)
            print("[mini-eval]", stats)
    else:
        stats = mini_eval_iou(gt, preds)
        json.dump(stats, open(RUNS_DIR / "mini_eval.json", "w"), indent=2)
        print("[mini-eval]", stats)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
