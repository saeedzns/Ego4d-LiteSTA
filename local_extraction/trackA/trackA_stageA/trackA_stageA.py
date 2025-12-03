#!/usr/bin/env python3
"""
Track A — Stage A (Detector, last frame) — local-only runner

Inputs (local):
  - Frames: local_extraction/<version>/extracted_frames/<uid>/<frame:07d>.jpg
  - Optional labels (for oracle mode): local_extraction/<version>/yolo_labels_540/{clips|videos}/<uid>/<frame:07d>.txt

Outputs (local):
  - Run folder: local_extraction/runs/Track_A/StageA_<RUN_NAME>/
    * candidates.jsonl  (one JSON per line: {uid, frame, boxes:[{x1,y1,x2,y2,conf,cls}]})
    * per-image CSV (optional toggle)
    * summary.json

Configuration is loaded from configs/trackA.yaml
"""

from __future__ import annotations

import os
import json
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime
import sys

# ================== YAML CONFIG LOADING ==================
# Add parent directories to path for core imports
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
SPACE = _cfg.get('stage_a.oracle.label_space', 'clips')

# Where to read frames + (optionally) labels from
FRAMES_ROOT = Path(_cfg.get('paths.extracted_frames', f'local_extraction/{VERSION}/extracted_frames'))
LABELS_ROOT = Path(_cfg.get('paths.yolo_labels', f'local_extraction/{VERSION}/yolo_labels_540'))

# Which images to process
LAST_FRAME_ONLY = _cfg.get('stage_a.last_frame_only', True)
MAX_IMAGES = _cfg.get('stage_a.max_images', None)

# Candidate generation mode
DETECTION_MODE = _cfg.get('stage_a.mode', 'yolo')
K = _cfg.get('stage_a.k', 6)

# YOLO settings
YOLO_WEIGHTS = _cfg.get('stage_a.yolo.weights', 'yolov8s.pt')
YOLO_IMGSZ = _cfg.get('stage_a.yolo.imgsz', 960)
YOLO_CONF = _cfg.get('stage_a.yolo.conf_thresh', 0.05)
YOLO_IOU = _cfg.get('stage_a.yolo.nms_iou', 0.45)

# Save options
SAVE_PER_IMAGE_CSV = _cfg.get('stage_a.output.save_per_image_csv', False)

# Demo/limits and progress
DEMO_MODE = _cfg.get('demo.enabled', False)
DEMO_N = _cfg.get('demo.max_samples', 50)
PRINT_PROGRESS = _cfg.get('runtime.print_progress', True)

# Run folder (date-stamped with script prefix)
FILE_PREFIX = _cfg.get('stage_a.output.run_prefix', Path(__file__).stem)
RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = Path(_cfg.get('paths.runs', 'local_extraction/runs')) / "Track_A" / f"{FILE_PREFIX}_{RUN_STAMP}"
# =========================================================

# -------- Environment overrides (optional, still supported) --------
_env_mode = os.environ.get("STAGEA_MODE")
if _env_mode:
    DETECTION_MODE = _env_mode.strip().lower()
_env_k = os.environ.get("STAGEA_K")
if _env_k:
    try:
        K = int(_env_k)
    except Exception:
        pass
_env_max = os.environ.get("STAGEA_MAX_IMAGES")
if _env_max:
    try:
        MAX_IMAGES = int(_env_max)
    except Exception:
        pass
_env_demo = os.environ.get("STAGEA_DEMO_MODE")
if _env_demo in {"1","true","True"}:
    DEMO_MODE = True
_env_demo_n = os.environ.get("STAGEA_DEMO_N")
if _env_demo_n:
    try:
        DEMO_N = int(_env_demo_n)
    except Exception:
        pass


def _read_size_raw(img_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        with open(img_path, 'rb') as f:
            head = f.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                f.seek(16); w = int.from_bytes(f.read(4), 'big'); h = int.from_bytes(f.read(4), 'big'); return w, h
            if head[0:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    if not b: break
                    while b != b"\xff":
                        b = f.read(1)
                        if not b: return None, None
                    while True:
                        m = f.read(1)
                        if m != b"\xff": break
                    if not m: break
                    marker = m[0]
                    if marker in (0xD8, 0xD9):
                        continue
                    seglen_b = f.read(2)
                    if len(seglen_b) != 2: break
                    seglen = int.from_bytes(seglen_b, 'big')
                    if seglen < 2: return None, None
                    if (0xC0 <= marker <= 0xC3) or (0xC5 <= marker <= 0xC7) or (0xC9 <= marker <= 0xCB) or (0xCD <= marker <= 0xCF):
                        f.read(1); h = int.from_bytes(f.read(2), 'big'); w = int.from_bytes(f.read(2), 'big'); return w, h
                    f.seek(seglen - 2, 1)
    except Exception:
        return None, None
    return None, None


def iter_images(frames_root: Path) -> Iterable[Path]:
    for uid_dir in sorted(frames_root.iterdir()):
        if not uid_dir.is_dir():
            continue
        imgs = sorted([p for p in uid_dir.iterdir() if p.is_file() and p.suffix.lower() == '.jpg'])
        if not imgs:
            continue
        if LAST_FRAME_ONLY:
            yield imgs[-1]
        else:
            for p in imgs:
                yield p


_YOLO_MODEL = None  # lazy singleton

def _ensure_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError("Ultralytics YOLO not available. Install 'ultralytics' or switch DETECTION_MODE to 'oracle'.") from e
    _YOLO_MODEL = YOLO(YOLO_WEIGHTS)
    return _YOLO_MODEL

def yolo_detect_one(image_path: Path, model) -> List[Dict]:
    """Return a list of {x1,y1,x2,y2,conf,cls} using an already-instantiated Ultralytics YOLO model."""
    res = model.predict(source=str(image_path), imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
    boxes_out: List[Dict] = []
    for r in res:
        if not hasattr(r, 'boxes') or r.boxes is None:
            continue
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy() if hasattr(b, 'xyxy') else []
        confs = b.conf.cpu().numpy() if hasattr(b, 'conf') else []
        clss = b.cls.cpu().numpy() if hasattr(b, 'cls') else []
        n = min(len(xyxy), len(confs), len(clss))
        for i in range(n):
            x1, y1, x2, y2 = map(float, xyxy[i].tolist())
            boxes_out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(confs[i]), "cls": int(clss[i])})
    # Top-K by conf
    boxes_out.sort(key=lambda d: d.get('conf', 0.0), reverse=True)
    return boxes_out[:K]


def oracle_boxes(image_path: Path, labels_root: Path, space: str) -> List[Dict]:
    """Use existing YOLO label file as proposals; convert to pixel coords."""
    uid = image_path.parent.name
    try:
        frame_idx = int(image_path.stem)
    except Exception:
        return []
    lbl = labels_root / space / uid / f"{frame_idx:07d}.txt"
    if not lbl.exists():
        return []
    # Read image size to denormalize
    W, H = _read_size_raw(image_path)
    if not (W and H and W > 0 and H > 0):
        return []
    out: List[Dict] = []
    for ln in lbl.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx = float(parts[1]); cy = float(parts[2]); ww = float(parts[3]); hh = float(parts[4])
        except Exception:
            continue
        px, py = cx * W, cy * H
        pw, ph = ww * W, hh * H
        x1, y1 = px - pw / 2.0, py - ph / 2.0
        x2, y2 = px + pw / 2.0, py + ph / 2.0
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": 1.0, "cls": int(cls)})
    # Keep at most K
    return out[:K]


def ensure_run_dir(create_candidates: bool = False) -> Path:
    """Create the run directory; optionally create the per-image candidates/ subfolder.

    The empty candidates folder you observed appears because previously we always created it
    even when SAVE_PER_IMAGE_CSV was False. Now it's created only if requested.
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if create_candidates:
        (RUN_DIR / "candidates").mkdir(parents=True, exist_ok=True)
    return RUN_DIR


def main() -> int:
    t0 = time.time()
    ensure_run_dir(create_candidates=SAVE_PER_IMAGE_CSV)
    images = list(iter_images(FRAMES_ROOT))
    if MAX_IMAGES is not None:
        images = images[: int(MAX_IMAGES)]
    if DEMO_MODE:
        images = images[: int(DEMO_N)]
    if not images:
        print("[stageA] No images found under:", FRAMES_ROOT)
        return 2
    total_images = len(images)
    print(f"[stageA] Images to process: {total_images} | mode={DETECTION_MODE} | K={K}")

    # Prepare outputs
    jsonl_path = RUN_DIR / "candidates.jsonl"
    n_images = 0
    n_boxes = 0
    # Try tqdm progress bar
    pbar = None
    HAS_TQDM = False
    if PRINT_PROGRESS:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total_images, unit='img', desc='StageA', dynamic_ncols=True)
            HAS_TQDM = True
        except Exception:
            HAS_TQDM = False

    # Pre-instantiate YOLO model once (if using YOLO mode)
    yolo_model = None
    if DETECTION_MODE == "yolo":
        try:
            yolo_model = _ensure_yolo_model()
            print(f"[stageA] YOLO model loaded once: {YOLO_WEIGHTS}")
        except Exception as e:
            print(f"[stageA] Failed to load YOLO model: {e}")
            return 3

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for idx, img in enumerate(images, 1):
            step_t0 = time.time()
            uid = img.parent.name
            try:
                frame_idx = int(img.stem)
            except Exception:
                continue
            if DETECTION_MODE == "yolo":
                try:
                    boxes = yolo_detect_one(img, yolo_model)
                except Exception as e:
                    print(f"[stageA] YOLO failed for {img}: {e}")
                    boxes = []
            else:
                boxes = oracle_boxes(img, LABELS_ROOT, SPACE)

            record = {"uid": uid, "frame": frame_idx, "boxes": boxes}
            jf.write(json.dumps(record) + "\n")
            n_images += 1
            n_boxes += len(boxes)

            if SAVE_PER_IMAGE_CSV:
                csvp = RUN_DIR / "candidates" / f"{uid}_{frame_idx:07d}.csv"
                csvp.parent.mkdir(parents=True, exist_ok=True)  # lazy ensure
                with open(csvp, "w", encoding="utf-8") as cf:
                    cf.write("x1,y1,x2,y2,conf,cls\n")
                    for b in boxes:
                        cf.write(f"{b['x1']:.1f},{b['y1']:.1f},{b['x2']:.1f},{b['y2']:.1f},{b.get('conf',0):.4f},{b.get('cls',-1)}\n")
            if PRINT_PROGRESS:
                step_dt = time.time() - step_t0
                elapsed = time.time() - t0
                eta = (elapsed / idx) * (total_images - idx) if idx > 0 else 0.0
                if HAS_TQDM and pbar is not None:
                    pbar.set_postfix(step=f"{step_dt:5.2f}s", elapsed=f"{elapsed:6.1f}s", eta=f"{eta:6.1f}s", boxes=len(boxes))
                    pbar.update(1)
                else:
                    # Single-line progress without flooding the console
                    percent = 100.0 * idx / max(1, total_images)
                    sys.stdout.write(
                        f"\r[StageA] {idx}/{total_images} ({percent:5.1f}%) | step {step_dt:5.2f}s | elapsed {elapsed:6.1f}s | ETA ~{eta:6.1f}s | boxes={len(boxes)}"
                    )
                    sys.stdout.flush()

    if PRINT_PROGRESS and (not HAS_TQDM):
        # End the single-line progress with a newline
        sys.stdout.write("\n")
        sys.stdout.flush()
    if PRINT_PROGRESS and HAS_TQDM and pbar is not None:
        pbar.close()

    # Summary
    t1 = time.time()
    summary = {
        "images": n_images,
        "avg_boxes_per_image": (n_boxes / n_images) if n_images else 0.0,
        "mode": DETECTION_MODE,
        "K": K,
        "space": SPACE,
        "frames_root": str(FRAMES_ROOT),
        "labels_root": str(LABELS_ROOT),
        "duration_sec": round(t1 - t0, 2),
        "file_prefix": FILE_PREFIX,
        "run_stamp": RUN_STAMP,
        "script_path": str(Path(__file__).resolve()),
    }
    # Persist YOLO-related config when using YOLO mode
    if DETECTION_MODE.lower() == "yolo":
        summary["yolo_config"] = {
            "weights": str(YOLO_WEIGHTS),
            "imgsz": int(YOLO_IMGSZ),
            "conf_thresh": float(YOLO_CONF),
            "nms_iou": float(YOLO_IOU),
            "loaded_once": bool(_YOLO_MODEL is not None),
        }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Avoid non-ASCII arrows for Windows console compatibility
    print("[stageA] Summary ->", RUN_DIR / "summary.json")
    
    # Log run using RunLogger
    try:
        from core import RunLogger
        run_logger = RunLogger(track='trackA', stage='stageA', run_dir=RUN_DIR)
        run_logger.log_config(summary)
        run_logger.log_metrics({
            'images': n_images,
            'avg_boxes_per_image': summary['avg_boxes_per_image'],
            'duration_sec': summary['duration_sec'],
        })
        run_logger.log_artifacts([str(RUN_DIR / "candidates.jsonl"), str(RUN_DIR / "summary.json")])
        run_logger.log_end(success=True)
        run_logger.print_summary()
    except Exception as e:
        print(f"[stageA] WARNING: failed to write run log: {e}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
