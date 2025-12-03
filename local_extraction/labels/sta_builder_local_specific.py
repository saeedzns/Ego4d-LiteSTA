#!/usr/bin/env python3
"""
STA → YOLO label generator for specific UIDs only (local-first).

Reads STA annotations from your mounted Drive, filters to a small set of
specific UIDs (set below in TARGET_UIDS), and generates YOLO labels only for
those UIDs. Labels are written under your local_extraction tree. Existing
labels are skipped unless OVERWRITE_EXISTING_LABELS=True.

All toggles are in-code; you don't need CLI flags.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# =============== TOGGLES (edit here) =================
# UIDs to process (interpretation depends on SPACE)
TARGET_UIDS: List[str] = ["0c8eee05-5cdb-4036-bf3d-3e9666b8980e"]

# Space to use: 'clips' or 'videos'
SPACE = 'clips'

# Use height-540 scaled JSONs (recommended to match 540p frames)
USE_HEIGHT_540 = True
VERSION = 'v2'

# Overwrite behavior
OVERWRITE_EXISTING_LABELS = False

# Restrict to these splits; set to ('train','val') to include both
SPLITS = ('train', 'val')

# Local output roots
OUTPUT_LABEL_ROOT = Path('local_extraction') / VERSION / 'yolo_labels_540'
FRAMES_IMAGE_ROOT = Path('local_extraction') / VERSION / 'extracted_frames'

# Missing image log (CSV of uid,frame,path) — set to None to disable
MISSING_LOG = Path('local_extraction') / f'missing_images_specific_{SPACE}.txt'
# ====================================================


def detect_drive_root() -> str:
    env = os.environ.get('EGO4D_ROOT')
    if env and os.path.isdir(env):
        return env
    for p in (
        'H:/My Drive/ego4d_data',
        'D:/My Drive/ego4d_data',
        '/mnt/h/My Drive/ego4d_data',
        '/mnt/d/My Drive/ego4d_data',
        '/content/drive/MyDrive/ego4d_data',
        '/drive/MyDrive/ego4d_data',
    ):
        if os.path.isdir(p):
            return p
    return 'H:/My Drive/ego4d_data'


def _iter_list(obj):
    if isinstance(obj, dict):
        if 'annotations' in obj and isinstance(obj['annotations'], list):
            return obj['annotations']
        if 'data' in obj and isinstance(obj['data'], list):
            return obj['data']
    if isinstance(obj, list):
        return obj
    return []


def _get_dims(rec: dict) -> Tuple[Optional[int], Optional[int]]:
    for (kw, kh) in [('width', 'height'), ('frame_width', 'frame_height'), ('img_w', 'img_h')]:
        if kw in rec and kh in rec:
            try:
                return int(rec[kw]), int(rec[kh])
            except Exception:
                pass
    return None, None


def _as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _image_size_from_local(path: Path) -> Tuple[Optional[int], Optional[int]]:
    # Try Pillow
    try:
        from PIL import Image  # type: ignore
        with Image.open(str(path)) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        pass
    # Try imageio
    try:
        import imageio.v2 as imageio  # type: ignore
        im = imageio.imread(str(path))
        if hasattr(im, 'shape') and len(im.shape) >= 2:
            h, w = int(im.shape[0]), int(im.shape[1])
            return w, h
    except Exception:
        pass
    # Fallback minimal header parsing for PNG/JPEG
    try:
        with open(path, 'rb') as f:
            head = f.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                f.seek(16); w = int.from_bytes(f.read(4), 'big'); h = int.from_bytes(f.read(4), 'big'); return w, h
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
                    seglen = int.from_bytes(seglen_b, 'big')
                    if seglen < 2:
                        return None, None
                    if (0xC0 <= marker <= 0xC3) or (0xC5 <= marker <= 0xC7) or (0xC9 <= marker <= 0xCB) or (0xCD <= marker <= 0xCF):
                        f.read(1); h = int.from_bytes(f.read(2), 'big'); w = int.from_bytes(f.read(2), 'big'); return w, h
                    f.seek(seglen - 2, 1)
    except Exception:
        pass
    return None, None


def load_sta_records(ego4d_root: str, version: str, use_540: bool) -> List[dict]:
    ann_dir = os.path.join(ego4d_root, version, 'annotations_540ss' if use_540 else 'annotations')
    files = [
        ('fho_sta_train_height-540.json' if use_540 else 'fho_sta_train.json', 'train'),
        ('fho_sta_val_height-540.json' if use_540 else 'fho_sta_val.json', 'val'),
    ]
    out: List[dict] = []
    for name, split in files:
        if split not in SPLITS:
            continue
        path = os.path.join(ann_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            raw = json.load(f)
        for rec in _iter_list(raw):
            vuid = rec.get('video_uid'); cuid = rec.get('clip_uid')
            vfr = _as_int(rec.get('frame')); cfr = _as_int(rec.get('clip_frame'))
            wj, hj = _get_dims(rec)
            for obj in rec.get('objects', []) or []:
                box = obj.get('box') or obj.get('bbox') or obj.get('gt_box')
                if not (isinstance(box, (list, tuple)) and len(box) == 4):
                    continue
                try:
                    x1, y1, x2, y2 = map(float, box)
                except Exception:
                    continue
                noun = _as_int(obj.get('noun_category_id', obj.get('noun_id', 0))) or 0
                out.append({
                    'split': split,
                    'video_uid': vuid,
                    'clip_uid': cuid,
                    'video_frame': vfr,
                    'clip_frame': cfr,
                    'box': (x1, y1, x2, y2),
                    'noun_id': int(noun),
                    'json_w': wj,
                    'json_h': hj,
                })
    return out


def main() -> int:
    if not TARGET_UIDS:
        print('[cfg] Please set TARGET_UIDS with one or more UIDs to process.')
        return 2

    ego4d_root = detect_drive_root()
    print(f"[cfg] EGO4D_ROOT={ego4d_root}")
    print(f"[cfg] SPACE={SPACE} | SPLITS={SPLITS} | USE_HEIGHT_540={USE_HEIGHT_540}")
    print(f"[cfg] OUTPUT_LABEL_ROOT={OUTPUT_LABEL_ROOT}")
    print(f"[cfg] FRAMES_IMAGE_ROOT={FRAMES_IMAGE_ROOT}")
    print(f"[cfg] TARGET_UIDS: {len(TARGET_UIDS)} item(s)")

    recs = load_sta_records(ego4d_root, VERSION, USE_HEIGHT_540)
    if not recs:
        print('[err] No STA records loaded — check paths.')
        return 2

    # Filter to target UIDs and group by image
    key_name = 'clip_frame' if SPACE == 'clips' else 'video_frame'
    id_name = 'clip_uid' if SPACE == 'clips' else 'video_uid'

    groups: Dict[Tuple[str, int], List[dict]] = {}
    for r in recs:
        uid = r.get(id_name)
        if not uid or str(uid) not in TARGET_UIDS:
            continue
        fr = r.get(key_name)
        if fr is None:
            continue
        groups.setdefault((str(uid), int(fr)), []).append(r)

    if not groups:
        print('[info] No images found for provided TARGET_UIDS and settings.')
        return 0

    written = 0
    skipped_exist = 0
    missing_img = 0
    missing_dims = 0
    overwritten = 0
    missing_rows: List[str] = []

    for (uid, fr), rs in groups.items():
        img_path = FRAMES_IMAGE_ROOT / uid / f"{int(fr):07d}.jpg"
        if not img_path.exists():
            missing_img += 1
            if MISSING_LOG:
                missing_rows.append(f"{uid},{int(fr):07d},{img_path}")
            continue
        lbl_dir = OUTPUT_LABEL_ROOT / SPACE / uid
        lbl_path = lbl_dir / f"{int(fr):07d}.txt"
        if lbl_path.exists() and not OVERWRITE_EXISTING_LABELS:
            skipped_exist += 1
            continue
        if lbl_path.exists() and OVERWRITE_EXISTING_LABELS:
            overwritten += 1

        # Get image size
        W, H = _image_size_from_local(img_path)
        if not (W and H and W > 0 and H > 0):
            missing_dims += 1
            continue

        lbl_dir.mkdir(parents=True, exist_ok=True)
        with open(lbl_path, 'w') as g:
            for r in rs:
                x1, y1, x2, y2 = r['box']
                wj, hj = r.get('json_w'), r.get('json_h')
                # rescale from JSON dims if necessary
                if wj and hj and (wj > 0 and hj > 0) and (wj != W or hj != H):
                    sx = W / float(wj); sy = H / float(hj)
                    x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
                cx = x1 + bw / 2.0; cy = y1 + bh / 2.0
                cls = int(r.get('noun_id', 0))
                g.write(f"{cls} {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}\n")
        written += 1

    # Missing log
    if MISSING_LOG and missing_rows:
        try:
            MISSING_LOG.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        Path(MISSING_LOG).write_text("\n".join(missing_rows) + "\n", encoding='utf-8')
        print(f"[log] Missing images → {MISSING_LOG} ({len(missing_rows)} lines)")

    print(
        f"[done] images: {len(groups)} | written: {written} | skipped_exist: {skipped_exist} | overwritten: {overwritten} | missing_img: {missing_img} | missing_dims: {missing_dims}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
