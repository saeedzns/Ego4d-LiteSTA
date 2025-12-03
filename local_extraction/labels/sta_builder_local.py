#!/usr/bin/env python3
"""
Ego4D STA — ONE-CELL BUILDER (local outputs)

Inputs (read-only, from Google Drive on Windows):
  H:/My Drive/ego4d_data/<version>/annotations[_540ss]/...
  H:/My Drive/ego4d_data/<version>/tmp_frame_lists/...

Outputs (local-only, under this repo's local_extraction/<version>/...):
  - UID files                    → <OUTPUT_ROOT>/sta_uids.txt, sta_clip_uids.txt
  - Frame lists (videos/clips)   → <OUTPUT_ROOT>/<version>/tmp_frame_lists/{videos,clips}
  - YOLO labels (optional)       → <OUTPUT_ROOT>/<version>/yolo_labels_540/{videos,clips}
  - Head manifests (optional)    → <OUTPUT_ROOT>/<version>/manifests

Image naming convention (expected by downstream training):
  For a frame index k, filename is f"{k:07d}.jpg" (e.g., 23 → 0000023.jpg).

Notes
- This script never writes to your Drive H: path; all outputs go to the repo's
  local_extraction folder so they won't be pushed to Git.
- Set EGO4D_ROOT to override the Drive input root; set OUTPUT_ROOT to override
  where local outputs are written (defaults to <repo>/local_extraction).
"""

from __future__ import annotations

import os
import json
import collections
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# ---------------- Path helpers ----------------
def detect_repo_local_root() -> str:
    here = Path(__file__).resolve()
    # Walk up until we find the 'local_extraction' folder regardless of nesting
    for p in [here] + list(here.parents):
        if p.name == 'local_extraction':
            return str(p)
    # Fallback: parent of current file
    return str(here.parent)


def detect_drive_root() -> str:
    env = os.environ.get("EGO4D_ROOT")
    if env:
        return env
    candidates = [
        "H:/My Drive/ego4d_data",
        "D:/My Drive/ego4d_data",
        "/mnt/h/My Drive/ego4d_data",
        "/mnt/d/My Drive/ego4d_data",
        "/content/drive/MyDrive/ego4d_data",
        "/drive/MyDrive/ego4d_data",
    ]
    for p in candidates:
        try:
            if os.path.isdir(p):
                return p
        except Exception:
            pass
    return "H:/My Drive/ego4d_data"


# ---------------- USER TOGGLES ----------------
EGO4D_ROOT = detect_drive_root()
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", detect_repo_local_root())
VERSION = os.environ.get("EGO4D_VERSION", "v2")

# Use 540-scaled JSONs you produced (True) OR original full-res JSONs (False)
USE_HEIGHT_540 = True

# Build lists (recommended True): video lists, clip lists, and UID files
BUILD_VIDEO_LISTS = False
BUILD_CLIP_LISTS = False

# Optional artifacts
GENERATE_YOLO_LABELS = False    # normalized YOLO txt (requires width/height per record)
# If True, overwrite existing YOLO label files; if False, skip already present files
OVERWRITE_EXISTING_LABELS = False
DETECTOR_CLASS_POLICY = "noun"  # "noun" or "single" (single -> class 0)
"""
Environment overrides supported:
- EGO4D_GENERATE_HEAD_MANIFESTS=1|0 to enable/disable head manifests without editing code
"""
GENERATE_HEAD_MANIFESTS = (
    os.environ.get("EGO4D_GENERATE_HEAD_MANIFESTS", "1").strip() in {"1", "true", "True"}
)  # head manifests for videos + clips
SELECT_MIN_TTC_FOR_HEAD = True   # for head manifests: pick one object per (id,frame) by min-TTC


# ---------------- PATHS ----------------
ANN_DIR_FULL = os.path.join(EGO4D_ROOT, VERSION, "annotations")
ANN_540_DIR = os.path.join(EGO4D_ROOT, VERSION, "annotations_540ss")

# Expected file names
FHO_TRAIN_NAME_540 = "fho_sta_train_height-540.json"
FHO_VAL_NAME_540 = "fho_sta_val_height-540.json"
FHO_TRAIN_NAME_FULL = "fho_sta_train.json"
FHO_VAL_NAME_FULL = "fho_sta_val.json"

# Where we write lists/labels/manifests (LOCAL OUTPUTS)
TMP_BASE = os.path.join(OUTPUT_ROOT, VERSION, "tmp_frame_lists")
VIDEO_LIST_DIR = os.path.join(TMP_BASE, "videos")
CLIP_LIST_DIR = os.path.join(TMP_BASE, "clips")

UID_VID_FILE = os.path.join(OUTPUT_ROOT, "sta_uids.txt")
UID_CLIP_FILE = os.path.join(OUTPUT_ROOT, "sta_clip_uids.txt")

YOLO_BASE_DIR = os.path.join(OUTPUT_ROOT, VERSION, "yolo_labels_540")  # ok for both 540/full (labels are normalized)
YOLO_VID_DIR = os.path.join(YOLO_BASE_DIR, "videos")
YOLO_CLIP_DIR = os.path.join(YOLO_BASE_DIR, "clips")

MANIFEST_DIR = os.path.join(OUTPUT_ROOT, VERSION, "manifests")
HEAD_TRAIN_VIDEO = os.path.join(MANIFEST_DIR, "head_train_video.json")
HEAD_VAL_VIDEO = os.path.join(MANIFEST_DIR, "head_val_video.json")
HEAD_TRAIN_CLIP = os.path.join(MANIFEST_DIR, "head_train_clip.json")
HEAD_VAL_CLIP = os.path.join(MANIFEST_DIR, "head_val_clip.json")

# Where your extracted images should live locally (not created here)
# This matches the extractor's default: <OUTPUT_ROOT>/<version>/extracted_frames/<uid>/<frame:07d>.jpg
FRAMES_IMAGE_ROOT = os.path.join(OUTPUT_ROOT, VERSION, "extracted_frames")

for d in [VIDEO_LIST_DIR, CLIP_LIST_DIR, YOLO_VID_DIR, YOLO_CLIP_DIR, MANIFEST_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------- DATA MODEL ----------------
@dataclass
class StaRec:
    video_uid: Optional[str]
    clip_uid: Optional[str]
    video_frame: Optional[int]  # "frame" (global video frame index)
    clip_frame: Optional[int]   # "clip_frame" (0-based within canonical clip @ 30FPS)
    box: Optional[List[float]]  # [x1,y1,x2,y2] (in current JSON pixel space)
    verb_id: Optional[int]
    noun_id: Optional[int]
    ttc: Optional[float]
    split: str  # "train" | "val"
    width: Optional[int] = None
    height: Optional[int] = None


def _iter_list(obj):
    if isinstance(obj, dict):
        if "annotations" in obj and isinstance(obj["annotations"], list):
            return obj["annotations"]
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
    if isinstance(obj, list):
        return obj
    return []


def _get_dims(rec: dict) -> Tuple[Optional[int], Optional[int]]:
    # Try several common keys
    for (kw, kh) in [("width", "height"), ("frame_width", "frame_height"), ("img_w", "img_h")]:
        if kw in rec and kh in rec:
            try:
                return int(rec[kw]), int(rec[kh])
            except Exception:
                pass
    return None, None


def _image_size_from_local(path: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        try:
            import imageio.v2 as imageio  # type: ignore
            im = imageio.imread(path)
            # imageio returns HxWxC
            if hasattr(im, "shape") and len(im.shape) >= 2:
                h, w = im.shape[0], im.shape[1]
                return int(w), int(h)
        except Exception:
            pass
    return None, None


def _as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _load_one_file(path: str, split: str) -> List[StaRec]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing annotation file: {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    out: List[StaRec] = []
    for rec in _iter_list(raw):
        vuid = rec.get("video_uid")
        cuid = rec.get("clip_uid")
        vfr = _as_int(rec.get("frame"))
        cfr = _as_int(rec.get("clip_frame"))
        w, h = _get_dims(rec)

        objs = rec.get("objects", [])
        for obj in objs:
            box = obj.get("box") or obj.get("bbox") or obj.get("gt_box")
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            try:
                x1, y1, x2, y2 = map(float, box)
            except Exception:
                continue

            verb = _as_int(obj.get("verb_category_id", obj.get("verb_id", 0)))
            noun = _as_int(obj.get("noun_category_id", obj.get("noun_id", 0)))
            ttc = _as_float(obj.get("time_to_contact", obj.get("ttc", 0.0)))

            out.append(
                StaRec(
                    video_uid=vuid,
                    clip_uid=cuid,
                    video_frame=vfr,
                    clip_frame=cfr,
                    box=[x1, y1, x2, y2],
                    verb_id=verb,
                    noun_id=noun,
                    ttc=ttc,
                    split=split,
                    width=w,
                    height=h,
                )
            )
    return out


def load_all_records(use_540: bool) -> List[StaRec]:
    if use_540:
        train = _load_one_file(os.path.join(ANN_540_DIR, FHO_TRAIN_NAME_540), "train")
        val = _load_one_file(os.path.join(ANN_540_DIR, FHO_VAL_NAME_540), "val")
    else:
        train = _load_one_file(os.path.join(ANN_DIR_FULL, FHO_TRAIN_NAME_FULL), "train")
        val = _load_one_file(os.path.join(ANN_DIR_FULL, FHO_VAL_NAME_FULL), "val")
    return train + val


def _choose_min_ttc(objs: List[StaRec]) -> StaRec:
    return min(objs, key=lambda r: (1e9 if r.ttc is None else r.ttc))


def _class_id_for_detector(noun_id: Optional[int]) -> int:
    if DETECTOR_CLASS_POLICY == "single":
        return 0
    return 0 if noun_id is None else int(noun_id)


# ---------------- MAIN ----------------
def main() -> int:
    print(
        f"Reading {'540-scaled' if USE_HEIGHT_540 else 'full-resolution'} STA JSONs from {EGO4D_ROOT} ..."
    )
    recs = load_all_records(USE_HEIGHT_540)
    print(f"Loaded {len(recs)} object records.")

    # --------- Build FRAME LISTS + UID FILES (LOCAL) ---------
    if BUILD_VIDEO_LISTS:
        by_video = collections.defaultdict(set)  # video_uid -> set(video_frame)
        for r in recs:
            if r.video_uid is not None and r.video_frame is not None:
                by_video[str(r.video_uid)].add(int(r.video_frame))

        for vuid, frames in by_video.items():
            outp = os.path.join(VIDEO_LIST_DIR, f"{vuid}_frames.txt")
            with open(outp, "w") as f:
                for k in sorted(frames):
                    f.write(f"{k}\n")

        with open(UID_VID_FILE, "w") as f:
            f.write("\n".join(sorted(by_video.keys())))

        print(f"[VIDEO] {len(by_video)} videos → lists at {VIDEO_LIST_DIR}")
        print(f"[VIDEO] UID file → {UID_VID_FILE}")

    if BUILD_CLIP_LISTS:
        by_clip = collections.defaultdict(set)  # clip_uid -> set(clip_frame)
        for r in recs:
            if r.clip_uid is not None and r.clip_frame is not None:
                by_clip[str(r.clip_uid)].add(int(r.clip_frame))

        for cuid, frames in by_clip.items():
            outp = os.path.join(CLIP_LIST_DIR, f"{cuid}_clip_frames.txt")
            with open(outp, "w") as f:
                for k in sorted(frames):
                    f.write(f"{k}\n")

        with open(UID_CLIP_FILE, "w") as f:
            f.write("\n".join(sorted(by_clip.keys())))

        print(f"[CLIP]  {len(by_clip)} clips → lists at {CLIP_LIST_DIR}")
        print(f"[CLIP]  UID file → {UID_CLIP_FILE}")

    # (clip_frame → filename mapping intentionally removed; extractor now names files by true clip_frame)

    # --------- Optional: YOLO LABELS (normalized, LOCAL) ----------
    # Only needs width/height from JSON record; if missing, sample is skipped.
    if GENERATE_YOLO_LABELS:
        written_vid, skipped_vid = 0, 0
        written_clip, skipped_clip = 0, 0
        skipped_vid_exist, skipped_clip_exist = 0, 0
        overwritten_vid, overwritten_clip = 0, 0

        # For videos (per (video_uid, video_frame) we can emit one label line per object)
        per_img = collections.defaultdict(list)  # key: ("video", vuid, vframe)
        for r in recs:
            if r.video_uid is not None and r.video_frame is not None:
                per_img[("video", r.video_uid, r.video_frame)].append(r)

        for (tag, vuid, vframe), rs in per_img.items():
            # Prefer dims from JSON; otherwise read from the extracted image on disk
            W = next((x.width for x in rs if x.width), None)
            H = next((x.height for x in rs if x.height), None)
            if not (W and H and W > 0 and H > 0):
                probe_path = os.path.join(FRAMES_IMAGE_ROOT, str(vuid), f"{int(vframe):07d}.jpg")
                if os.path.exists(probe_path):
                    W, H = _image_size_from_local(probe_path)
            if not (W and H and W > 0 and H > 0):
                skipped_vid += 1
                continue

            out_dir = os.path.join(YOLO_VID_DIR, str(vuid))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{int(vframe):07d}.txt")
            if os.path.exists(out_path):
                if OVERWRITE_EXISTING_LABELS:
                    overwritten_vid += 1
                else:
                    skipped_vid_exist += 1
                    continue
            with open(out_path, "w") as g:
                for r in rs:
                    x1, y1, x2, y2 = r.box
                    w = max(1.0, x2 - x1)
                    h = max(1.0, y2 - y1)
                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0
                    cls_id = _class_id_for_detector(r.noun_id)
                    g.write(f"{cls_id} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}\n")
            written_vid += 1

        # For clips (rarely used for detection; included for completeness)
        per_clip_img = collections.defaultdict(list)  # key: ("clip", cuid, cframe)
        for r in recs:
            if r.clip_uid is not None and r.clip_frame is not None:
                per_clip_img[("clip", r.clip_uid, r.clip_frame)].append(r)

        for (tag, cuid, cframe), rs in per_clip_img.items():
            W = next((x.width for x in rs if x.width), None)
            H = next((x.height for x in rs if x.height), None)
            if not (W and H and W > 0 and H > 0):
                probe_path = os.path.join(FRAMES_IMAGE_ROOT, str(cuid), f"{int(cframe):07d}.jpg")
                if os.path.exists(probe_path):
                    W, H = _image_size_from_local(probe_path)
            if not (W and H and W > 0 and H > 0):
                skipped_clip += 1
                continue

            out_dir = os.path.join(YOLO_CLIP_DIR, str(cuid))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{int(cframe):07d}.txt")
            if os.path.exists(out_path):
                if OVERWRITE_EXISTING_LABELS:
                    overwritten_clip += 1
                else:
                    skipped_clip_exist += 1
                    continue
            with open(out_path, "w") as g:
                for r in rs:
                    x1, y1, x2, y2 = r.box
                    w = max(1.0, x2 - x1)
                    h = max(1.0, y2 - y1)
                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0
                    cls_id = _class_id_for_detector(r.noun_id)
                    g.write(f"{cls_id} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}\n")
            written_clip += 1

        print(f"[YOLO] videos: written {written_vid}, skipped_missing {skipped_vid}, skipped_exist {skipped_vid_exist}, overwritten {overwritten_vid}")
        print(f"[YOLO] clips : written {written_clip}, skipped_missing {skipped_clip}, skipped_exist {skipped_clip_exist}, overwritten {overwritten_clip}")

    # --------- Optional: HEAD MANIFESTS (videos + clips, LOCAL) ----------
    # For each (id, frame) we pick ONE object with min-TTC (if SELECT_MIN_TTC_FOR_HEAD=True),
    # otherwise we keep all objects for that image as separate entries.
    if GENERATE_HEAD_MANIFESTS:
        head_train_video: List[Dict] = []
        head_val_video: List[Dict] = []
        head_train_clip: List[Dict] = []
        head_val_clip: List[Dict] = []

        # group by (video_uid, video_frame) and (clip_uid, clip_frame)
        g_video: Dict[Tuple[str, int, str], List[StaRec]] = collections.defaultdict(list)
        g_clip: Dict[Tuple[str, int, str], List[StaRec]] = collections.defaultdict(list)
        for r in recs:
            if r.video_uid and r.video_frame is not None:
                g_video[(r.video_uid, r.video_frame, r.split)].append(r)
            if r.clip_uid and r.clip_frame is not None:
                g_clip[(r.clip_uid, r.clip_frame, r.split)].append(r)

        def _emit_entries(rs: List[StaRec], space: str) -> List[Dict]:
            # Return list of entries for head; either min-TTC single or all
            out: List[Dict] = []
            entries = [(_choose_min_ttc(rs),)] if SELECT_MIN_TTC_FOR_HEAD else [(x,) for x in rs]
            for (r,) in entries:
                if space == "video":
                    img_path = os.path.join(
                        FRAMES_IMAGE_ROOT, str(r.video_uid), f"{int(r.video_frame):07d}.jpg"
                    )
                else:
                    img_path = os.path.join(
                        FRAMES_IMAGE_ROOT, str(r.clip_uid), f"{int(r.clip_frame):07d}.jpg"
                    )
                out.append(
                    {
                        "image": img_path,
                        "gt_box": r.box,
                        "verb_id": r.verb_id if r.verb_id is not None else 0,
                        "noun_id": r.noun_id if r.noun_id is not None else 0,
                        "ttc": r.ttc if r.ttc is not None else 0.0,
                    }
                )
            return out

        # videos
        for (vuid, vframe, split), rs in g_video.items():
            items = _emit_entries(rs, space="video")
            if (split or "").lower() == "val":
                head_val_video.extend(items)
            else:
                head_train_video.extend(items)

        # clips
        for (cuid, cframe, split), rs in g_clip.items():
            items = _emit_entries(rs, space="clip")
            if (split or "").lower() == "val":
                head_val_clip.extend(items)
            else:
                head_train_clip.extend(items)

        # dump
        os.makedirs(MANIFEST_DIR, exist_ok=True)
        json.dump(head_train_video, open(HEAD_TRAIN_VIDEO, "w"))
        json.dump(head_val_video, open(HEAD_VAL_VIDEO, "w"))
        json.dump(head_train_clip, open(HEAD_TRAIN_CLIP, "w"))
        json.dump(head_val_clip, open(HEAD_VAL_CLIP, "w"))

        print(f"[HEAD] video: train={len(head_train_video)} → {HEAD_TRAIN_VIDEO}")
        print(f"[HEAD] video:   val={len(head_val_video)}   → {HEAD_VAL_VIDEO}")
        print(f"[HEAD] clip : train={len(head_train_clip)}  → {HEAD_TRAIN_CLIP}")
        print(f"[HEAD] clip :   val={len(head_val_clip)}    → {HEAD_VAL_CLIP}")

    print("\nDone.")
    print(f"Inputs (Drive):   {EGO4D_ROOT}")
    print(f"Outputs (local):  {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
