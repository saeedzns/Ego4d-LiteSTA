#!/usr/bin/env python3
"""
Build YOLO TXT labels and Head manifests from full Ego4D v2 STA JSONs.

Outputs:
 - YOLO images (flattened, optional) and labels (TXT per image)
 - head_train.json and head_val.json with fields used by the head stage

Assumptions:
 - Frames are extracted under {data_v2_root}/frames/<video_uid>/%07d.jpg
 - Full STA JSONs live under {data_v2_root}/annotations/fho_sta_{train,val}.json
 - If --images-out is set, images are copied/linked to a flat folder using
   base names: {video_uid}_{frame:07d}.jpg and labels use the same base name.

Usage (Colab):
  python scripts/sta_build_labels_and_manifests.py \
    --data-v2-root /content/drive/MyDrive/ego4d_data/v2 \
    --images-out   /content/drive/MyDrive/ego4d_data/v2/yolo_images \
    --labels-out   /content/drive/MyDrive/ego4d_data/v2/yolo_labels \
    --head-out     /content/drive/MyDrive/ego4d_data/v2/manifests
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image


def load_sta_json(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            return data["annotations"]
        # Some files use different top-level keys; try naive fallback
        # Not expected for STA, but keeping robust
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return []
    elif isinstance(data, list):
        return data
    return []


def frame_image_path(frames_root: Path, video_uid: str, frame_idx: int) -> Optional[Path]:
    for ext in (".jpg", ".png", ".jpeg"):
        p = frames_root / video_uid / f"{frame_idx:07d}{ext}"
        if p.exists():
            return p
    return None


def choose_object(obj_list: List[Dict]) -> Optional[Dict]:
    if not obj_list:
        return None
    # Prefer minimum time_to_contact > 0 if present, else first
    with_ttc = [o for o in obj_list if isinstance(o.get("time_to_contact"), (int, float))]
    if with_ttc:
        return sorted(with_ttc, key=lambda o: o.get("time_to_contact", 9e9))[0]
    return obj_list[0]


def xyxy_to_yolo(x1, y1, x2, y2, w, h) -> Tuple[float, float, float, float]:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, (x2 - x1))
    bh = max(1.0, (y2 - y1))
    return cx / w, cy / h, bw / w, bh / h


def process_split(
    name: str,
    anns: List[Dict],
    frames_root: Path,
    images_out: Optional[Path],
    labels_out: Optional[Path],
    head_items: List[Dict],
    class_mode: str = "noun",
    link_instead_of_copy: bool = False,
):
    labels_out.mkdir(parents=True, exist_ok=True) if labels_out else None
    images_out.mkdir(parents=True, exist_ok=True) if images_out else None

    written_labels = 0
    missing_frames = 0

    for ann in anns:
        video_uid = ann.get("video_uid") or ann.get("video_id")
        frame_idx = ann.get("frame") or ann.get("clip_frame")
        if video_uid is None or frame_idx is None:
            continue

        img_path = frame_image_path(frames_root, video_uid, int(frame_idx))
        if not img_path:
            missing_frames += 1
            continue

        # Load object choice
        obj = choose_object(ann.get("objects") or [])
        if not obj or not isinstance(obj.get("box"), (list, tuple)) or len(obj["box"]) != 4:
            continue
        x1, y1, x2, y2 = obj["box"]

        # Load image to normalize
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue

        cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)

        # Class id
        if class_mode == "noun":
            cls = int(obj.get("noun_category_id", 0))
        elif class_mode == "verb":
            cls = int(obj.get("verb_category_id", 0))
        else:
            cls = 0

        # Determine base name (flattened)
        base = f"{video_uid}_{int(frame_idx):07d}"

        # Optionally copy/link image to flat folder
        flat_img_path = None
        if images_out:
            flat_img_path = images_out / f"{base}{img_path.suffix.lower()}"
            if not flat_img_path.exists():
                if link_instead_of_copy:
                    try:
                        os.link(img_path, flat_img_path)
                    except OSError:
                        shutil.copy2(img_path, flat_img_path)
                else:
                    shutil.copy2(img_path, flat_img_path)
        else:
            flat_img_path = img_path

        # Write YOLO label
        if labels_out is not None:
            lbl_path = labels_out / f"{base}.txt"
            lbl_path.write_text(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            written_labels += 1

        # Add head item (train split)
        if name == "train":
            head_items.append({
                "image": str(flat_img_path),
                "gt_box": [float(x1), float(y1), float(x2), float(y2)],
                "verb_id": int(obj.get("verb_category_id", -1)),
                "noun_id": int(obj.get("noun_category_id", -1)),
                "ttc": float(obj.get("time_to_contact", -1.0)),
            })

    print(f"[{name}] labels written: {written_labels}, missing frames: {missing_frames}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-v2-root", default="/content/drive/MyDrive/ego4d_data/v2", help="Root folder containing videos, frames, annotations")
    ap.add_argument("--frames-root", default=None, help="Override frames root; default {data_v2_root}/frames")
    ap.add_argument("--sta-train-json", default=None, help="Path to full STA train JSON; default {data_v2_root}/annotations/fho_sta_train.json")
    ap.add_argument("--sta-val-json", default=None, help="Path to full STA val JSON; default {data_v2_root}/annotations/fho_sta_val.json")
    ap.add_argument("--images-out", default=None, help="Optional flat images output folder (copied/linked from frames)")
    ap.add_argument("--labels-out", default=None, help="YOLO TXT labels output folder (flat)")
    ap.add_argument("--head-out", default=None, help="Folder to write head_train.json and head_val.json")
    ap.add_argument("--class-mode", default="noun", choices=["noun", "verb", "single"], help="Which class id to write into YOLO labels")
    ap.add_argument("--link", action="store_true", help="Hardlink images instead of copying when possible")
    args = ap.parse_args()

    data_root = Path(args.data_v2_root)
    frames_root = Path(args.frames_root) if args.frames_root else data_root / "frames"
    sta_train_path = Path(args.sta_train_json) if args.sta_train_json else data_root / "annotations" / "fho_sta_train.json"
    sta_val_path = Path(args.sta_val_json) if args.sta_val_json else data_root / "annotations" / "fho_sta_val.json"
    images_out = Path(args.images_out) if args.images_out else None
    labels_out = Path(args.labels_out) if args.labels_out else None
    head_out = Path(args.head_out) if args.head_out else None

    print("data_root =", data_root)
    print("frames_root =", frames_root)
    print("sta_train =", sta_train_path)
    print("sta_val   =", sta_val_path)
    if images_out: print("images_out =", images_out)
    if labels_out: print("labels_out =", labels_out)
    if head_out: print("head_out =", head_out)

    # Load JSONs
    if not sta_train_path.exists() or not sta_val_path.exists():
        print("ERROR: Full STA JSONs not found. Please supply --sta-*-json or place files under data_v2_root/annotations.")
        return
    train_anns = load_sta_json(sta_train_path)
    val_anns = load_sta_json(sta_val_path)
    print(f"Loaded: train={len(train_anns)} val={len(val_anns)}")

    # Head items
    head_train: List[Dict] = []
    head_val: List[Dict] = []

    # Process
    process_split("train", train_anns, frames_root, images_out, labels_out, head_train, class_mode=args.class_mode, link_instead_of_copy=args.link)
    process_split("val", val_anns, frames_root, images_out, labels_out, head_val, class_mode=args.class_mode, link_instead_of_copy=args.link)

    # Write manifests if requested
    if head_out:
        head_out.mkdir(parents=True, exist_ok=True)
        (head_out / "head_train.json").write_text(json.dumps(head_train, indent=2), encoding="utf-8")
        (head_out / "head_val.json").write_text(json.dumps(head_val, indent=2), encoding="utf-8")
        print("Wrote:", head_out / "head_train.json", head_out / "head_val.json")


if __name__ == "__main__":
    main()

