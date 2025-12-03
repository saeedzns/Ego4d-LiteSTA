#!/usr/bin/env python3
"""
Noun/Verb Sanity Check

Given STA annotations (train/val) and the main narration file, this script lets you
inspect a specific frame (or a random sample) to verify that noun/verb labels
and boxes line up with the narration and critical frames.

What it does:
- Load STA annotations (fho_sta_{train,val}_height-540.json)
- Load narration metadata (fho_main.json)
- Load noun/verb taxonomies for human-readable labels
- Select a frame (explicit --uid/--frame, or random). If your frames are keyed by
  clip_uid/clip_frame, pass the clip UID via --uid; the script will try video_uid,
  clip_uid, and main_uid combinations automatically.
- Show the frame with bounding boxes and print:
  * noun/verb labels from STA
  * narration text and structured/freeform verbs for overlapping actions
  * action start/end frames so you can see if the frame falls inside the action window

Usage examples (from repo root):

  # Inspect a specific frame
  python local_extraction/noun_verb_sanity_check/check_labels.py \
    --uid 7d072357-cabe-4ed8-bbec-52475b890301 --frame 9177

  # Random sample from val split
  python local_extraction/noun_verb_sanity_check/check_labels.py --split val
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
from PIL import Image

# Optional hardcoded batch pairs; set tuples of (uid, frame) to avoid passing via CLI.
DEFAULT_BATCH_PAIRS: List[Tuple[str, int]] = [("71e2f5f3-fb8e-4fdd-822f-644a8996fb3e", 8736), ("ebbe9fdf-babf-46e8-87ac-c4adf19a34e8", 4830), ("7d072357-cabe-4ed8-bbec-52475b890301", 9177), ("0d56e589-e0c5-48c6-a005-101f1493d9ea", 4832), ("4f9caa52-51cd-4026-9393-261a3f10a95f", 8374)]


def load_tax_labels(path: Path, kind: str) -> Dict[int, str]:
    """Load {id: label} from JSON (with 'nouns'/'verbs') or CSV (row index as id)."""
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        arr = obj.get("nouns" if kind == "noun" else "verbs")
        if not isinstance(arr, list):
            raise ValueError(f"Expected key '{'nouns' if kind=='noun' else 'verbs'}' in {path}")
        return {i: str(x) for i, x in enumerate(arr)}
    # CSV fallback
    labels: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    # skip header if present
    start = 1 if lines and ("label" in lines[0] or "group" in lines[0]) else 0
    for i, ln in enumerate(lines[start:]):
        # assume "label,group" and take first token (label)
        parts = ln.split(",")
        label = parts[0].strip('" ')
        labels[i] = label
    return labels


def load_sta_records(sta_path: Path) -> List[Dict[str, Any]]:
    if not sta_path.exists():
        raise FileNotFoundError(f"STA file not found: {sta_path}")
    data = json.loads(sta_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "annotations" in data:
        return list(data["annotations"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unexpected STA format in {sta_path}")


def load_fho_main(main_path: Path) -> Dict[str, Any]:
    if not main_path.exists():
        raise FileNotFoundError(f"Main narration file not found: {main_path}")
    obj = json.loads(main_path.read_text(encoding="utf-8"))
    videos = obj.get("videos") or []
    idx = {}
    for v in videos:
        vid = v.get("video_uid") or v.get("video_uid_")
        if vid:
            idx[str(vid)] = v
    return idx


def find_record(records: List[Dict[str, Any]], uid: str, frame: int, clip_only: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Attempt to find a record by matching:
      - video_uid + frame
      - clip_uid + clip_frame (or frame)
      - main_uid + frame
      - packed uid field (prefix match)
    Returns (record, match_type) where match_type describes which key matched.
    """
    for r in records:
        vid_uid = str(r.get("video_uid"))
        clip_uid = str(r.get("clip_uid")) if r.get("clip_uid") else None
        main_uid = str(r.get("main_uid")) if r.get("main_uid") else None
        packed_uid = str(r.get("uid")) if r.get("uid") else None
        f_global = int(r.get("frame", -1))
        f_clip = int(r.get("clip_frame", -1)) if r.get("clip_frame") is not None else None

        if not clip_only and vid_uid == uid and f_global == frame:
            return r, "video_uid+frame"
        if clip_uid == uid and (f_clip == frame or f_global == frame):
            return r, "clip_uid"
        if not clip_only and main_uid == uid and f_global == frame:
            return r, "main_uid"
        if packed_uid and packed_uid.startswith(f"{uid}_") and f_global == frame:
            return r, "packed_uid"
    return None, None


def pick_random(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return random.choice(records)


def find_narrations(main_idx: Dict[str, Any], video_uid: str, frame: int) -> List[Dict[str, Any]]:
    v = main_idx.get(video_uid)
    if not v:
        return []
    out = []
    for act in v.get("narrated_actions", []):
        try:
            start = int(act.get("action_start_frame", -1))
            end = int(act.get("action_end_frame", -1))
        except Exception:
            continue
        if start <= frame <= end:
            out.append(act)
    return out


def find_nearest_narrations(main_idx: Dict[str, Any], video_uid: str, frame: int, k: int = 3) -> List[Tuple[Dict[str, Any], int]]:
    v = main_idx.get(video_uid)
    if not v:
        return []
    best: List[Tuple[Dict[str, Any], int]] = []
    for act in v.get("narrated_actions", []):
        try:
            start = int(act.get("action_start_frame", -1))
            end = int(act.get("action_end_frame", -1))
        except Exception:
            continue
        if start <= frame <= end:
            best.append((act, 0))
            continue
        # distance to interval
        if frame < start:
            dist = start - frame
        else:
            dist = frame - end
        best.append((act, dist))
    best.sort(key=lambda x: x[1])
    return best[:k]


def resolve_image_path(record: Dict[str, Any], frames_root: Path, user_uid: Optional[str], clip_only: bool) -> Path:
    """
    Try multiple directory/frame combinations to locate the JPEG:
      - user_uid / frame
      - video_uid / frame
      - clip_uid / frame
      - clip_uid / clip_frame
    """
    vid_uid = str(record.get("video_uid"))
    clip_uid = str(record.get("clip_uid")) if record.get("clip_uid") else None
    frame = int(record.get("frame", -1))
    clip_frame = int(record.get("clip_frame", -1)) if record.get("clip_frame") is not None else None

    candidates: List[Tuple[Optional[str], int]] = []
    if user_uid:
        candidates.append((user_uid, frame))
    if clip_uid:
        candidates.append((clip_uid, frame))
        if clip_frame is not None and clip_frame >= 0:
            candidates.append((clip_uid, clip_frame))
    if not clip_only:
        candidates.append((vid_uid, frame))

    for uid, fr in candidates:
        if uid is None:
            continue
        p = frames_root / uid / f"{fr:07d}.jpg"
        if p.exists():
            return p
    # fallback to video_uid path even if missing
    return frames_root / vid_uid / f"{frame:07d}.jpg"


def draw_sample(record: Dict[str, Any], noun_labels: Dict[int, str], verb_labels: Dict[int, str], frames_root: Path, narrs: List[Dict[str, Any]], user_uid: Optional[str], match_type: Optional[str], main_idx: Dict[str, Any], clip_only: bool) -> None:
    uid = str(record.get("video_uid"))
    frame = int(record.get("frame", -1))
    img_path = resolve_image_path(record, frames_root, user_uid, clip_only)
    boxes = []
    for obj in record.get("objects", []) or []:
        box = obj.get("box") or obj.get("bbox") or obj.get("gt_box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            boxes.append((box, obj))

    # Print info
    print(f"\n=== Sample uid={uid} frame={frame} (matched via {match_type or 'video_uid'}) ===")
    if user_uid and user_uid != uid:
        print(f"[info] user_uid={user_uid}")
    print(f"Image path: {img_path}")
    print(f"Objects: {len(boxes)}")
    for i, (box, obj) in enumerate(boxes, 1):
        n = obj.get("noun_category_id")
        v = obj.get("verb_category_id")
        print(f"  Obj {i}: box={box} noun_id={n} ({noun_labels.get(int(n), '<unk>') if n is not None else '<none>'}) "
              f"verb_id={v} ({verb_labels.get(int(v), '<unk>') if v is not None else '<none>'}) "
              f"ttc={obj.get('time_to_contact')}")
    if narrs:
        print("\nNarrations covering this frame (source of truth):")
        for j, a in enumerate(narrs, 1):
            # Compare STA verb to structured_verb for a quick sanity check
            structured = a.get('structured_verb')
            structured_norm = norm_str(structured)
            sta_verbs = [obj.get("verb_category_id") for _, obj in boxes]
            verb_mismatch = []
            for vid in sta_verbs:
                lbl = norm_str(verb_labels.get(int(vid), "")) if vid is not None else ""
                if lbl and structured_norm and lbl != structured_norm:
                    verb_mismatch.append(f"{vid}->{verb_labels.get(int(vid), '')}")
            print(f"  [{j}] narration_text='{a.get('narration_text')}' "
                  f"structured_verb={a.get('structured_verb')} freeform_verb={a.get('freeform_verb')} "
                  f"action_start_frame={a.get('action_start_frame')} action_end_frame={a.get('action_end_frame')}")
            if verb_mismatch:
                print(f"      [warn] STA verb labels differ from structured_verb: {verb_mismatch}")
    else:
        nearest = find_nearest_narrations(main_idx, uid, frame, k=3)
        print("\nNo narrated actions covering this frame. Nearest narrated actions (for reference):")
        if not nearest:
            print("  <none for this video>")
        for j, (a, dist) in enumerate(nearest, 1):
            print(f"  [{j}] dist={dist} frames | narration_text='{a.get('narration_text')}' "
                  f"structured_verb={a.get('structured_verb')} freeform_verb={a.get('freeform_verb')} "
                  f"action_start_frame={a.get('action_start_frame')} action_end_frame={a.get('action_end_frame')}")

    if not img_path.exists():
        print("[warn] image not found on disk; skipping visualization.")
        return

    try:
        img = Image.open(str(img_path)).convert("RGB")
    except Exception as e:
        print(f"[warn] failed to open image {img_path}: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    for box, obj in boxes:
        x1, y1, x2, y2 = box
        n = obj.get("noun_category_id")
        v = obj.get("verb_category_id")
        lbl = f"{noun_labels.get(int(n), n)} | {verb_labels.get(int(v), v)}"
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, y1 - 5, lbl, color="lime", fontsize=8, bbox=dict(facecolor="black", alpha=0.5, pad=1))
    ax.set_title(f"{uid} frame {frame}")
    ax.axis("off")
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity check noun/verb labels against narration and STA metadata.")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"], help="STA split to use.")
    ap.add_argument("--uid", type=str, default=None, help="Video UID to inspect.")
    ap.add_argument("--frame", type=int, default=None, help="Frame index to inspect.")
    ap.add_argument("--batch_pairs", type=str, nargs="*", default=None, help="Optional list of uid:frame entries to inspect (e.g., 002e11bc-...:8939).")
    ap.add_argument("--k", type=int, default=1, help="Number of random samples if uid/frame not provided.")
    ap.add_argument("--org_root", type=str, default="local_extraction/v2/org_annotations", help="Root folder for STA and main JSONs.")
    ap.add_argument("--frames_root", type=str, default="local_extraction/v2/extracted_frames", help="Root of extracted frames.")
    ap.add_argument("--noun_tax", type=str, default="local_extraction/v2/org_annotations/fho_main_taxonomy.json", help="Noun taxonomy (CSV or JSON).")
    ap.add_argument("--verb_tax", type=str, default="local_extraction/v2/org_annotations/fho_lta_taxonomy.json", help="Verb taxonomy (CSV or JSON).")
    ap.add_argument("--clip_only", action="store_true", help="If set, only match clip_uid/clip_frame (ignore video_uid/main_uid).")
    args = ap.parse_args()

    org_root = Path(args.org_root)
    main_path = org_root / "fho_main.json"
    frames_root = Path(args.frames_root)

    noun_labels = load_tax_labels(Path(args.noun_tax), kind="noun")
    verb_labels = load_tax_labels(Path(args.verb_tax), kind="verb")
    main_idx = load_fho_main(main_path)

    # Pick samples
    samples: List[Tuple[Dict[str, Any], Optional[str]]] = []
    split_used = args.split
    sta_path = org_root / f"fho_sta_{split_used}_height-540.json"
    sta_records = load_sta_records(sta_path)

    pairs: List[Tuple[str, int]] = list(DEFAULT_BATCH_PAIRS)
    if args.batch_pairs:
        for entry in args.batch_pairs:
            if ":" not in entry:
                print(f"[warn] Ignoring malformed pair '{entry}' (expected uid:frame)")
                continue
            uid_str, frame_str = entry.split(":", 1)
            try:
                frame_i = int(frame_str)
                pairs.append((uid_str, frame_i))
            except Exception:
                print(f"[warn] Ignoring malformed frame in pair '{entry}'")

    match_type: Optional[str] = None
    if pairs:
        for uid_in, frame_in in pairs:
            rec, mtype = find_record(sta_records, uid_in, frame_in, clip_only=args.clip_only)
            used_split = split_used
            if rec is None:
                other_split = "train" if split_used == "val" else "val"
                other_path = org_root / f"fho_sta_{other_split}_height-540.json"
                try:
                    other_records = load_sta_records(other_path)
                    rec, mtype = find_record(other_records, uid_in, frame_in, clip_only=args.clip_only)
                    if rec is not None:
                        used_split = other_split
                        print(f"[info] Found uid={uid_in} frame={frame_in} in {other_split} split (not in {split_used}).")
                except Exception:
                    pass
            if rec is None:
                print(f"[warn] No STA record found for uid={uid_in} frame={frame_in} in train or val splits.")
                continue
            samples.append((rec, mtype))
            print(f"[info] Using split={used_split} for uid={uid_in} frame={frame_in}")
    elif args.uid and args.frame is not None:
        rec, match_type = find_record(sta_records, args.uid, args.frame, clip_only=args.clip_only)
        if rec is None:
            # try the other split
            other_split = "train" if split_used == "val" else "val"
            other_path = org_root / f"fho_sta_{other_split}_height-540.json"
            try:
                other_records = load_sta_records(other_path)
                rec, match_type = find_record(other_records, args.uid, args.frame, clip_only=args.clip_only)
                if rec is not None:
                    split_used = other_split
                    sta_records = other_records
                    print(f"[info] Found uid={args.uid} frame={args.frame} in {other_split} split (not in {args.split}).")
            except Exception:
                pass
        if rec is None:
            raise ValueError(f"No STA record found for uid={args.uid} frame={args.frame} in train or val splits.")
        samples = [(rec, match_type)]
        print(f"[info] Using split={split_used}")
    else:
        for _ in range(max(1, args.k)):
            samples.append((pick_random(sta_records), "video_uid"))

    for rec, mtype in samples:
        uid = str(rec.get("video_uid"))
        frame = int(rec.get("frame", -1))
        narrs = find_narrations(main_idx, uid, frame)
        draw_sample(rec, noun_labels, verb_labels, frames_root, narrs, args.uid, mtype, main_idx, args.clip_only)


if __name__ == "__main__":
    main()
