#!/usr/bin/env python3
"""
Clip-centric noun/verb sanity checker

This script works strictly in clip_uid space. It cross-references STA clip
annotations (fho_sta_*_height-540.json) with narrations from fho_main.json to
check noun/verb consistency and visualize frames.

Features:
- Clip index from fho_main.json (narrations per clip_uid with clip-relative intervals)
- STA records filtered to entries with clip_uid and clip_frame
- Matching and visualization by (clip_uid, clip_frame) only (optional fallback off)
- Nearest narrated actions if none cover the frame
- Simple heuristics/flags (no_narration, verb_mismatch, suspicious_box)
- Interactive mode (single clip/frame or random samples)
- Batch mode to dump flagged rows to CSV/JSONL
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Rectangle, Patch  # type: ignore
from PIL import Image

# Optional hardcoded batch pairs; set tuples of (clip_uid, clip_frame) to avoid passing via CLI.
DEFAULT_BATCH_PAIRS: List[Tuple[str, int]] = [("71e2f5f3-fb8e-4fdd-822f-644a8996fb3e", 8736), ("ebbe9fdf-babf-46e8-87ac-c4adf19a34e8", 4830), ("7d072357-cabe-4ed8-bbec-52475b890301", 9177), ("0d56e589-e0c5-48c6-a005-101f1493d9ea", 4832), ("4f9caa52-51cd-4026-9393-261a3f10a95f", 8374)]


# ---------------------- Taxonomy helpers ---------------------- #
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
    start = 1 if lines and ("label" in lines[0] or "group" in lines[0]) else 0
    for i, ln in enumerate(lines[start:]):
        parts = ln.split(",")
        label = parts[0].strip('" ')
        labels[i] = label
    return labels


def norm_str(s: Optional[str]) -> str:
    return str(s or "").strip().lower().replace("_", " ")


# ---------------------- Clip index ---------------------- #
def build_clip_index(main_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build clip-level index from fho_main.json.

    Returns:
      clip_index[clip_uid] = {
        "video_uid": str,
        "actions": [ {clip_start_frame, clip_end_frame, start_frame, end_frame,
                      narration_text, structured_verb, freeform_verb}, ... ]
      }
    """
    if not main_path.exists():
        raise FileNotFoundError(f"Main narration file not found: {main_path}")
    obj = json.loads(main_path.read_text(encoding="utf-8"))
    videos = obj.get("videos") or []
    idx: Dict[str, Dict[str, Any]] = {}
    for v in videos:
        vid = v.get("video_uid") or v.get("video_uid_")
        intervals = v.get("annotated_intervals") or []
        for inter in intervals:
            clip_uid = inter.get("clip_uid")
            if not clip_uid:
                continue
            actions = inter.get("narrated_actions") or []
            act_list = []
            for a in actions:
                act_list.append({
                    "clip_start_frame": a.get("clip_start_frame"),
                    "clip_end_frame": a.get("clip_end_frame"),
                    "start_frame": a.get("action_start_frame"),
                    "end_frame": a.get("action_end_frame"),
                    "narration_text": a.get("narration_text"),
                    "structured_verb": a.get("structured_verb"),
                    "freeform_verb": a.get("freeform_verb"),
                })
            idx[str(clip_uid)] = {
                "video_uid": vid,
                "actions": act_list,
            }
    return idx


def find_clip_narrations(clip_index: Dict[str, Any], clip_uid: str, clip_frame: int) -> List[Dict[str, Any]]:
    clip = clip_index.get(clip_uid)
    if not clip:
        return []
    hits: List[Dict[str, Any]] = []
    for a in clip.get("actions", []):
        cs = a.get("clip_start_frame")
        ce = a.get("clip_end_frame")
        if cs is not None and ce is not None:
            if int(cs) <= clip_frame <= int(ce):
                hits.append(a)
        else:
            # fallback to global action frames if clip frames missing
            gs = a.get("start_frame")
            ge = a.get("end_frame")
            if gs is not None and ge is not None and int(gs) <= clip_frame <= int(ge):
                hits.append(a)
    return hits


def find_nearest_clip_narrations(clip_index: Dict[str, Any], clip_uid: str, clip_frame: int, k: int = 3) -> List[Tuple[Dict[str, Any], int]]:
    clip = clip_index.get(clip_uid)
    if not clip:
        return []
    best: List[Tuple[Dict[str, Any], int]] = []
    for a in clip.get("actions", []):
        cs = a.get("clip_start_frame")
        ce = a.get("clip_end_frame")
        if cs is not None and ce is not None:
            cs_i, ce_i = int(cs), int(ce)
        else:
            cs_i = int(a.get("start_frame", -1))
            ce_i = int(a.get("end_frame", -1))
        if cs_i <= clip_frame <= ce_i:
            best.append((a, 0))
            continue
        dist = min(abs(clip_frame - cs_i), abs(clip_frame - ce_i))
        best.append((a, dist))
    best.sort(key=lambda x: x[1])
    return best[:k]


# ---------------------- STA load ---------------------- #
def load_sta_clip_records(sta_path: Path) -> List[Dict[str, Any]]:
    if not sta_path.exists():
        raise FileNotFoundError(f"STA file not found: {sta_path}")
    data = json.loads(sta_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data.get("annotations") or []
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unexpected STA format in {sta_path}")
    out: List[Dict[str, Any]] = []
    for r in records:
        if r.get("clip_uid") is None or r.get("clip_frame") is None:
            continue
        out.append(r)
    return out


# ---------------------- Matching ---------------------- #
def find_record_clip(records: List[Dict[str, Any]], clip_uid: str, clip_frame: int) -> Optional[Dict[str, Any]]:
    for r in records:
        if str(r.get("clip_uid")) == clip_uid and int(r.get("clip_frame", -1)) == int(clip_frame):
            return r
    return None


# ---------------------- Paths ---------------------- #
def resolve_clip_image_path(clip_uid: str, clip_frame: int, frames_root: Path) -> Path:
    p = frames_root / clip_uid / f"{clip_frame:07d}.jpg"
    return p


# ---------------------- Heuristics ---------------------- #
def suspicious_box(box: List[float]) -> bool:
    if len(box) != 4:
        return True
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return True
    if x1 < -5 or y1 < -5:
        return True
    # heuristic: huge boxes in 1080/1440 frames
    w = x2 - x1
    h = y2 - y1
    if w > 2200 or h > 1800:
        return True
    return False


def verb_mismatch_flag(verb_id: Optional[int], structured_verb: Optional[str], verb_labels: Dict[int, str]) -> bool:
    if verb_id is None or structured_verb is None:
        return False
    gt_label = norm_str(verb_labels.get(int(verb_id), ""))
    narr_label = norm_str(structured_verb)
    return gt_label != "" and narr_label != "" and gt_label != narr_label


# ---------------------- Visualization ---------------------- #
def draw_sample(
    record: Dict[str, Any],
    noun_labels: Dict[int, str],
    verb_labels: Dict[int, str],
    frames_root: Path,
    clip_index: Dict[str, Any],
) -> None:
    clip_uid = str(record.get("clip_uid"))
    clip_frame = int(record.get("clip_frame", -1))
    video_uid = str(record.get("video_uid"))
    img_path = resolve_clip_image_path(clip_uid, clip_frame, frames_root)
    narrs = find_clip_narrations(clip_index, clip_uid, clip_frame)

    print(f"\n=== Clip {clip_uid} frame {clip_frame} (video_uid={video_uid}) ===")
    print(f"Image path: {img_path}")
    boxes = []
    for obj in record.get("objects", []) or []:
        box = obj.get("box") or obj.get("bbox") or obj.get("gt_box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            boxes.append((box, obj))
    print(f"Objects: {len(boxes)}")
    for i, (box, obj) in enumerate(boxes, 1):
        n = obj.get("noun_category_id")
        v = obj.get("verb_category_id")
        print(f"  Obj {i}: box={box} noun_id={n} ({noun_labels.get(int(n), '<unk>') if n is not None else '<none>'}) "
              f"verb_id={v} ({verb_labels.get(int(v), '<unk>') if v is not None else '<none>'}) "
              f"ttc={obj.get('time_to_contact')}")

    if narrs:
        print("\nNarrations covering this clip frame:")
        for j, a in enumerate(narrs, 1):
            print(f"  [{j}] narration_text='{a.get('narration_text')}' "
                  f"structured_verb={a.get('structured_verb')} freeform_verb={a.get('freeform_verb')} "
                  f"clip_start_frame={a.get('clip_start_frame')} clip_end_frame={a.get('clip_end_frame')}")
    else:
        nearest = find_nearest_clip_narrations(clip_index, clip_uid, clip_frame, k=3)
        print("\nNo narrated actions covering this frame. Nearest in this clip:")
        if not nearest:
            print("  <none for this clip_uid>")
        for j, (a, dist) in enumerate(nearest, 1):
            print(f"  [{j}] dist={dist} frames | narration_text='{a.get('narration_text')}' "
                  f"structured_verb={a.get('structured_verb')} freeform_verb={a.get('freeform_verb')} "
                  f"clip_start_frame={a.get('clip_start_frame')} clip_end_frame={a.get('clip_end_frame')}")

    if not img_path.exists():
        print("[warn] image not found on disk; skipping visualization.")
        return
    try:
        img = Image.open(str(img_path)).convert("RGB")
    except Exception as e:
        print(f"[warn] failed to open image {img_path}: {e}")
        return

    main_narr = narrs[0] if narrs else None
    narr_struct = main_narr.get("structured_verb") if main_narr else None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    for box, obj in boxes:
        x1, y1, x2, y2 = box
        n = obj.get("noun_category_id")
        v = obj.get("verb_category_id")
        lbl = f"{noun_labels.get(int(n), n)} | {verb_labels.get(int(v), v)}"
        if narr_struct:
            lbl += f" | narr:{narr_struct}"
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, y1 - 5, lbl, color="lime", fontsize=8, bbox=dict(facecolor="black", alpha=0.5, pad=1))
    ax.set_title(f"{clip_uid} frame {clip_frame}")
    legend_patches = [
        Patch(facecolor="lime", edgecolor="lime", label="STA noun/verb (bbox+label)"),
        Patch(facecolor="orange", edgecolor="orange", label="Narration verb (structured)"),
    ]
    if narr_struct:
        ax.text(5, 15, f"Narration: {narr_struct}", color="orange", fontsize=9, bbox=dict(facecolor="black", alpha=0.4, pad=2))
    ax.legend(handles=legend_patches, loc="lower right")
    ax.axis("off")
    plt.show()


# ---------------------- Batch scan ---------------------- #
def scan_records(
    records: List[Dict[str, Any]],
    clip_index: Dict[str, Any],
    noun_labels: Dict[int, str],
    verb_labels: Dict[int, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in records:
        clip_uid = str(r.get("clip_uid"))
        clip_frame = int(r.get("clip_frame", -1))
        video_uid = str(r.get("video_uid"))
        narrs = find_clip_narrations(clip_index, clip_uid, clip_frame)
        nearest = find_nearest_clip_narrations(clip_index, clip_uid, clip_frame, k=1)
        flags: List[str] = []
        if not narrs:
            flags.append("no_narration")
        # For first object only (heuristic)
        obj = (r.get("objects") or [{}])[0]
        box = obj.get("box") or obj.get("bbox") or obj.get("gt_box") or []
        n_id = obj.get("noun_category_id")
        v_id = obj.get("verb_category_id")
        if suspicious_box(box):
            flags.append("suspicious_box")
        best_narr = narrs[0] if narrs else (nearest[0][0] if nearest else None)
        structured = best_narr.get("structured_verb") if best_narr else None
        if verb_mismatch_flag(v_id, structured, verb_labels):
            flags.append("verb_mismatch")
        rows.append({
            "clip_uid": clip_uid,
            "clip_frame": clip_frame,
            "video_uid": video_uid,
            "noun_id": n_id,
            "noun_label": noun_labels.get(int(n_id), "") if n_id is not None else "",
            "verb_id": v_id,
            "verb_label": verb_labels.get(int(v_id), "") if v_id is not None else "",
            "has_covering_narration": bool(narrs),
            "narration_text": best_narr.get("narration_text") if best_narr else "",
            "structured_verb": structured,
            "freeform_verb": best_narr.get("freeform_verb") if best_narr else "",
            "flags": ",".join(flags) if flags else "",
        })
    return rows


def write_rows(rows: List[Dict[str, Any]], out_csv: Optional[Path], out_jsonl: Optional[Path]) -> None:
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[info] wrote CSV: {out_csv}")
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[info] wrote JSONL: {out_jsonl}")


# ---------------------- CLI ---------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Clip-centric noun/verb sanity checker.")
    ap.add_argument("--clip_uid", type=str, default=None, help="Clip UID to inspect.")
    ap.add_argument("--clip_frame", type=int, default=None, help="Clip-relative frame index to inspect.")
    ap.add_argument("--k", type=int, default=1, help="Number of random samples if clip_uid/frame not provided.")
    ap.add_argument("--mode", type=str, default="interactive", choices=["interactive", "batch"], help="interactive=visualize, batch=write flagged rows.")
    ap.add_argument("--out_csv", type=str, default=None, help="Batch mode: CSV output path.")
    ap.add_argument("--out_jsonl", type=str, default=None, help="Batch mode: JSONL output path.")
    ap.add_argument("--org_root", type=str, default="local_extraction/v2/org_annotations", help="Root folder for STA and main JSONs.")
    ap.add_argument("--frames_root", type=str, default="local_extraction/v2/extracted_frames", help="Root of extracted frames (clip_uid/0001234.jpg).")
    ap.add_argument("--noun_tax", type=str, default="local_extraction/v2/org_annotations/fho_main_taxonomy.json", help="Noun taxonomy (CSV or JSON).")
    ap.add_argument("--verb_tax", type=str, default="local_extraction/v2/org_annotations/fho_lta_taxonomy.json", help="Verb taxonomy (CSV or JSON).")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "both"], help="STA split to use (or both).")
    ap.add_argument("--batch_pairs", type=str, nargs="*", default=None, help="Optional list of clip_uid:clip_frame entries to inspect.")
    ap.add_argument("--visualize_batch", action="store_true", help="After batch scan, visualize the selected pairs with overlays.")
    ap.add_argument("--no_default_pairs", action="store_true", help="Ignore DEFAULT_BATCH_PAIRS; rely on --batch_pairs or random/k.")
    args = ap.parse_args()

    org_root = Path(args.org_root)
    main_path = org_root / "fho_main.json"
    sta_paths: List[Path] = []
    if args.split in ("train", "both"):
        sta_paths.append(org_root / "fho_sta_train_height-540.json")
    if args.split in ("val", "both"):
        sta_paths.append(org_root / "fho_sta_val_height-540.json")
    sta_records: List[Dict[str, Any]] = []
    for sp in sta_paths:
        try:
            sta_records.extend(load_sta_clip_records(sp))
        except FileNotFoundError as e:
            print(f"[warn] {e}")
    frames_root = Path(args.frames_root)

    noun_labels = load_tax_labels(Path(args.noun_tax), kind="noun")
    verb_labels = load_tax_labels(Path(args.verb_tax), kind="verb")
    clip_index = build_clip_index(main_path)

    # Build pair list (hardcoded + CLI)
    pairs: List[Tuple[str, int]] = [] if args.no_default_pairs else list(DEFAULT_BATCH_PAIRS)
    if args.batch_pairs:
        for entry in args.batch_pairs:
            if ":" not in entry:
                print(f"[warn] Ignoring malformed pair '{entry}' (expected clip_uid:clip_frame)")
                continue
            cu, fr = entry.split(":", 1)
            try:
                pairs.append((cu, int(fr)))
            except Exception:
                print(f"[warn] Ignoring malformed frame in pair '{entry}'")

    if args.mode == "batch":
        selected: List[Dict[str, Any]] = []
        if pairs:
            for cu, fr in pairs:
                rec = find_record_clip(sta_records, cu, fr)
                if rec:
                    selected.append(rec)
                else:
                    print(f"[warn] No STA record found for clip_uid={cu} clip_frame={fr} in split={args.split}")
        else:
            selected = sta_records

        rows = scan_records(selected, clip_index, noun_labels, verb_labels)
        out_csv = Path(args.out_csv) if args.out_csv else None
        out_jsonl = Path(args.out_jsonl) if args.out_jsonl else None
        if rows:
            write_rows(rows, out_csv, out_jsonl)
        else:
            print("[warn] no rows to write")
        if args.visualize_batch and selected:
            for rec in selected:
                draw_sample(rec, noun_labels, verb_labels, frames_root, clip_index)
        return

    # interactive
    samples: List[Dict[str, Any]] = []

    if pairs:
        for cu, fr in pairs:
            rec = find_record_clip(sta_records, cu, fr)
            if rec:
                samples.append(rec)
            else:
                print(f"[warn] No STA record found for clip_uid={cu} clip_frame={fr} in split={args.split}")
    elif args.clip_uid and args.clip_frame is not None:
        rec = find_record_clip(sta_records, args.clip_uid, args.clip_frame)
        if rec is None:
            raise ValueError(f"No STA record found for clip_uid={args.clip_uid} clip_frame={args.clip_frame} in split={args.split}")
        samples = [rec]
    else:
        # random clip-based sampling for diversity
        by_clip: Dict[str, List[Dict[str, Any]]] = {}
        for r in sta_records:
            cu = str(r.get("clip_uid"))
            by_clip.setdefault(cu, []).append(r)
        clips = list(by_clip.keys())
        for _ in range(max(1, args.k)):
            if not clips:
                break
            cu = random.choice(clips)
            rec = random.choice(by_clip[cu])
            samples.append(rec)

    for rec in samples:
        draw_sample(rec, noun_labels, verb_labels, frames_root, clip_index)


if __name__ == "__main__":
    main()
