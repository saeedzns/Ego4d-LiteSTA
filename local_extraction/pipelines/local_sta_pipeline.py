#!/usr/bin/env python3
"""
Local STA data pipeline (no downloads, no extraction).

Reads already-extracted frames and locally generated YOLO labels under
  - frames: local_extraction/<version>/extracted_frames/<uid>/<frame:07d>.jpg
  - labels: local_extraction/<version>/yolo_labels_540/{clips|videos}/<uid>/<frame:07d>.txt

Provides:
  - sanity: group frames by image_dim (short side), summarize dims, write Markdown
  - stats:  label/object counts, class distribution, images without labels
  - sample: draw YOLO boxes on images and save a few visualizations
  - labels: create missing YOLO labels from Drive annotations, skipping ones that already exist
  - (optional) torch dataset: if torch available, build a Dataset/DataLoader

Usage examples (from repo root):
  # Sanity report
  python local_extraction/local_sta_pipeline.py sanity \
    --frames-root local_extraction/v2/extracted_frames \
    --out local_extraction/sanity_local_report.md

  # Label stats report
  python local_extraction/local_sta_pipeline.py stats \
    --label-root local_extraction/v2/yolo_labels_540 \
    --frames-root local_extraction/v2/extracted_frames \
    --space clips --out local_extraction/label_stats_report.md

  # Visualize 8 samples
  python local_extraction/local_sta_pipeline.py sample \
    --label-root local_extraction/v2/yolo_labels_540 \
    --frames-root local_extraction/v2/extracted_frames \
    --space clips --num 8 --out-dir local_extraction/samples

  # Create ONLY missing labels (idempotent)
  python local_extraction/local_sta_pipeline.py labels \
    --label-root local_extraction/v2/yolo_labels_540 \
    --frames-root local_extraction/v2/extracted_frames \
    --space clips --use-540 --missing-log local_extraction/missing_images_clips.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Tuple, Optional
import random

# ----------- In-code toggles (edit here, no CLI needed) -----------
DEFAULT_CMD = 'sample'  # one of: 'sanity', 'stats', 'sample', 'labels'

DEFAULT_FRAMES_ROOT = Path('local_extraction') / 'v2' / 'extracted_frames'
DEFAULT_LABEL_ROOT  = Path('local_extraction') / 'v2' / 'yolo_labels_540'

# For stats/sample/labels
DEFAULT_SPACE       = 'clips'   # 'clips' or 'videos'

# For sanity/stats
DEFAULT_SANITY_OUT  = Path('local_extraction') / 'sanity_local_report.md'
DEFAULT_STATS_OUT   = Path('local_extraction') / 'label_stats_report.md'
DEFAULT_SAMPLE_OUT_DIR = Path('local_extraction') / 'samples'
DEFAULT_SAMPLE_NUM  = 8
DEFAULT_SAMPLE_PER_GROUP = 8

# For labels creation
DEFAULT_USE_540     = True      # use height-540 JSONs
DEFAULT_OVERWRITE   = False     # do not overwrite existing labels
DEFAULT_MISSING_LOG = Path('local_extraction') / 'logs' / 'missing_images_clips.txt'


# ---------- image size helpers (no external deps required) ----------
def _read_size_raw(img_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        with open(img_path, 'rb') as f:
            head = f.read(32)
            # PNG
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                f.seek(16)
                w = int.from_bytes(f.read(4), 'big')
                h = int.from_bytes(f.read(4), 'big')
                return w, h
            # JPEG
            if head[0:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    if not b:
                        break
                    while b != b"\xff":
                        b = f.read(1)
                        if not b:
                            return None, None
                    # skip padding FFs
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
                        f.read(1)  # precision
                        h = int.from_bytes(f.read(2), 'big')
                        w = int.from_bytes(f.read(2), 'big')
                        return w, h
                    f.seek(seglen - 2, 1)
    except Exception:
        return None, None
    return None, None


def read_image_size(img_path: Path) -> Tuple[Optional[int], Optional[int]]:
    # Try Pillow
    try:
        from PIL import Image  # type: ignore
        with Image.open(str(img_path)) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        pass
    # Try imageio
    try:
        import imageio.v2 as imageio  # type: ignore
        im = imageio.imread(str(img_path))
        if hasattr(im, 'shape') and len(im.shape) >= 2:
            h, w = int(im.shape[0]), int(im.shape[1])
            return w, h
    except Exception:
        pass
    # Fallback
    return _read_size_raw(img_path)


# ---------- Drive annotation helpers (for on-demand label creation) ----------
def detect_drive_root() -> str:
    env = os.environ.get("EGO4D_ROOT")
    if env:
        return env
    for p in [
        "H:/My Drive/ego4d_data",
        "D:/My Drive/ego4d_data",
        "/mnt/h/My Drive/ego4d_data",
        "/mnt/d/My Drive/ego4d_data",
        "/content/drive/MyDrive/ego4d_data",
        "/drive/MyDrive/ego4d_data",
    ]:
        try:
            if os.path.isdir(p):
                return p
        except Exception:
            pass
    return "H:/My Drive/ego4d_data"


def _iter_list(obj):
    if isinstance(obj, dict):
        if "annotations" in obj and isinstance(obj["annotations"], list):
            return obj["annotations"]
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
    if isinstance(obj, list):
        return obj
    return []


def _get_dims(rec: dict):
    for (kw, kh) in [("width","height"),("frame_width","frame_height"),("img_w","img_h")]:
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


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def load_sta_records(ego4d_root: str, version: str, use_540: bool) -> list:
    import json
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
        with open(path, 'r') as f:
            raw = json.load(f)
        for rec in _iter_list(raw):
            vuid = rec.get("video_uid"); cuid = rec.get("clip_uid")
            vfr = _as_int(rec.get("frame")); cfr = _as_int(rec.get("clip_frame"))
            wj, hj = _get_dims(rec)
            for obj in rec.get("objects", []) or []:
                box = obj.get("box") or obj.get("bbox") or obj.get("gt_box")
                if not (isinstance(box, (list,tuple)) and len(box)==4):
                    continue
                try:
                    x1,y1,x2,y2 = map(float, box)
                except Exception:
                    continue
                noun = _as_int(obj.get("noun_category_id", obj.get("noun_id", 0)))
                out.append({
                    'split': split,
                    'video_uid': vuid,
                    'clip_uid': cuid,
                    'video_frame': vfr,
                    'clip_frame': cfr,
                    'box': (x1,y1,x2,y2),
                    'noun_id': 0 if noun is None else int(noun),
                    'json_w': wj,
                    'json_h': hj,
                })
    return out


def write_missing_labels(label_root: Path, frames_root: Path, space: str, use_540: bool, overwrite: bool=False, missing_log: Optional[Path]=None) -> None:
    ego4d_root = detect_drive_root()
    version = 'v2'
    recs = load_sta_records(ego4d_root, version, use_540)
    if not recs:
        print(f"[labels] No STA records loaded from {ego4d_root}/... (use_540={use_540})")
        return
    print(f"[labels] Loaded {len(recs)} records from STA JSONs (use_540={use_540})")

    # group
    groups: Dict[Tuple[str,int], List[dict]] = defaultdict(list)
    for r in recs:
        if space=='videos' and (r['video_uid'] and r['video_frame'] is not None):
            key = (str(r['video_uid']), int(r['video_frame']))
            groups[key].append(r)
        if space=='clips' and (r['clip_uid'] and r['clip_frame'] is not None):
            key = (str(r['clip_uid']), int(r['clip_frame']))
            groups[key].append(r)

    written = 0
    skipped_exist = 0
    skipped_missing_img = 0
    skipped_nodims = 0
    missing_entries: List[str] = []

    for (uid, frame) in groups.keys():
        img_path = frames_root / uid / f"{int(frame):07d}.jpg"
        if not img_path.exists():
            skipped_missing_img += 1
            missing_entries.append(f"{uid},{int(frame):07d},{img_path}")
            continue
        lbl_path = label_root / space / uid / f"{int(frame):07d}.txt"
        if lbl_path.exists() and not overwrite:
            skipped_exist += 1
            continue
        W, H = read_image_size(img_path)
        if not (W and H and W>0 and H>0):
            skipped_nodims += 1
            continue
        items = groups[(uid, frame)]
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lbl_path, 'w') as g:
            for r in items:
                x1,y1,x2,y2 = r['box']
                wj, hj = r.get('json_w'), r.get('json_h')
                # scale from JSON dims if known
                if wj and hj and (wj>0 and hj>0) and (wj!=W or hj!=H):
                    sx = W/float(wj); sy = H/float(hj)
                    x1,y1,x2,y2 = x1*sx, y1*sy, x2*sx, y2*sy
                w = max(1.0, x2-x1); h = max(1.0, y2-y1)
                cx = x1 + w/2.0; cy = y1 + h/2.0
                cls = int(r.get('noun_id',0))
                g.write(f"{cls} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}\n")
        written += 1

    print(f"[labels] Done. Written={written} | skipped_exist={skipped_exist} | missing_img={skipped_missing_img} | nodims={skipped_nodims}")
    # Write missing images log if requested (or if default file provided by caller)
    if missing_log and missing_entries:
        try:
            missing_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        missing_log.write_text("\n".join(missing_entries) + "\n", encoding='utf-8')
        print(f"[labels] Missing images log → {missing_log} ({len(missing_entries)} lines)")

# ---------- data iterators ----------
def iter_frames(frames_root: Path) -> Iterable[Path]:
    for uid_dir in sorted(frames_root.iterdir()):
        if not uid_dir.is_dir():
            continue
        for p in uid_dir.iterdir():
            if p.is_file() and p.suffix.lower() == '.jpg':
                yield p


def yolo_label_path(label_root: Path, space: str, uid: str, frame: int) -> Path:
    return label_root / space / uid / f"{int(frame):07d}.txt"


def parse_yolo_label(path: Path) -> List[Tuple[int, float, float, float, float]]:
    out: List[Tuple[int, float, float, float, float]] = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0])); cx = float(parts[1]); cy = float(parts[2]); ww = float(parts[3]); hh = float(parts[4])
            out.append((cls, cx, cy, ww, hh))
        except Exception:
            continue
    return out


# ---------- commands ----------
def cmd_sanity(frames_root: Path, out_md: Path, sample_per_group: int = 8) -> None:
    total = 0
    unknown = 0
    by_short: Counter[int] = Counter()
    by_pair: Counter[Tuple[int, int]] = Counter()
    examples: Dict[int, List[Path]] = defaultdict(list)

    for img in iter_frames(frames_root):
        total += 1
        w, h = read_image_size(img)
        if not (w and h and w > 0 and h > 0):
            unknown += 1
            continue
        short = min(w, h)
        by_short[short] += 1
        by_pair[(w, h)] += 1
        if len(examples[short]) < sample_per_group:
            examples[short].append(img)

    lines: List[str] = []
    lines.append(f"# Sanity Check (local)\n")
    lines.append(f"Root: `{frames_root}`\n")
    lines.append("## Overview")
    lines.append(f"- Total frames: {total}")
    lines.append(f"- Unknown (no size): {unknown}\n")
    lines.append("## By short side")
    for s, c in by_short.most_common():
        pct = 100.0 * c / max(1, total)
        lines.append(f"- {s}: {c} ({pct:.1f}%)")
    lines.append("")
    lines.append("## Top dimension pairs")
    for (w, h), c in by_pair.most_common(12):
        pct = 100.0 * c / max(1, total)
        lines.append(f"- {w}x{h}: {c} ({pct:.1f}%)")
    lines.append("")
    lines.append("## Examples per short side")
    for s, ex in sorted(examples.items(), key=lambda kv: (-by_short[kv[0]], kv[0])):
        lines.append(f"- {s}:")
        for p in ex:
            lines.append(f"  - `{p}`")
    lines.append("")
    if by_short:
        dom, cnt = by_short.most_common(1)[0]
        pct = 100.0 * cnt / max(1, total)
        lines.append("## Interpretation")
        if dom == 540:
            lines.append(f"- Dominant short side is 540 ({pct:.1f}%). Looks consistent with 540p normalization.")
        else:
            lines.append(f"- Dominant short side is {dom} ({pct:.1f}%). Consider consistent resizing to 540 if needed.")

    md = "\n".join(lines) + "\n"
    print(md)
    out_md.write_text(md, encoding='utf-8')
    print(f"[sanity] wrote → {out_md}")


def cmd_stats(label_root: Path, frames_root: Path, space: str, out_md: Path) -> None:
    total_imgs = 0
    imgs_with_labels = 0
    total_objs = 0
    class_counts: Counter[int] = Counter()
    missing_img = 0
    missing_label = 0
    invalid_label = 0

    # Enumerate by scanning frames to ensure image existence; then look for labels
    for img in iter_frames(frames_root):
        # Expect path: <frames_root>/<uid>/<frame>.jpg
        uid = img.parent.name
        try:
            frame_idx = int(img.stem)
        except Exception:
            continue
        total_imgs += 1
        lbl = yolo_label_path(label_root, space, uid, frame_idx)
        if not lbl.exists():
            missing_label += 1
            continue
        labels = parse_yolo_label(lbl)
        if not labels:
            # could be empty file
            continue
        imgs_with_labels += 1
        for (cls, cx, cy, ww, hh) in labels:
            total_objs += 1
            class_counts[int(cls)] += 1

    lines: List[str] = []
    lines.append(f"# Label Stats ({space})\n")
    lines.append(f"Frames root: `{frames_root}`")
    lines.append(f"Label root : `{label_root}`\n")
    lines.append("## Overview")
    lines.append(f"- Total images found: {total_imgs}")
    lines.append(f"- Images with one-or-more labels: {imgs_with_labels}")
    lines.append(f"- Total labeled objects: {total_objs}")
    lines.append(f"- Images missing labels: {missing_label}\n")

    if class_counts:
        lines.append("## Top classes (id: count)")
        for cls, cnt in class_counts.most_common(20):
            lines.append(f"- {cls}: {cnt}")
        lines.append("")

    md = "\n".join(lines) + "\n"
    print(md)
    out_md.write_text(md, encoding='utf-8')
    print(f"[stats] wrote → {out_md}")


def cmd_sample(label_root: Path, frames_root: Path, space: str, out_dir: Path, num: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build a list of candidate (img,label) pairs by scanning frames
    pairs: List[Tuple[Path, Path]] = []
    for img in iter_frames(frames_root):
        uid = img.parent.name
        try:
            frame_idx = int(img.stem)
        except Exception:
            continue
        lbl = yolo_label_path(label_root, space, uid, frame_idx)
        if lbl.exists():
            pairs.append((img, lbl))
    if not pairs:
        print("[sample] No pairs found.")
        return
    random.shuffle(pairs)
    chosen = pairs[:max(1, int(num))]

    # Try to draw with Pillow
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        print("[sample] Pillow not installed; cannot draw samples. pip install pillow")
        return

    for img_path, lbl_path in chosen:
        labels = parse_yolo_label(lbl_path)
        if not labels:
            continue
        try:
            im = Image.open(str(img_path)).convert('RGB')
        except Exception:
            continue
        W, H = im.size
        draw = ImageDraw.Draw(im)
        for (cls, cx, cy, ww, hh) in labels:
            px, py = cx * W, cy * H
            pw, ph = ww * W, hh * H
            x1, y1 = px - pw / 2.0, py - ph / 2.0
            x2, y2 = px + pw / 2.0, py + ph / 2.0
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        outp = out_dir / f"{img_path.parent.name}_{img_path.stem}_vis.jpg"
        im.save(str(outp), quality=90)
        print(f"[sample] saved → {outp}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local STA data pipeline (no downloads/extraction)")
    # Allow running with no subcommand; we'll fall back to DEFAULT_CMD
    sub = ap.add_subparsers(dest='cmd', required=False)

    # sanity
    ap_sa = sub.add_parser('sanity', help='Summarize frame dimensions and write Markdown')
    ap_sa.add_argument('--frames-root', type=str, default=str(DEFAULT_FRAMES_ROOT))
    ap_sa.add_argument('--out', type=str, default=str(DEFAULT_SANITY_OUT))
    ap_sa.add_argument('--sample-per-group', type=int, default=int(DEFAULT_SAMPLE_PER_GROUP))

    # stats
    ap_st = sub.add_parser('stats', help='Label/object statistics and top classes')
    ap_st.add_argument('--label-root', type=str, default=str(DEFAULT_LABEL_ROOT))
    ap_st.add_argument('--frames-root', type=str, default=str(DEFAULT_FRAMES_ROOT))
    ap_st.add_argument('--space', type=str, default=str(DEFAULT_SPACE), choices=['clips','videos'])
    ap_st.add_argument('--out', type=str, default=str(DEFAULT_STATS_OUT))

    # sample
    ap_sp = sub.add_parser('sample', help='Visualize a few labeled images to verify alignment')
    ap_sp.add_argument('--label-root', type=str, default=str(DEFAULT_LABEL_ROOT))
    ap_sp.add_argument('--frames-root', type=str, default=str(DEFAULT_FRAMES_ROOT))
    ap_sp.add_argument('--space', type=str, default=str(DEFAULT_SPACE), choices=['clips','videos'])
    ap_sp.add_argument('--out-dir', type=str, default=str(DEFAULT_SAMPLE_OUT_DIR))
    ap_sp.add_argument('--num', type=int, default=int(DEFAULT_SAMPLE_NUM))

    # labels (create missing only)
    ap_lb = sub.add_parser('labels', help='Create missing YOLO labels from Drive annotations (skip existing)')
    ap_lb.add_argument('--label-root', type=str, default=str(DEFAULT_LABEL_ROOT))
    ap_lb.add_argument('--frames-root', type=str, default=str(DEFAULT_FRAMES_ROOT))
    ap_lb.add_argument('--space', type=str, default=str(DEFAULT_SPACE), choices=['clips','videos'])
    ap_lb.add_argument('--use-540', action='store_true', default=DEFAULT_USE_540, help='Use height-540 annotations (recommended to match 540p frames)')
    ap_lb.add_argument('--overwrite', action='store_true', default=DEFAULT_OVERWRITE, help='Overwrite existing label files (default: skip)')
    ap_lb.add_argument('--missing-log', type=str, default=str(DEFAULT_MISSING_LOG), help='Write CSV of missing images (uid,frame,path) to this file')

    # Do not require a subcommand; allow default from in-file toggle
    args = ap.parse_args(argv)
    if getattr(args, 'cmd', None) is None and DEFAULT_CMD:
        # Synthesize default command invocation with in-file defaults
        args.cmd = DEFAULT_CMD
        # Attach default namespaces similar to parsed args for target command
        if DEFAULT_CMD == 'sanity':
            args.frames_root = str(DEFAULT_FRAMES_ROOT)
            args.out = str(DEFAULT_SANITY_OUT)
            args.sample_per_group = int(DEFAULT_SAMPLE_PER_GROUP)
        elif DEFAULT_CMD == 'stats':
            args.label_root = str(DEFAULT_LABEL_ROOT)
            args.frames_root = str(DEFAULT_FRAMES_ROOT)
            args.space = str(DEFAULT_SPACE)
            args.out = str(DEFAULT_STATS_OUT)
        elif DEFAULT_CMD == 'sample':
            args.label_root = str(DEFAULT_LABEL_ROOT)
            args.frames_root = str(DEFAULT_FRAMES_ROOT)
            args.space = str(DEFAULT_SPACE)
            args.out_dir = str(DEFAULT_SAMPLE_OUT_DIR)
            args.num = int(DEFAULT_SAMPLE_NUM)
        elif DEFAULT_CMD == 'labels':
            args.label_root = str(DEFAULT_LABEL_ROOT)
            args.frames_root = str(DEFAULT_FRAMES_ROOT)
            args.space = str(DEFAULT_SPACE)
            args.use_540 = bool(DEFAULT_USE_540)
            args.overwrite = bool(DEFAULT_OVERWRITE)
            args.missing_log = str(DEFAULT_MISSING_LOG)

    if args.cmd == 'sanity':
        cmd_sanity(Path(args.frames_root), Path(args.out), int(args.sample_per_group))
    elif args.cmd == 'stats':
        cmd_stats(Path(args.label_root), Path(args.frames_root), str(args.space), Path(args.out))
    elif args.cmd == 'sample':
        cmd_sample(Path(args.label_root), Path(args.frames_root), str(args.space), Path(args.out_dir), int(args.num))
    elif args.cmd == 'labels':
        # default missing log path if not provided
        miss_path = Path(args.missing_log) if args.missing_log else (Path('local_extraction')/f'missing_images_{args.space}.txt')
        write_missing_labels(Path(args.label_root), Path(args.frames_root), str(args.space), bool(args.use_540), bool(args.overwrite), miss_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
