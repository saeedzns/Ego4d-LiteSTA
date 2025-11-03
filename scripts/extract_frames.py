#!/usr/bin/env python3
"""
Extract frames from Ego4D videos.

Supports three main modes:
  1) all        — dump every frame for given video_uids
  2) keyframes  — extract discrete frames discovered from a sample CSV (ego4d_samples/v2)
  3) windows    — extract frame ranges discovered from a sample CSV

This script is Colab-friendly and uses ffmpeg under the hood.

Examples
--------
# All frames for a few videos
python scripts/extract_frames.py \
  --videos-root /content/drive/MyDrive/ego4d_data/v2/videos \
  --out-frames-root /content/drive/MyDrive/ego4d_data/v2/frames \
  --mode all --uids 77cc4654-....,3e08beb0-....

# Keyframes/windows based on a sample CSV (e.g., STA val sample)
python scripts/extract_frames.py \
  --videos-root /content/drive/MyDrive/ego4d_data/v2/videos \
  --out-frames-root /content/drive/MyDrive/ego4d_data/v2/frames \
  --mode keyframes \
  --sample-csv ego4d_samples/v2/annotations/fho_sta_val_json.csv

python scripts/extract_frames.py \
  --videos-root /content/drive/MyDrive/ego4d_data/v2/videos \
  --out-frames-root /content/drive/MyDrive/ego4d_data/v2/frames \
  --mode windows \
  --sample-csv ego4d_samples/v2/annotations/moments_val_json.csv
"""

import argparse
import ast
import csv
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def ffmpeg_present() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def find_video(videos_root: Path, video_uid: str, recursive: bool = False) -> Optional[Path]:
    exts = (".mp4", ".mov", ".mkv")
    for ext in exts:
        p = videos_root / f"{video_uid}{ext}"
        if p.exists():
            return p
    if recursive:
        for ext in exts:
            hits = list(videos_root.rglob(f"{video_uid}{ext}"))
            if hits:
                return hits[0]
    return None


def extract_all(video_path: Path, out_dir: Path, pad: int = 7, ext: str = "jpg", start_number: int = 0) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"%0{pad}d.{ext}"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-start_number", str(start_number), str(pattern),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 0


def extract_one(video_path: Path, out_dir: Path, frame_idx: int, pad: int = 7, ext: str = "jpg") -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"f{frame_idx:0{pad}d}.{ext}"
    if out_path.exists():
        return True
    expr = f"select='eq(n,{frame_idx})'"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", expr, "-vsync", "vfr", "-vframes", "1", str(out_path),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 0 and out_path.exists()


def extract_between(video_path: Path, out_dir: Path, start: int, end: int, pad: int = 7, ext: str = "jpg") -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    expr = f"select='between(n,{start},{end})'"
    pattern = out_dir / f"%0{pad}d.{ext}"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", expr, "-vsync", "vfr", str(pattern),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 0


def parse_jsonish(cell: str):
    t = cell.strip()
    if (t[:1] in ("'", '"')) and (t[-1:] in ("'", '"')):
        t = t[1:-1]
    try:
        return ast.literal_eval(t)
    except Exception:
        return None


def collect_from_sample_csv(sample_csv: Path, max_items: int = 200) -> Tuple[Dict[str, Set[int]], Dict[str, List[Tuple[int, int]]]]:
    """Return per-video keyframes and windows discovered from a sample CSV."""
    per_video_frames: Dict[str, Set[int]] = defaultdict(set)
    per_video_windows: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    pairs = [
        ("video_start_frame", "video_end_frame"),
        ("clip_start_frame", "clip_end_frame"),
        ("action_start_frame", "action_end_frame"),
        ("interval_start_frame", "interval_end_frame"),
        ("clip_parent_start_frame", "clip_parent_end_frame"),
    ]

    def rec_collect(x, frames: Set[int], wins: List[Tuple[int, int]]):
        if isinstance(x, dict):
            # keyframes
            for k, v in x.items():
                kl = k.lower()
                if isinstance(v, int) and (
                    kl == "frame"
                    or kl.endswith("_frame")
                    or kl in ("pnr_frame", "clip_pnr_frame", "video_frame_number", "frame_number", "clip_frame_number")
                ):
                    if kl not in ("frame_height", "frame_width"):
                        frames.add(int(v))
            # windows
            for a, b in pairs:
                if a in x and b in x and isinstance(x[a], int) and isinstance(x[b], int):
                    s, e = int(x[a]), int(x[b])
                    if e >= s:
                        wins.append((s, e))
            for v in x.values():
                rec_collect(v, frames, wins)
        elif isinstance(x, list):
            for v in x:
                rec_collect(v, frames, wins)

    with sample_csv.open("r", encoding="utf-8") as f:
        _header = f.readline().strip()
        r = csv.reader(f)
        processed = 0
        for row in r:
            if not row or len(row) != 1:
                continue
            d = parse_jsonish(row[0])
            if not isinstance(d, dict):
                continue
            vu = d.get("video_uid")
            if not vu and isinstance(d.get("clips"), list) and d["clips"]:
                vu = d["clips"][0].get("video_uid")
            if not vu:
                continue
            frames: Set[int] = set()
            wins: List[Tuple[int, int]] = []
            rec_collect(d, frames, wins)
            if frames:
                per_video_frames[vu].update(frames)
            if wins:
                per_video_windows[vu].extend(wins)
            processed += 1
            if processed >= max_items:
                break

    return per_video_frames, per_video_windows


def main():
    ap = argparse.ArgumentParser(description="Extract frames from Ego4D videos")
    ap.add_argument("--videos-root", required=True, help="Folder containing <video_uid>.mp4/.mov/.mkv")
    ap.add_argument("--out-frames-root", required=True, help="Output frames root (per-video subfolders)")
    ap.add_argument("--mode", choices=["all", "keyframes", "windows"], required=True)
    ap.add_argument("--uids", default=None, help="Comma-separated video_uids for mode=all")
    ap.add_argument("--uids-file", default=None, help="File with one video_uid per line for mode=all")
    ap.add_argument("--sample-csv", default=None, help="Sample CSV (ego4d_samples/v2/annotations/*.csv) for keyframes/windows")
    ap.add_argument("--recursive", action="store_true", help="Search videos recursively under videos-root")
    ap.add_argument("--pad", type=int, default=7)
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--start-number", type=int, default=0)
    args = ap.parse_args()

    if not ffmpeg_present():
        print("ERROR: ffmpeg not found in PATH. Install it (e.g., !apt-get install -y ffmpeg in Colab).")
        return

    videos_root = Path(args.videos_root)
    out_root = Path(args.out_frames_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "all":
        if not args.uids and not args.uids_file:
            print("ERROR: Provide --uids or --uids-file for mode=all")
            return
        uids: List[str] = []
        if args.uids:
            uids.extend([u.strip() for u in args.uids.split(",") if u.strip()])
        if args.uids_file:
            for line in Path(args.uids_file).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    uids.append(line)
        uids = sorted(set(uids))
        print("UIDs:", len(uids))
        for vu in uids:
            src = find_video(videos_root, vu, recursive=args.recursive)
            if not src:
                print("skip (video not found):", vu)
                continue
            out_dir = out_root / vu
            ok = extract_all(src, out_dir, pad=args.pad, ext=args.ext, start_number=args.start_number)
            print("extract_all", vu, ok)
        return

    # keyframes/windows
    if not args.sample_csv:
        print("ERROR: Provide --sample-csv for mode=keyframes/windows")
        return
    sample_csv = Path(args.sample_csv)
    if not sample_csv.exists():
        print("ERROR: sample CSV not found:", sample_csv)
        return

    per_video_frames, per_video_windows = collect_from_sample_csv(sample_csv)
    print("videos with keyframes:", sum(1 for v in per_video_frames if per_video_frames[v]))
    print("videos with windows  :", sum(1 for v in per_video_windows if per_video_windows[v]))

    for vu in sorted(set(list(per_video_frames.keys()) + list(per_video_windows.keys()))):
        src = find_video(videos_root, vu, recursive=args.recursive)
        if not src:
            print("skip (video not found):", vu)
            continue
        # keyframes
        if args.mode == "keyframes":
            frames = sorted(per_video_frames.get(vu, []))
            out_dir = out_root / vu
            for fi in frames:
                extract_one(src, out_dir, fi, pad=args.pad, ext=args.ext)
            print("keyframes", vu, len(frames))
        elif args.mode == "windows":
            wins = per_video_windows.get(vu, [])
            for (s, e) in wins:
                out_dir = out_root / vu / f"win_{s}_{e}"
                extract_between(src, out_dir, s, e, pad=args.pad, ext=args.ext)
            print("windows", vu, len(wins))


if __name__ == "__main__":
    main()

