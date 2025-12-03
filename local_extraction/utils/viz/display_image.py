#!/usr/bin/env python3
"""
Show the first image in a Google Drive folder on Windows (or any OS).
Default folder (Windows Google Drive for Desktop):
  H:/My Drive/ego4d_data/v2/extracted_frames/00be9fe7-617c-46cf-a0de-7432896c1705
Usage examples:
  python scripts/show_drive_first_image.py
  python scripts/show_drive_first_image.py --dir "H:/My Drive/ego4d_data/v2/extracted_frames/<clip_uid>"
Behavior:
  - Finds the first image by lexicographic filename order
  - Tries to display via Pillow (if installed), else opens via OS default viewer
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import subprocess
import shutil
from typing import Iterable
DEFAULT_DIR = Path(
    # Use forward slashes for cross-platform-safe Windows paths
    "H:/My Drive/ego4d_data/v2/extracted_frames/00be9fe7-617c-46cf-a0de-7432896c1705"
)
IMAGE_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
def find_first_image(folder: Path, exts: Iterable[str] = IMAGE_EXTS) -> Path | None:
    if not folder.exists():
        return None
    if not folder.is_dir():
        return None
    # Gather files with desired extensions (case-insensitive)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        # Also look one level down if the frames are stored in subfolders
        for sub in sorted([d for d in folder.iterdir() if d.is_dir()]):
            files = [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in exts]
            if files:
                break
    if not files:
        return None
    files.sort(key=lambda p: p.name)
    return files[0]
def show_with_pillow(img_path: Path) -> bool:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return False
    try:
        im = Image.open(str(img_path))
        im.show(title=str(img_path.name))  # Uses OS default viewer via Pillow helper
        return True
    except Exception as e:
        print(f"[warn] Pillow failed to display image: {e}")
        return False
def open_with_os_default(img_path: Path) -> bool:
    try:
        if os.name == "nt":
            os.startfile(str(img_path))  # type: ignore[attr-defined]
            return True
        # macOS
        if sys.platform == "darwin" and shutil.which("open"):
            subprocess.run(["open", str(img_path)], check=False)
            return True
        # Linux / others
        if shutil.which("xdg-open"):
            subprocess.run(["xdg-open", str(img_path)], check=False)
            return True
    except Exception as e:
        print(f"[warn] OS open failed: {e}")
    return False
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show first image in a folder")
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=str(DEFAULT_DIR),
        help="Folder containing extracted frames (default: the provided Google Drive path)",
    )
    args = parser.parse_args(argv)
    folder = Path(args.dir)
    print(f"[info] Looking for images in: {folder}")
    img = find_first_image(folder)
    if not img:
        print("[error] No image files found in the folder (or subfolders).")
        return 2
    print(f"[info] Showing: {img}")
    # Try Pillow first (gives a quick preview); fallback to OS default app
    if show_with_pillow(img):
        return 0
    if open_with_os_default(img):
        return 0
    print("[error] Could not display the image. Install Pillow: pip install pillow")
    return 1
if __name__ == "__main__":
    raise SystemExit(main())
