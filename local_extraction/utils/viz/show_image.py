#!/usr/bin/env python3
"""
Show any image given its path.

Usage (from repo root or anywhere):
  python local_extraction/show_image.py "D:/path/to/image.jpg"
  python local_extraction/show_image.py --path "local_extraction/v2/extracted_frames/<uid>/0000003.jpg"

Behavior:
  - Prints basic info (exists, size, dimensions if readable)
  - Tries to display via Pillow (if installed), else opens with OS default viewer
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Set your default image path here so you don't need to pass it on the CLI.
# Example: r"D:\Thesis\Ego4d-LiteSTA\local_extraction\v2\extracted_frames\<uid>\0000003.jpg"
DEFAULT_IMAGE_PATH: str = r"local_extraction\v2\extracted_frames\964575bd-afec-4815-af83-e93e60d46602\0001478.jpg"


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
    # Fallback minimal header parse for PNG/JPEG
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
                    # skip fill FFs
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
                        f.read(1)
                        h = int.from_bytes(f.read(2), 'big')
                        w = int.from_bytes(f.read(2), 'big')
                        return w, h
                    f.seek(seglen - 2, 1)
    except Exception:
        pass
    return None, None


def open_with_os(img_path: Path) -> bool:
    try:
        if os.name == 'nt':
            os.startfile(str(img_path))  # type: ignore[attr-defined]
            return True
        if sys.platform == 'darwin':
            return os.system(f"open '{img_path}'") == 0
        return os.system(f"xdg-open '{img_path}' >/dev/null 2>&1") == 0
    except Exception:
        return False


def show_with_pillow(img_path: Path) -> bool:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return False
    try:
        im = Image.open(str(img_path))
        im.show(title=img_path.name)
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Display an image by path')
    ap.add_argument('path', nargs='?', default=None, help='Path to the image file')
    ap.add_argument('--path', dest='path_kw', default=None, help='Path to the image file (named arg)')
    args = ap.parse_args(argv)

    # Use CLI arg if provided; otherwise fall back to the default above
    path_str = args.path_kw or args.path or DEFAULT_IMAGE_PATH
    if not path_str:
        print('Provide a path: python local_extraction/show_image.py "C:/path/to/image.jpg"')
        return 2

    img_path = Path(path_str)
    print(f"[info] Path: {img_path}")
    if not img_path.exists():
        print('[error] File does not exist.')
        return 2

    try:
        size = img_path.stat().st_size
        print(f"[info] Size: {size} bytes")
    except Exception:
        pass

    w, h = read_image_size(img_path)
    if w and h:
        print(f"[info] Dimensions: {w}x{h}")
    else:
        print('[warn] Could not determine image dimensions')

    if show_with_pillow(img_path):
        print('[info] Opened via Pillow')
        return 0
    if open_with_os(img_path):
        print('[info] Opened with OS default viewer')
        return 0

    print('[error] Could not open the image. Install Pillow: pip install pillow')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
