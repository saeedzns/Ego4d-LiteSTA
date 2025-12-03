#!/usr/bin/env python3
"""
Minimal YOLO label explainer for a single STA frame.

Defaults target the clip you referenced:
  clip_uid = 00be9fe7-617c-46cf-a0de-7432896c1705
  frame    = 99  (file 0000099.txt / 0000099.jpg)

It loads the label file, optionally reads the corresponding image to get W,H,
then prints how YOLO numbers are computed to/from pixel boxes.

Run (from repo root):
  python local_extraction/test_yolo_label_one.py \
    --clip 00be9fe7-617c-46cf-a0de-7432896c1705 --frame 99

Optional flags:
  --space clips|videos               (default: clips)
  --label-root local_extraction/v2/yolo_labels_540
  --frames-root local_extraction/v2/extracted_frames
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple


def _read_size_raw(img_path: Path) -> Tuple[int | None, int | None]:
    """Read image size without external deps for PNG/JPEG."""
    try:
        p = str(img_path)
        with open(p, 'rb') as f:
            head = f.read(32)
            # PNG
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                # IHDR: width/height are 4 bytes each at offset 16
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
                    # find marker 0xFF
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
                    # SOI/EOI have no length; 0xD8, 0xD9
                    if marker in (0xD8, 0xD9):
                        continue
                    # read segment length
                    seglen_bytes = f.read(2)
                    if len(seglen_bytes) != 2:
                        break
                    seglen = int.from_bytes(seglen_bytes, 'big')
                    if seglen < 2:
                        return None, None
                    # SOF0..SOF3, SOF5..SOF7, SOF9..SOF11, SOF13..SOF15 contain size
                    if (0xC0 <= marker <= 0xC3) or (0xC5 <= marker <= 0xC7) or (0xC9 <= marker <= 0xCB) or (0xCD <= marker <= 0xCF):
                        f.read(1)  # precision
                        h = int.from_bytes(f.read(2), 'big')
                        w = int.from_bytes(f.read(2), 'big')
                        return w, h
                    else:
                        # skip this segment
                        f.seek(seglen - 2, 1)
    except Exception:
        return None, None
    return None, None


def read_image_size(img_path: Path) -> Tuple[int | None, int | None]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(str(img_path)) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        try:
            import imageio.v2 as imageio  # type: ignore
            im = imageio.imread(str(img_path))
            if hasattr(im, "shape") and len(im.shape) >= 2:
                h, w = int(im.shape[0]), int(im.shape[1])
                return w, h
        except Exception:
            pass
    # Fallback to raw header parsing (PNG/JPEG)
    w, h = _read_size_raw(img_path)
    return w, h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Explain a single YOLO label")
    ap.add_argument("--clip", type=str, default="00be9fe7-617c-46cf-a0de-7432896c1705", help="clip_uid or video_uid")
    ap.add_argument("--frame", type=int, default=99, help="frame index (e.g., 99 for 0000099)")
    ap.add_argument("--space", type=str, default="clips", choices=["clips", "videos"], help="label/image space")
    ap.add_argument("--label-root", type=str, default="local_extraction/v2/yolo_labels_540", help="root for YOLO labels")
    ap.add_argument("--frames-root", type=str, default="local_extraction/v2/extracted_frames", help="root for extracted frames")
    ap.add_argument("--out", type=str, default=None, help="optional path to save a visualization image")
    ap.add_argument("--show", action="store_true", help="open the visualization with the OS viewer after saving")
    args = ap.parse_args(argv)

    frame_name = f"{int(args.frame):07d}"
    label_path = Path(args.label_root) / args.space / args.clip / f"{frame_name}.txt"
    image_path = Path(args.frames_root) / args.clip / f"{frame_name}.jpg"

    print(f"Label file : {label_path}")
    print(f"Image file : {image_path}")

    if not label_path.exists():
        print("[error] Label file not found.")
        return 2

    W = H = None
    if image_path.exists():
        W, H = read_image_size(image_path)
        if W and H:
            print(f"Image size: W={W}, H={H}")
        else:
            print("[warn] Could not read image size (will show only normalized values)")
    else:
        print("[warn] Image not found; will show only normalized values.")

    lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
    if not lines:
        print("[warn] Label file is empty")
        return 0

    print("\nYOLO format per line: class_id cx cy w h (normalized 0..1)\n")

    boxes = []  # (cls, x1,y1,x2,y2)
    for i, ln in enumerate(lines, 1):
        parts = ln.split()
        if len(parts) != 5:
            print(f"[skip] line {i}: unexpected format -> {ln}")
            continue
        cls = int(float(parts[0]))
        cx = float(parts[1])
        cy = float(parts[2])
        ww = float(parts[3])
        hh = float(parts[4])

        print(f"- Object {i}:")
        print(f"  class_id = {cls}")
        print(f"  normalized: cx={cx:.6f} cy={cy:.6f} w={ww:.6f} h={hh:.6f}")

        if W and H and W > 0 and H > 0:
            px = cx * W
            py = cy * H
            pw = ww * W
            ph = hh * H
            x1 = px - pw / 2.0
            y1 = py - ph / 2.0
            x2 = px + pw / 2.0
            y2 = py + ph / 2.0

            print(f"  pixel box: x1={x1:.1f} y1={y1:.1f} x2={x2:.1f} y2={y2:.1f}")

            boxes.append((cls, x1, y1, x2, y2))

            # Verify round-trip back to YOLO
            v_w = max(1.0, x2 - x1)
            v_h = max(1.0, y2 - y1)
            v_cx = x1 + v_w / 2.0
            v_cy = y1 + v_h / 2.0
            print(
                f"  re-norm:  cx={v_cx/W:.6f} cy={v_cy/H:.6f} w={v_w/W:.6f} h={v_h/H:.6f}"
            )

    print("\nFormulas used:")
    print("  cx = (x1 + x2)/2 / W")
    print("  cy = (y1 + y2)/2 / H")
    print("   w = (x2 - x1) / W")
    print("   h = (y2 - y1) / H")

    # Optional visualization
    if boxes and W and H and image_path.exists():
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
            im = Image.open(str(image_path)).convert("RGB")
            draw = ImageDraw.Draw(im)
            for (cls, x1, y1, x2, y2) in boxes:
                # clamp to image
                x1i = max(0, min(W - 1, int(round(x1))))
                y1i = max(0, min(H - 1, int(round(y1))))
                x2i = max(0, min(W - 1, int(round(x2))))
                y2i = max(0, min(H - 1, int(round(y2))))
                draw.rectangle([x1i, y1i, x2i, y2i], outline=(255, 0, 0), width=2)
                label = f"{cls}"
                try:
                    draw.text((x1i + 2, max(0, y1i - 12)), label, fill=(255, 0, 0))
                except Exception:
                    pass

            out_path = Path(args.out) if args.out else (image_path.parent / f"{frame_name}_vis.jpg")
            im.save(str(out_path), quality=90)
            print(f"\nSaved visualization → {out_path}")
            if args.show:
                # Try OS viewer
                try:
                    if os.name == "nt":
                        os.startfile(str(out_path))  # type: ignore[attr-defined]
                    elif sys.platform == "darwin":
                        os.system(f"open '{out_path}'")
                    else:
                        os.system(f"xdg-open '{out_path}' >/dev/null 2>&1 || true")
                except Exception:
                    pass
        except Exception as e:
            print(f"[warn] Visualization failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
