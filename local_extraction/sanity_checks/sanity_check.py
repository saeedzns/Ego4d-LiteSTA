#!/usr/bin/env python3
"""
Sanity check for extracted frames.

Scans a frames root (default: local_extraction/v2/extracted_frames),
reads image sizes, groups by short-side dimension ("image_dim": min(W,H)),
and writes a detailed Markdown report with counts, percentages, and notes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Tuple, Optional

def _read_size_raw(img_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        with open(img_path, 'rb') as f:
            head = f.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                f.seek(16)
                w = int.from_bytes(f.read(4), 'big')
                h = int.from_bytes(f.read(4), 'big')
                return w, h
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
        return None, None
    return None, None

def read_image_size(img_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(str(img_path)) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        pass
    try:
        import imageio.v2 as imageio  # type: ignore
        im = imageio.imread(str(img_path))
        if hasattr(im, 'shape') and len(im.shape) >= 2:
            h, w = int(im.shape[0]), int(im.shape[1])
            return w, h
    except Exception:
        pass
    return _read_size_raw(img_path)

def iter_jpgs(frames_root: Path) -> Iterable[Path]:
    if not frames_root.is_dir():
        return []
    for uid_dir in sorted(frames_root.iterdir()):
        if not uid_dir.is_dir():
            continue
        for p in uid_dir.iterdir():
            if p.is_file() and p.suffix.lower() == '.jpg':
                yield p

def build_report(frames_root: Path, sample_per_group: int = 8) -> Tuple[str, Dict[int, int]]:
    total = 0
    unknown = 0
    short_counts: Counter[int] = Counter()
    pair_counts: Counter[Tuple[int, int]] = Counter()
    examples_by_short: Dict[int, List[Path]] = defaultdict(list)

    for img_path in iter_jpgs(frames_root):
        total += 1
        w, h = read_image_size(img_path)
        if not (w and h and w > 0 and h > 0):
            unknown += 1
            continue
        short = min(w, h)
        short_counts[short] += 1
        pair_counts[(w, h)] += 1
        if len(examples_by_short[short]) < sample_per_group:
            examples_by_short[short].append(img_path)

    lines: List[str] = []
    lines.append(f"# Sanity Check: Extracted Frames\n")
    lines.append(f"Root: `{frames_root}`\n")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Total frames scanned: {total}")
    lines.append(f"- Unknown/failed to read size: {unknown}")
    lines.append(f"- Unique short-side groups: {len(short_counts)}")
    lines.append(f"- Unique dimension pairs: {len(pair_counts)}\n")

    if total > 0:
        lines.append("## Group by Short Side (image_dim)")
        for s, c in short_counts.most_common():
            pct = (100.0 * c) / max(1, total)
            lines.append(f"- {s}: {c} frames ({pct:.1f}%)")
        lines.append("")

    if pair_counts:
        lines.append("## Top Dimension Pairs (WxH)")
        for (w, h), c in pair_counts.most_common(12):
            pct = (100.0 * c) / max(1, total)
            lines.append(f"- {w}x{h}: {c} ({pct:.1f}%)")
        lines.append("")

    if examples_by_short:
        lines.append("## Examples per Short-Side Group")
        for s, examples in sorted(examples_by_short.items(), key=lambda kv: (-short_counts[kv[0]], kv[0])):
            lines.append(f"- {s} (showing up to {sample_per_group}):")
            for p in examples:
                lines.append(f"  - `{p}`")
        lines.append("")

    dominant = None
    if short_counts:
        dominant, _ = short_counts.most_common(1)[0]
    lines.append("## Interpretation")
    if dominant is not None:
        pct = (100.0 * short_counts[dominant]) / max(1, total)
        if dominant == 540:
            lines.append(f"- Majority of frames have short side 540 ({pct:.1f}%). This matches expected 540p normalization.")
        else:
            lines.append(f"- Majority short-side is {dominant} ({pct:.1f}%). Consider standardizing to 540 for consistency if desired.")
    if any(s != 540 for s in short_counts.keys()):
        lines.append("- Mixed sizes detected. Potential causes: partial runs without resizing, different sources, or earlier settings.")
        lines.append("- Recommendation: Re-run extraction with consistent resizing (e.g., short side 540) for unified training inputs.")
    if unknown:
        lines.append("- Some files failed to read dimensions. Ensure images are valid and accessible.")

    md = "\n".join(lines) + "\n"
    return md, dict(short_counts)

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sanity-check extracted frames by image dimensions")
    ap.add_argument("--frames-root", type=str, default=str(Path("local_extraction") / "v2" / "extracted_frames"))
    ap.add_argument("--out", type=str, default=str(Path("local_extraction") / "sanity_check_report.md"))
    ap.add_argument("--sample-per-group", type=int, default=8, help="example paths per group in the report")
    args = ap.parse_args(argv)

    frames_root = Path(args.frames_root)
    md, _ = build_report(frames_root, sample_per_group=int(args.sample_per_group))
    print(md)
    out_path = Path(args.out)
    out_path.write_text(md, encoding='utf-8')
    print(f"[report] Saved Markdown → {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
