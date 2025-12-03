"""
Stage B quick sanity report.

This script scans the most recent Stage B run folder under
local_extraction/runs/Track_A/trackA_stageB_* and prints:
 - Positive/negative counts and ratio for head_train.jsonl and head_val.jsonl (if present)
 - Recall metrics from summary.json
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

try:
    from PIL import Image, ImageDraw  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore


def main() -> int:
    runs_root = Path("local_extraction/runs/Track_A")
    if not runs_root.exists():
        print("No Stage B runs root found at:", runs_root)
        return 2
    run_dirs = sorted(runs_root.glob("trackA_stageB_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        print("No Stage B runs found.")
        return 1

    rd = run_dirs[0]
    print("Inspecting run:", rd)

    for split in ("head_train.jsonl", "head_val.jsonl"):
        f = rd / split
        if not f.exists():
            print(split, "not found.")
            continue
        pos = 0
        neg = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("is_positive"):
                pos += 1
            else:
                neg += 1
        total = pos + neg
        ratio = round(pos / total, 4) if total > 0 else 0.0
        print(f"{split} -> positives {pos} negatives {neg} pos_ratio {ratio}")

    summary_data = None
    sumf = rd / "summary.json"
    if sumf.exists():
        try:
            summary_data = json.loads(sumf.read_text(encoding="utf-8"))
            print("Recall metrics:", summary_data.get("recall_metrics"))
        except Exception:
            print("Recall metrics: could not parse summary.json")
    else:
        print("summary.json not found.")

    for split in ("head_train.jsonl", "head_val.jsonl"):
        f = rd / split
        if f.exists():
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", f"(Get-Content '{f.as_posix()}').Count"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"Line count {split}: {result.stdout.strip()}")
            except Exception as exc:  # pragma: no cover
                print(f"Could not count lines for {split}: {exc}")

    manifest_path = rd / "manifest.jsonl"
    if manifest_path.exists() and Image is not None and ImageDraw is not None:
        sample = None
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sample = rec
                if rec.get("is_positive"):
                    break
        if sample:
            frames_root_str = summary_data.get("frames_root") if summary_data else None
            frames_root = Path(frames_root_str) if frames_root_str else Path("local_extraction") / "v2" / "extracted_frames"
            frame_name = f"{int(sample['frame']):07d}.jpg"
            frame_path = Path(frames_root) / sample["uid"] / frame_name
            crop_rel = sample.get("crop_path")
            crop_path = rd / Path(crop_rel) if crop_rel else None
            try:
                if frame_path.exists():
                    img = Image.open(frame_path).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([(float(sample["x1"]), float(sample["y1"])), (float(sample["x2"]), float(sample["y2"]))], outline="red", width=3)
                    overlay_path = rd / "overlay_preview.jpg"
                    img.save(overlay_path)
                    if crop_path and crop_path.exists():
                        crop_img = Image.open(crop_path).convert("RGB")
                        overlay_img = Image.open(overlay_path).convert("RGB")
                        combined_w = crop_img.width + overlay_img.width
                        combined_h = max(crop_img.height, overlay_img.height)
                        combined = Image.new("RGB", (combined_w, combined_h), (32, 32, 32))
                        combined.paste(crop_img, (0, 0))
                        combined.paste(overlay_img, (crop_img.width, 0))
                        combined_path = rd / "comparison_preview.jpg"
                        combined.save(combined_path)
                        print("Opening combined preview (crop | overlay):", combined_path)
                        subprocess.Popen(["powershell", "-NoProfile", "-Command", f"Start-Process '{combined_path.as_posix()}'"])
                    else:
                        print("Crop image not found; opening overlay only.")
                        subprocess.Popen(["powershell", "-NoProfile", "-Command", f"Start-Process '{overlay_path.as_posix()}'"])
                else:
                    print("Frame not found for overlay:", frame_path)
            except Exception as exc:  # pragma: no cover
                print(f"Failed to create overlay: {exc}")
        else:
            print("No manifest records available for overlay preview.")
    elif not manifest_path.exists():
        print("manifest.jsonl not found; skipping overlay preview.")
    else:
        print("Pillow not available; cannot generate overlay preview.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())