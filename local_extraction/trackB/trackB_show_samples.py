#!/usr/bin/env python3
"""
Visual sanity check for Stage B manifests.

Given one or more Stage B head manifests (head_train.jsonl / head_val.jsonl)
and noun/verb taxonomy label files, this script samples K candidates and
displays the corresponding image crop with noun/verb labels.

Usage example (from repo root):

  python local_extraction/trackB/trackB_show_samples.py ^
      --manifests local_extraction/runs/Track_A/trackA_stageB_20251117_184342/head_train.jsonl ^
      --org_root local_extraction/v2/org_annotations ^
      --k 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore
from PIL import Image

def load_sta_label_maps(org_root: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Load noun/verb label maps from STA height-540 JSONs (train + val)."""
    noun_map: Dict[int, str] = {}
    verb_map: Dict[int, str] = {}
    for name in ["fho_sta_val_height-540.json", "fho_sta_train_height-540.json"]:
        p = org_root / name
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                for x in obj.get("noun_categories") or []:
                    if "id" in x:
                        noun_map[int(x["id"])] = str(x.get("name", ""))
                for x in obj.get("verb_categories") or []:
                    if "id" in x:
                        verb_map[int(x["id"])] = str(x.get("name", ""))
        except Exception as e:
            print(f"[warn] failed to load labels from {p}: {e}")
    return noun_map, verb_map


def _iter_manifest_records(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield dict records from JSON or JSONL manifest."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict):
                    yield rec
            return
    except Exception:
        pass
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            yield rec


def collect_stageb_candidates(manifest_paths: List[Path], require_positive: bool = True) -> List[Dict[str, Any]]:
    """Collect Stage B candidate entries from manifests.

    If require_positive=True, only keep entries with is_positive==True when present.
    """
    samples: List[Dict[str, Any]] = []
    for path in manifest_paths:
        for rec in _iter_manifest_records(path):
            if not isinstance(rec, dict):
                continue
            # Stage B head manifests typically have per-candidate rows
            img = rec.get("image_path")
            box = rec.get("candidate_box")
            is_pos = rec.get("is_positive")
            if require_positive and (is_pos is not None) and (not bool(is_pos)):
                continue
            if isinstance(img, str) and isinstance(box, (list, tuple)) and len(box) == 4:
                samples.append({
                    "image_path": img,
                    "box": [float(x) for x in box],
                    "noun_id": rec.get("noun_id"),
                    "verb_id": rec.get("verb_id"),
                    "ttc": rec.get("ttc"),
                    "manifest": path,
                    "is_positive": is_pos,
                })
                continue
            # Alternatively, some manifests may have an image + candidates list
            if "image" in rec and "candidates" in rec:
                img_path = rec.get("image")
                if not isinstance(img_path, str):
                    continue
                for cand in rec.get("candidates") or []:
                    if not isinstance(cand, dict):
                        continue
                    box2 = cand.get("box") or cand.get("candidate_box")
                    if not isinstance(box2, (list, tuple)) or len(box2) < 4:
                        continue
                    is_pos2 = cand.get("is_positive")
                    if require_positive and (is_pos2 is not None) and (not bool(is_pos2)):
                        continue
                    samples.append({
                        "image_path": img_path,
                        "box": [float(x) for x in box2[:4]],
                        "noun_id": cand.get("noun_id"),
                        "verb_id": cand.get("verb_id"),
                        "ttc": cand.get("ttc"),
                        "manifest": path,
                        "is_positive": is_pos2,
                    })
    return samples


def resolve_image_path(image_path: str, frames_root: Optional[Path]) -> Path:
    """Return a Path to the image (try absolute, then frames_root)."""
    p = Path(image_path)
    if p.exists():
        return p
    if frames_root is not None:
        alt = frames_root / image_path
        if alt.exists():
            return alt
        # also try relative name (e.g., only UID/frame)
        try:
            rel = Path(*Path(image_path).parts[-2:])
            alt2 = frames_root / rel
            if alt2.exists():
                return alt2
        except Exception:
            pass
    return p  # may not exist; caller handles


def show_sample(sample: Dict[str, Any], noun_labels: Dict[int, str], verb_labels: Dict[int, str], frames_root: Optional[Path]) -> None:
    image_path = sample["image_path"]
    box = sample["box"]
    noun_id = sample.get("noun_id")
    verb_id = sample.get("verb_id")
    noun_label = noun_labels.get(int(noun_id), "<unknown>") if noun_id is not None else "<none>"
    verb_label = verb_labels.get(int(verb_id), "<unknown>") if verb_id is not None else "<none>"

    resolved = resolve_image_path(image_path, frames_root)
    if not resolved.exists():
        print(f"[warn] Image not found: {resolved}")
        return

    try:
        img = Image.open(str(resolved)).convert("RGB")
    except Exception as e:
        print(f"[warn] Failed to open {resolved}: {e}")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    x1, y1, x2, y2 = box
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
    title = f"noun {noun_id} ({noun_label})\nverb {verb_id} ({verb_label})"
    if sample.get("ttc") is not None:
        title += f"\nTTC={sample['ttc']}"
    ax.set_title(title)
    ax.axis("off")
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visual sanity check: StageB image + noun/verb labels.")
    ap.add_argument("--manifests", type=str, nargs="+", required=True, help="Stage B manifest paths (head_train/head_val, .json or .jsonl).")
    ap.add_argument("--org_root", type=str, default="local_extraction/v2/org_annotations", help="Root containing STA height-540 JSONs (for label maps).")
    ap.add_argument("--frames_root", type=str, default=None, help="Optional frames root for resolving relative image paths.")
    ap.add_argument("--k", type=int, default=3, help="Number of random samples to display.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    ap.add_argument("--positives_only", action="store_true", help="Sample only positive candidates (is_positive==True when present).")
    args = ap.parse_args()

    random.seed(args.seed)

    manifest_paths: List[Path] = []
    for p in args.manifests:
        path = Path(p)
        if path.exists():
            manifest_paths.append(path)
            continue
        # try alternate extension .jsonl/.json
        if path.suffix.lower() == ".json":
            alt = path.with_suffix(".jsonl")
        elif path.suffix.lower() == ".jsonl":
            alt = path.with_suffix(".json")
        else:
            alt = path
        if alt.exists():
            manifest_paths.append(alt)
        else:
            print(f"[warn] manifest not found: {path} (also tried {alt})")
    if not manifest_paths:
        print("[error] no manifest files found. Please provide valid .json/.jsonl paths.")
        return
    noun_labels, verb_labels = load_sta_label_maps(Path(args.org_root))
    frames_root = Path(args.frames_root) if args.frames_root else None

    samples = collect_stageb_candidates(manifest_paths, require_positive=args.positives_only)
    if not samples:
        print("No StageB-style candidate entries found in the provided manifests.")
        return

    n_show = min(max(args.k, 0), len(samples))
    if n_show == 0:
        print("Nothing to show; k=0 or no samples available.")
        return

    chosen = random.sample(samples, n_show)
    for i, sample in enumerate(chosen, 1):
        print(f"\nSample {i}/{n_show}")
        print(f"  manifest   : {sample['manifest']}")
        print(f"  image_path : {sample['image_path']}")
        print(f"  noun_id    : {sample.get('noun_id')} ({noun_labels.get(int(sample['noun_id']), '<unknown>') if sample.get('noun_id') is not None else '<none>'})")
        print(f"  verb_id    : {sample.get('verb_id')} ({verb_labels.get(int(sample['verb_id']), '<unknown>') if sample.get('verb_id') is not None else '<none>'})")
        print(f"  box        : {sample['box']}")
        if sample.get("ttc") is not None:
            print(f"  ttc        : {sample['ttc']}")
        if sample.get("is_positive") is not None:
            print(f"  is_positive: {sample.get('is_positive')}")
        show_sample(sample, noun_labels, verb_labels, frames_root)


if __name__ == "__main__":
    main()
