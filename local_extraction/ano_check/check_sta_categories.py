#!/usr/bin/env python3
"""
STA annotation sanity helper (no CLI).

Configure the toggles below and run:
  python local_extraction/ano_check/check_sta_categories.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_categories(path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    obj = load_json(path)
    nouns_raw = obj.get("noun_categories") or []
    verbs_raw = obj.get("verb_categories") or []
    noun_map = {int(it["id"]): str(it["name"]) for it in nouns_raw if "id" in it and "name" in it}
    verb_map = {int(it["id"]): str(it["name"]) for it in verbs_raw if "id" in it and "name" in it}
    return noun_map, verb_map


def compare_vocab(base: Dict[int, str], height: Dict[int, str], kind: str, split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ids = set(base.keys()) | set(height.keys())
    for i in sorted(ids):
        b = base.get(i)
        h = height.get(i)
        if b is None:
            status = "missing_in_base"
        elif h is None:
            status = "missing_in_height"
        elif b == h:
            status = "match"
        else:
            status = "mismatch"
        rows.append({
            "split": split,
            "type": kind,
            "id": i,
            "base_name": b or "",
            "height540_name": h or "",
            "status": status,
        })
    return rows


def print_samples(path: Path, k: int, label: str) -> None:
    obj = load_json(path)
    if isinstance(obj, dict):
        records = obj.get("annotations") or []
    elif isinstance(obj, list):
        records = obj
    else:
        records = []
    print(f"\n=== First {k} records from {label} ({path}) ===")
    if not records:
        print("  [warn] no records found in this file.")
        return
    k = min(k, len(records))
    for idx, rec in enumerate(records[:k], 1):
        clip_uid = rec.get("clip_uid") or rec.get("video_uid")
        clip_frame = rec.get("clip_frame", rec.get("frame"))
        objs = rec.get("objects") or []
        if objs:
            o0 = objs[0]
            noun_id = o0.get("noun_category_id")
            verb_id = o0.get("verb_category_id")
        else:
            noun_id = verb_id = None
        print(f"[{idx}] clip_uid={clip_uid} clip_frame={clip_frame} noun_id={noun_id} verb_id={verb_id} ttc={objs[0].get('time_to_contact') if objs else None}")


def main() -> None:
    # --------- Toggles ---------
    org_root = Path("local_extraction/v2/org_annotations")
    k = 3  # how many records to print from each selected file
    out_csv = Path("local_extraction/ano_check/results/category_diffs.csv")
    compare = False  # set to False to skip vocab comparison
    source = "both"  # one of: val_height, train_height, both
    print_base = False  # also print first k from base val/train JSONs
    # ---------------------------

    org_root = Path(org_root)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    val_base = org_root / "fho_sta_val.json"
    train_base = org_root / "fho_sta_train.json"
    val_h = org_root / "fho_sta_val_height-540.json"
    train_h = org_root / "fho_sta_train_height-540.json"

    # Print samples from requested height-540 files
    sources = []
    if source in ("val_height", "both"):
        sources.append((val_h, "val_height-540"))
    if source in ("train_height", "both"):
        sources.append((train_h, "train_height-540"))
    for p, lbl in sources:
        try:
            print_samples(p, k, lbl)
        except FileNotFoundError as e:
            print(f"[warn] {e}")
    if print_base:
        for p, lbl in [(val_base, "val_base"), (train_base, "train_base")]:
            try:
                print_samples(p, k, lbl)
            except FileNotFoundError as e:
                print(f"[warn] {e}")

    if not compare:
        return

    rows: List[Dict[str, Any]] = []
    # Compare vocab for val and train separately (often identical but we check both)
    for base_p, h_p, split in [(val_base, val_h, "val"), (train_base, train_h, "train")]:
        try:
            base_n, base_v = load_categories(base_p)
            h_n, h_v = load_categories(h_p)
            rows.extend(compare_vocab(base_n, h_n, "noun", split))
            rows.extend(compare_vocab(base_v, h_v, "verb", split))
        except FileNotFoundError as e:
            print(f"[warn] {e}")
            continue

    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[info] wrote vocab comparison CSV: {out_csv}")
    else:
        print("[warn] no rows to write; check your paths.")


if __name__ == "__main__":
    main()
