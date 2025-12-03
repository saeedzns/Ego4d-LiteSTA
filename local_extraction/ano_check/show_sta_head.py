#!/usr/bin/env python3
"""
Show the first K records from STA height-540 JSONs (train/val).

Run:
  python local_extraction/ano_check/show_sta_head.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def show_head(path: Path, k: int) -> None:
    obj = load_json(path)
    noun_lookup = {}
    verb_lookup = {}
    if isinstance(obj, dict):
        records: List[Dict[str, Any]] = obj.get("annotations") or []
        noun_lookup = {int(x["id"]): x.get("name", "") for x in obj.get("noun_categories") or [] if "id" in x}
        verb_lookup = {int(x["id"]): x.get("name", "") for x in obj.get("verb_categories") or [] if "id" in x}
    elif isinstance(obj, list):
        records = obj
    else:
        records = []
    print(f"\n=== First {k} records from {path} ===")
    if not records:
        print("  [warn] no records found")
        return
    k = min(k, len(records))
    for idx, rec in enumerate(records[:k], 1):
        clip_uid = rec.get("clip_uid") or rec.get("video_uid")
        clip_frame = rec.get("clip_frame", rec.get("frame"))
        print(f"\n[{idx}] clip_uid={clip_uid} clip_frame={clip_frame}")
        for j, obj_i in enumerate(rec.get("objects") or [], 1):
            n_id = obj_i.get("noun_category_id")
            v_id = obj_i.get("verb_category_id")
            n_name = noun_lookup.get(int(n_id)) if n_id is not None else None
            v_name = verb_lookup.get(int(v_id)) if v_id is not None else None
            ttc = obj_i.get("time_to_contact")
            print(f"  obj{j}: noun_id={n_id} ({n_name}) verb_id={v_id} ({v_name}) ttc={ttc} box={obj_i.get('box')}")


def main() -> None:
    k = 3
    org_root = Path("local_extraction/v2/org_annotations")
    files = [
        org_root / "fho_sta_val_height-540.json",
        org_root / "fho_sta_train_height-540.json",
    ]
    for p in files:
        try:
            show_head(p, k)
        except FileNotFoundError as e:
            print(f"[warn] {e}")


if __name__ == "__main__":
    main()
