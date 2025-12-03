#!/usr/bin/env python3
"""
Sanity check: compare noun/verb categories between STA train/val height-540 JSONs.

Runs with in-code defaults; adjust paths if needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_vocab(path: Path) -> Dict[str, Dict[int, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    noun_map = {}
    verb_map = {}
    if isinstance(obj, dict):
        for x in obj.get("noun_categories") or []:
            if "id" in x:
                noun_map[int(x["id"])] = str(x.get("name", ""))
        for x in obj.get("verb_categories") or []:
            if "id" in x:
                verb_map[int(x["id"])] = str(x.get("name", ""))
    return {"nouns": noun_map, "verbs": verb_map}


def compare_maps(a: Dict[int, str], b: Dict[int, str]) -> Dict[str, int]:
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    mismatches = {k for k in keys_a & keys_b if a[k] != b[k]}
    return {
        "only_a": len(only_a),
        "only_b": len(only_b),
        "mismatch": len(mismatches),
    }


def main() -> None:
    org_root = Path("local_extraction/v2/org_annotations")
    val_path = org_root / "fho_sta_val_height-540.json"
    train_path = org_root / "fho_sta_train_height-540.json"

    if not val_path.exists() or not train_path.exists():
        print(f"[warn] missing files: {val_path} or {train_path}")
        return

    val = load_vocab(val_path)
    train = load_vocab(train_path)

    noun_cmp = compare_maps(val["nouns"], train["nouns"])
    verb_cmp = compare_maps(val["verbs"], train["verbs"])

    print("=== STA height-540 vocab match ===")
    print(f"val nouns: {len(val['nouns'])}, train nouns: {len(train['nouns'])}, diffs: {noun_cmp}")
    print(f"val verbs: {len(val['verbs'])}, train verbs: {len(train['verbs'])}, diffs: {verb_cmp}")
    if noun_cmp["only_a"] or noun_cmp["only_b"] or noun_cmp["mismatch"]:
        print("  [warn] noun categories differ between val/train.")
    else:
        print("  nouns match exactly.")
    if verb_cmp["only_a"] or verb_cmp["only_b"] or verb_cmp["mismatch"]:
        print("  [warn] verb categories differ between val/train.")
    else:
        print("  verbs match exactly.")


if __name__ == "__main__":
    main()
