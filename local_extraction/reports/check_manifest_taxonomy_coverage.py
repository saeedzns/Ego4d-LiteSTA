#!/usr/bin/env python3
"""
Check whether STA manifests cover all noun/verb IDs from Ego4D taxonomies.

Usage example (from repo root):
  python local_extraction/reports/check_manifest_taxonomy_coverage.py ^
      --noun_taxonomy "H:/My Drive/ego4d_data/v2/annotations/narration_noun_taxonomy.csv" ^
      --verb_taxonomy "H:/My Drive/ego4d_data/v2/annotations/narration_verb_taxonomy.csv" ^
      --manifests local_extraction/runs/Track_A/trackA_stageB_20251117_184342/head_train.jsonl ^
                 local_extraction/runs/Track_A/trackA_stageB_20251117_184342/head_val.jsonl

You can point --manifests either to Stage B head manifests (head_train/head_val.jsonl)
or to the earlier label-builder head manifests under local_extraction/v2/manifests.

The script will print how many noun/verb IDs exist in the taxonomy, how many are
actually used in your manifests, and which IDs (if any) are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, Set, Dict, Any, List, Tuple


def load_taxonomy_labels(path: Path, kind: str, id_col: str = "id") -> Dict[int, str]:
    """
    Load (id -> label) mapping from a taxonomy file.

    Supports:
      - CSV with an explicit ID column (e.g., narration_*_taxonomy_csv.csv with 'id')
      - CSV without an ID column (e.g., narration_*_taxonomy.csv with 'label,group'):
        in this case, the row index (0-based) is treated as the class ID.
      - JSON with a top-level list under 'nouns' or 'verbs' (e.g., fho_main_taxonomy.json,
        fho_lta_taxonomy.json); IDs are indices into that list.
    """
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    labels: Dict[int, str] = {}
    suffix = path.suffix.lower()
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(obj)}")
        # Prefer STA-style categories if present
        cat_key = "noun_categories" if kind == "noun" else "verb_categories"
        if cat_key in obj and isinstance(obj.get(cat_key), list):
            for item in obj.get(cat_key) or []:
                if not isinstance(item, dict):
                    continue
                if "id" not in item:
                    continue
                try:
                    idx = int(item["id"])
                except Exception:
                    continue
                labels[idx] = str(item.get("name", ""))
            return labels
        # Fallback: top-level list under 'nouns' / 'verbs'
        arr = obj.get("nouns" if kind == "noun" else "verbs")
        if not isinstance(arr, list):
            raise ValueError(f"Expected key '{'nouns' if kind=='noun' else 'verbs'}' with a list in {path}")
        for i, name in enumerate(arr):
            labels[i] = str(name)
        return labels

    # CSV case
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        rows = list(reader)
        # If an explicit ID column exists, use it
        if id_col in reader.fieldnames:
            for row in rows:
                raw = row.get(id_col)
                if raw is None or raw == "":
                    continue
                try:
                    idx = int(raw)
                except Exception:
                    continue
                label = row.get("label") or row.get("name") or f"{kind}_{idx}"
                labels[idx] = str(label)
            return labels
        # Fallback: use row indices as IDs (Ego4D narration_*_taxonomy.csv style)
        for i, row in enumerate(rows):
            label = row.get("label") or row.get("name") or f"{kind}_{i}"
            labels[i] = str(label)
        return labels


def load_ids_from_taxonomy(path: Path, kind: str, id_col: str = "id") -> Set[int]:
    """Compatibility helper: return just the ID set."""
    labels = load_taxonomy_labels(path, kind=kind, id_col=id_col)
    return set(labels.keys())


def _iter_manifest_records(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield dict records from a manifest file.

    Supports:
      - JSON array: [ {...}, {...}, ... ]
      - JSONL: one JSON object per line
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Try JSON array first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict):
                    yield rec
            return
    except Exception:
        pass
    # Fallback: JSONL
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


def collect_ids_from_manifests(manifest_paths: Iterable[Path]) -> Dict[str, Set[int]]:
    """
    Scan manifests and collect noun/verb IDs.

    Handles both:
      - Stage B head manifests:  verb_id / noun_id
      - Raw STA-style records:   verb_category_id / noun_category_id
    """
    used_nouns: Set[int] = set()
    used_verbs: Set[int] = set()

    for path in manifest_paths:
        for rec in _iter_manifest_records(path):
            # Stage B head manifests: fields live at top level
            n = rec.get("noun_id", rec.get("noun_category_id"))
            v = rec.get("verb_id", rec.get("verb_category_id"))
            # Some manifests may nest per-object info; handle that too
            objs = rec.get("objects") or rec.get("candidates") or []
            if n is not None:
                try:
                    nid = int(n)
                    if nid >= 0:
                        used_nouns.add(nid)
                except Exception:
                    pass
            if v is not None:
                try:
                    vid = int(v)
                    if vid >= 0:
                        used_verbs.add(vid)
                except Exception:
                    pass
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                on = obj.get("noun_id", obj.get("noun_category_id"))
                ov = obj.get("verb_id", obj.get("verb_category_id"))
                if on is not None:
                    try:
                        nid = int(on)
                        if nid >= 0:
                            used_nouns.add(nid)
                    except Exception:
                        pass
                if ov is not None:
                    try:
                        vid = int(ov)
                        if vid >= 0:
                            used_verbs.add(vid)
                    except Exception:
                        pass

    return {"nouns": used_nouns, "verbs": used_verbs}


def _collect_stageB_examples(
    manifest_paths: Iterable[Path],
    used_nouns: Set[int],
    used_verbs: Set[int],
    k: int,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Collect up to k random Stage B‑style examples from manifests.

    We look for records that have:
      - image_path
      - noun_id and/or verb_id
      - candidate_box (Stage B head manifests)
    """
    if k <= 0:
        return []
    candidates: List[Tuple[Path, Dict[str, Any]]] = []
    for path in manifest_paths:
        for rec in _iter_manifest_records(path):
            if not isinstance(rec, dict):
                continue
            img = rec.get("image_path")
            box = rec.get("candidate_box")
            nid = rec.get("noun_id")
            vid = rec.get("verb_id")
            if not isinstance(img, str):
                continue
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            try:
                nid_int = int(nid) if nid is not None else None
                vid_int = int(vid) if vid is not None else None
            except Exception:
                nid_int = vid_int = None
            # Prefer examples whose IDs we know are in the used set
            if (nid_int is not None and nid_int in used_nouns) or (vid_int is not None and vid_int in used_verbs):
                candidates.append((path, rec))
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates
    return random.sample(candidates, k)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check coverage of STA manifests vs STA height-540 vocab.")
    ap.add_argument("--org_root", type=str, default="local_extraction/v2/org_annotations",
                    help="Root containing fho_sta_{train,val}_height-540.json.")
    ap.add_argument("--manifests", type=str, nargs="+", required=True,
                    help="One or more manifest paths (JSON or JSONL).")
    ap.add_argument("--examples_k", type=int, default=0,
                    help="If >0, print K random StageB-style examples (image_path, box, noun/verb labels).")

    args = ap.parse_args()

    org_root = Path(args.org_root)
    noun_tax_path = org_root / "fho_sta_val_height-540.json"
    verb_tax_path = noun_tax_path  # same file carries verb_categories
    manifest_paths = [Path(p) for p in args.manifests]

    noun_labels = load_taxonomy_labels(noun_tax_path, kind="noun", id_col="id")
    verb_labels = load_taxonomy_labels(verb_tax_path, kind="verb", id_col="id")
    # Include both val/train height-540 vocab if present
    noun_labels_train = {}
    verb_labels_train = {}
    train_path = org_root / "fho_sta_train_height-540.json"
    if train_path.exists():
        noun_labels_train = load_taxonomy_labels(train_path, kind="noun", id_col="id")
        verb_labels_train = load_taxonomy_labels(train_path, kind="verb", id_col="id")
    noun_labels.update(noun_labels_train)
    verb_labels.update(verb_labels_train)

    all_noun_ids = set(noun_labels.keys())
    all_verb_ids = set(verb_labels.keys())

    coverage = collect_ids_from_manifests(manifest_paths)
    used_nouns = coverage["nouns"]
    used_verbs = coverage["verbs"]

    missing_nouns = sorted(all_noun_ids - used_nouns)
    missing_verbs = sorted(all_verb_ids - used_verbs)

    print("=== Noun coverage ===")
    print(f"Taxonomy nouns: {len(all_noun_ids)}")
    print(f"Used in manifests: {len(used_nouns)}")
    if all_noun_ids:
        print(f"Coverage: {100.0 * len(used_nouns) / len(all_noun_ids):.2f}%")
    if missing_nouns:
        print(f"Missing noun IDs ({len(missing_nouns)}): {missing_nouns[:20]}{' ...' if len(missing_nouns) > 20 else ''}")
    else:
        print("No missing nouns: all taxonomy noun IDs appear in the manifests.")

    print("\n=== Verb coverage ===")
    print(f"Taxonomy verbs: {len(all_verb_ids)}")
    print(f"Used in manifests: {len(used_verbs)}")
    if all_verb_ids:
        print(f"Coverage: {100.0 * len(used_verbs) / len(all_verb_ids):.2f}%")
    if missing_verbs:
        print(f"Missing verb IDs ({len(missing_verbs)}): {missing_verbs[:20]}{' ...' if len(missing_verbs) > 20 else ''}")
    else:
        print("No missing verbs: all taxonomy verb IDs appear in the manifests.")

    # Optional: show random examples to visually inspect taxonomy vs frames
    if args.examples_k > 0:
        print(f"\n=== Random StageB-style examples (K={args.examples_k}) ===")
        examples = _collect_stageB_examples(manifest_paths, used_nouns, used_verbs, args.examples_k)
        if not examples:
            print("No suitable StageB-style records with image_path + candidate_box found in the provided manifests.")
        else:
            for i, (mpath, rec) in enumerate(examples, 1):
                img = rec.get("image_path")
                box = rec.get("candidate_box")
                nid = rec.get("noun_id")
                vid = rec.get("verb_id")
                try:
                    nid_int = int(nid) if nid is not None else None
                except Exception:
                    nid_int = None
                try:
                    vid_int = int(vid) if vid is not None else None
                except Exception:
                    vid_int = None
                noun_label = noun_labels.get(nid_int, "<unknown>") if nid_int is not None else "<none>"
                verb_label = verb_labels.get(vid_int, "<unknown>") if vid_int is not None else "<none>"
                print(f"\nExample {i}:")
                print(f"  manifest     : {mpath}")
                print(f"  image_path   : {img}")
                print(f"  candidate_box: {box}")
                print(f"  noun_id      : {nid_int} ({noun_label})")
                print(f"  verb_id      : {vid_int} ({verb_label})")


if __name__ == "__main__":
    main()
