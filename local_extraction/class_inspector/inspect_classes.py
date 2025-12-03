#!/usr/bin/env python3
"""
Class/label inspector and dataset.yaml builder (local-only).

Modes (edit toggles below; no CLI needed):
  - labels       Scan YOLO .txt labels under local_extraction/<ver>/yolo_labels_540
  - annotations  Parse Ego4D STA annotations from mounted Drive (noun ids)
  - detections   Parse pre-extracted detections JSON from Drive (.../v2/sta_models/object_detections.json)
  - all          Run all three and merge what we find

Outputs are written to: local_extraction/class_inspector/<RUN_STAMP>/
  - labels_classes.json / .csv (if labels mode)
  - annotations_classes.json / .csv (if annotations mode)
  - detections_classes.json / .csv (if detections mode)
  - merged_names.json (best-effort id->name mapping)
  - dataset.yaml (YOLO format; names + basic paths)
  - summary.json
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
from collections import Counter
from datetime import datetime


# ================== TOGGLES (edit here) ==================
MODE = 'labels'                 # 'labels' | 'annotations' | 'detections' | 'all'
VERSION = 'v2'

# Labels scan
LABEL_ROOT = Path('local_extraction') / VERSION / 'yolo_labels_540'
LABEL_SPACES = ('clips', 'videos')  # which subtrees to scan

# Annotations scan
USE_HEIGHT_540 = True        # read from annotations_540ss when True
SPLITS = ('train', 'val')

# Detections JSON on mounted Drive
DETECTIONS_REL_PATH = Path('v2') / 'sta_models' / 'object_detections.json'

# Dataset.yaml build
FRAMES_ROOT = Path('local_extraction') / VERSION / 'extracted_frames'
DATASET_REL_TRAIN = 'train'    # placeholders; adjust to your structure if needed
DATASET_REL_VAL = 'val'

# Run folder
RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR = Path('local_extraction') / 'class_inspector' / RUN_STAMP
# =========================================================


def detect_drive_root() -> str:
    env = os.environ.get('EGO4D_ROOT')
    if env and os.path.isdir(env):
        return env
    for p in (
        'H:/My Drive/ego4d_data',
        'D:/My Drive/ego4d_data',
        '/mnt/h/My Drive/ego4d_data',
        '/mnt/d/My Drive/ego4d_data',
        '/content/drive/MyDrive/ego4d_data',
        '/drive/MyDrive/ego4d_data',
    ):
        if os.path.isdir(p):
            return p
    return 'H:/My Drive/ego4d_data'


def to_int_or_none(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def scan_yolo_labels(label_root: Path, spaces: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for sp in spaces:
        root = label_root / sp
        if not root.is_dir():
            continue
        for uid_dir in root.rglob('*'):
            if not uid_dir.is_dir():
                continue
            for txt in uid_dir.glob('*.txt'):
                try:
                    for ln in txt.read_text().splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        cls = to_int_or_none(ln.split()[0])
                        if cls is None:
                            continue
                        counts[cls] += 1
                except Exception:
                    continue
    return counts


def scan_annotations(ego4d_root: str, version: str, use_540: bool, splits: Iterable[str]) -> Tuple[Counter, Dict[int, str]]:
    ann_dir = Path(ego4d_root) / version / ('annotations_540ss' if use_540 else 'annotations')
    files = []
    if 'train' in splits:
        files.append(ann_dir / ('fho_sta_train_height-540.json' if use_540 else 'fho_sta_train.json'))
    if 'val' in splits:
        files.append(ann_dir / ('fho_sta_val_height-540.json' if use_540 else 'fho_sta_val.json'))
    counts: Counter = Counter()

    # Best-effort taxonomy lookup
    name_map: Dict[int, str] = {}
    for cand in (
        ann_dir / 'fho_sta_taxonomy.json',
        ann_dir / 'noun_taxonomy.json',
        ann_dir.parent / 'annotations' / 'fho_sta_taxonomy.json',
        ann_dir.parent / 'annotations' / 'noun_taxonomy.json',
    ):
        try:
            if cand.exists():
                data = json.loads(cand.read_text())
                # Accept common shapes: {id: name} or {'nouns':[{'id':..,'name':..}, ...]}
                if isinstance(data, dict) and 'nouns' in data and isinstance(data['nouns'], list):
                    for n in data['nouns']:
                        nid = to_int_or_none(n.get('id'))
                        nm = n.get('name')
                        if nid is not None and isinstance(nm, str):
                            name_map[nid] = nm
                elif isinstance(data, dict):
                    for k, v in data.items():
                        nid = to_int_or_none(k)
                        if nid is not None and isinstance(v, str):
                            name_map[nid] = v
        except Exception:
            pass

    import json as _json
    for f in files:
        if not f.exists():
            continue
        try:
            raw = _json.loads(f.read_text())
        except Exception:
            continue
        # Collect noun ids from objects
        def it(obj):
            if isinstance(obj, dict):
                if 'annotations' in obj and isinstance(obj['annotations'], list):
                    return obj['annotations']
                if 'data' in obj and isinstance(obj['data'], list):
                    return obj['data']
            if isinstance(obj, list):
                return obj
            return []
        for rec in it(raw):
            for ob in (rec.get('objects') or []):
                nid = to_int_or_none(ob.get('noun_category_id', ob.get('noun_id', 0)))
                if nid is None:
                    continue
                counts[nid] += 1

    return counts, name_map


def scan_detections_json(ego4d_root: str, detections_rel: Path, version: str) -> Tuple[Counter, Dict[int, str]]:
    p = Path(ego4d_root) / detections_rel
    counts: Counter = Counter()
    names: Dict[int, str] = {}
    if not p.exists():
        return counts, names
    try:
        data = json.loads(p.read_text())
    except Exception:
        return counts, names

    # Flexible parsing of common shapes
    def walk(obj):
        if isinstance(obj, dict):
            # Look for class fields
            cid = None
            if 'class_id' in obj:
                cid = to_int_or_none(obj['class_id'])
            elif 'category_id' in obj:
                cid = to_int_or_none(obj['category_id'])
            elif 'cls' in obj:
                cid = to_int_or_none(obj['cls'])
            if cid is not None:
                counts[cid] += 1
                # optional name
                for k in ('class_name', 'name', 'label'):
                    nm = obj.get(k)
                    if isinstance(nm, str):
                        names.setdefault(cid, nm)
            # Recurse
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)
    return counts, names


def write_counter(out_prefix: Path, counts: Counter):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # JSON
    (out_prefix.with_suffix('.json')).write_text(json.dumps(counts, indent=2), encoding='utf-8')
    # CSV
    lines = ['class_id,count'] + [f'{int(k)},{int(v)}' for k, v in counts.most_common()]
    (out_prefix.with_suffix('.csv')).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_names_map(*name_maps: Dict[int, str]) -> Dict[int, str]:
    merged: Dict[int, str] = {}
    for nm in name_maps:
        for k, v in (nm or {}).items():
            if isinstance(k, int) and isinstance(v, str):
                merged.setdefault(k, v)
    return merged


def build_dataset_yaml(out_path: Path, names: Dict[int, str]):
    # Construct YOLO dataset.yaml content
    # path can be set to frames root; train/val are placeholders unless you have split folders
    names_sorted = {int(k): str(v) for k, v in sorted(names.items(), key=lambda kv: kv[0])}
    if not names_sorted:
        # If no names available, create from observed ids in other counters
        names_sorted = {}
    nc = len(names_sorted)
    yaml_lines = []
    yaml_lines.append(f"path: {FRAMES_ROOT}")
    yaml_lines.append(f"train: {DATASET_REL_TRAIN}")
    yaml_lines.append(f"val: {DATASET_REL_VAL}")
    yaml_lines.append(f"nc: {nc}")
    yaml_lines.append("names:")
    if names_sorted:
        for k, v in names_sorted.items():
            yaml_lines.append(f"  {k}: {v}")
    else:
        yaml_lines.append("  {}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(yaml_lines) + '\n', encoding='utf-8')


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ego4d_root = detect_drive_root()

    results = {}
    merged_names: Dict[int, str] = {}

    if MODE in ('labels', 'all'):
        lbl_counts = scan_yolo_labels(LABEL_ROOT, LABEL_SPACES)
        write_counter(OUT_DIR / 'labels_classes', lbl_counts)
        results['labels'] = dict(lbl_counts)

    ann_counts = Counter()
    ann_names: Dict[int, str] = {}
    if MODE in ('annotations', 'all'):
        ann_counts, ann_names = scan_annotations(ego4d_root, VERSION, USE_HEIGHT_540, SPLITS)
        write_counter(OUT_DIR / 'annotations_classes', ann_counts)
        results['annotations'] = dict(ann_counts)
        merged_names = build_names_map(merged_names, ann_names)

    det_counts = Counter()
    det_names: Dict[int, str] = {}
    if MODE in ('detections', 'all'):
        det_counts, det_names = scan_detections_json(ego4d_root, DETECTIONS_REL_PATH, VERSION)
        write_counter(OUT_DIR / 'detections_classes', det_counts)
        results['detections'] = dict(det_counts)
        merged_names = build_names_map(merged_names, det_names)

    # Prefer names from annotations; fill missing with detections; fallback to id strings
    if not merged_names:
        # fall back to observed ids from labels/detections/annotations
        observed_ids = set()
        for src in ('labels', 'annotations', 'detections'):
            if src in results:
                observed_ids.update(int(k) for k in results[src].keys())
        merged_names = {i: f'class_{i}' for i in sorted(observed_ids)}

    (OUT_DIR / 'merged_names.json').write_text(json.dumps(merged_names, indent=2), encoding='utf-8')

    # Build dataset.yaml
    build_dataset_yaml(OUT_DIR / 'dataset.yaml', merged_names)

    # Summary
    summary = {
        'mode': MODE,
        'version': VERSION,
        'label_root': str(LABEL_ROOT),
        'frames_root': str(FRAMES_ROOT),
        'drive_root': ego4d_root,
        'detections_path': str(Path(ego4d_root) / DETECTIONS_REL_PATH),
        'outputs': str(OUT_DIR),
        'classes_found': {
            'labels': list(results.get('labels', {}).keys()),
            'annotations': list(results.get('annotations', {}).keys()),
            'detections': list(results.get('detections', {}).keys()),
        },
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('[inspector] Done. Outputs →', OUT_DIR)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

