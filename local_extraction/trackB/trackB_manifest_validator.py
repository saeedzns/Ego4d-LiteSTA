#!/usr/bin/env python3
"""Track B Manifest Validator

Reports:
 - Total records, total candidates
 - Key coverage frequencies
 - TTC stats (mean/std/min/max) and % of records with TTC
 - Sample records (first 3) with condensed candidate info
 - Writes ttc_stats.json next to manifest
 - Writes key_coverage.json listing encountered keys

Configuration loaded from configs/trackB.yaml

Usage:
  python local_extraction/trackB/trackB_manifest_validator.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# ================== YAML CONFIG LOADING ==================
def _load_yaml_config():
    """Load YAML config for Track B."""
    _THIS_DIR = Path(__file__).resolve().parent
    _LOCAL_EXTRACTION = _THIS_DIR.parent
    for p in [str(_LOCAL_EXTRACTION), str(_LOCAL_EXTRACTION.parent), str(_THIS_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from core import load_config
        return load_config('trackB')
    except Exception:
        return None

_cfg = _load_yaml_config()
# =========================================================

from trackB_dataset import (
    load_manifest,
    discover_manifest,
    latest_stageB_run,
    resolve_stageB_manifest,
)


def summarize_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in ['x1','y1','x2','y2','xmin','ymin','xmax','ymax','left','top','right','bottom']:
        if k in c:
            out[k] = c[k]
    if 'cls' in c:
        out['cls'] = c['cls']
    if 'label' in c and 'cls' not in out:
        out['label'] = c['label']
    if 'ttc' in c:
        out['ttc'] = c['ttc']
    elif 'time_to_contact' in c:
        out['ttc'] = c['time_to_contact']
    return out


@dataclass
class ValidatorConfig:
    """
    Toggle between base-dataset manifests and Stage A StageB run manifests.
    Configuration is loaded from configs/trackB.yaml.
    """

    # Explicit manifest override (takes precedence over everything else)
    manifest_path: Optional[Path] = None

    # Toggle: use latest Stage A StageB run manifests vs base dataset manifests
    use_stageA_run: bool = True

    # Where to look for Stage A StageB runs (trackA_stageB_*)
    trackA_runs_root: Path = field(
        default_factory=lambda: Path(_cfg.get('paths.runs', 'local_extraction/runs') if _cfg else 'local_extraction/runs') / "Track_A"
    )

    # If not None, use this specific StageB run directory instead of auto-discovery
    stageA_run: Optional[Path] = field(
        default_factory=lambda: Path(_cfg.get('data.stageB_run')) if (_cfg and _cfg.get('data.stageB_run')) else None
    )

    # Base name of the manifest inside a StageB run (e.g., 'head_train' -> head_train.json[l])
    stageA_manifest_base: str = "head_train"

    # Base dataset manifests root (used when use_stageA_run is False or as fallback)
    base_manifests_root: Path = field(
        default_factory=lambda: Path(_cfg.get('paths.manifests', 'local_extraction/v2/manifests') if _cfg else 'local_extraction/v2/manifests')
    )


def main():
    cfg = ValidatorConfig()

    # 1) Explicit manifest path wins
    mp: Optional[Path] = cfg.manifest_path
    if mp is not None and not isinstance(mp, Path):
        mp = Path(mp)

    # 2) If requested, try Stage A StageB run manifests
    stageB_run: Optional[Path] = None
    if mp is None and cfg.use_stageA_run:
        stageB_run = cfg.stageA_run or latest_stageB_run(cfg.trackA_runs_root)
        if stageB_run is not None:
            mp = resolve_stageB_manifest(stageB_run, cfg.stageA_manifest_base)
            if mp is None:
                print(f"[validator] StageA run found at {stageB_run} but no manifest for base='{cfg.stageA_manifest_base}'. Falling back to base dataset manifests.")
        else:
            print(f"[validator] No TrackA StageB run found under {cfg.trackA_runs_root}; falling back to base dataset manifests.")

    # 3) Fallback: base dataset manifests under v2/manifests
    manif_root = cfg.base_manifests_root
    if mp is None:
        mp = discover_manifest(manif_root)

    if mp is None:
        print("[validator] No manifest found under", manif_root)
        return
    records = load_manifest(mp)
    if not records:
        print("[validator] Manifest empty or unreadable:", mp)
        return

    key_counter = Counter()
    candidate_key_counter = Counter()
    ttc_values: List[float] = []

    for r in records:
        key_counter.update(r.keys())
        cands = r.get('candidates') or r.get('boxes') or r.get('objects') or []
        for c in cands:
            if isinstance(c, dict):
                candidate_key_counter.update(c.keys())
                if 'ttc' in c and isinstance(c['ttc'], (int,float)):
                    ttc_values.append(float(c['ttc']))
                elif 'time_to_contact' in c and isinstance(c['time_to_contact'], (int,float)):
                    ttc_values.append(float(c['time_to_contact']))

    total_records = len(records)
    total_candidates = sum(candidate_key_counter.values())  # rough

    if ttc_values:
        mean = sum(ttc_values)/len(ttc_values)
        var = sum((x-mean)**2 for x in ttc_values) / max(1,len(ttc_values)-1)
        std = var**0.5
        stats = {
            'count': len(ttc_values),
            'mean': mean,
            'std': std,
            'min': min(ttc_values),
            'max': max(ttc_values),
        }
    else:
        stats = {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

    print(f"[validator] Manifest: {mp}")
    print(f"[validator] Records: {total_records}")
    print(f"[validator] TTC samples: {stats['count']} mean={stats['mean']:.3f} std={stats['std']:.3f} min={stats['min']:.3f} max={stats['max']:.3f}")
    print("[validator] Top record keys:")
    for k,v in key_counter.most_common(15):
        print(f"  {k}: {v}")
    print("[validator] Candidate keys:")
    for k,v in candidate_key_counter.most_common(15):
        print(f"  {k}: {v}")

    # Sample records
    print("[validator] Sample records (3):")
    for r in records[:3]:
        cands = r.get('candidates') or r.get('boxes') or r.get('objects') or []
        sample_c = [summarize_candidate(c) for c in cands[:2]]
        slim = {k: r.get(k) for k in ['uid','video_uid','video_id','frame','frame_idx','frame_index','frame_name','image'] if k in r}
        slim['candidates_sample'] = sample_c
        print(json.dumps(slim)[:500])

    # Write stats files
    cache_root = Path("local_extraction") / "runs" / "Track_B" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    stats_path = cache_root / 'ttc_stats.json'
    with stats_path.open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    kc_path = cache_root / 'key_coverage.json'
    with kc_path.open('w', encoding='utf-8') as f:
        json.dump({
            'record_keys': dict(key_counter),
            'candidate_keys': dict(candidate_key_counter)
        }, f, indent=2)

    print("[validator] Wrote:", stats_path)
    print("[validator] Wrote:", kc_path)


if __name__ == '__main__':
    main()
