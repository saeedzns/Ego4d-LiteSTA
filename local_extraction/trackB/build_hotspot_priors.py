#!/usr/bin/env python3
"""
Build hotspot priors (PEAR-style) from StageB manifests.

Output JSON format (consumed by trackB_eval.py):
{
  "pairs": {
    "noun_id,verb_id": score,
    "12,3": 0.9,
    "48,7": 0.2
  },
  "default": 0.0
}

Keys are global STA noun/verb IDs as seen in head_train/head_val manifests.
Score is the empirical P(next-active | noun,verb) estimated from labels.

Configuration loaded from configs/trackB.yaml

CLI arguments override YAML settings:
- --manifests: StageB head manifests (JSON/JSONL)
- --out: output path for the prior JSON
- --method: "prob" (P(pos|pair)) or "logodds" (smoothed log odds)
- --min_total: skip pairs with total count below this
- --alpha: smoothing for log-odds
"""

from __future__ import annotations

import sys
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional


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


def _iter_manifest_records(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield dict records from JSON/JSONL."""
    if not path.exists():
        print(f"[warn] manifest not found: {path}")
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Try JSON array
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


def main() -> None:
    # -------- Config from YAML (fallback to hardcoded defaults) --------
    USE_CONFIG = True
    
    # Defaults from YAML or fallback
    SOURCE_MODE = "train"  # "train", "val", "both", or "custom"
    
    # Get stageB_run from YAML if available
    stageB_run_str = _cfg.get('data.stageB_run') if _cfg else None
    if stageB_run_str:
        STAGEB_RUN = Path(stageB_run_str)
    else:
        # Auto-discover latest
        runs_root = Path(_cfg.get('paths.runs', 'local_extraction/runs') if _cfg else 'local_extraction/runs') / "Track_A"
        stageb_runs = sorted([d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith('trackA_stageB_')]) if runs_root.exists() else []
        STAGEB_RUN = stageb_runs[-1] if stageb_runs else Path("local_extraction/runs/Track_A/trackA_stageB_20251117_184342")
    
    # Custom manifest paths (used when SOURCE_MODE == "custom")
    CUSTOM_MANIFESTS = []
    
    # Get defaults from YAML hotspot_priors section
    DEFAULT_OUT = _cfg.get('evaluation.hotspot_priors.path', 'local_extraction/v2/hotspot_priors_{source}_{method}_min{min_total}.json') if _cfg else 'local_extraction/v2/hotspot_priors_{source}_{method}_min{min_total}.json'
    DEFAULT_METHOD = "logodds"   # "prob" or "logodds"
    DEFAULT_MIN_TOTAL = 10
    DEFAULT_ALPHA = _cfg.get('evaluation.hotspot_priors.alpha', 1.0) if _cfg else 1.0

    # -------------------------------------------------------------------

    ap = argparse.ArgumentParser(description="Build hotspot prior JSON from StageB manifests.")
    ap.add_argument("--manifests", type=str, nargs="+", required=not USE_CONFIG, help="StageB manifest paths (head_train/head_val .json or .jsonl).")
    # Format default out using current defaults
    default_out_value = DEFAULT_OUT.format(source=SOURCE_MODE, method=DEFAULT_METHOD, min_total=DEFAULT_MIN_TOTAL)
    ap.add_argument("--out", type=str, default=default_out_value, help="Output JSON path for priors.")
    ap.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=["prob", "logodds"], help="Scoring method: prob = P(pos|pair), logodds = log((pos+alpha)/(neg+alpha)).")
    ap.add_argument("--min_total", type=int, default=DEFAULT_MIN_TOTAL, help="Skip pairs with total count < min_total.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Smoothing for log-odds (pseudocount).")
    args = ap.parse_args()

    if USE_CONFIG:
        manifests = []
        if SOURCE_MODE == "custom":
            manifests = [Path(p) for p in CUSTOM_MANIFESTS]
        else:
            if SOURCE_MODE in ("train", "both"):
                manifests.append(STAGEB_RUN / "head_train.jsonl")
            if SOURCE_MODE in ("val", "both"):
                manifests.append(STAGEB_RUN / "head_val.jsonl")
        if not manifests:
            print("[error] no manifests configured. Set CUSTOM_MANIFESTS or adjust SOURCE_MODE.")
            return
        args.manifests = [str(p) for p in manifests]
        # Build default out name based on source/method if not overridden
        if args.out == default_out_value or args.out == DEFAULT_OUT:
            args.out = DEFAULT_OUT.format(source=SOURCE_MODE, method=args.method, min_total=args.min_total)
    else:
        # If CLI default still has placeholders, format with current args
        if "{" in args.out and "}" in args.out:
            args.out = args.out.format(source="cli", method=args.method, min_total=args.min_total)

    pair_pos = Counter()
    pair_total = Counter()

    for mpath in args.manifests:
        for rec in _iter_manifest_records(Path(mpath)):
            # StageB manifests: per-candidate
            lbl = rec.get("is_positive")
            nid = rec.get("noun_id")
            vid = rec.get("verb_id")
            try:
                nid_int = int(nid) if nid is not None else -1
                vid_int = int(vid) if vid is not None else -1
                lbl_int = int(lbl) if lbl is not None else None
            except Exception:
                continue
            if nid_int < 0 or vid_int < 0 or lbl_int is None:
                continue
            pair = (nid_int, vid_int)
            pair_total[pair] += 1
            if lbl_int == 1:
                pair_pos[pair] += 1

    if not pair_total:
        print("[error] no valid pairs found in manifests; nothing to write.")
        return

    # Global positive rate used for a neutral default in "prob" mode
    total_pos = sum(pair_pos.values())
    total_tot = sum(pair_total.values())
    global_rate = total_pos / (total_tot + 1e-6)

    pairs_json: Dict[str, float] = {}
    for (n, v), tot in pair_total.items():
        if tot < args.min_total:
            continue
        pos = pair_pos[(n, v)]
        neg = tot - pos
        if args.method == "prob":
            score = pos / (tot + 1e-6)
        else:  # logodds
            score = (pos + args.alpha) / (neg + args.alpha)
            # convert to log-odds
            score = float(json.loads(json.dumps(score)))  # ensure JSON-safe float
            if score > 0:
                score = float(__import__("math").log(score))
            else:
                score = -999.0
        pairs_json[f"{n},{v}"] = float(score)

    # Default prior: for prob mode use global P(pos); for logodds neutral is 0.0 (odds=1)
    if args.method == "prob":
        default_value = float(global_rate)
    else:
        default_value = 0.0

    data = {
        "pairs": pairs_json,
        "default": default_value,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[info] wrote hotspot priors: {out_path} (pairs={len(pairs_json)})")


if __name__ == "__main__":
    main()
