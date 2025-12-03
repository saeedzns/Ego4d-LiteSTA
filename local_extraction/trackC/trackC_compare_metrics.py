#!/usr/bin/env python3
"""
Compare Track B (unpruned) and Track C (pruned) metrics.

This script loads:
  - One or more Track B metrics JSON files under local_extraction/runs/Track_B/metrics
  - One or more Track C metrics JSON files under local_extraction/runs/Track_C/metrics

and prints a side-by-side comparison for key metrics:
  - accuracy, mAP, ttc_mae_seconds
  - N_mAP, Nv_mAP, N_delta_mAP, All_mAP
  - N_top5_acc/mAP, Nv_top5_acc/mAP, N_delta_top5_acc/mAP, All_top5_acc/mAP

Usage (from repo root):

  # Compare latest Track B vs Track C (single pair):
  python local_extraction/trackC/trackC_compare_metrics.py

  # Compare all runs from both tracks and save to TSV:
  python local_extraction/trackC/trackC_compare_metrics.py --all --out comparison.tsv

You can override specific files with:

  --b_metrics path/to/metrics_val_YYYY.json
  --c_metrics path/to/trackC_val_rateXX_YYYY.json

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


KEYS_ORDER = [
    "N_top5_mAP",
    "Nv_top5_mAP",
    "N_delta_top5_mAP",
    "All_top5_mAP",
    "accuracy",
    "mAP",
    "ttc_mae_seconds",
    "N_mAP",
    "Nv_mAP",
    "N_delta_mAP",
    "All_mAP",
    "N_top5_acc",
    "Nv_top5_acc",
    "N_delta_top5_acc",
    "All_top5_acc",
]


def _latest_json(dir_path: Path, prefix: str) -> Optional[Path]:
    if not dir_path.exists():
        return None
    files = sorted(dir_path.glob(prefix + "*.json"))
    return files[-1] if files else None


def _all_jsons(dir_path: Path, prefix: str) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted(dir_path.glob(prefix + "*.json"))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.6f}"
    if isinstance(x, int):
        return str(x)
    return "-"


def _build_row(track: str, path: Path, data: Dict[str, Any]) -> Dict[str, str]:
    row: Dict[str, str] = {
        "track": track,
        "file": path.name,
        "checkpoint": Path(data.get("checkpoint", "")).name or "-",
        "timestamp": data.get("timestamp") or "-",
    }
    for key in KEYS_ORDER:
        row[key] = _fmt(data.get(key))
    return row


def compare_single(b_path: Path, c_path: Path) -> None:
    """Print a single pair comparison with metrics as columns."""
    b = _load_json(b_path)
    c = _load_json(c_path)

    print(f"Track B: {b_path}")
    print(f"Track C: {c_path}\n")

    # Header row
    header = ["track"] + KEYS_ORDER + ["file", "checkpoint", "timestamp"]
    print("\t".join(header))
    print("-" * 120)

    b_row = _build_row("B", b_path, b)
    c_row = _build_row("C", c_path, c)

    print("\t".join(b_row.get(h, "-") for h in header))
    print("\t".join(c_row.get(h, "-") for h in header))


def compare_all(b_dir: Path, c_dir: Path, out_path: Optional[Path]) -> None:
    """Load all Track B and C metrics, print table, optionally save to TSV."""
    b_files = _all_jsons(b_dir, "metrics_val_")
    c_files = _all_jsons(c_dir, "trackC_val_rate")

    if not b_files and not c_files:
        print("[compare] No metrics files found in Track B or Track C.")
        return

    rows: List[Dict[str, str]] = []

    for p in b_files:
        try:
            data = _load_json(p)
            rows.append(_build_row("B", p, data))
        except Exception as e:
            print(f"[compare] Skipping {p}: {e}")

    for p in c_files:
        try:
            data = _load_json(p)
            rows.append(_build_row("C", p, data))
        except Exception as e:
            print(f"[compare] Skipping {p}: {e}")

    if not rows:
        print("[compare] No valid metrics loaded.")
        return

    header = ["track"] + KEYS_ORDER + ["file", "checkpoint", "timestamp"]

    # Print to console
    print("\t".join(header))
    print("-" * 160)
    for row in rows:
        print("\t".join(row.get(h, "-") for h in header))

    # Save to TSV if requested
    if out_path:
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(header) + "\n")
            for row in rows:
                f.write("\t".join(row.get(h, "-") for h in header) + "\n")
        print(f"\n[compare] Saved comparison table -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Track B vs Track C metrics.")
    ap.add_argument(
        "--b_metrics",
        type=str,
        default=None,
        help="Track B metrics JSON path; defaults to latest metrics_val_*.json under runs/Track_B/metrics.",
    )
    ap.add_argument(
        "--c_metrics",
        type=str,
        default=None,
        help="Track C metrics JSON path; defaults to latest trackC_val_rate*.json under runs/Track_C/metrics.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Compare all runs from both Track B and Track C together.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output TSV file path (used with --all).",
    )
    args = ap.parse_args()

    b_dir = Path("local_extraction") / "runs" / "Track_B" / "metrics"
    c_dir = Path("local_extraction") / "runs" / "Track_C" / "metrics"

    if args.all:
        out_path = Path(args.out) if args.out else None
        compare_all(b_dir, c_dir, out_path)
    else:
        b_path = Path(args.b_metrics) if args.b_metrics else _latest_json(b_dir, "metrics_val_")
        c_path = Path(args.c_metrics) if args.c_metrics else _latest_json(c_dir, "trackC_val_rate")

        if b_path is None or not b_path.exists():
            raise FileNotFoundError(f"Track B metrics not found (searched in {b_dir}); specify --b_metrics.")
        if c_path is None or not c_path.exists():
            raise FileNotFoundError(f"Track C metrics not found (searched in {c_dir}); specify --c_metrics.")

        compare_single(b_path, c_path)


if __name__ == "__main__":
    main()

