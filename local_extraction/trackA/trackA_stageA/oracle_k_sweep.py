#!/usr/bin/env python3
"""
Oracle K sweep runner for Track A/B.
- Runs Stage A in oracle mode for a list of K values
- Runs Stage B after each
- Collects recall metrics from Stage B summaries

Env overrides supported by Stage A (already implemented):
  STAGEA_MODE=oracle
  STAGEA_K=<int>
  STAGEA_MAX_IMAGES=<int>

Optional env for this script:
  ORACLE_SWEEP_MAX_IMAGES: cap Stage A images (default 500)
  ORACLE_SWEEP_KS: comma-separated list of Ks (default "1,2,3,5,8,10")
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]  # repo root
STAGEA = ROOT / "local_extraction" / "trackA_stageA" / "trackA_stageA.py"
STAGEB = ROOT / "local_extraction" / "trackA_stageB" / "trackA_stageB.py"
RUNS = ROOT / "local_extraction" / "runs" / "Track_A"


def latest_run(prefix: str) -> Path | None:
    cand = [p for p in RUNS.glob(f"{prefix}_*") if p.is_dir()]
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def run_stage(script: Path, extra_env: dict[str, str] | None = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env, capture_output=True, text=True)
    # Print minimal trace
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
    return proc.returncode


def main() -> int:
    ks = os.environ.get("ORACLE_SWEEP_KS", "1,2,3,5,8,10")
    k_list = [int(x) for x in ks.split(",") if x.strip()]
    max_images = int(os.environ.get("ORACLE_SWEEP_MAX_IMAGES", "500"))

    results = []
    for k in k_list:
        print(f"\n[SWEEP] Running Stage A (oracle) with K={k} (max_images={max_images}) ...", flush=True)
        rc = run_stage(STAGEA, {
            "STAGEA_MODE": "oracle",
            "STAGEA_K": str(k),
            "STAGEA_MAX_IMAGES": str(max_images),
        })
        if rc != 0:
            print(f"[SWEEP] Stage A failed for K={k} (rc={rc})", file=sys.stderr)
            continue
        a_run = latest_run("trackA_stageA")
        if not a_run:
            print("[SWEEP] Could not find Stage A run directory", file=sys.stderr)
            continue
        print(f"[SWEEP] Stage A run: {a_run}")

        print(f"[SWEEP] Running Stage B for K={k} ...", flush=True)
        rc = run_stage(STAGEB)
        if rc != 0:
            print(f"[SWEEP] Stage B failed for K={k} (rc={rc})", file=sys.stderr)
            continue
        b_run = latest_run("trackA_stageB")
        if not b_run:
            print("[SWEEP] Could not find Stage B run directory", file=sys.stderr)
            continue
        print(f"[SWEEP] Stage B run: {b_run}")

        sum_path = b_run / "summary.json"
        if not sum_path.exists():
            print(f"[SWEEP] Missing summary at {sum_path}", file=sys.stderr)
            continue
        s = json.loads(sum_path.read_text(encoding="utf-8"))
        rec = s.get("recall_metrics") or {}
        results.append({
            "K": k,
            "images_with_gt": rec.get("images_with_gt"),
            "images_with_hit": rec.get("images_with_hit"),
            "recall_at_K": rec.get("recall_at_K"),
            "mean_best_iou": rec.get("mean_best_iou"),
            "run_dir": str(b_run),
        })
        # small pause to avoid identical timestamps
        time.sleep(0.5)

    # Print table-like output
    print("\n[SWEEP] Results:")
    if results:
        width = {
            "K": 3, "images_with_gt": 15, "images_with_hit": 15, "recall_at_K": 12, "mean_best_iou": 14
        }
        header = f"{'K':>{width['K']}}  {'images_with_gt':>{width['images_with_gt']}}  {'images_with_hit':>{width['images_with_hit']}}  {'recall_at_K':>{width['recall_at_K']}}  {'mean_best_iou':>{width['mean_best_iou']}}"
        print(header)
        print("-" * len(header))
        for r in sorted(results, key=lambda x: x['K']):
            print(f"{r['K']:>{width['K']}}  {r['images_with_gt']:>{width['images_with_gt']}}  {r['images_with_hit']:>{width['images_with_hit']}}  {r['recall_at_K']:>{width['recall_at_K']}.6f}  {r['mean_best_iou']:>{width['mean_best_iou']}.6f}")
    else:
        print("No results collected.")

    # Dump JSON for further processing
    out_json = ROOT / "local_extraction" / "runs" / "Track_A" / f"oracle_k_sweep_{int(time.time())}.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[SWEEP] Saved JSON: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
