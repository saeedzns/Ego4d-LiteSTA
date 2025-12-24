#!/usr/bin/env python3
"""Track A smoke test runner.

This is intentionally lightweight and opt-in:
- Reads `smoke_test.*` from configs/trackA.yaml
- Runs Stage A and Stage B with small caps
- Checks for expected output files

Usage:
  python -m trackA.trackA_smoke_test

Notes:
- This does NOT change normal StageA/StageB behavior unless you run it.
- Uses env overrides STAGEA_MAX_IMAGES / STAGEB_MAX_IMAGES.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runs_root(root: Path) -> Path:
    return root / "local_extraction" / "runs" / "Track_A"


def _latest_run(runs_root: Path, prefix: str) -> Path | None:
    cands = [p for p in runs_root.glob(f"{prefix}_*") if p.is_dir()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _run_py(script: Path, cwd: Path, env: dict[str, str]) -> int:
    proc = subprocess.run([sys.executable, str(script)], cwd=str(cwd), env=env)
    return int(proc.returncode)


def main() -> int:
    root = _repo_root()
    try:
        # make core importable
        sys.path.insert(0, str(root / "local_extraction"))
        sys.path.insert(0, str(root))
        from core import load_config

        cfg = load_config("trackA")
    except Exception as e:
        print(f"[trackA.smoke] failed to load config: {e}")
        return 2

    if not bool(cfg.get("smoke_test.enabled", False)):
        print("[trackA.smoke] smoke_test.enabled is false; nothing to do")
        return 0

    max_samples = int(cfg.get("smoke_test.max_samples", 10))
    expected = cfg.get("smoke_test.expected_outputs", ["candidates.jsonl", "summary.json"]) or []

    stagea_script = root / "local_extraction" / "trackA" / "trackA_stageA" / "trackA_stageA.py"
    stageb_script = root / "local_extraction" / "trackA" / "trackA_stageB" / "trackA_stageB.py"

    run_prefix_a = cfg.get("stage_a.output.run_prefix", "trackA_stageA")
    run_prefix_b = cfg.get("stage_b.output.run_prefix", "trackA_stageB")

    env = os.environ.copy()
    env["STAGEA_MAX_IMAGES"] = str(max_samples)
    env["STAGEB_MAX_IMAGES"] = str(max_samples)

    print(f"[trackA.smoke] Running Stage A with STAGEA_MAX_IMAGES={max_samples}")
    rc = _run_py(stagea_script, cwd=root, env=env)
    if rc != 0:
        print(f"[trackA.smoke] Stage A failed (rc={rc})")
        return rc

    runs_root = _runs_root(root)
    a_run = _latest_run(runs_root, str(run_prefix_a))
    if a_run is None:
        print("[trackA.smoke] Could not locate latest Stage A run directory")
        return 3

    missing = [name for name in expected if not (a_run / name).exists()]
    if missing:
        print(f"[trackA.smoke] Missing Stage A outputs in {a_run}: {missing}")
        return 4

    print(f"[trackA.smoke] Stage A outputs OK in {a_run}")

    print(f"[trackA.smoke] Running Stage B with STAGEB_MAX_IMAGES={max_samples}")
    rc = _run_py(stageb_script, cwd=root, env=env)
    if rc != 0:
        print(f"[trackA.smoke] Stage B failed (rc={rc})")
        return rc

    b_run = _latest_run(runs_root, str(run_prefix_b))
    if b_run is None:
        print("[trackA.smoke] Could not locate latest Stage B run directory")
        return 5

    # Minimal sanity: Stage B should emit a manifest.
    if not (b_run / "manifest.jsonl").exists():
        print(f"[trackA.smoke] Missing Stage B manifest.jsonl in {b_run}")
        return 6

    print(f"[trackA.smoke] Stage B outputs OK in {b_run}")
    print("[trackA.smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
