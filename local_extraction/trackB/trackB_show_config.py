#!/usr/bin/env python3
"""
Inspect Track B checkpoint configuration.

This script loads a Track B `.pt` checkpoint (produced by trackB_train_loader.py)
and prints the stored TrainConfig fields so you can see which settings were used
for that run (epochs, lr, candidate_limit, multi-task flags, loss weights, etc.).

Usage (from repo root):

  python local_extraction/trackB/trackB_show_config.py \
      --ckpt local_extraction/runs/Track_B/checkpoints/trackB_best_mAP_0.1234_20251120_123456.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Show TrainConfig stored in a Track B checkpoint.")
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to Track B checkpoint (.pt) produced by trackB_train_loader.py",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    obj = torch.load(str(ckpt_path), map_location="cpu")
    cfg: Dict[str, Any] = obj.get("train_config") or {}

    print(f"Checkpoint: {ckpt_path}")
    if not cfg:
        print("No 'train_config' found in this checkpoint.")
        return

    print("\nStored TrainConfig fields:")
    # Print keys in sorted order for readability
    for k in sorted(cfg.keys()):
        print(f"  {k}: {cfg[k]}")


if __name__ == "__main__":
    main()

