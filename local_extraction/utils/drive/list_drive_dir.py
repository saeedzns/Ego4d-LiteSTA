#!/usr/bin/env python3
"""
List a Google Drive directory and (optionally) create a local link in the workspace.

Examples (PowerShell):
  # List a specific Windows Drive path
  python local_extraction/list_drive_dir.py --path "H:/My Drive/ego4d_data"

  # Auto-detect common paths (Windows/WSL/Colab)
  python local_extraction/list_drive_dir.py

  # Also create a workspace link named 'ego4d_data' to that path
  python local_extraction/list_drive_dir.py --link-name ego4d_data

Examples (Colab):
  !python local_extraction/list_drive_dir.py --path \
      /content/drive/MyDrive/ego4d_data --link-name ego4d_data
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Iterable


COMMON_CANDIDATES = (
    # Windows Google Drive for desktop (via WSL path)
    "/mnt/h/My Drive/ego4d_data",
    "/mnt/d/My Drive/ego4d_data",
    # Windows native path (if running Python on Windows)
    "H:/My Drive/ego4d_data",
    "D:/My Drive/ego4d_data",
    # Colab mounts
    "/content/drive/MyDrive/ego4d_data",
    "/drive/MyDrive/ego4d_data",
)


def detect_path() -> Path | None:
    for p in COMMON_CANDIDATES:
        path = Path(p)
        try:
            if path.exists() and path.is_dir():
                return path
        except Exception:
            continue
    return None


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"


def list_dir(path: Path, recursive: bool = False, max_depth: int = 2) -> None:
    def _walk(p: Path, depth: int = 0):
        try:
            entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            print(f"[skip] Permission denied: {p}")
            return
        except FileNotFoundError:
            print(f"[skip] Not found: {p}")
            return
        for e in entries:
            try:
                if e.is_dir():
                    print(f"[D] {e}")
                    if recursive and depth + 1 < max_depth:
                        _walk(e, depth + 1)
                elif e.is_file():
                    size = e.stat().st_size
                    print(f"[F] {e}  ({sizeof_fmt(size)})")
                else:
                    print(f"[?] {e}")
            except Exception as ex:
                print(f"[warn] {e}: {ex}")

    print(f"[list] {path}")
    _walk(path, 0)


def create_workspace_link(target: Path, link_name: str) -> bool:
    # Resolve repo root as parent of local_extraction
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    link = repo_root / link_name
    # Remove existing
    try:
        if link.exists() or link.is_symlink():
            if link.is_dir() and not link.is_symlink() and os.name == "nt":
                # Windows directory junction or real dir
                # Try rmdir (junction) or remove tree
                try:
                    subprocess.run(["cmd", "/c", "rmdir", str(link)], check=False)
                except Exception:
                    pass
            if link.exists() or link.is_symlink():
                if link.is_symlink() or link.is_file():
                    link.unlink(missing_ok=True)
                elif link.is_dir():
                    # As a last resort, do not remove real dirs automatically
                    print(f"[link] Path exists and is a real directory, not removing: {link}")
                    return False
    except Exception as ex:
        print(f"[link] Could not clean existing path {link}: {ex}")
        return False

    # Try symlink first
    try:
        link.symlink_to(target, target_is_directory=True)
        print(f"[link] Created symlink: {link} -> {target}")
        return True
    except Exception as ex:
        # On Windows, non-admin often fails; try junction
        if os.name == "nt":
            try:
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True)
                print(f"[link] Created junction: {link} -> {target}")
                return True
            except Exception as ex2:
                print(f"[link] mklink failed: {ex2}")
        print(f"[link] symlink failed: {ex}")
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="List a Drive directory and optionally link it into the workspace")
    ap.add_argument("--path", "-p", type=str, default=None, help="Drive directory to list (auto-detect if omitted)")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recursively list a couple of levels")
    ap.add_argument("--max-depth", type=int, default=2, help="Max recursion depth when using --recursive")
    ap.add_argument("--link-name", type=str, default=None, help="Create a workspace link (symlink/junction) with this name")
    args = ap.parse_args(argv)

    target = Path(args.path) if args.path else detect_path()
    if not target:
        print("[error] Could not auto-detect a Drive path. Pass --path explicitly.")
        print("        Common examples: 'H:/My Drive/ego4d_data' or '/content/drive/MyDrive/ego4d_data'")
        return 2
    if not target.exists() or not target.is_dir():
        print(f"[error] Not a directory: {target}")
        return 2

    list_dir(target, recursive=bool(args.recursive), max_depth=int(args.max_depth))

    if args.link_name:
        ok = create_workspace_link(target, args.link_name)
        if not ok:
            print("[warn] Link creation failed or skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
