#!/usr/bin/env python3
"""Scan YAML config files for '# Options:' blocks with too few examples.

This repo uses a convention like:

  some_key: 123
  # Options:
  # - 1
  # - 2

This script reports blocks that have exactly 1 bullet (and optionally 0 bullets).

Usage (from repo root):
  python local_extraction/utils/scan_yaml_options_blocks.py

Output format:
  <relpath>\t<line>\t<bullets>\t<inferred_yaml_path>
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


def _iter_target_yaml_files(repo_root: Path) -> List[Path]:
    configs_dir = repo_root / "local_extraction" / "configs"
    candidates = sorted(configs_dir.glob("*.yaml"))
    return [
        p
        for p in candidates
        if p.name == "base.yaml" or p.name.startswith(("trackA", "trackB", "trackC"))
    ]


_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+)\s*:\s*(?P<rest>.*)$")
_OPT_RE = re.compile(r"^\s*#\s*Options\s*:\s*$")
_BULLET_RE = re.compile(r"^\s*#\s*-\s+.+$")


def _iter_options_blocks(fp: Path) -> Iterator[Dict[str, object]]:
    lines = fp.read_text(encoding="utf-8").splitlines()

    stack: List[Tuple[int, str]] = []
    last_key_stack: List[Tuple[int, str]] | None = None

    def update_stack(indent_len: int, key: str) -> None:
        nonlocal stack
        while stack and stack[-1][0] >= indent_len:
            stack.pop()
        stack.append((indent_len, key))

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip() and not line.lstrip().startswith("#"):
            m = _KEY_RE.match(line)
            if m:
                indent_len = len(m.group("indent").replace("\t", "    "))
                update_stack(indent_len, m.group("key"))
                last_key_stack = list(stack)

        if _OPT_RE.match(line):
            j = i + 1
            bullets = 0
            while j < len(lines):
                nxt = lines[j]
                if _BULLET_RE.match(nxt):
                    bullets += 1
                    j += 1
                    continue
                if nxt.strip() == "":
                    j += 1
                    continue
                # Another comment (NOTE:, etc.) ends the options list
                if nxt.lstrip().startswith("#"):
                    break
                # A non-comment line ends the options list
                break

            key_path = "<unknown>"
            if last_key_stack:
                key_path = ".".join(k for _, k in last_key_stack)

            yield {
                "line": i + 1,
                "bullets": bullets,
                "key_path": key_path,
            }

            i = j
            continue

        i += 1


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    files = _iter_target_yaml_files(repo_root)

    blocks: List[Tuple[Path, Dict[str, object]]] = []
    for fp in files:
        for b in _iter_options_blocks(fp):
            blocks.append((fp, b))

    single = [(fp, b) for fp, b in blocks if int(b["bullets"]) == 1]
    zero = [(fp, b) for fp, b in blocks if int(b["bullets"]) == 0]

    print(f"FILES\t{len(files)}")
    print(f"OPTIONS_BLOCKS\t{len(blocks)}")
    print(f"SINGLE_OPTION_BLOCKS\t{len(single)}")
    print(f"ZERO_OPTION_BLOCKS\t{len(zero)}")
    print("---")

    for fp, b in single:
        rel = fp.relative_to(repo_root).as_posix()
        print(f"{rel}\t{b['line']}\t{b['bullets']}\t{b['key_path']}")

    if zero:
        print("---ZERO---")
        for fp, b in zero:
            rel = fp.relative_to(repo_root).as_posix()
            print(f"{rel}\t{b['line']}\t{b['bullets']}\t{b['key_path']}")


if __name__ == "__main__":
    main()
