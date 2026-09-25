#!/usr/bin/env python3
"""
Ego4D-LiteSTA Path Utilities

Centralized path resolution for local and Colab environments.
Handles path normalization, existence checks, and automatic discovery.

Usage:
    from core.paths import Paths

    paths = Paths.from_config(cfg)
    frames_root = paths.frames_root

    # Or standalone
    paths = Paths(repo_root="/content/Ego4d-LiteSTA")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


# =============================================================================
# Path Dataclass
# =============================================================================

@dataclass
class Paths:
    """Centralized path management for Ego4D-LiteSTA."""

    repo_root: Path = field(default_factory=lambda: Path(".").resolve())
    version: str = "v2"

    # Computed paths (set in __post_init__)
    local_extraction: Path = field(init=False)
    frames_root: Path = field(init=False)
    labels_root: Path = field(init=False)
    manifests_root: Path = field(init=False)
    org_annotations: Path = field(init=False)
    runs_root: Path = field(init=False)
    configs_root: Path = field(init=False)

    # Track-specific run directories
    track_a_runs: Path = field(init=False)
    track_b_runs: Path = field(init=False)
    track_c_runs: Path = field(init=False)

    def __post_init__(self):
        self.repo_root = Path(self.repo_root).resolve()
        self.local_extraction = self.repo_root / "local_extraction"

        v = self.version
        self.frames_root = self.local_extraction / v / "extracted_frames"
        self.labels_root = self.local_extraction / v / "yolo_labels_540"
        self.manifests_root = self.local_extraction / v / "manifests"
        self.org_annotations = self.local_extraction / v / "org_annotations"
        self.runs_root = self.local_extraction / "runs"
        self.configs_root = self.local_extraction / "configs"

        self.track_a_runs = self.runs_root / "Track_A"
        self.track_b_runs = self.runs_root / "Track_B"
        self.track_c_runs = self.runs_root / "Track_C"

    # Property aliases for notebook compatibility
    @property
    def extracted_frames(self) -> Path:
        """Alias for frames_root."""
        return self.frames_root

    @property
    def yolo_labels(self) -> Path:
        """Alias for labels_root."""
        return self.labels_root

    @property
    def manifests(self) -> Path:
        """Alias for manifests_root."""
        return self.manifests_root

    @classmethod
    def from_config(cls, config) -> "Paths":
        """Create Paths from a Config object."""
        repo_root = config.get("paths.repo_root", ".")
        version = config.get("version", "v2")
        return cls(repo_root=repo_root, version=version)

    @classmethod
    def auto_detect(cls) -> "Paths":
        """Auto-detect repo root by searching for marker files."""
        markers = ["local_extraction", "Docs", "scripts", "notebooks"]

        # Start from current working directory
        current = Path.cwd()

        # Walk up until we find markers or hit root
        for _ in range(10):
            if all((current / m).exists() for m in markers[:2]):
                return cls(repo_root=current)
            parent = current.parent
            if parent == current:
                break
            current = parent

        # Fallback: assume cwd is repo root
        return cls(repo_root=Path.cwd())

    def ensure_dirs(self) -> None:
        """Create essential directories if they don't exist."""
        for path in [
            self.frames_root,
            self.labels_root,
            self.manifests_root,
            self.runs_root,
            self.track_a_runs,
            self.track_b_runs,
            self.track_c_runs,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def run_dir(self, track: str, prefix: str, create: bool = True) -> Path:
        """
        Create a timestamped run directory.

        Args:
            track: 'A', 'B', or 'C'
            prefix: Run prefix (e.g., 'trackA_stageA')
            create: Whether to create the directory

        Returns:
            Path to run directory
        """
        track_dir = {
            "A": self.track_a_runs,
            "B": self.track_b_runs,
            "C": self.track_c_runs,
        }.get(track.upper(), self.runs_root)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = track_dir / f"{prefix}_{stamp}"

        if create:
            run_path.mkdir(parents=True, exist_ok=True)

        return run_path

    def latest_run(self, track: str, prefix: Optional[str] = None) -> Optional[Path]:
        """
        Find the latest run directory for a track.

        Args:
            track: 'A', 'B', 'C', 'Track_A', 'Track_B', 'Track_C', 'trackA', etc.
            prefix: Optional prefix filter (e.g., 'trackA_stageA')

        Returns:
            Path to latest run directory, or None if not found
        """
        # Normalize track name
        track_upper = track.upper().replace('TRACK_', '').replace('TRACK', '')
        if track_upper:
            track_upper = track_upper[0]  # Get first letter: A, B, or C

        track_dir = {
            "A": self.track_a_runs,
            "B": self.track_b_runs,
            "C": self.track_c_runs,
        }.get(track_upper)

        if not track_dir or not track_dir.exists():
            return None

        pattern = f"{prefix}_*" if prefix else "*"
        candidates = sorted(track_dir.glob(pattern), reverse=True)

        for c in candidates:
            if c.is_dir():
                return c

        return None

    def find_checkpoint(self, pattern: str = "trackB_best_*.pt") -> Optional[Path]:
        """Find latest matching checkpoint in Track B."""
        ckpt_dir = self.track_b_runs / "checkpoints"
        if not ckpt_dir.exists():
            return None

        candidates = sorted(ckpt_dir.glob(pattern), reverse=True)
        return candidates[0] if candidates else None

    def find_manifest(self, name: str) -> Optional[Path]:
        """Find manifest file by name (with common extension fallbacks)."""
        for ext in ["", ".json", ".jsonl"]:
            path = self.manifests_root / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def to_dict(self) -> Dict[str, str]:
        """Export all paths as string dictionary."""
        return {
            "repo_root": str(self.repo_root),
            "local_extraction": str(self.local_extraction),
            "frames_root": str(self.frames_root),
            "labels_root": str(self.labels_root),
            "manifests_root": str(self.manifests_root),
            "org_annotations": str(self.org_annotations),
            "runs_root": str(self.runs_root),
            "configs_root": str(self.configs_root),
            "track_a_runs": str(self.track_a_runs),
            "track_b_runs": str(self.track_b_runs),
            "track_c_runs": str(self.track_c_runs),
        }

    def validate(self) -> List[str]:
        """Check which essential paths exist. Returns list of missing paths."""
        essential = [
            ("repo_root", self.repo_root),
            ("local_extraction", self.local_extraction),
        ]
        missing = [name for name, path in essential if not path.exists()]
        return missing

    def __repr__(self) -> str:
        return f"Paths(repo_root={self.repo_root}, version={self.version})"


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """
    Normalize a path, optionally relative to a base.

    Handles:
    - Windows/Unix path separators
    - Relative vs absolute paths
    - Environment variables
    """
    path_str = str(path)

    # Expand environment variables
    path_str = os.path.expandvars(path_str)

    # Convert to Path
    p = Path(path_str)

    # Make absolute if relative and base provided
    if not p.is_absolute() and base:
        p = base / p

    return p.resolve()


def is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def colab_paths() -> Paths:
    """Get Paths configured for Colab environment."""
    return Paths(
        repo_root=Path("/content/Ego4d-LiteSTA"),
        version="v2"
    )


def local_paths() -> Paths:
    """Get Paths for local environment via auto-detection."""
    return Paths.auto_detect()


def get_paths() -> Paths:
    """Get appropriate Paths based on runtime environment."""
    if is_colab():
        return colab_paths()
    return local_paths()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI tool to print paths."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ego4D-LiteSTA Path Utilities")
    parser.add_argument("--colab", action="store_true", help="Use Colab paths")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--validate", action="store_true", help="Check for missing paths")
    parser.add_argument("--ensure", action="store_true", help="Create missing directories")

    args = parser.parse_args()

    paths = colab_paths() if args.colab else local_paths()

    if args.ensure:
        paths.ensure_dirs()
        print("Created essential directories.")

    if args.validate:
        missing = paths.validate()
        if missing:
            print(f"Missing paths: {', '.join(missing)}")
        else:
            print("All essential paths exist.")
        return

    if args.json:
        print(json.dumps(paths.to_dict(), indent=2))
    else:
        for name, path in paths.to_dict().items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"{exists} {name}: {path}")


if __name__ == "__main__":
    main()
