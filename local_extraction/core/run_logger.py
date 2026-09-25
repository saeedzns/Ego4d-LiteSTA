#!/usr/bin/env python3
"""
Run Logger — Unified logging for all track runs

Creates a structured log file for each run containing:
- Timestamp and run ID
- Full configuration snapshot (from YAML + overrides)
- Git info (commit, branch, dirty status)
- System info (Python version, device, CUDA)
- Summary results (metrics, outputs)
- Timing information

Usage:
    from core.run_logger import RunLogger

    logger = RunLogger(track='trackA', stage='stageA')
    logger.log_config(config_dict)
    logger.log_start()

    # ... run code ...

    logger.log_metrics({'mAP': 0.45, 'accuracy': 0.82})
    logger.log_artifacts(['output.json', 'model.pt'])
    logger.log_end()
    logger.save()
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib


def _get_git_info() -> Dict[str, Any]:
    """Get current git commit, branch, and dirty status."""
    info = {
        'commit': None,
        'branch': None,
        'dirty': None,
        'remote': None,
    }

    # Skip git commands in Colab (often fails or times out)
    try:
        import google.colab  # noqa: F401
        info['note'] = 'git info skipped in Colab'
        return info
    except ImportError:
        pass

    try:
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=os.getcwd()
        )
        if result.returncode == 0:
            info['commit'] = result.stdout.strip()[:12]

        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=os.getcwd()
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5,
            cwd=os.getcwd()
        )
        if result.returncode == 0:
            info['dirty'] = len(result.stdout.strip()) > 0

        # Get remote URL
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True, text=True, timeout=5,
            cwd=os.getcwd()
        )
        if result.returncode == 0:
            info['remote'] = result.stdout.strip()
    except Exception:
        pass
    return info


def _get_system_info() -> Dict[str, Any]:
    """Get system and environment information."""
    # Detect Colab
    is_colab = False
    try:
        import google.colab  # noqa: F401
        is_colab = True
    except ImportError:
        pass

    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'hostname': platform.node() if not is_colab else 'colab-vm',
        'cwd': os.getcwd(),
        'is_colab': is_colab,
    }

    # CUDA info
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        info['torch_version'] = None
        info['cuda_available'] = False

    return info


def _serialize_value(v: Any) -> Any:
    """Convert value to JSON-serializable format."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    if hasattr(v, '__dict__'):
        return {k: _serialize_value(val) for k, val in v.__dict__.items() if not k.startswith('_')}
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


@dataclass
class RunLog:
    """Structured log for a single run."""

    # Identity
    run_id: str = ""
    track: str = ""
    stage: Optional[str] = None
    timestamp: str = ""

    # Environment
    git: Dict[str, Any] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    cli_args: List[str] = field(default_factory=list)

    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Results
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Notes
    notes: str = ""


class RunLogger:
    """Logger for tracking run configuration and results."""

    def __init__(
        self,
        track: str,
        stage: Optional[str] = None,
        run_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize run logger.

        Args:
            track: Track name ('trackA', 'trackB', 'trackC')
            stage: Optional stage ('stageA', 'stageB') for Track A
            run_dir: Directory for this run (auto-created if None)
            run_id: Custom run ID (auto-generated if None)
        """
        self.track = track
        self.stage = stage
        self.timestamp = datetime.now()

        # Generate run ID
        if run_id:
            self.run_id = run_id
        else:
            ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
            stage_suffix = f"_{stage}" if stage else ""
            self.run_id = f"{track}{stage_suffix}_{ts}"

        # Determine run directory
        if run_dir:
            self.run_dir = Path(run_dir)
        else:
            # Default: local_extraction/runs/Track_X/run_id/
            track_folder = track.replace('track', 'Track_')
            base = Path("local_extraction") / "runs" / track_folder
            self.run_dir = base / self.run_id

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "run_log.json"

        # Initialize log structure
        self.log = RunLog(
            run_id=self.run_id,
            track=track,
            stage=stage,
            timestamp=self.timestamp.isoformat(),
            git=_get_git_info(),
            system=_get_system_info(),
            cli_args=sys.argv[1:] if len(sys.argv) > 1 else [],
        )

        self._start_dt: Optional[datetime] = None

    def log_config(
        self,
        config: Union[Dict[str, Any], Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "RunLogger":
        """Log the configuration used for this run."""
        self.log.config = _serialize_value(config)
        if overrides:
            self.log.config_overrides = _serialize_value(overrides)
        return self

    def log_start(self) -> "RunLogger":
        """Mark the start of the run."""
        self._start_dt = datetime.now()
        self.log.start_time = self._start_dt.isoformat()
        self.log.status = "running"
        self._autosave()
        return self

    def log_end(self, success: bool = True, error: Optional[str] = None) -> "RunLogger":
        """Mark the end of the run."""
        end_dt = datetime.now()
        self.log.end_time = end_dt.isoformat()

        if self._start_dt:
            self.log.duration_seconds = (end_dt - self._start_dt).total_seconds()

        if success:
            self.log.status = "completed"
        else:
            self.log.status = "failed"
            self.log.error = error

        self._autosave()
        return self

    def log_metrics(self, metrics: Dict[str, Any]) -> "RunLogger":
        """Log evaluation metrics."""
        self.log.metrics.update(_serialize_value(metrics))
        return self

    def log_metric(self, name: str, value: Any) -> "RunLogger":
        """Log a single metric."""
        self.log.metrics[name] = _serialize_value(value)
        return self

    def log_artifacts(self, paths: List[Union[str, Path]]) -> "RunLogger":
        """Log output artifact paths."""
        for p in paths:
            path_str = str(p)
            if path_str not in self.log.artifacts:
                self.log.artifacts.append(path_str)
        return self

    def log_output(self, key: str, value: Any) -> "RunLogger":
        """Log an output value (e.g., paths, counts, summaries)."""
        self.log.outputs[key] = _serialize_value(value)
        return self

    def log_warning(self, message: str) -> "RunLogger":
        """Log a warning."""
        self.log.warnings.append(message)
        return self

    def log_note(self, note: str) -> "RunLogger":
        """Add a note to the run."""
        if self.log.notes:
            self.log.notes += "\n" + note
        else:
            self.log.notes = note
        return self

    def _autosave(self) -> None:
        """Auto-save log after status changes."""
        try:
            self.save()
        except Exception:
            pass

    def save(self) -> Path:
        """Save the log to JSON file."""
        data = asdict(self.log)
        with self.log_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return self.log_path

    def to_dict(self) -> Dict[str, Any]:
        """Return log as dictionary."""
        return asdict(self.log)

    def print_summary(self) -> None:
        """Print a summary of the run."""
        print(f"\n{'='*60}")
        print(f"Run: {self.run_id}")
        print(f"{'='*60}")
        print(f"Track: {self.track}" + (f" / {self.stage}" if self.stage else ""))
        print(f"Status: {self.log.status}")
        print(f"Log: {self.log_path}")

        if self.log.duration_seconds:
            mins = int(self.log.duration_seconds // 60)
            secs = int(self.log.duration_seconds % 60)
            print(f"Duration: {mins}m {secs}s")

        if self.log.metrics:
            print(f"\nMetrics:")
            for k, v in self.log.metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        if self.log.artifacts:
            print(f"\nArtifacts: {len(self.log.artifacts)} files")

        if self.log.warnings:
            print(f"\nWarnings: {len(self.log.warnings)}")
            for w in self.log.warnings[:3]:
                print(f"  - {w}")

        print(f"{'='*60}\n")

    @classmethod
    def load(cls, log_path: Path) -> "RunLogger":
        """Load a run logger from an existing log file."""
        with log_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        logger = cls(
            track=data.get('track', ''),
            stage=data.get('stage'),
            run_dir=log_path.parent,
            run_id=data.get('run_id', ''),
        )

        # Restore log data
        for key, value in data.items():
            if hasattr(logger.log, key):
                setattr(logger.log, key, value)

        return logger


def create_run_logger(
    track: str,
    stage: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None,
) -> RunLogger:
    """
    Convenience function to create and initialize a run logger.

    Args:
        track: Track name
        stage: Optional stage
        config: Configuration dict to log
        run_dir: Override run directory (useful in notebooks)

    Returns:
        Initialized RunLogger
    """
    logger = RunLogger(track=track, stage=stage, run_dir=run_dir)
    if config:
        logger.log_config(config)
    return logger


# =============================================================================
# Integration helpers for existing track scripts
# =============================================================================

def log_trackA_run(
    stage: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    artifacts: List[str],
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Log a Track A run (stageA or stageB).

    Returns:
        Path to the log file
    """
    logger = RunLogger(track='trackA', stage=stage, run_dir=run_dir)
    logger.log_config(config)
    logger.log_start()
    logger.log_metrics(results)
    logger.log_artifacts(artifacts)
    logger.log_end(success=True)
    logger.print_summary()
    return logger.save()


def log_trackB_run(
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: List[str],
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Log a Track B training/eval run.

    Returns:
        Path to the log file
    """
    logger = RunLogger(track='trackB', run_dir=run_dir)
    logger.log_config(config)
    logger.log_start()
    logger.log_metrics(metrics)
    logger.log_artifacts(artifacts)
    logger.log_end(success=True)
    logger.print_summary()
    return logger.save()


def log_trackC_run(
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: List[str],
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Log a Track C pruning run.

    Returns:
        Path to the log file
    """
    logger = RunLogger(track='trackC', run_dir=run_dir)
    logger.log_config(config)
    logger.log_start()
    logger.log_metrics(metrics)
    logger.log_artifacts(artifacts)
    logger.log_end(success=True)
    logger.print_summary()
    return logger.save()


# =============================================================================
# Entry point for testing
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    print("RunLogger Demo")
    print("-" * 40)

    # Create a test logger
    logger = RunLogger(track='trackB', stage=None)

    # Log configuration
    logger.log_config({
        'training': {
            'epochs': 20,
            'batch_size': 8,
            'lr': 1e-3,
        },
        'model': {
            'fusion_layers': 2,
            'token_dim': 256,
        }
    })

    # Log start
    logger.log_start()

    # Simulate some metrics
    import time
    time.sleep(0.5)

    logger.log_metrics({
        'mAP': 0.4523,
        'accuracy': 0.8234,
        'ttc_mae': 0.312,
    })

    logger.log_artifacts([
        'checkpoints/best.pt',
        'predictions/val.json',
    ])

    logger.log_output('total_samples', 1234)
    logger.log_warning('Low GPU memory detected')

    # Log end
    logger.log_end(success=True)

    # Print summary
    logger.print_summary()

    print(f"Log saved to: {logger.log_path}")
