#!/usr/bin/env python3
"""
Ego4D-LiteSTA Configuration Loader

Unified configuration management with support for:
- YAML config files with inheritance (_base_)
- Variable interpolation (${paths.local_extraction})
- CLI argument overrides
- Environment variable overrides
- Runtime environment detection (Colab vs local)

Usage:
    from core.config_loader import load_config, Config

    # Load Track A config
    cfg = load_config("trackA")

    # Load with overrides
    cfg = load_config("trackA", overrides={"stage_a.k": 10})

    # Load for Colab
    cfg = load_config("trackA", colab=True)
"""
from __future__ import annotations

import os
import re
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import argparse


# =============================================================================
# YAML Loading (with fallback if PyYAML not installed)
# =============================================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file with fallback to basic parsing."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Basic YAML parser for simple configs
        return _basic_yaml_parse(path)


def _basic_yaml_parse(path: Path) -> Dict[str, Any]:
    """Minimal YAML parser for configs without PyYAML dependency."""
    result: Dict[str, Any] = {}
    stack: List[tuple] = [(result, -1)]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Skip comments and empty lines
            stripped = line.split("#")[0].rstrip()
            if not stripped:
                continue

            # Calculate indentation
            indent = len(line) - len(line.lstrip())

            # Pop stack to correct level
            while stack and stack[-1][1] >= indent:
                stack.pop()

            current = stack[-1][0] if stack else result

            # Parse key: value
            if ":" in stripped:
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip()

                if value:
                    # Parse value types
                    if value.lower() == "true":
                        current[key] = True
                    elif value.lower() == "false":
                        current[key] = False
                    elif value.lower() == "null" or value == "~":
                        current[key] = None
                    elif value.startswith("[") and value.endswith("]"):
                        # Simple list parsing
                        items = value[1:-1].split(",")
                        current[key] = [_parse_value(i.strip()) for i in items if i.strip()]
                    elif value.startswith('"') and value.endswith('"'):
                        current[key] = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        current[key] = value[1:-1]
                    else:
                        current[key] = _parse_value(value)
                else:
                    # Nested dict
                    current[key] = {}
                    stack.append((current[key], indent))

    return result


def _parse_value(value: str) -> Any:
    """Parse a string value to appropriate Python type."""
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.lower() == "null" or value == "~":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# =============================================================================
# Config Merging and Variable Interpolation
# =============================================================================

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _interpolate_vars(config: Dict, root: Optional[Dict] = None) -> Dict:
    """Interpolate ${var.path} references in config values."""
    if root is None:
        root = config

    def resolve(value: Any) -> Any:
        if isinstance(value, str):
            # Find all ${...} patterns
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            for match in matches:
                # Resolve variable path
                parts = match.split(".")
                resolved = root
                try:
                    for part in parts:
                        resolved = resolved[part]
                    if isinstance(resolved, str):
                        value = value.replace(f"${{{match}}}", resolved)
                except (KeyError, TypeError):
                    pass  # Leave unresolved
            return value
        elif isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return resolve(config)


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict to dot-notation keys."""
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict:
    """Unflatten dot-notation keys to nested dict."""
    result: Dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


# =============================================================================
# Environment Detection
# =============================================================================

def is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_device() -> str:
    """Get best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# =============================================================================
# Config Dataclass
# =============================================================================

@dataclass
class Config:
    """Configuration container with attribute access."""
    _data: Dict[str, Any] = field(default_factory=dict)
    _flat: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._flat = _flatten_dict(self._data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return Config(_data=value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Support dot-notation access: cfg['paths.frames_root']"""
        if "." in key:
            parts = key.split(".")
            value = self._data
            for part in parts:
                value = value[part]
            return value
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get with default, supporting dot-notation."""
        try:
            return self[key]
        except (KeyError, TypeError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Return raw dictionary."""
        return copy.deepcopy(self._data)

    def flat(self) -> Dict[str, Any]:
        """Return flattened dictionary with dot-notation keys."""
        return copy.deepcopy(self._flat)


# =============================================================================
# Main Loader
# =============================================================================

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(
    name: str,
    configs_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    colab: Optional[bool] = None,
    cli_args: Optional[List[str]] = None,
) -> Config:
    """
    Load configuration by name with inheritance and overrides.

    Args:
        name: Config name ('trackA', 'trackB', 'trackC', 'extraction', 'base')
        configs_dir: Directory containing YAML configs (default: local_extraction/configs)
        overrides: Dict of overrides (supports dot-notation keys)
        colab: Force Colab mode (None = auto-detect)
        cli_args: CLI arguments to parse for overrides

    Returns:
        Config object with merged settings
    """
    if configs_dir is None:
        configs_dir = _CONFIGS_DIR

    # Load main config
    config_path = configs_dir / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _load_yaml(config_path)

    # Handle inheritance
    if "_base_" in config:
        base_name = config.pop("_base_").replace(".yaml", "")
        base_config = load_config(base_name, configs_dir=configs_dir, colab=colab).to_dict()
        config = _deep_merge(base_config, config)

    # Apply Colab overrides
    if colab is None:
        colab = is_colab()

    if colab and "paths" in config:
        # Swap local paths to Colab paths
        if "colab_drive_root" in config["paths"]:
            config["paths"]["repo_root"] = "/content/Ego4d-LiteSTA"
            config["paths"]["local_extraction"] = "/content/Ego4d-LiteSTA/local_extraction"

    # Apply environment variable overrides (EGO4D_LiteSTA_*)
    for key, value in os.environ.items():
        if key.startswith("EGO4D_LITESTA_"):
            config_key = key[14:].lower().replace("__", ".")
            override_dict = _unflatten_dict({config_key: _parse_value(value)})
            config = _deep_merge(config, override_dict)

    # Apply explicit overrides
    if overrides:
        override_dict = _unflatten_dict(overrides) if any("." in k for k in overrides.keys()) else overrides
        config = _deep_merge(config, override_dict)

    # Apply CLI overrides
    if cli_args:
        cli_overrides = _parse_cli_overrides(cli_args)
        config = _deep_merge(config, cli_overrides)

    # Interpolate variables
    config = _interpolate_vars(config)

    # Set runtime device
    if config.get("runtime", {}).get("device") == "auto":
        config.setdefault("runtime", {})["device"] = get_device()

    return Config(_data=config)


def _parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """Parse --key=value CLI arguments into dict."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)

    # Parse known args, treat rest as overrides
    known, unknown = parser.parse_known_args(args)

    overrides: Dict[str, Any] = {}
    for arg in unknown:
        if arg.startswith("--") and "=" in arg:
            key, _, value = arg[2:].partition("=")
            overrides[key] = _parse_value(value)

    return _unflatten_dict(overrides)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI tool to print/validate configs."""
    import sys
    import json

    parser = argparse.ArgumentParser(description="Ego4D-LiteSTA Config Loader")
    parser.add_argument("config", type=str, help="Config name (trackA, trackB, trackC, extraction)")
    parser.add_argument("--colab", action="store_true", help="Force Colab mode")
    parser.add_argument("--flat", action="store_true", help="Print flattened config")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args, unknown = parser.parse_known_args()

    try:
        cfg = load_config(args.config, colab=args.colab, cli_args=unknown)
        data = cfg.flat() if args.flat else cfg.to_dict()

        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            _print_config(data)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _print_config(data: Dict, indent: int = 0):
    """Pretty-print config dict."""
    for key, value in data.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


if __name__ == "__main__":
    main()
