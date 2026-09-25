"""
Ego4D-LiteSTA Core Module

Shared utilities for configuration and path management.
"""
from .config_loader import load_config, Config, is_colab, get_device
from .paths import Paths, get_paths, normalize_path
from .config_adapter import (
    apply_config_to_module,
    get_track_config,
    create_train_config,
    create_rgtp_config,
    create_eval_config,
    create_fusion_config,
    create_head_config,
    create_tokenizer_config,
    create_trackb_eval_config,
)
from .run_logger import (
    RunLogger,
    RunLog,
    create_run_logger,
    log_trackA_run,
    log_trackB_run,
    log_trackC_run,
)

__all__ = [
    # Config loading
    "load_config",
    "Config",
    "is_colab",
    "get_device",
    # Paths
    "Paths",
    "get_paths",
    "normalize_path",
    # Config adapter - generic
    "apply_config_to_module",
    "get_track_config",
    # Config adapter - Track B
    "create_train_config",
    "create_fusion_config",
    "create_head_config",
    "create_tokenizer_config",
    "create_trackb_eval_config",
    # Config adapter - Track C
    "create_rgtp_config",
    "create_eval_config",
    # Run logging
    "RunLogger",
    "RunLog",
    "create_run_logger",
    "log_trackA_run",
    "log_trackB_run",
    "log_trackC_run",
]
