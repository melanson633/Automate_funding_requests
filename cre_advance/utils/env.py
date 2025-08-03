from __future__ import annotations

"""Configuration loading utilities.

This module provides :func:`get_config` which merges the default
configuration with lender-specific overrides. Environment variables
are loaded from a ``.env`` file using :mod:`python-dotenv`.

Example
-------
>>> from cre_advance.utils.env import get_config
>>> cfg = get_config("example_lender")
>>> cfg["logging"]["level"]
'INFO'
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .errors import ConfigError

_REQUIRED_ENV_VARS = ["GOOGLE_API_KEY"]
_REQUIRED_KEYS = [
    "logging",
    "mapping_coverage_threshold",
    "unmatched_threshold",
    "min_confidence",
    "gemini_model",
    "gemini_temperature",
    "gemini_max_retries",
]

_MODEL_TIERS = {"flash", "pro"}


def _merge_dicts(base: dict, overrides: dict) -> dict:
    """Recursively merge two dictionaries."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def get_config(lender: str) -> dict:
    """Return merged configuration for ``lender``.

    Parameters
    ----------
    lender:
        Name of the lender whose config file exists under
        ``configs/lenders/<lender>.yaml``.

    Raises
    ------
    ConfigError
        If configuration files are missing or required keys are absent.
    """
    project_root = Path(__file__).resolve().parents[2]

    # Load .env variables
    load_dotenv(project_root / ".env")

    for var in _REQUIRED_ENV_VARS:
        if os.getenv(var) is None:
            raise ConfigError(f"Environment variable '{var}' is required")

    defaults_path = project_root / "configs" / "defaults.yaml"
    try:
        with defaults_path.open("r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Default config not found: {defaults_path}") from exc

    lender_path = project_root / "configs" / "lenders" / f"{lender}.yaml"
    if lender_path.is_file():
        with lender_path.open("r") as f:
            lender_cfg = yaml.safe_load(f) or {}
        config = _merge_dicts(config, lender_cfg)
    else:
        raise ConfigError(f"Lender config not found: {lender_path}")

    model_tier = config.get("model_tier")
    if model_tier is not None:
        if model_tier not in _MODEL_TIERS:
            raise ConfigError("'model_tier' must be one of {'flash', 'pro'}")
        config["gemini_model"] = f"gemini-2.5-{model_tier}"
    elif "gemini_model" not in config:
        raise ConfigError("Either 'model_tier' or 'gemini_model' must be provided")

    for key in _REQUIRED_KEYS:
        if key not in config:
            raise ConfigError(f"Missing required config key: {key}")

    return config
