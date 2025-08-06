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
        msg = f"Default config not found: {defaults_path}"
        raise ConfigError(msg) from exc

    lender_path = project_root / "configs" / "lenders" / f"{lender}.yaml"
    if lender_path.is_file():
        with lender_path.open("r") as f:
            lender_cfg = yaml.safe_load(f) or {}
        lender_reports = lender_cfg.pop("report_types", {})
        config = _merge_dicts(config, lender_cfg)
        config["report_types"] = _merge_dicts(
            config.get("report_types", {}), lender_reports
        )
    else:
        raise ConfigError(f"Lender config not found: {lender_path}")

    # Environment variables take precedence over YAML values
    env_model_tier = os.getenv("MODEL_TIER")
    env_gemini_model = os.getenv("GEMINI_MODEL")
    if env_model_tier:
        config["model_tier"] = env_model_tier.lower()
    if env_gemini_model:
        config["gemini_model"] = env_gemini_model

    model_tier = config.get("model_tier")
    gemini_model = config.get("gemini_model")

    if model_tier:
        if model_tier not in _MODEL_TIERS:
            raise ConfigError("'model_tier' must be one of {'flash', 'pro'}")
        gemini_model = f"gemini-2.5-{model_tier}"
    elif not gemini_model:
        gemini_model = "gemini-2.5-pro"

    config["gemini_model"] = gemini_model

    for key in _REQUIRED_KEYS:
        if key not in config:
            raise ConfigError(f"Missing required config key: {key}")

    config["unmatched_threshold"] = float(config.get("unmatched_threshold", 0.4))
    config["min_confidence"] = float(config.get("min_confidence", 0.0))

    # Normalise packager configuration
    pkg_cfg = config.setdefault("packager", {})
    pkg_cfg["vendor_ratio_threshold"] = float(
        pkg_cfg.get("vendor_ratio_threshold", 0.8)
    )
    pkg_cfg["amount_tolerance"] = float(pkg_cfg.get("amount_tolerance", 0.01))
    pkg_cfg["score_threshold"] = float(pkg_cfg.get("score_threshold", 2.0))

    # Normalise PDF configuration
    pdf_cfg = config.setdefault("pdf", {})
    pdf_cfg.setdefault("use_vision", False)
    pdf_cfg.setdefault("vision_model", "gemini-2.5-pro")
    pdf_cfg["max_pages_per_request"] = int(
        pdf_cfg.get("max_pages_per_request", 3000)
    )
    pdf_cfg["classification_confidence_threshold"] = float(
        pdf_cfg.get("classification_confidence_threshold", 0.5)
    )
    pdf_cfg["min_confidence"] = float(
        pdf_cfg.get("min_confidence", config["min_confidence"])
    )
    pdf_cfg["unmatched_threshold"] = float(
        pdf_cfg.get("unmatched_threshold", config["unmatched_threshold"])
    )

    # Normalise OCR configuration
    ocr_cfg = config.setdefault("ocr", {})
    langs = ocr_cfg.get("langs", [])
    if isinstance(langs, str):
        ocr_cfg["langs"] = [lang.strip() for lang in langs.split(",") if lang.strip()]
    ocr_cfg["psm"] = int(ocr_cfg.get("psm", 6))
    ocr_cfg["oem"] = int(ocr_cfg.get("oem", 1))
    ocr_cfg["deskew"] = bool(ocr_cfg.get("deskew", False))

    # Normalise prompts configuration
    prompts_cfg = config.setdefault("prompts", {})
    for key, value in list(prompts_cfg.items()):
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        prompts_cfg[key] = str(path)

    # Normalise scoring configuration
    scoring_cfg = config.setdefault("scoring", {})
    for key, value in list(scoring_cfg.items()):
        scoring_cfg[key] = float(value)

    return config
