"""Logging helpers for the CRE Advance package.

Example
-------
>>> from cre_advance.utils.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Ready")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict


def _level_from_cfg(cfg: Dict[str, Any] | None = None) -> int:
    """Return logging level from ``cfg`` or environment."""

    level_str = os.getenv("LOG_LEVEL")
    if not level_str and cfg:
        level_str = str(cfg.get("logging", {}).get("level", "INFO"))
    level_str = (level_str or "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def get_logger(
    name: str, cfg: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None
) -> logging.LoggerAdapter:
    """Return a configured ``LoggerAdapter`` with optional context."""

    logger = logging.getLogger(name)

    level = _level_from_cfg(cfg)
    if logger.handlers:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
    else:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(context)s - %(message)s"
        )

        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        log_file = None
        if cfg:
            log_file = cfg.get("logging", {}).get("file")
        if not log_file:
            log_file = Path("logs") / f"{name}.log"
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    adapter = logging.LoggerAdapter(logger, extra={"context": context or ""})
    return adapter
