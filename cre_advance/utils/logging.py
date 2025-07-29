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


def get_logger(name: str) -> logging.Logger:
    """Return a ``logging.Logger`` configured for console and file output.

    The log level is determined by the ``LOG_LEVEL`` environment variable and
    defaults to ``INFO``.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
