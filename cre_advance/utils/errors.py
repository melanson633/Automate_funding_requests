"""Custom error classes used throughout the project."""

from __future__ import annotations


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


class PDFSegmentationError(Exception):
    """Raised when Gemini is unable to segment a PDF into invoices."""


class NormalizationError(Exception):
    """Raised when Excel normalization encounters invalid data."""
