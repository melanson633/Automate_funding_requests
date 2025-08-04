from __future__ import annotations

"""Vision-based PDF segmentation using Gemini 2.5 Vision."""

import time
from pathlib import Path
from typing import List

import fitz
from google.genai import types

from . import ai_gemini
from .pdf_segmenter import _derive_ranges
from .utils.logging import get_logger

logger = get_logger(__name__)


def segment(pdf_path: str | Path, cfg: dict, metrics: dict | None = None) -> List[dict]:
    """Segment a multi-invoice PDF using Gemini 2.5 Vision.

    Args:
        pdf_path: Path to the PDF to process.
        cfg: Configuration dictionary for Gemini and processing options.
        metrics: Optional dictionary for recording timing and page counts.

    Returns:
        List of manifest dictionaries with start/end page and invoice metadata.
        Returns ``None`` on failure to signal fallback.
    """
    start = time.perf_counter()
    try:
        with fitz.open(str(pdf_path)) as doc:
            parts = []
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                png_bytes = pix.tobytes("png")
                parts.append(
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png")
                )

        prompt = (
            "You are an accounts payable assistant. Given the following pages from a CRE "
            "funding package (pages are 1-indexed), identify each invoice's starting page "
            "and extract vendor name, invoice number, invoice date, and amount. Return a "
            "JSON array with objects having start_page, vendor, invoice_number, date, "
            "amount, confidence."
        )
        contents = [prompt, *parts]

        if metrics is not None:
            metrics["vision_pages"] = len(parts)

        response = ai_gemini.invoke_multimodal(contents, cfg)
        manifest = ai_gemini.parse_manifest_response(response)

        manifest = sorted(manifest, key=lambda m: m.get("start_page", 0))
        manifest = _derive_ranges(manifest, len(parts))
        for item in manifest:
            item["start_page"] = int(item.get("start_page", 0))
            item["end_page"] = int(item.get("end_page", 0))

        elapsed = time.perf_counter() - start
        if metrics is not None:
            metrics["vision_seconds"] = elapsed
        return manifest
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        if metrics is not None:
            metrics["vision_seconds"] = elapsed
        logger.error(
            "Vision segmentation failed: %s", exc, extra={"context": "segment"}
        )
        return None
