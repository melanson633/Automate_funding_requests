from __future__ import annotations

"""PDF segmentation utilities."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List

import pytesseract
from PIL import Image
from pypdf import PdfReader

from . import ai_gemini
from .utils.errors import PDFSegmentationError
from .utils.logging import get_logger

logger = get_logger(__name__)


def _page_text(page: Any, ocr_cfg: dict) -> str:
    """Return extracted text or OCR result for ``page``."""
    text = page.extract_text() or ""
    if text.strip():
        return text

    tesseract_cmd = ocr_cfg.get("tesseract_cmd")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    texts = []
    for img in page.images:
        try:
            pil_img = img.image if hasattr(img, "image") else Image.open(img.data)
            texts.append(pytesseract.image_to_string(pil_img, lang="eng"))
        except Exception as exc:  # noqa: BLE001 broad to avoid OCR crash
            logger.warning("OCR failed on image: %s", exc)
    return "\n".join(texts)


def _derive_ranges(manifest: List[dict], total_pages: int) -> List[dict]:
    """Add ``end_page`` to each invoice entry."""
    manifest = sorted(manifest, key=lambda m: m.get("start_page", 0))
    for idx, item in enumerate(manifest):
        next_start = (
            manifest[idx + 1]["start_page"]
            if idx + 1 < len(manifest)
            else total_pages + 1
        )
        item["end_page"] = next_start - 1
    return manifest


def _validate(manifest: List[dict], total_pages: int, cfg: dict) -> bool:
    pdf_cfg = cfg.get("pdf", {})
    min_conf = float(pdf_cfg.get("min_confidence", cfg.get("min_confidence", 0.0)))
    unmatched_threshold = float(
        pdf_cfg.get("unmatched_threshold", cfg.get("unmatched_threshold", 0.4))
    )

    low_conf_pages = 0
    covered_pages: set[int] = set()
    for item in manifest:
        start = int(item.get("start_page", 0))
        end = int(item.get("end_page", start))
        conf = float(item.get("confidence", 1.0))
        if conf < min_conf:
            low_conf_pages += max(0, end - start + 1)
        covered_pages.update(range(start, end + 1))

    unmatched = total_pages - len(covered_pages)
    if total_pages and (
        low_conf_pages / total_pages > unmatched_threshold
        or unmatched / total_pages > unmatched_threshold
    ):
        return False
    return True


def segment(pdf_path: str | Path, cfg: dict) -> List[dict]:
    """Return invoice manifest with page ranges for ``pdf_path``."""
    reader = PdfReader(str(pdf_path))
    ocr_cfg = cfg.get("ocr", {})
    with ThreadPoolExecutor() as ex:
        texts = list(ex.map(lambda p: _page_text(p, ocr_cfg), reader.pages))

    manifest = ai_gemini.segment_pdf(texts, cfg)

    if not manifest:
        logger.warning("Gemini returned no invoices; using page-per-invoice fallback")
        manifest = [
            {
                "start_page": i + 1,
                "end_page": i + 1,
                "vendor": "",
                "invoice_number": "",
                "date": "",
                "amount": "",
                "confidence": 1.0,
            }
            for i in range(len(reader.pages))
        ]
    else:
        manifest = _derive_ranges(manifest, len(reader.pages))

    if not _validate(manifest, len(reader.pages), cfg):
        pdf_cfg = cfg.get("pdf", {})
        if pdf_cfg.get("split_on_low_confidence"):
            logger.warning(
                "Low confidence manifest detected; splitting PDF into single pages"
            )
            min_conf = float(
                pdf_cfg.get("min_confidence", cfg.get("min_confidence", 0.0))
            )
            pages_covered: set[int] = set()
            high_conf_manifest: List[dict] = []
            for item in manifest:
                start = int(item.get("start_page", 0))
                end = int(item.get("end_page", start))
                conf = float(item.get("confidence", 1.0))
                if conf >= min_conf:
                    high_conf_manifest.append(item)
                    pages_covered.update(range(start, end + 1))
            for page_num in range(1, len(reader.pages) + 1):
                if page_num not in pages_covered:
                    high_conf_manifest.append(
                        {
                            "start_page": page_num,
                            "end_page": page_num,
                            "vendor": "",
                            "invoice_number": "",
                            "date": "",
                            "amount": "",
                            "confidence": 0.0,
                        }
                    )
            manifest = sorted(high_conf_manifest, key=lambda m: m["start_page"])
        else:
            raise PDFSegmentationError("PDF segmentation confidence too low")

    return manifest
