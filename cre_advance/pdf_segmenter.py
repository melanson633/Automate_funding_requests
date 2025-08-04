from __future__ import annotations

"""PDF segmentation utilities."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List
import copy

import pytesseract
from PIL import Image
from pypdf import PdfReader

from . import ai_gemini
from .classifiers import GeminiClassifier, HeuristicClassifier, PageClassifier
from .metrics import log_metric
from .segmenters import InvoiceSegmenter
from .utils.errors import PDFSegmentationError
from .utils.logging import get_logger

logger = get_logger(__name__)


def _page_text(page: Any, ocr_cfg: dict) -> tuple[str, bool]:
    """Return extracted text or OCR result flag for ``page``."""
    text = page.extract_text() or ""
    if text.strip():
        return text, False

    tesseract_cmd = ocr_cfg.get("tesseract_cmd")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    texts = []
    for img in page.images:
        try:
            pil_img = img.image if hasattr(img, "image") else Image.open(img.data)
            texts.append(pytesseract.image_to_string(pil_img, lang="eng"))
        except Exception as exc:  # noqa: BLE001 broad to avoid OCR crash
            logger.warning(
                "OCR failed on image: %s", exc, extra={"context": "segment"}
            )
    return "\n".join(texts), True


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


def _validate(
    manifest: List[dict], total_pages: int, cfg: dict, metrics: dict | None = None
) -> bool:
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
    if metrics is not None:
        metrics["low_conf_pages"] = low_conf_pages
        metrics["unmatched_pages"] = unmatched
        metrics["total_pages"] = total_pages
    if total_pages and (
        low_conf_pages / total_pages > unmatched_threshold
        or unmatched / total_pages > unmatched_threshold
    ):
        return False
    return True

def _finalize(
    manifest: List[dict],
    page_map: List[int],
    cfg: dict,
    metrics: dict | None = None,
) -> List[dict]:
    """Finalize manifest with proper page numbers."""
    if metrics is not None:
        metrics["total_pages"] = len(page_map)

    if not manifest:
        logger.warning(
            "Gemini returned no invoices; using page-per-invoice fallback",
            extra={"context": "segment"},
        )
        manifest = [
            {
                "start_page": p,
                "end_page": p,
                "vendor": "",
                "invoice_number": "",
                "date": "",
                "amount": "",
                "confidence": 1.0,
            }
            for p in page_map
        ]
    else:
        if any("end_page" not in m for m in manifest):
            manifest = _derive_ranges(manifest, len(page_map))
        for item in manifest:
            start_idx = int(item.get("start_page", 1)) - 1
            end_idx = int(item.get("end_page", start_idx + 1)) - 1
            item["start_page"] = page_map[start_idx]
            item["end_page"] = page_map[end_idx]

    if metrics is not None:
        metrics["invoice_count"] = len(manifest)

    return manifest


def segment(
    pdf_path: str | Path,
    cfg: dict,
    metrics: dict | None = None,
    classifier: PageClassifier | None = None,
    segmenter: InvoiceSegmenter | None = None,
) -> List[dict]:
    """Return invoice manifest with page ranges for ``pdf_path``.

    Args:
        pdf_path: Path to source PDF.
        cfg: Configuration dictionary.
        metrics: Optional metrics dictionary.
        classifier: Page classifier instance. Defaults to ``GeminiClassifier``.
        segmenter: Invoice segmenter instance. Defaults to ``InvoiceSegmenter``.
    """
    logger.info(
        "Starting PDF segmentation",
        extra={"context": {"file": str(pdf_path)}},
    )
    start_all = time.perf_counter()
    pdf_cfg = cfg.get("pdf", {})
    use_vision = pdf_cfg.get("use_vision", False)

    reader = PdfReader(str(pdf_path))
    page_map_all = list(range(1, len(reader.pages) + 1))

    manifest: List[dict] | None = None
    page_map: List[int] = page_map_all

    if use_vision:
        t0 = time.perf_counter()
        try:
            from . import vision_segmenter

            manifest = vision_segmenter.segment(pdf_path, cfg, metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Vision segmentation error: %s",
                exc,
                extra={"context": {"file": str(pdf_path)}},
            )
            manifest = None
        if metrics is not None:
            metrics["pdf_seconds"] = time.perf_counter() - t0
        if manifest:
            logger.info(
                "Vision segmentation succeeded",
                extra={"context": {"file": str(pdf_path)}},
            )
            manifest = _finalize(manifest, page_map_all, cfg, metrics)

    if manifest is None:
        ocr_cfg = cfg.get("ocr", {})
        with ThreadPoolExecutor() as ex:
            text_results = list(
                ex.map(lambda p: _page_text(p, ocr_cfg), reader.pages)
            )
        all_texts = [t for t, _ in text_results]
        ocr_pages = sum(1 for _, flag in text_results if flag)
        if metrics is not None:
            metrics["ocr_pages"] = ocr_pages
            metrics["ocr_quality"] = 1 - ocr_pages / max(1, len(text_results))

        classifier = classifier or GeminiClassifier()
        segmenter = segmenter or InvoiceSegmenter()

        def run_with_classifier(cls: PageClassifier) -> tuple[List[dict], List[int]]:
            texts = all_texts
            page_map_local = page_map_all
            if pdf_cfg.get("remove_invoice_register", True):
                classified: List[dict] | None = None
                try:
                    classified = cls.classify(all_texts, cfg)
                    if metrics is not None and classified:
                        conf_vals = [
                            float(c.get("confidence", 0.0)) for c in classified
                        ]
                        if conf_vals:
                            metrics["classification_confidence"] = sum(conf_vals) / len(
                                conf_vals
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Page classification error: %s", exc, extra={"context": "segment"}
                    )
                if not classified and not isinstance(cls, HeuristicClassifier):
                    logger.info(
                        "Falling back to heuristic page classification",
                        extra={"context": "segment"},
                    )
                    classified = HeuristicClassifier().classify(all_texts, cfg)
                if not classified:
                    raise PDFSegmentationError("Page classification failed")

                threshold = float(
                    pdf_cfg.get("classification_confidence_threshold", 0.0)
                )
                removed: Dict[str, int] = {}
                keep_pages: List[int] = []
                by_number = {int(c.get("page_number", 0)): c for c in classified}
                for num in range(1, len(all_texts) + 1):
                    cls_res = by_number.get(
                        num, {"keep": True, "category": "unknown", "confidence": 1.0}
                    )
                    keep = bool(cls_res.get("keep"))
                    conf = float(cls_res.get("confidence", 0.0))
                    if keep and conf >= threshold:
                        keep_pages.append(num)
                    else:
                        cat = str(cls_res.get("category", "unknown"))
                        removed[cat] = removed.get(cat, 0) + 1
                if removed:
                    logger.info(
                        "Removed %s pages: %s",
                        sum(removed.values()),
                        removed,
                        extra={"context": "segment"},
                    )
                texts = [all_texts[i - 1] for i in keep_pages]
                page_map_local = keep_pages
                if not texts:
                    raise PDFSegmentationError("No invoice pages after classification")

            man = segmenter.segment_invoices(texts, cfg)
            man = _finalize(man, page_map_local, cfg, metrics)
            return man, page_map_local

        manifest, page_map = run_with_classifier(classifier)

        fallback_used = "none"
        if not _validate(manifest, len(page_map), cfg, metrics):
            relaxed_cfg = copy.deepcopy(cfg)
            relaxed_pdf = relaxed_cfg.setdefault("pdf", {})
            relaxed_pdf["min_confidence"] = 0.0
            relaxed_pdf["unmatched_threshold"] = 1.0
            if _validate(manifest, len(page_map), relaxed_cfg, metrics):
                fallback_used = "lower_confidence"
            else:
                manifest, page_map = run_with_classifier(HeuristicClassifier())
                if _validate(manifest, len(page_map), relaxed_cfg, metrics):
                    fallback_used = "heuristic_classifier"
                else:
                    manifest = [
                        {
                            "start_page": p,
                            "end_page": p,
                            "vendor": "",
                            "invoice_number": "",
                            "date": "",
                            "amount": "",
                            "confidence": 0.0,
                        }
                        for p in page_map
                    ]
                    if metrics is not None:
                        metrics["invoice_count"] = len(manifest)
                    fallback_used = "page_per_invoice"
        else:
            fallback_used = "none"

        if metrics is not None:
            metrics["fallback_used"] = fallback_used
            total = metrics.get("total_pages", 0)
            unmatched = metrics.get("unmatched_pages", 0)
            if total:
                metrics["segmentation_confidence"] = 1 - unmatched / total
            metrics["processing_seconds"] = time.perf_counter() - start_all
            for key, value in metrics.items():
                log_metric(
                    f"pdf_{key}", value, tags={"file": str(pdf_path)}
                )
        for item in manifest:
            item["fallback_used"] = fallback_used
        logger.info(
            "Finished PDF segmentation",
            extra={"context": {"file": str(pdf_path)}},
        )
        return manifest

    # vision path where manifest is not None
    fallback_used = "none"
    if not _validate(manifest, len(page_map_all), cfg, metrics):
        relaxed_cfg = copy.deepcopy(cfg)
        relaxed_pdf = relaxed_cfg.setdefault("pdf", {})
        relaxed_pdf["min_confidence"] = 0.0
        relaxed_pdf["unmatched_threshold"] = 1.0
        if _validate(manifest, len(page_map_all), relaxed_cfg, metrics):
            fallback_used = "lower_confidence"
        else:
            # Need OCR + heuristic fallback
            ocr_cfg = cfg.get("ocr", {})
            with ThreadPoolExecutor() as ex:
                text_results = list(
                    ex.map(lambda p: _page_text(p, ocr_cfg), reader.pages)
                )
            all_texts = [t for t, _ in text_results]
            ocr_pages = sum(1 for _, flag in text_results if flag)
            if metrics is not None:
                metrics["ocr_pages"] = ocr_pages
                metrics["ocr_quality"] = 1 - ocr_pages / max(1, len(text_results))
            classifier = HeuristicClassifier()
            segmenter = segmenter or InvoiceSegmenter()
            manifest, page_map = run_with_classifier(classifier)
            if _validate(manifest, len(page_map), relaxed_cfg, metrics):
                fallback_used = "heuristic_classifier"
            else:
                manifest = [
                    {
                        "start_page": p,
                        "end_page": p,
                        "vendor": "",
                        "invoice_number": "",
                        "date": "",
                        "amount": "",
                        "confidence": 0.0,
                    }
                    for p in page_map
                ]
                if metrics is not None:
                    metrics["invoice_count"] = len(manifest)
                fallback_used = "page_per_invoice"

    if metrics is not None:
        metrics["fallback_used"] = fallback_used
        total = metrics.get("total_pages", 0)
        unmatched = metrics.get("unmatched_pages", 0)
        if total:
            metrics["segmentation_confidence"] = 1 - unmatched / total
        metrics["processing_seconds"] = time.perf_counter() - start_all
        for key, value in metrics.items():
            log_metric(f"pdf_{key}", value, tags={"file": str(pdf_path)})
    for item in manifest:
        item["fallback_used"] = fallback_used
    logger.info(
        "Finished PDF segmentation",
        extra={"context": {"file": str(pdf_path)}},
    )
    return manifest
