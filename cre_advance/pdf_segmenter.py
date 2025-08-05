from __future__ import annotations

"""PDF segmentation utilities."""

import copy
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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


class OCRService:
    """Simple OCR helper around ``pytesseract``."""

    def __init__(self, cfg: Optional[dict] = None) -> None:
        self.cfg = cfg or {}
        cmd = self.cfg.get("tesseract_cmd")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        self.langs = "+".join(self.cfg.get("langs", ["eng"]))
        self.psm = int(self.cfg.get("psm", 6))
        self.oem = int(self.cfg.get("oem", 1))
        self.deskew = bool(self.cfg.get("deskew", False))
        self.tess_config = f"--psm {self.psm} --oem {self.oem}"

    def _deskew_image(self, image: Image.Image) -> Image.Image:
        if not self.deskew:
            return image
        try:
            osd = pytesseract.image_to_osd(image)
            rotate = int(re.search(r"Rotate: (\d+)", osd).group(1))
            if rotate:
                image = image.rotate(360 - rotate, expand=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Deskew failed: %s", exc, extra={"context": "segment"})
        return image

    def extract(self, page: Any) -> tuple[str, bool]:
        """Return text for ``page`` and whether OCR was used."""
        text = page.extract_text() or ""
        if text.strip():
            return text, False

        texts: list[str] = []
        for img in page.images:
            try:
                pil_img = img.image if hasattr(img, "image") else Image.open(img.data)
                pil_img = self._deskew_image(pil_img)
                try:
                    text = pytesseract.image_to_string(
                        pil_img, lang=self.langs, config=self.tess_config
                    )
                except TypeError:
                    text = pytesseract.image_to_string(pil_img, lang=self.langs)
                texts.append(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OCR failed on image: %s", exc, extra={"context": "segment"}
                )
        return "\n".join(texts), True


class PDFDocument:
    """Thin wrapper around ``pypdf.PdfReader``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.reader = PdfReader(str(self.path))

    @property
    def pages(self) -> list[Any]:
        return list(self.reader.pages)

    @property
    def page_map(self) -> list[int]:
        return list(range(1, len(self.reader.pages) + 1))


@dataclass
class Manifest:
    """Container for invoice segmentation results."""

    items: list[dict] = field(default_factory=list)

    @staticmethod
    def _derive_ranges(manifest: list[dict], total_pages: int) -> list[dict]:
        manifest = sorted(manifest, key=lambda m: m.get("start_page", 0))
        for idx, item in enumerate(manifest):
            next_start = (
                manifest[idx + 1]["start_page"]
                if idx + 1 < len(manifest)
                else total_pages + 1
            )
            item["end_page"] = next_start - 1
        return manifest

    def finalize(
        self, page_map: list[int], cfg: dict, metrics: Optional[dict] = None
    ) -> None:
        if metrics is not None:
            metrics["total_pages"] = len(page_map)

        if not self.items:
            logger.warning(
                "Gemini returned no invoices; using page-per-invoice fallback",
                extra={"context": "segment"},
            )
            self.items = [
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
            if any("end_page" not in m for m in self.items):
                self.items = self._derive_ranges(self.items, len(page_map))
            for item in self.items:
                s_idx = int(item.get("start_page", 1)) - 1
                e_idx = int(item.get("end_page", s_idx + 1)) - 1
                item["start_page"] = page_map[s_idx]
                item["end_page"] = page_map[e_idx]

        if metrics is not None:
            metrics["invoice_count"] = len(self.items)

    def validate(
        self, total_pages: int, cfg: dict, metrics: Optional[dict] = None
    ) -> bool:
        pdf_cfg = cfg.get("pdf", {})
        min_conf = float(pdf_cfg.get("min_confidence", cfg.get("min_confidence", 0.0)))
        threshold = float(
            pdf_cfg.get("unmatched_threshold", cfg.get("unmatched_threshold", 0.4))
        )

        low_conf = 0
        covered: set[int] = set()
        for item in self.items:
            start = int(item.get("start_page", 0))
            end = int(item.get("end_page", start))
            conf = float(item.get("confidence", 1.0))
            if conf < min_conf:
                low_conf += max(0, end - start + 1)
            covered.update(range(start, end + 1))

        unmatched = total_pages - len(covered)
        if metrics is not None:
            metrics["low_conf_pages"] = low_conf
            metrics["unmatched_pages"] = unmatched
            metrics["total_pages"] = total_pages

        if total_pages and (
            low_conf / total_pages > threshold or unmatched / total_pages > threshold
        ):
            return False
        return True

    def to_list(self) -> list[dict]:
        return list(self.items)


def create_services(cfg: dict) -> tuple[OCRService, PageClassifier, InvoiceSegmenter]:
    """Factory returning default OCR, classifier and segmenter."""
    return OCRService(cfg.get("ocr", {})), GeminiClassifier(), InvoiceSegmenter()


def segment(
    pdf_path: str | Path,
    cfg: dict,
    metrics: Optional[dict] = None,
    classifier: Optional[PageClassifier] = None,
    segmenter: Optional[InvoiceSegmenter] = None,
    ocr_service: Optional[OCRService] = None,
) -> list[dict]:
    """Return invoice manifest with page ranges for ``pdf_path``."""
    logger.info(
        "Starting PDF segmentation", extra={"context": {"file": str(pdf_path)}}
    )
    start_all = time.perf_counter()
    pdf_cfg = cfg.get("pdf", {})
    use_vision = pdf_cfg.get("use_vision", False)

    document = PDFDocument(pdf_path)
    page_map_all = document.page_map

    manifest_obj: Optional[Manifest] = None
    page_map: list[int] = page_map_all

    if use_vision:
        t0 = time.perf_counter()
        try:
            from . import vision_segmenter

            man_list = vision_segmenter.segment(pdf_path, cfg, metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Vision segmentation error: %s",
                exc,
                extra={"context": {"file": str(pdf_path)}},
            )
            man_list = None
        if metrics is not None:
            metrics["pdf_seconds"] = time.perf_counter() - t0
        if man_list:
            manifest_obj = Manifest(man_list)
            manifest_obj.finalize(page_map_all, cfg, metrics)

    if manifest_obj is None:
        ocr_service = ocr_service or OCRService(cfg.get("ocr", {}))
        with ThreadPoolExecutor() as ex:
            text_results = list(ex.map(lambda p: ocr_service.extract(p), document.pages))
        all_texts = [t for t, _ in text_results]
        ocr_pages = sum(1 for _, flag in text_results if flag)
        if metrics is not None:
            metrics["ocr_pages"] = ocr_pages
            metrics["ocr_quality"] = 1 - ocr_pages / max(1, len(text_results))

        classifier = classifier or GeminiClassifier()
        segmenter = segmenter or InvoiceSegmenter()

        def run_with_classifier(cls: PageClassifier) -> tuple[Manifest, list[int]]:
            texts = all_texts
            page_map_local = page_map_all
            if pdf_cfg.get("remove_invoice_register", True):
                classified: list[dict] | None = None
                try:
                    classified = cls.classify(all_texts, cfg)
                    if metrics is not None and classified:
                        conf_vals = [float(c.get("confidence", 0.0)) for c in classified]
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
                removed: dict[str, int] = {}
                keep_pages: list[int] = []
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

            man_list = segmenter.segment_invoices(texts, cfg)
            manifest = Manifest(man_list)
            manifest.finalize(page_map_local, cfg, metrics)
            return manifest, page_map_local

        manifest_obj, page_map = run_with_classifier(classifier)

        fallback_used = "none"
        if not manifest_obj.validate(len(page_map), cfg, metrics):
            relaxed_cfg = copy.deepcopy(cfg)
            relaxed_pdf = relaxed_cfg.setdefault("pdf", {})
            relaxed_pdf["min_confidence"] = 0.0
            relaxed_pdf["unmatched_threshold"] = 1.0
            if manifest_obj.validate(len(page_map), relaxed_cfg, metrics):
                fallback_used = "lower_confidence"
            else:
                manifest_obj, page_map = run_with_classifier(HeuristicClassifier())
                if manifest_obj.validate(len(page_map), relaxed_cfg, metrics):
                    fallback_used = "heuristic_classifier"
                else:
                    manifest_obj = Manifest(
                        [
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
                    )
                    if metrics is not None:
                        metrics["invoice_count"] = len(manifest_obj.items)
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
                log_metric(f"pdf_{key}", value, tags={"file": str(pdf_path)})
        for item in manifest_obj.items:
            item["fallback_used"] = fallback_used
        logger.info(
            "Finished PDF segmentation",
            extra={"context": {"file": str(pdf_path)}},
        )
        return manifest_obj.to_list()

    # vision path where manifest_obj is not None
    fallback_used = "none"
    if not manifest_obj.validate(len(page_map_all), cfg, metrics):
        relaxed_cfg = copy.deepcopy(cfg)
        relaxed_pdf = relaxed_cfg.setdefault("pdf", {})
        relaxed_pdf["min_confidence"] = 0.0
        relaxed_pdf["unmatched_threshold"] = 1.0
        if manifest_obj.validate(len(page_map_all), relaxed_cfg, metrics):
            fallback_used = "lower_confidence"
        else:
            ocr_service = ocr_service or OCRService(cfg.get("ocr", {}))
            with ThreadPoolExecutor() as ex:
                text_results = list(
                    ex.map(lambda p: ocr_service.extract(p), document.pages)
                )
            all_texts = [t for t, _ in text_results]
            ocr_pages = sum(1 for _, flag in text_results if flag)
            if metrics is not None:
                metrics["ocr_pages"] = ocr_pages
                metrics["ocr_quality"] = 1 - ocr_pages / max(1, len(text_results))
            classifier = HeuristicClassifier()
            segmenter = segmenter or InvoiceSegmenter()
            manifest_obj, page_map = run_with_classifier(classifier)
            if manifest_obj.validate(len(page_map), relaxed_cfg, metrics):
                fallback_used = "heuristic_classifier"
            else:
                manifest_obj = Manifest(
                    [
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
                )
                if metrics is not None:
                    metrics["invoice_count"] = len(manifest_obj.items)
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
    for item in manifest_obj.items:
        item["fallback_used"] = fallback_used
    logger.info(
        "Finished PDF segmentation", extra={"context": {"file": str(pdf_path)}}
    )
    return manifest_obj.to_list()
