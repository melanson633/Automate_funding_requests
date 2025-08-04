from __future__ import annotations

"""PDF text extraction with OCR fallback."""

from pathlib import Path
from typing import List

import fitz
import numpy as np
import pytesseract
from PIL import Image

from .utils.logging import get_logger

logger = get_logger(__name__)


def _deskew_image(img: Image.Image, cfg: dict) -> Image.Image:
    """Return image rotated based on Tesseract orientation data.

    Args:
        img: Image to deskew.
        cfg: Application configuration.

    Returns:
        Deskewed copy of ``img`` or the original image when orientation
        detection fails.
    """
    import cv2

    ocr_cfg = cfg.get("ocr", {})
    tesseract_cmd = ocr_cfg.get("tesseract_cmd")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return img

    rotate = osd.get("rotate", 0)
    if rotate not in {90, 180, 270}:
        return img

    array = np.array(img)
    if rotate == 90:
        rotated = cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == 180:
        rotated = cv2.rotate(array, cv2.ROTATE_180)
    else:  # 270
        rotated = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
    return Image.fromarray(rotated)


class PDFDocument:
    """Wrapper around ``fitz.Document`` providing page text extraction."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.doc = fitz.open(self.path)
        self.logger = logger

    def extract_pages_text(self, cfg: dict) -> List[str]:
        """Return list of page texts, falling back to OCR when necessary."""
        texts: List[str] = []
        ocr_cfg = cfg.get("ocr", {})
        tesseract_cmd = ocr_cfg.get("tesseract_cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        langs = "+".join(ocr_cfg.get("langs", []))
        for page_num, page in enumerate(self.doc, start=1):
            text = (page.get_text("text") or "").strip()
            if not text:
                self.logger.debug("Page %s yielded no text; performing OCR", page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if ocr_cfg.get("deskew"):
                    img = _deskew_image(img, cfg)
                text = pytesseract.image_to_string(img, lang=langs)
            texts.append(text)
        return texts
