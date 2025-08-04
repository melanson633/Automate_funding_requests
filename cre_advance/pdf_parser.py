from __future__ import annotations

"""PDF text extraction with OCR fallback."""

from pathlib import Path
from typing import List

import fitz
import numpy as np
from PIL import Image
import pytesseract

from .utils.logging import get_logger

logger = get_logger(__name__)


def _deskew_image(img: Image.Image) -> Image.Image:
    """Return a deskewed copy of ``img`` using OpenCV."""
    import cv2

    array = np.array(img)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = array.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        array, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
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
                self.logger.debug(
                    "Page %s yielded no text; performing OCR", page_num
                )
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if ocr_cfg.get("deskew"):
                    img = _deskew_image(img)
                text = pytesseract.image_to_string(img, lang=langs)
            texts.append(text)
        return texts
