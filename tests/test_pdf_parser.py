from __future__ import annotations

"""Tests for PDFDocument text extraction."""

import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cre_advance.pdf_parser import PDFDocument


class DummyPix:
    width = 10
    height = 10
    samples = b"\x00" * width * height * 3


class DummyPage:
    def get_text(self, mode: str) -> str:  # pragma: no cover - simple stub
        return ""

    def get_pixmap(self) -> DummyPix:  # pragma: no cover - simple stub
        return DummyPix()


class DummyDoc:
    def __iter__(self):  # pragma: no cover - simple stub
        return iter([DummyPage()])


def test_extract_pages_text_uses_ocr_when_text_empty() -> None:
    """Ensure OCR is invoked when direct text extraction yields nothing."""
    with (
        patch("cre_advance.pdf_parser.fitz.open", return_value=DummyDoc()),
        patch(
            "cre_advance.pdf_parser.Image.frombytes",
            return_value=Image.new("RGB", (10, 10)),
        ),
        patch(
            "cre_advance.pdf_parser.pytesseract.image_to_string",
            return_value="OCR TEXT",
        ) as ocr_mock,
    ):
        doc = PDFDocument("dummy.pdf")
        cfg = {"ocr": {"langs": ["eng"], "deskew": False}}
        texts = doc.extract_pages_text(cfg)
        assert texts == ["OCR TEXT"]
        ocr_mock.assert_called_once()
