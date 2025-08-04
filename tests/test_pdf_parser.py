from __future__ import annotations

"""Tests for PDFDocument text extraction."""

import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cre_advance.pdf_parser import PDFDocument  # noqa: E402


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


def test_deskew_improves_ocr_output() -> None:
    """Rotated images yield better OCR once deskewed."""
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    base = Image.new("RGB", (400, 100), "white")
    draw = ImageDraw.Draw(base)
    font = ImageFont.truetype(font_path, 32)
    draw.text((10, 10), "HELLO WORLD", fill="black", font=font)
    rotated_img = base.rotate(90, expand=True)

    class RotatedPix:
        width, height = rotated_img.size
        samples = b"\x00" * (width * height * 3)

    class RotatedPage:
        def get_text(self, mode: str) -> str:  # pragma: no cover - simple stub
            return ""

        def get_pixmap(self) -> RotatedPix:  # pragma: no cover - simple stub
            return RotatedPix()

    class RotatedDoc:
        def __iter__(self):  # pragma: no cover - simple stub
            return iter([RotatedPage()])

    with (
        patch("cre_advance.pdf_parser.fitz.open", return_value=RotatedDoc()),
        patch(
            "cre_advance.pdf_parser.Image.frombytes",
            return_value=rotated_img,
        ),
        patch(
            "cre_advance.pdf_parser._deskew_image",
            side_effect=lambda im, cfg: im.rotate(-90, expand=True),
        ),
        patch(
            "cre_advance.pdf_parser.pytesseract.image_to_string",
            side_effect=lambda im, **_: (
                "HELLO WORLD" if im.size == (400, 100) else "GQTYOM O7114H"
            ),
        ),
    ):
        doc = PDFDocument("dummy.pdf")
        cfg = {"ocr": {"langs": ["eng"], "deskew": False}}
        bad = doc.extract_pages_text(cfg)[0].strip().lower()
        cfg["ocr"]["deskew"] = True
        good = doc.extract_pages_text(cfg)[0].strip().lower()

    assert "hello world" not in bad
    assert "hello world" in good
