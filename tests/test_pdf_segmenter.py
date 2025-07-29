from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.modules.setdefault("google", types.ModuleType("google"))  # noqa: E402
sys.modules.setdefault(
    "google.generativeai", types.ModuleType("generativeai")
)  # noqa: E402
from cre_advance import pdf_segmenter  # noqa: E402


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text
        self.images = []

    def extract_text(self) -> str:
        return self._text


class FakeReader:
    def __init__(self, pages) -> None:
        self.pages = pages


def test_segment_success(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))

    def fake_segment(texts, cfg=None):
        return [
            {
                "start_page": 1,
                "vendor": "A",
                "invoice_number": "1",
                "date": "2024-01-01",
                "amount": "100",
                "confidence": 0.9,
            },
            {
                "start_page": 2,
                "vendor": "B",
                "invoice_number": "2",
                "date": "2024-02-01",
                "amount": "200",
                "confidence": 0.95,
            },
        ]

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "segment_pdf", fake_segment)

    manifest = pdf_segmenter.segment("dummy.pdf", {"pdf": {"min_conf": 0.8}})

    assert manifest[0]["end_page"] == 1
    assert manifest[1]["end_page"] == 2


def test_segment_low_conf(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))

    def fake_segment(texts, cfg=None):
        return [
            {"start_page": 1, "confidence": 0.1},
            {"start_page": 2, "confidence": 0.95},
        ]

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "segment_pdf", fake_segment)

    with pytest.raises(pdf_segmenter.PDFSegmentationError):
        pdf_segmenter.segment("dummy.pdf", {"pdf": {"min_conf": 0.8}})


def test_page_text_uses_ocr(monkeypatch):
    page = types.SimpleNamespace(
        extract_text=lambda: "",
        images=[types.SimpleNamespace(image="img")],
    )

    called = {}

    def fake_ocr(img, lang="eng"):
        called["ocr"] = True
        return "hello"

    monkeypatch.setattr(pdf_segmenter.pytesseract, "image_to_string", fake_ocr)

    text = pdf_segmenter._page_text(page, {"tesseract_cmd": None})

    assert text.strip() == "hello"
    assert called.get("ocr")
