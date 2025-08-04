from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import google.api_core  # noqa: F401
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

# Mock the genai module structure
google_mock = types.ModuleType("google")
genai_mock = types.ModuleType("genai")
genai_types_mock = types.ModuleType("types")
genai_mock.types = genai_types_mock
google_mock.genai = genai_mock

sys.modules.setdefault("google", google_mock)
sys.modules.setdefault("google.genai", genai_mock)
sys.modules.setdefault("google.genai.types", genai_types_mock)

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


def _keep_all(_text: str) -> bool:
    return True


def test_segment_success(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_page", _keep_all)
    monkeypatch.setattr(
        pdf_segmenter.ai_gemini, "detect_invoice_starts", lambda texts: [0, 1]
    )

    manifest = pdf_segmenter.segment("dummy.pdf", {"pdf": {"min_confidence": 0.8}})

    assert manifest[0]["end_page"] == 1
    assert manifest[1]["end_page"] == 2


def test_segment_low_conf(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_page", _keep_all)
    monkeypatch.setattr(
        pdf_segmenter.ai_gemini, "detect_invoice_starts", lambda texts: [0, 1]
    )
    monkeypatch.setattr(pdf_segmenter, "_validate", lambda m, t, c, metrics=None: False)

    with pytest.raises(pdf_segmenter.PDFSegmentationError):
        pdf_segmenter.segment("dummy.pdf", {"pdf": {"min_confidence": 0.8}})


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


def test_segment_fallback(monkeypatch):
    pages = [FakePage("A"), FakePage("B"), FakePage("C")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_page", _keep_all)
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "detect_invoice_starts", lambda t: [])

    manifest = pdf_segmenter.segment("dummy.pdf", {"pdf": {}})

    assert len(manifest) == 3
    assert all(
        m["start_page"] == i + 1 and m["end_page"] == i + 1
        for i, m in enumerate(manifest)
    )


def test_low_conf_split(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_page", _keep_all)
    monkeypatch.setattr(
        pdf_segmenter.ai_gemini, "detect_invoice_starts", lambda texts: [0, 1]
    )
    monkeypatch.setattr(pdf_segmenter, "_validate", lambda m, t, c, metrics=None: False)

    manifest = pdf_segmenter.segment(
        "dummy.pdf",
        {"pdf": {"min_confidence": 0.8, "split_on_low_confidence": True}},
    )

    assert len(manifest) == 2
    assert manifest[0]["start_page"] == 1 and manifest[0]["end_page"] == 1
    assert manifest[1]["start_page"] == 2


def test_segment_filters_pages(monkeypatch) -> None:
    pages = [
        FakePage("Invoice Register\nWorkflow Approval"),
        FakePage("Invoice #123\nBill To"),
        FakePage("From: Craig\nSent: today"),
    ]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))

    def fake_classify(text: str) -> bool:
        return "invoice #" in text.lower()

    called = {}

    def fake_detect(texts):
        called["texts"] = list(texts)
        return [0]

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_page", fake_classify)
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "detect_invoice_starts", fake_detect)

    manifest = pdf_segmenter.segment("dummy.pdf", {"pdf": {}})

    assert called["texts"] == ["Invoice #123\nBill To"]
    assert manifest[0]["start_page"] == 2
    assert manifest[0]["end_page"] == 2


def test_segment_heuristic_fallback(monkeypatch) -> None:
    pages = [
        FakePage("Invoice Register\nWorkflow Approval"),
        FakePage("Invoice #123\nBill To"),
    ]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))

    def fail_classify(text: str) -> bool:
        raise RuntimeError("boom")

    called = {}

    def fake_detect(texts):
        called["texts"] = list(texts)
        return [0]

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_page", fail_classify)
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "detect_invoice_starts", fake_detect)

    manifest = pdf_segmenter.segment("dummy.pdf", {"pdf": {}})

    assert called["texts"] == ["Invoice #123\nBill To"]
    assert manifest[0]["start_page"] == 2

