from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import google.api_core  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
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


def _keep_all(texts, cfg=None):
    return [
        {
            "page_number": i + 1,
            "category": "invoice",
            "keep": True,
            "confidence": 1.0,
        }
        for i in range(len(texts))
    ]


def test_segment_success(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_pages", _keep_all)
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

    manifest = pdf_segmenter.segment("dummy.pdf", {"pdf": {"min_confidence": 0.8}})

    assert manifest[0]["end_page"] == 1
    assert manifest[1]["end_page"] == 2


def test_segment_low_conf(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_pages", _keep_all)
    def fake_segment(texts, cfg=None):
        return [
            {"start_page": 1, "confidence": 0.1},
            {"start_page": 2, "confidence": 0.95},
        ]

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "segment_pdf", fake_segment)

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
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_pages", _keep_all)
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "segment_pdf", lambda t, cfg=None: [])

    manifest = pdf_segmenter.segment("dummy.pdf", {"pdf": {}})

    assert len(manifest) == 3
    assert all(
        m["start_page"] == i + 1 and m["end_page"] == i + 1
        for i, m in enumerate(manifest)
    )


def test_low_conf_split(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_pages", _keep_all)
    def fake_segment(texts, cfg=None):
        return [
            {"start_page": 1, "confidence": 0.1},
            {"start_page": 2, "confidence": 0.95},
        ]

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "segment_pdf", fake_segment)

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

    def fake_classify(texts, cfg=None):
        return [
            {
                "page_number": 1,
                "category": "invoice_register",
                "keep": False,
                "confidence": 0.9,
            },
            {
                "page_number": 2,
                "category": "invoice",
                "keep": True,
                "confidence": 0.95,
            },
            {
                "page_number": 3,
                "category": "email_approval",
                "keep": False,
                "confidence": 0.8,
            },
        ]

    called = {}

    def fake_segment(texts, cfg=None):
        called["texts"] = list(texts)
        return [
            {
                "start_page": 1,
                "vendor": "V",
                "invoice_number": "123",
                "date": "",
                "amount": "",
                "confidence": 1.0,
            }
        ]

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_pages", fake_classify)
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "segment_pdf", fake_segment)

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

    def fail_classify(texts, cfg=None):
        raise RuntimeError("boom")

    called = {}

    def fake_segment(texts, cfg=None):
        called["texts"] = list(texts)
        return [
            {
                "start_page": 1,
                "vendor": "V",
                "invoice_number": "123",
                "date": "",
                "amount": "",
                "confidence": 1.0,
            }
        ]

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_pages", fail_classify)
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "segment_pdf", fake_segment)

    manifest = pdf_segmenter.segment("dummy.pdf", {"pdf": {}})

    assert called["texts"] == ["Invoice #123\nBill To"]
    assert manifest[0]["start_page"] == 2
