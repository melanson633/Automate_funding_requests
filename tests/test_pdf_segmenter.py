from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import google.api_core  # noqa: F401

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


def test_segment_success(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(
        pdf_segmenter.ai_gemini, "detect_invoice_starts", lambda texts: [0, 1]
    )

    manifest = pdf_segmenter.segment(
        "dummy.pdf",
        {"pdf": {"min_confidence": 0.8}},
        classifier=pdf_segmenter.HeuristicClassifier(),
    )

    assert manifest[0]["end_page"] == 1
    assert manifest[1]["end_page"] == 2


def test_segment_low_conf(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(
        pdf_segmenter.ai_gemini, "detect_invoice_starts", lambda texts: [0, 1]
    )
    monkeypatch.setattr(pdf_segmenter, "_validate", lambda m, t, c, metrics=None: False)

    with pytest.raises(pdf_segmenter.PDFSegmentationError):
        pdf_segmenter.segment(
            "dummy.pdf",
            {"pdf": {"min_confidence": 0.8}},
            classifier=pdf_segmenter.HeuristicClassifier(),
        )


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
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "detect_invoice_starts", lambda t: [])

    manifest = pdf_segmenter.segment(
        "dummy.pdf", {"pdf": {}}, classifier=pdf_segmenter.HeuristicClassifier()
    )

    assert len(manifest) == 3
    assert all(
        m["start_page"] == i + 1 and m["end_page"] == i + 1
        for i, m in enumerate(manifest)
    )


def test_low_conf_split(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    monkeypatch.setattr(
        pdf_segmenter.ai_gemini, "detect_invoice_starts", lambda texts: [0, 1]
    )
    monkeypatch.setattr(pdf_segmenter, "_validate", lambda m, t, c, metrics=None: False)

    manifest = pdf_segmenter.segment(
        "dummy.pdf",
        {"pdf": {"min_confidence": 0.8, "split_on_low_confidence": True}},
        classifier=pdf_segmenter.HeuristicClassifier(),
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
    called = {}

    def fake_detect(texts):
        called["texts"] = list(texts)
        return [0]

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "detect_invoice_starts", fake_detect)

    manifest = pdf_segmenter.segment(
        "dummy.pdf", {"pdf": {}}, classifier=pdf_segmenter.HeuristicClassifier()
    )

    assert called["texts"] == ["Invoice #123\nBill To"]
    assert manifest[0]["start_page"] == 2
    assert manifest[0]["end_page"] == 2


def test_segment_heuristic_fallback(monkeypatch) -> None:
    pages = [
        FakePage("Invoice Register\nWorkflow Approval"),
        FakePage("Invoice #123\nBill To"),
    ]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))
    called = {}

    def fake_detect(texts):
        called["texts"] = list(texts)
        return [0]

    def fail_classify_pages(pages, cfg):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(pdf_segmenter.ai_gemini, "classify_pages", fail_classify_pages)
    monkeypatch.setattr(pdf_segmenter.ai_gemini, "detect_invoice_starts", fake_detect)

    manifest = pdf_segmenter.segment("dummy.pdf", {"pdf": {}})

    assert called["texts"] == ["Invoice #123\nBill To"]
    assert manifest[0]["start_page"] == 2


def test_segment_vision_bypasses_ocr(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))

    fake_manifest = [
        {
            "start_page": 1,
            "end_page": 2,
            "vendor": "",
            "invoice_number": "",
            "date": "",
            "amount": "",
            "confidence": 1.0,
        }
    ]

    from cre_advance import vision_segmenter as vs

    monkeypatch.setattr(vs, "segment", lambda *a, **k: fake_manifest)
    monkeypatch.setattr(
        pdf_segmenter,
        "_page_text",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("OCR called")),
    )
    monkeypatch.setattr(pdf_segmenter, "_validate", lambda m, t, c, metrics=None: True)

    manifest = pdf_segmenter.segment(
        "dummy.pdf",
        {"pdf": {"use_vision": True}},
        classifier=pdf_segmenter.HeuristicClassifier(),
    )

    assert manifest == fake_manifest


def test_segment_vision_none_falls_back_to_ocr(monkeypatch) -> None:
    pages = [FakePage("A"), FakePage("B")]
    monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: FakeReader(pages))

    from cre_advance import vision_segmenter as vs

    monkeypatch.setattr(vs, "segment", lambda *a, **k: None)

    calls = {"count": 0}

    def fake_page_text(page, cfg):
        calls["count"] += 1
        return "text"

    monkeypatch.setattr(pdf_segmenter, "_page_text", fake_page_text)
    monkeypatch.setattr(
        pdf_segmenter.ai_gemini, "detect_invoice_starts", lambda texts: [0, 1]
    )
    monkeypatch.setattr(pdf_segmenter, "_validate", lambda m, t, c, metrics=None: True)

    manifest = pdf_segmenter.segment(
        "dummy.pdf",
        {"pdf": {"use_vision": True}},
        classifier=pdf_segmenter.HeuristicClassifier(),
    )

    assert calls["count"] == 2
    assert len(manifest) == 2
    assert manifest[0]["start_page"] == 1 and manifest[1]["start_page"] == 2

