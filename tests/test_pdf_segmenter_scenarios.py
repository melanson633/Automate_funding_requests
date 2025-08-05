from __future__ import annotations

import time

import pytest

from cre_advance import pdf_segmenter, segmenters


def _setup_ai_mocks(monkeypatch):
    monkeypatch.setattr(segmenters.ai_gemini, "detect_invoice_starts", lambda t: [0])
    monkeypatch.setattr(
        segmenters.ai_gemini,
        "extract_metadata",
        lambda text: {
            "vendor": "Vendor A",
            "invoice_number": "1001",
            "date": "2024-01-01",
            "amount": "10.00",
        },
    )


def test_scanned_pdf_uses_ocr(patch_pdf_reader, fake_page, monkeypatch):
    pages = [fake_page(has_image=True), fake_page(has_image=True)]
    patch_pdf_reader(pages)

    texts = [
        "Vendor A\nInvoice #1001\nTotal $10.00",
        "Page 2 of invoice 1001",
    ]

    def fake_ocr(img, lang="eng"):
        return texts.pop(0)

    monkeypatch.setattr(pdf_segmenter.pytesseract, "image_to_string", fake_ocr)
    _setup_ai_mocks(monkeypatch)

    metrics = {}
    manifest = pdf_segmenter.segment(
        "dummy.pdf",
        {"pdf": {}},
        metrics=metrics,
        classifier=pdf_segmenter.HeuristicClassifier(),
    )

    assert metrics.get("ocr_pages") == 2
    assert manifest[0]["start_page"] == 1 and manifest[0]["end_page"] == 2


def test_multi_page_invoice_continuation(patch_pdf_reader, fake_page, monkeypatch):
    pages = [
        fake_page("Vendor A\nInvoice #1001\nTotal $100.00"),
        fake_page("Continuation of invoice 1001"),
        fake_page("Vendor B\nInvoice #2002\nTotal $50.00"),
    ]
    patch_pdf_reader(pages)

    monkeypatch.setattr(segmenters.ai_gemini, "detect_invoice_starts", lambda t: [0, 2])

    def fake_extract(text: str) -> dict:
        if "1001" in text:
            return {
                "vendor": "Vendor A",
                "invoice_number": "1001",
                "date": "2024-01-01",
                "amount": "100.00",
            }
        return {
            "vendor": "Vendor B",
            "invoice_number": "2002",
            "date": "2024-02-01",
            "amount": "50.00",
        }

    monkeypatch.setattr(segmenters.ai_gemini, "extract_metadata", fake_extract)

    manifest = pdf_segmenter.segment(
        "dummy.pdf", {"pdf": {}}, classifier=pdf_segmenter.HeuristicClassifier()
    )

    assert manifest[0]["start_page"] == 1 and manifest[0]["end_page"] == 2
    assert manifest[1]["start_page"] == 3 and manifest[1]["end_page"] == 3


def test_reconcile_missing_invoice_number(patch_pdf_reader, fake_page, monkeypatch):
    pages = [fake_page("Vendor C\nTotal $30.00")]
    patch_pdf_reader(pages)

    _setup_ai_mocks(monkeypatch)
    monkeypatch.setattr(
        segmenters.ai_gemini,
        "extract_metadata",
        lambda text: {
            "vendor": "Vendor C",
            "invoice_number": "",
            "date": "",
            "amount": "30.00",
        },
    )

    cfg = {
        "excel_log": [
            {
                "vendor": "Vendor C",
                "invoice_number": "C-001",
                "date": "2024-03-01",
                "amount": "30.00",
            }
        ]
    }

    manifest = pdf_segmenter.segment(
        "dummy.pdf", cfg, classifier=pdf_segmenter.HeuristicClassifier()
    )

    assert manifest[0]["invoice_number"] == "C-001"
    assert manifest[0]["date"] == "2024-03-01"


def test_large_pdf_performance(patch_pdf_reader, fake_page, monkeypatch):
    pages = [fake_page(f"Vendor A\nInvoice #{i}\nTotal $1.00") for i in range(60)]
    patch_pdf_reader(pages)

    monkeypatch.setattr(
        segmenters.ai_gemini, "detect_invoice_starts", lambda t: list(range(0, 60))
    )

    def fake_extract(text: str) -> dict:
        num = text.split("#")[1].split("\n")[0]
        return {
            "vendor": "Vendor A",
            "invoice_number": num,
            "date": "2024-01-01",
            "amount": "1.00",
        }

    monkeypatch.setattr(segmenters.ai_gemini, "extract_metadata", fake_extract)

    metrics = {}
    start = time.perf_counter()
    manifest = pdf_segmenter.segment(
        "dummy.pdf",
        {"pdf": {}},
        metrics=metrics,
        classifier=pdf_segmenter.HeuristicClassifier(),
    )
    duration = time.perf_counter() - start

    assert len(manifest) == 60
    assert metrics.get("total_pages") == 60
    assert duration < 2
