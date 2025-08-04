from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import google.api_core  # noqa: F401
import pytest

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

from cre_advance import segmenters  # noqa: E402


def test_multi_page_invoice(monkeypatch) -> None:
    pages = [
        "Vendor A\nInvoice #1001\nTotal $100.00",
        "Page 2 of invoice 1001",
        "Vendor B\nInvoice #2002\nTotal $50.00",
    ]
    monkeypatch.setattr(segmenters.ai_gemini, "detect_invoice_starts", lambda t: [0, 2])

    calls = []

    def fake_extract(text: str) -> dict:
        calls.append(text)
        if "1001" in text:
            return {
                "vendor": "Vendor A",
                "invoice_number": "1001",
                "date": "2024-01-01",
                "amount": "100",
            }
        return {
            "vendor": "Vendor B",
            "invoice_number": "2002",
            "date": "2024-02-01",
            "amount": "50",
        }

    monkeypatch.setattr(segmenters.ai_gemini, "extract_metadata", fake_extract)

    seg = segmenters.InvoiceSegmenter()
    manifest = seg.segment_invoices(pages, {})

    assert [m["start_page"] for m in manifest] == [1, 3]
    assert manifest[0]["end_page"] == 2
    assert manifest[0]["vendor"] == "Vendor A"
    assert len(calls) == 2


def test_missing_metadata_reconciled(monkeypatch) -> None:
    pages = ["Invoice #3003\nTotal $300.00"]
    monkeypatch.setattr(segmenters.ai_gemini, "detect_invoice_starts", lambda t: [0])

    def fail_extract(text: str) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(segmenters.ai_gemini, "extract_metadata", fail_extract)
    cfg = {"excel_log": [{"invoice_number": "3003", "vendor": "Vendor C"}]}

    seg = segmenters.InvoiceSegmenter()
    manifest = seg.segment_invoices(pages, cfg)

    assert manifest[0]["invoice_number"] == "3003"
    assert manifest[0]["vendor"] == "Vendor C"
    assert manifest[0]["amount"] == "300.00"
    assert manifest[0]["date"] == ""


def test_segment_invoices_async(monkeypatch) -> None:
    seg = segmenters.InvoiceSegmenter()
    calls: list[list[str]] = []

    def fake_segment(self, texts, cfg):
        calls.append(texts)
        return [
            {
                "start_page": 1,
                "end_page": 1,
                "vendor": "",
                "invoice_number": "",
                "date": "",
                "amount": "",
                "confidence": 1.0,
            }
        ]

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(segmenters.InvoiceSegmenter, "segment_invoices", fake_segment)
    monkeypatch.setattr(segmenters.asyncio, "to_thread", fake_to_thread)

    res = asyncio.run(seg.segment_invoices_async([["a"], ["b"]], {}))
    assert len(res) == 2
    assert calls == [["a"], ["b"]]
