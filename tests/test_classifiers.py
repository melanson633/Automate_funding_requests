from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from cre_advance import ai_gemini
from cre_advance.classifiers import GeminiClassifier, HeuristicClassifier


@pytest.mark.parametrize(
    "text",
    [
        "Invoice Register\nWorkflow Approval",
        "INVOICE register workflow APPROVAL",
        "invoice    register for the WORKFLOW approval",
    ],
)
def test_invoice_register_detection(text) -> None:
    clf = HeuristicClassifier()
    res = clf.classify([text], {})
    assert res[0]["category"] == "invoice_register"
    assert res[0]["keep"] is False


@pytest.mark.parametrize(
    "text",
    [
        "From: Bob\nSent: Monday\nSubject: Payment",
        "subject: hi\nFROM: alice\nSent: Tue",
        "FROM: Carol\nTo: Dave\nSUBJECT: Hello",
    ],
)
def test_email_approval_detection(text) -> None:
    clf = HeuristicClassifier()
    res = clf.classify([text], {})
    assert res[0]["category"] == "email_approval"
    assert res[0]["keep"] is False


@pytest.mark.parametrize(
    "text",
    [
        "Invoice Packet Cover Sheet",
        "COVER SHEET",
        "   \n   ",
    ],
)
def test_blank_cover_detection(text) -> None:
    clf = HeuristicClassifier()
    res = clf.classify([text], {})
    assert res[0]["category"] == "blank_cover"
    assert res[0]["keep"] is False


def test_vendor_invoice_detection() -> None:
    clf = HeuristicClassifier()
    cfg = {"vendors": ["Acme Corp", "Foo LLC"]}
    text = "This is an invoice from ACME CORP for services"
    res = clf.classify([text], cfg)
    assert res[0]["category"] == "invoice"
    assert res[0]["keep"] is True


def test_gemini_classifier_batched(monkeypatch):
    clf = GeminiClassifier()
    seen: list[str] = []

    async def fake_async_generate_content(prompts, cfg, concurrency_limit=None):
        seen.extend([p[1]["parts"][0] for p in prompts])
        results = []
        for p in prompts:
            text = p[1]["parts"][0]
            if "\n---\n" in text:
                pages = text.split("\n---\n")
            else:
                pages = [seg.strip() for seg in text.split("Page") if seg.strip()]
            results.append(ai_gemini.classify_pages(pages, cfg))
        return results

    monkeypatch.setattr(
        ai_gemini, "async_generate_content", fake_async_generate_content
    )

    pages = [["Invoice", "Bill To", "Other"]]
    cfg = {"batch_size": 2}
    res = asyncio.run(clf.classify_async(pages, cfg))
    assert len(res[0]) == 3
    assert "Invoice" in seen[0] and "Bill To" in seen[0]
    assert "Other" in seen[1]
