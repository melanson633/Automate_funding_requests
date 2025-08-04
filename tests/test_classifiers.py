from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from cre_advance.classifiers import HeuristicClassifier


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
