from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
from cre_advance import file_packager  # noqa: E402


def test_match_invoices_order() -> None:
    df = pd.DataFrame(
        {
            "invoice_number": ["1", "2"],
            "vendor": ["A", "B"],
            "amount": [100.0, 200.0],
            "date": ["2024-01-01", "2024-02-01"],
        }
    )
    manifest = [
        {
            "invoice_number": "2",
            "vendor": "B",
            "amount": 200.0,
            "date": "2024-02-01",
            "start_page": 2,
            "end_page": 2,
        },
        {
            "invoice_number": "1",
            "vendor": "A",
            "amount": 100.0,
            "date": "2024-01-01",
            "start_page": 1,
            "end_page": 1,
        },
    ]

    ordered, unmatched_rows, unmatched_pdf = file_packager._match_invoices(df, manifest)

    assert [m["invoice_number"] for m in ordered] == ["1", "2"]
    assert unmatched_rows == []
    assert unmatched_pdf == []


def test_match_invoices_adjustable_scoring() -> None:
    df = pd.DataFrame(
        {
            "invoice_number": [""],
            "vendor": ["Acme Corp"],
            "amount": [100.0],
            "date": [""],
        }
    )
    manifest = [
        {
            "invoice_number": "",
            "vendor": "Acme Corporation",
            "amount": 100.02,
            "date": "",
            "start_page": 1,
            "end_page": 1,
        }
    ]

    ordered, unmatched_rows, _ = file_packager._match_invoices(df, manifest)
    assert unmatched_rows == [0]

    cfg = {
        "packager": {
            "vendor_ratio_threshold": 0.7,
            "amount_tolerance": 0.05,
            "score_threshold": 1.0,
        }
    }
    ordered2, unmatched_rows2, _ = file_packager._match_invoices(df, manifest, cfg)
    assert unmatched_rows2 == []
    assert [m["start_page"] for m in ordered2] == [1]


def test_package_orders_pdf(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "invoice_number": ["1", "2"],
            "vendor": ["A", "B"],
            "amount": [100.0, 200.0],
            "date": ["2024-01-01", "2024-02-01"],
        }
    )
    manifest = [
        {
            "invoice_number": "2",
            "vendor": "B",
            "amount": 200.0,
            "date": "2024-02-01",
            "start_page": 2,
            "end_page": 2,
        },
        {
            "invoice_number": "1",
            "vendor": "A",
            "amount": 100.0,
            "date": "2024-01-01",
            "start_page": 1,
            "end_page": 1,
        },
    ]

    pdf_order = []

    def fake_build_pdf(order, pdf, dest):
        pdf_order.extend([m["invoice_number"] for m in order])

    monkeypatch.setattr(file_packager, "_build_pdf", fake_build_pdf)
    monkeypatch.setattr(file_packager, "_write_excel", lambda *a, **k: None)

    summary = file_packager.package(
        df,
        manifest,
        "template.xlsx",
        "dummy.pdf",
        tmp_path,
        {"unmatched_threshold": 0.4},
    )

    assert pdf_order == ["1", "2"]
    assert summary["unmatched_rows"] == []


def test_package_warns_on_unmatched(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "invoice_number": ["1", "2"],
            "vendor": ["A", "B"],
            "amount": [100.0, 200.0],
            "date": ["2024-01-01", "2024-02-01"],
        }
    )
    manifest = [
        {
            "invoice_number": "1",
            "vendor": "A",
            "amount": 100.0,
            "date": "2024-01-01",
            "start_page": 1,
            "end_page": 1,
        }
    ]

    monkeypatch.setattr(file_packager, "_build_pdf", lambda *a, **k: None)
    monkeypatch.setattr(file_packager, "_write_excel", lambda *a, **k: None)

    summary = file_packager.package(
        df,
        manifest,
        "template.xlsx",
        "dummy.pdf",
        tmp_path,
        {"unmatched_threshold": 0.0},
    )

    assert len(summary["unmatched_rows"]) == 1
    assert summary["warning"] is not None
