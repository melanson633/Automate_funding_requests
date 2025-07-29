from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cre_advance import pipeline  # noqa: E402


def test_run_orchestrates(monkeypatch):
    called = {}

    def fake_get_config(lender):
        called["config"] = lender
        return {}

    def fake_normalize(yardi, cfg):
        called["normalize"] = list(yardi)
        return "df", "raw"

    def fake_segment(pdf, cfg):
        called["segment"] = pdf
        return ["manifest"]

    def fake_package(df, manifest, template, pdf, output, cfg):
        called["package"] = (template, pdf, output)
        return {
            "excel": "x.xlsx",
            "pdf": "p.pdf",
            "report": "r.json",
            "unmatched_rows": [],
        }

    monkeypatch.setattr(pipeline, "get_config", fake_get_config)
    monkeypatch.setattr(
        pipeline,
        "excel_normalizer",
        types.SimpleNamespace(normalize=fake_normalize),
    )
    monkeypatch.setattr(
        pipeline, "pdf_segmenter", types.SimpleNamespace(segment=fake_segment)
    )
    monkeypatch.setattr(
        pipeline, "file_packager", types.SimpleNamespace(package=fake_package)
    )

    args = types.SimpleNamespace(
        excel="template.xlsx",
        yardi=["a.xlsx", "b.xlsx"],
        pdf="invoices.pdf",
        lender="l1",
        output="out",
    )

    summary = pipeline.run(args)

    assert called["config"] == "l1"
    assert called["normalize"] == ["a.xlsx", "b.xlsx"]
    assert called["segment"] == "invoices.pdf"
    assert called["package"] == ("template.xlsx", "invoices.pdf", "out")
    assert summary["excel"] == "x.xlsx"
