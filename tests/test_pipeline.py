from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
import json
from cre_advance import pipeline  # noqa: E402


def test_run_orchestrates(monkeypatch):
    called = {}

    def fake_get_config(lender):
        called["config"] = lender
        return {}

    def fake_normalize(yardi, cfg, metrics=None):
        called["normalize"] = list(yardi)
        df = pd.DataFrame({"a": [1]})
        return df, df

    def fake_segment(pdf, cfg, metrics=None):
        called["segment"] = pdf
        return ["manifest"]

    def fake_package(df, manifest, template, pdf, output, cfg, metrics=None):
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
        resume=False,
        normalized=None,
        manifest=None,
    )

    summary = pipeline.run(args)

    assert called["config"] == "l1"
    assert called["normalize"] == ["a.xlsx", "b.xlsx"]
    assert called["segment"] == "invoices.pdf"
    assert called["package"] == ("template.xlsx", "invoices.pdf", "out")
    assert summary["excel"] == "x.xlsx"


def test_segment_failure_persists_df(monkeypatch, tmp_path):
    df = pd.DataFrame({"a": [1]})

    monkeypatch.setattr(pipeline, "get_config", lambda l: {})
    monkeypatch.setattr(
        pipeline,
        "excel_normalizer",
        types.SimpleNamespace(normalize=lambda y, c, metrics=None: (df, df)),
    )

    def fail_segment(pdf, cfg, metrics=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        pipeline, "pdf_segmenter", types.SimpleNamespace(segment=fail_segment)
    )

    args = types.SimpleNamespace(
        excel="t.xlsx",
        yardi=["y.xlsx"],
        pdf="p.pdf",
        lender="l",
        output=tmp_path,
        resume=False,
        normalized=None,
        manifest=None,
    )

    staging = Path("data/staging")
    if staging.exists():
        for f in staging.iterdir():
            f.unlink()
    staging.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError):
        pipeline.run(args)

    assert any(staging.glob("normalized_*.xlsx"))


def test_resume_skips_ai(monkeypatch, tmp_path):
    df = pd.DataFrame({"a": [1]})
    manifest = ["m"]

    staging = Path("data/staging")
    staging.mkdir(parents=True, exist_ok=True)
    norm_file = staging / "normalized_test.xlsx"
    manifest_file = staging / "manifest_test.json"
    df.to_excel(norm_file, index=False)
    manifest_file.write_text(json.dumps(manifest))

    called = {}

    monkeypatch.setattr(pipeline, "get_config", lambda l: {})
    monkeypatch.setattr(
        pipeline,
        "excel_normalizer",
        types.SimpleNamespace(
            normalize=lambda y, c, metrics=None: called.setdefault("norm", True)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "pdf_segmenter",
        types.SimpleNamespace(
            segment=lambda p, c, metrics=None: called.setdefault("seg", True)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "file_packager",
        types.SimpleNamespace(package=lambda *a, metrics=None, **k: {"excel": "x"}),
    )

    args = types.SimpleNamespace(
        excel="t.xlsx",
        yardi=["y.xlsx"],
        pdf="p.pdf",
        lender="l",
        output=tmp_path,
        resume=True,
        normalized=str(norm_file),
        manifest=str(manifest_file),
    )

    summary = pipeline.run(args)

    assert "norm" not in called and "seg" not in called
    assert summary["excel"] == "x"
