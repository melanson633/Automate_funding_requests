from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pandas as pd
import yaml
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.modules.setdefault("google", types.ModuleType("google"))  # noqa: E402
genai_stub = types.ModuleType("generativeai")
genai_stub.types = types.SimpleNamespace()
sys.modules.setdefault("google.generativeai", genai_stub)  # noqa: E402
from cre_advance import excel_normalizer  # noqa: E402


def _create_workbook(path: Path, header_row: int = 4) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Driver"
    for _ in range(header_row - 1):
        ws.append([])
    ws.append(["Date", "Amount"])
    ws.append(["2024-01-01", "100"])
    ws.append(["2024-02-01", "(200)"])
    wb.save(path)


def test_normalize_basic(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        excel_path = Path(tmp) / "test.xlsx"
        _create_workbook(excel_path)

        def fake_map(headers, samples, fields, cfg=None):
            return {"Date": "Date", "Amount": "Amount"}

        def fake_build(headers, samples, cfg=None):
            return {}

        monkeypatch.setattr(
            excel_normalizer.ai_gemini,
            "map_schema",
            fake_map,
        )
        monkeypatch.setattr(
            excel_normalizer.ai_gemini,
            "build_schema",
            fake_build,
        )

        cfg = {
            "lender": "example_lender",
            "excel": {"fields": ["Date", "Amount"]},
        }
        normalized, raw = excel_normalizer.normalize([excel_path], cfg)

        assert list(normalized.columns) == ["Date", "Amount"]
        assert pd.api.types.is_datetime64_any_dtype(normalized["Date"])
        assert normalized.loc[1, "Amount"] == -200.0
        assert not raw.empty


def test_header_row_config(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        excel_path = Path(tmp) / "hdr.xlsx"
        _create_workbook(excel_path, header_row=3)

        monkeypatch.setattr(
            excel_normalizer.ai_gemini,
            "map_schema",
            lambda h, s, f, cfg=None: {"Date": "Date", "Amount": "Amount"},
        )
        monkeypatch.setattr(
            excel_normalizer.ai_gemini, "build_schema", lambda h, s, cfg=None: {}
        )

        cfg = {"lender": "x", "excel": {"fields": ["Date", "Amount"], "header_row": 3}}
        normalized, _ = excel_normalizer.normalize([excel_path], cfg)

        assert list(normalized.columns) == ["Date", "Amount"]


def test_auto_save_schema(monkeypatch, tmp_path) -> None:
    excel_path = tmp_path / "auto.xlsx"
    _create_workbook(excel_path)

    monkeypatch.setattr(
        excel_normalizer.ai_gemini,
        "map_schema",
        lambda h, s, f, cfg=None: {},
    )

    monkeypatch.setattr(
        excel_normalizer.ai_gemini,
        "build_schema",
        lambda h, s, cfg=None: {
            "mapping": {"Date": "InvoiceDate"},
            "fields": ["InvoiceDate"],
        },
    )

    cfg = {"lender": "example_lender", "excel": {"fields": ["InvoiceDate"]}}

    version_dir = Path(__file__).resolve().parents[1] / "configs" / "schema_versions"
    if version_dir.exists():
        for f in version_dir.glob("example_lender_*.yaml"):
            f.unlink()

    normalized, _ = excel_normalizer.normalize([excel_path], cfg)

    files = list(version_dir.glob("example_lender_*.yaml"))
    assert len(files) == 1
    assert "InvoiceDate" in normalized.columns
    with files[0].open() as f:
        data = yaml.safe_load(f)
    assert data["excel"]["mapping"]["Date"] == "InvoiceDate"
    files[0].unlink()


def test_fuzzy_fallback(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        excel_path = Path(tmp) / "fuzzy.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "Driver"
        ws.append([])
        ws.append([])
        ws.append([])
        ws.append(["Date", "Amt"])
        ws.append(["2024-01-01", "100"])
        wb.save(excel_path)

        monkeypatch.setattr(
            excel_normalizer.ai_gemini,
            "map_schema",
            lambda h, s, f, cfg=None: {},
        )
        monkeypatch.setattr(
            excel_normalizer.ai_gemini, "build_schema", lambda h, s, cfg=None: {}
        )

        cfg = {
            "lender": "x",
            "excel": {"fields": ["Date", "Amount"], "fuzzy_ratio": 0.5},
        }

        normalized, _ = excel_normalizer.normalize([excel_path], cfg)

        assert list(normalized.columns) == ["Date", "Amount"]
