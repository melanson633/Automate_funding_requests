from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.modules.setdefault("google", types.ModuleType("google"))  # noqa: E402
genai_stub = types.ModuleType("generativeai")
genai_stub.types = types.SimpleNamespace()
sys.modules.setdefault("google.generativeai", genai_stub)  # noqa: E402
from cre_advance import excel_normalizer  # noqa: E402


def _create_workbook(path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Driver"
    for _ in range(3):
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
