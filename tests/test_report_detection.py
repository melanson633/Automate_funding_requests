from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from openpyxl import Workbook

from cre_advance import excel_normalizer


@pytest.mark.parametrize(
    "sheet_name, header_row, expected",
    [
        ("Report1", 6, "general_ledger"),
        ("Expense Distribution Report", 3, "expense_distribution"),
        ("DRIVER", 4, "funding_template"),
    ],
)
def test_detect_report_type_known(
    sheet_name: str, header_row: int, expected: str
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wb.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name
        for _ in range(header_row - 1):
            ws.append([])
        ws.append(["A", "B"])
        wb.save(path)

        result = excel_normalizer.detect_report_type(path)
        assert result == {
            "type": expected,
            "sheet_name": sheet_name,
            "header_row": header_row,
        }


def test_detect_report_type_unknown(tmp_path: Path) -> None:
    path = tmp_path / "unknown.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "MySheet"
    ws.append([])
    ws.append(["Col1", "Col2"])
    ws.append([1, 2])
    wb.save(path)

    result = excel_normalizer.detect_report_type(path)
    assert result["type"] == "unknown"
    assert result["sheet_name"] == "MySheet"
    assert result["header_row"] == 2


def test_detect_report_type_no_confidence(tmp_path: Path) -> None:
    path = tmp_path / "blank.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    wb.save(path)

    result = excel_normalizer.detect_report_type(path)
    assert result == {"type": "unknown", "sheet_name": None, "header_row": None}
