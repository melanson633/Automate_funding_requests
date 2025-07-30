from __future__ import annotations

"""Package funding request outputs.

This module aligns normalized Driver rows with segmented invoice pages,
building the final Excel workbook, reordered PDF, and a JSON report.
"""

from pathlib import Path
from typing import List

import json

import pandas as pd
from Levenshtein import ratio
from openpyxl import load_workbook
from pypdf import PdfReader, PdfWriter

from .utils.errors import NormalizationError
from .utils.logging import get_logger

logger = get_logger(__name__)


def _match_invoices(df: pd.DataFrame, manifest: List[dict]) -> tuple[List[dict], list[int]]:
    """Return invoices ordered to ``df`` rows and list of unmatched row indices."""
    used: set[int] = set()
    ordered: List[dict] = []
    unmatched: list[int] = []

    for idx, row in df.iterrows():
        row_num = str(row.get("invoice_number", "")).strip()
        row_vendor = str(row.get("vendor", "")).strip().lower()
        row_amt = float(row.get("amount", 0))
        row_date = (
            str(pd.to_datetime(row.get("date")).date()) if row.get("date") else ""
        )

        best = None
        best_score = 0.0
        for m_idx, item in enumerate(manifest):
            if m_idx in used:
                continue
            inv_num = str(item.get("invoice_number", "")).strip()
            inv_vendor = str(item.get("vendor", "")).strip().lower()
            inv_amt = float(item.get("amount", 0))
            inv_date = (
                str(pd.to_datetime(item.get("date")).date()) if item.get("date") else ""
            )

            score = 0.0
            if row_num and inv_num and row_num == inv_num:
                score += 3.0
            if row_vendor and inv_vendor:
                v_ratio = ratio(row_vendor, inv_vendor)
                if v_ratio >= 0.8:
                    score += v_ratio
            if abs(row_amt - inv_amt) < 0.01:
                score += 1.0
            if row_date and inv_date and row_date == inv_date:
                score += 1.0

            if score > best_score:
                best_score = score
                best = (m_idx, item)

        if best is not None and best_score >= 2.0:
            used.add(best[0])
            ordered.append(best[1])
        else:
            unmatched.append(idx)

    return ordered, unmatched


def _build_pdf(manifest: List[dict], pdf_path: Path, dest: Path) -> None:
    """Reorder pages from ``pdf_path`` according to ``manifest``."""
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for item in manifest:
        start = int(item.get("start_page", 1)) - 1
        end = int(item.get("end_page", start + 1)) - 1
        for page in reader.pages[start : end + 1]:
            writer.add_page(page)
    with dest.open("wb") as f:
        writer.write(f)


def _write_excel(
    driver_df: pd.DataFrame, invoice_df: pd.DataFrame, template: Path, dest: Path
) -> None:
    """Create workbook with hidden Driver and Invoice Log."""
    wb = load_workbook(template)
    if "Driver" not in wb.sheetnames:
        wb.create_sheet("Driver")
    if "Invoice Log" in wb.sheetnames:
        del wb["Invoice Log"]
    wb.create_sheet("Invoice Log")

    for row in driver_df.itertuples(index=False, name=None):
        wb["Driver"].append(list(row))
    wb["Driver"].sheet_state = "hidden"

    for row in invoice_df.itertuples(index=False, name=None):
        wb["Invoice Log"].append(list(row))

    if "amount" in invoice_df.columns:
        total = invoice_df["amount"].sum()
        wb["Invoice Log"].append(["Total", total])

    wb.save(dest)


def package(
    normalized_df: pd.DataFrame,
    manifest: List[dict],
    template_path: str | Path,
    pdf_path: str | Path,
    output_dir: str | Path,
    cfg: dict | None = None,
) -> dict:
    """Build funding package and return paths to artefacts."""
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered, unmatched = _match_invoices(normalized_df, manifest)
    unmatched_threshold = float(cfg.get("unmatched_threshold", 0.4))
    if len(normalized_df.index) and len(unmatched) / len(normalized_df.index) > unmatched_threshold:
        raise NormalizationError("Too many unmatched invoices")

    ordered_df = normalized_df.drop(index=unmatched).reset_index(drop=True)

    pdf_out = output_dir / "invoices.pdf"
    _build_pdf(ordered, Path(pdf_path), pdf_out)

    excel_out = output_dir / "request.xlsx"
    _write_excel(normalized_df, ordered_df, Path(template_path), excel_out)

    report = {
        "total_rows": len(normalized_df.index),
        "unmatched_rows": unmatched,
        "output_excel": str(excel_out),
        "output_pdf": str(pdf_out),
    }
    report_out = output_dir / "report.json"
    report_out.write_text(json.dumps(report, indent=2))

    return {
        "excel": str(excel_out),
        "pdf": str(pdf_out),
        "report": str(report_out),
        "unmatched_rows": unmatched,
    }
