from __future__ import annotations

"""Utilities for normalising Yardi Excel exports."""

import difflib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import yaml

from . import ai_gemini
from .metrics import log_metric
from .utils.errors import NormalizationError
from .utils.logging import get_logger

logger = get_logger(__name__)


def _analyze_sheet_structure(df: pd.DataFrame, sheet_name: str) -> dict:
    """Return a guessed header row and confidence for ``sheet_name``.

    The function inspects the first 10 rows of ``df`` (which should be loaded
    without headers) and scores each row based on:

    * Density of non-empty cells
    * Ratio of text to numeric values (header rows are mostly text)
    * Uniqueness of values within the row (headers rarely repeat)

    Parameters
    ----------
    df:
        DataFrame representing the sheet with no headers applied.
    sheet_name:
        Name of the sheet being analysed.

    Returns
    -------
    dict
        ``{"sheet_name": str, "header_row": int | None, "confidence": float}``
        where ``header_row`` is 1-indexed.
    """
    if df.empty:
        return {"sheet_name": sheet_name, "header_row": None, "confidence": 0.0}

    best_row = None
    best_score = 0.0
    max_rows = min(10, len(df))
    for idx in range(max_rows):
        row = df.iloc[idx]
        non_empty = row.dropna()
        non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
        non_empty_count = len(non_empty)
        if non_empty_count == 0:
            continue

        numeric_count = pd.to_numeric(non_empty, errors="coerce").notna().sum()
        text_count = non_empty_count - numeric_count
        text_ratio = text_count / non_empty_count
        unique_ratio = non_empty.nunique() / non_empty_count
        non_empty_ratio = non_empty_count / max(len(row), 1)
        score = non_empty_ratio * text_ratio * unique_ratio

        if score > best_score:
            best_score = score
            best_row = idx + 1  # Convert to 1-indexed

    return {"sheet_name": sheet_name, "header_row": best_row, "confidence": best_score}


def detect_report_type(file_path: Path) -> dict:
    """Detect the report type and header row for a workbook.

    The function first checks for known Yardi report sheet names.  If none are
    found, each sheet is analysed heuristically to guess the most likely header
    row.  When no confident match is found, ``sheet_name`` and ``header_row``
    are returned as ``None``.

    Parameters
    ----------
    file_path:
        Path to the Excel workbook.

    Returns
    -------
    dict
        ``{"type": str, "sheet_name": str | None, "header_row": int | None}``
        where ``type`` is one of ``general_ledger``, ``expense_distribution``,
        ``funding_template`` or ``unknown``.
    """
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    sheets = xls.sheet_names

    if "Report1" in sheets:
        return {"type": "general_ledger", "sheet_name": "Report1", "header_row": 6}
    if "Expense Distribution Report" in sheets:
        return {
            "type": "expense_distribution",
            "sheet_name": "Expense Distribution Report",
            "header_row": 3,
        }
    if "DRIVER" in sheets:
        return {"type": "funding_template", "sheet_name": "DRIVER", "header_row": 4}

    best = {"confidence": 0.0, "sheet_name": None, "header_row": None}
    for sheet in sheets:
        df = xls.parse(sheet, header=None, nrows=10)
        info = _analyze_sheet_structure(df, sheet)
        if info["confidence"] > best["confidence"]:
            best = info

    if best["confidence"] >= 0.6 and best["sheet_name"]:
        return {
            "type": "unknown",
            "sheet_name": best["sheet_name"],
            "header_row": best["header_row"],
        }
    return {"type": "unknown", "sheet_name": None, "header_row": None}


def _fuzzy_match(
    headers: list[str], target_fields: list[str], threshold: float
) -> dict:
    """Return heuristic header → field mapping using simple string similarity."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for header in headers:
        best_field = None
        best_score = 0.0
        for field in target_fields:
            if field in used:
                continue
            score = difflib.SequenceMatcher(None, header.lower(), field.lower()).ratio()
            if score > best_score:
                best_score = score
                best_field = field
        if best_field and best_score >= threshold:
            mapping[header] = best_field
            used.add(best_field)
    return mapping


def _read_workbook(path: Path, sheet_name: str, header_row: int) -> pd.DataFrame:
    """Return DataFrame from ``sheet_name`` using ``header_row``.

    The function validates that ``sheet_name`` exists and that the provided
    ``header_row`` contains mostly non-empty cells.  If the row is sparsely
    populated, the first sufficiently populated row is used instead and a
    warning is logged.
    """

    xls = pd.ExcelFile(path, engine="openpyxl")
    if sheet_name not in xls.sheet_names:
        raise ValueError(f"Sheet '{sheet_name}' not found in '{path}'")

    df = xls.parse(sheet_name, header=None)

    def _non_empty_ratio(row: pd.Series) -> float:
        non_empty = row.dropna().astype(str).str.strip()
        non_empty = non_empty[non_empty != ""]
        return len(non_empty) / max(len(row), 1)

    ratio = _non_empty_ratio(df.iloc[header_row - 1])
    if ratio < 0.5:
        logger.warning(
            "Header row %s in sheet '%s' is mostly empty; falling back to first non-empty row",
            header_row,
            sheet_name,
            extra={"context": {"file": str(path)}},
        )
        for idx, row in df.iterrows():
            if _non_empty_ratio(row) >= 0.5:
                header_row = idx + 1
                break
        ratio = _non_empty_ratio(df.iloc[header_row - 1])
        if ratio < 0.5:
            raise ValueError(
                f"No suitable header row found in sheet '{sheet_name}' of '{path}'"
            )

    headers = df.iloc[header_row - 1].tolist()
    data = df.iloc[header_row:]
    data.columns = headers
    return data


def _apply_casts(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and cast date and amount columns."""
    df = df.dropna(axis=1, how="all")
    for col in df.columns:
        lower = col.lower()
        if "date" in lower:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        if "amount" in lower:
            series = df[col].astype(str)
            series = series.str.replace(",", "", regex=False)
            series = series.str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
            df[col] = pd.to_numeric(series, errors="coerce")
    return df


def normalize(
    workbooks: Iterable[str] | str,
    cfg: dict,
    metrics: Dict[str, Any] | None = None,
    template_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized and raw DataFrames from Yardi workbooks.

    Parameters
    ----------
    workbooks:
        One or more Excel file paths.
    cfg:
        Configuration dictionary. Keys under ``excel`` include ``header_row``
        (default ``4``), ``sheet_name``, ``force_sheet_detection`` and
        ``fields``. The lender name should be stored in ``cfg['lender']``.
        If ``force_sheet_detection`` is true (default), each workbook is
        analysed via :func:`detect_report_type` to determine the appropriate
        sheet and header row.  Overrides for specific report types can be
        provided under ``excel['overrides']``.

    Raises
    ------
    NormalizationError
        If more than 40% of required columns are unmapped.
    """
    t0 = time.perf_counter()
    wb_paths = (
        [Path(workbooks)]
        if isinstance(workbooks, str)
        else [Path(p) for p in workbooks]
    )
    logger.info(
        "Starting Excel normalization",
        extra={"context": {"file": str(wb_paths[0])}},
    )
    excel_cfg = cfg.get("excel", {})
    force_detection = excel_cfg.get("force_sheet_detection", True)
    default_header = int(excel_cfg.get("header_row", 4))
    default_sheet = excel_cfg.get("sheet_name", "Driver")

    frames_by_type: dict[str, list[pd.DataFrame]] = {}
    detections: list[dict[str, Any]] = []

    for path in wb_paths:
        report_type = "unknown"
        sheet = default_sheet
        header = default_header

        if force_detection:
            detection = detect_report_type(path)
            report_type = detection.get("type", "unknown")
            if report_type != "unknown":
                sheet = detection.get("sheet_name") or default_sheet
                header = int(detection.get("header_row") or default_header)
            else:
                sheet = default_sheet
                header = default_header

        overrides = excel_cfg.get("overrides", {}).get(report_type, {})
        sheet = overrides.get("sheet_name", sheet)
        header = int(overrides.get("header_row", header))

        df = _read_workbook(path, sheet, header)
        frames_by_type.setdefault(report_type, []).append(df)
        detections.append(
            {
                "file": str(path),
                "type": report_type,
                "sheet_name": sheet,
                "header_row": header,
            }
        )
        logger.info(
            "Workbook loaded",
            extra={
                "context": {
                    "file": str(path),
                    "type": report_type,
                    "sheet": sheet,
                    "header_row": header,
                }
            },
        )

    frames = [pd.concat(dfs, ignore_index=True) for dfs in frames_by_type.values()]
    raw_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if metrics is not None:
        metrics["report_detections"] = detections

    headers = list(raw_df.columns)
    samples = raw_df.head(5).to_dict(orient="records")
    target_fields = excel_cfg.get("fields", [])

    # Try AI mapping first, fallback to manual mapping if it fails
    manual_mapping = excel_cfg.get("manual_mapping", {})
    mapping = ai_gemini.map_headers(headers, samples, target_fields)

    # If AI mapping failed, use manual mapping
    if not mapping and manual_mapping:
        logger.info(
            "AI mapping failed, using manual mapping", extra={"context": "normalize"}
        )
        mapping = {k: v for k, v in manual_mapping.items() if k in headers}

    logger.debug("Initial mapping: %s", mapping, extra={"context": "normalize"})
    coverage = len(mapping) / len(headers) if headers else 0
    if metrics is not None:
        metrics["mapping_coverage"] = coverage
        metrics["total_columns"] = len(headers)
    mapping_threshold = float(cfg.get("mapping_coverage_threshold", 0.6))
    if coverage < mapping_threshold or excel_cfg.get("force_schema_builder"):
        logger.info("Generating schema via Gemini", extra={"context": "normalize"})
        proposal = ai_gemini.build_schema(headers, samples)
        mapping = proposal.get("mapping", {})
        if proposal:
            lender = cfg.get("lender", "unknown")
            dest = (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "schema_versions"
                / f"{lender}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.yaml"
            )
            if cfg.get("auto_save_schemas", True):
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("w") as f:
                    yaml.safe_dump({"excel": proposal}, f)
                logger.warning(
                    (
                        "Low mapping coverage or schema builder forced. "
                        "Saved inferred schema to %s. Review this file."
                    ),
                    dest,
                )
            else:
                logger.warning(
                    "Low mapping coverage or schema builder forced but "
                    "auto_save_schemas is disabled."
                )

    coverage = len(mapping) / len(headers) if headers else 0
    if coverage < mapping_threshold:
        fuzzy_ratio = float(excel_cfg.get("fuzzy_ratio", 0.6))
        unmapped = [h for h in headers if h not in mapping]
        mapping.update(_fuzzy_match(unmapped, target_fields, fuzzy_ratio))

    normalized = raw_df.rename(columns=mapping)

    missing = [f for f in target_fields if f not in normalized.columns]
    unmatched_threshold = float(cfg.get("unmatched_threshold", 0.4))
    if target_fields and len(missing) / len(target_fields) >= unmatched_threshold:
        raise NormalizationError("Too many columns unmapped")
    if metrics is not None:
        metrics["unmatched_columns"] = len(missing)
        metrics["total_fields"] = len(target_fields)

    normalized = _apply_casts(normalized)

    # Filter previously funded invoices if template provided (after normalization)
    if template_path and cfg.get("filter_funded", True):
        logger.info(
            "Filtering previously funded invoices", extra={"context": "normalize"}
        )
        template_cfg = cfg.get("template", {})
        template_df = _read_workbook(
            Path(template_path),
            template_cfg.get("sheet_name", "INVOICES"),
            template_cfg.get("header_row", 6),
        )

        # Apply template column mapping if provided
        template_column_mapping = template_cfg.get("column_mapping", {})
        if template_column_mapping:
            template_df = template_df.rename(columns=template_column_mapping)

        # Check for both old and new column names for compatibility
        vendor_col = "vendor_name" if "vendor_name" in normalized.columns else "vendor"
        invoice_col = "invoice_number"

        if (
            not template_df.empty
            and invoice_col in normalized.columns
            and vendor_col in normalized.columns
        ):
            # Create composite key for exact matching
            normalized["_key"] = (
                normalized[invoice_col].astype(str).str.strip()
                + "|"
                + normalized[vendor_col].astype(str).str.strip()
            )
            template_df["_key"] = (
                template_df[invoice_col].astype(str).str.strip()
                + "|"
                + template_df[vendor_col].astype(str).str.strip()
            )

            # Capture excluded invoices before filtering
            excluded_invoices = normalized[normalized["_key"].isin(template_df["_key"])]

            # Filter using pandas set operations (super fast!)
            before_count = len(normalized)
            normalized = normalized[~normalized["_key"].isin(template_df["_key"])].drop(
                "_key", axis=1
            )
            filtered_count = before_count - len(normalized)

            if filtered_count > 0:
                logger.info(
                    f"Filtered {filtered_count} previously funded invoices",
                    extra={"context": {"file": str(wb_paths[0])}},
                )
                if metrics is not None:
                    metrics["filtered_invoices"] = filtered_count
                    excluded_data = excluded_invoices[
                        ["invoice_number", vendor_col, "amount"]
                    ].to_dict("records")
                    metrics["excluded_invoices"] = excluded_data
                    excluded_summary = [
                        f"{item['invoice_number']}|{item[vendor_col]}|${item['amount']}"
                        for item in excluded_data
                    ]
                    logger.info(
                        f"Excluded invoice details: {excluded_summary}",
                        extra={"context": {"file": str(wb_paths[0])}},
                    )

    duration = time.perf_counter() - t0
    if metrics is not None:
        metrics["processing_seconds"] = duration
        for key, value in metrics.items():
            log_metric(f"excel_{key}", value, tags={"file": str(wb_paths[0])})
    logger.info(
        "Finished Excel normalization",
        extra={"context": {"file": str(wb_paths[0])}},
    )
    return normalized, raw_df
