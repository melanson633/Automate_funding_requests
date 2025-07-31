from __future__ import annotations

"""Utilities for normalising Yardi Excel exports."""

import difflib
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import yaml

from . import ai_gemini
from .utils.errors import NormalizationError
from .utils.logging import get_logger

logger = get_logger(__name__)


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


def _read_workbook(path: Path, header_row: int) -> pd.DataFrame:
    """Return DataFrame from the Driver sheet using ``header_row``."""
    df = pd.read_excel(
        path,
        sheet_name="Driver",
        header=None,
        engine="openpyxl",
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized and raw DataFrames from Yardi workbooks.

    Parameters
    ----------
    workbooks:
        One or more Excel file paths.
    cfg:
        Configuration dictionary. Keys under ``excel`` include ``header_row``
        (default ``4``), ``force_schema_builder`` and ``fields``. The lender
        name should be stored in ``cfg['lender']``.

    Raises
    ------
    NormalizationError
        If more than 40% of required columns are unmapped.
    """
    wb_paths = (
        [Path(workbooks)]
        if isinstance(workbooks, str)
        else [Path(p) for p in workbooks]
    )
    excel_cfg = cfg.get("excel", {})
    header_row = int(excel_cfg.get("header_row", 4))

    frames = [_read_workbook(p, header_row) for p in wb_paths]
    raw_df = pd.concat(frames, ignore_index=True)

    headers = list(raw_df.columns)
    samples = raw_df.head(5).to_dict(orient="records")
    target_fields = excel_cfg.get("fields", [])

    mapping = ai_gemini.map_schema(headers, samples, target_fields, cfg)
    coverage = len(mapping) / len(headers) if headers else 0
    mapping_threshold = float(cfg.get("mapping_coverage_threshold", 0.6))
    if coverage < mapping_threshold or excel_cfg.get("force_schema_builder"):
        proposal = ai_gemini.build_schema(headers, samples, cfg)
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
                    "Low mapping coverage or schema builder forced. Saved inferred schema to %s. Review this file.",
                    dest,
                )
            else:
                logger.warning(
                    "Low mapping coverage or schema builder forced but auto_save_schemas is disabled."
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

    normalized = _apply_casts(normalized)
    return normalized, raw_df
