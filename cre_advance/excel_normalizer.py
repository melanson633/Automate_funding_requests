from __future__ import annotations

"""Utilities for normalising Yardi Excel exports."""

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import yaml

from . import ai_gemini
from .utils.errors import NormalizationError
from .utils.logging import get_logger

logger = get_logger(__name__)


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
                / "lenders"
                / f"{lender}.yaml"
            )
            prompt = f"Write new schema to {dest}? [y/N] "
            answer = input(prompt).strip().lower()
            if answer == "y":
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("w") as f:
                    yaml.safe_dump({"excel": proposal}, f)

    normalized = raw_df.rename(columns=mapping)

    missing = [f for f in target_fields if f not in normalized.columns]
    unmatched_threshold = float(cfg.get("unmatched_threshold", 0.4))
    if target_fields and len(missing) / len(target_fields) >= unmatched_threshold:
        raise NormalizationError("Too many columns unmapped")

    normalized = _apply_casts(normalized)
    return normalized, raw_df
