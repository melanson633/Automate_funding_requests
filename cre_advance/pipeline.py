from __future__ import annotations

"""Pipeline orchestrator for the CRE advance request process."""

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict
import json

from . import excel_normalizer, file_packager, pdf_segmenter
from .utils import get_config, get_logger

logger = get_logger(__name__)


def run(args: Namespace) -> Dict[str, Any]:
    """Run the normalization, segmentation and packaging phases.

    Parameters
    ----------
    args:
        Parsed command-line arguments with attributes ``excel``, ``yardi``,
        ``pdf``, ``lender`` and ``output``.

    Returns
    -------
    dict
        Mapping of artefact types to their output paths.
    """
    logger.info("Loading configuration for lender '%s'", args.lender)
    cfg = get_config(args.lender)

    staging_dir = Path("data/staging")
    staging_dir.mkdir(parents=True, exist_ok=True)

    normalized_df = None
    try:
        logger.info("Normalizing Excel workbooks")
        normalized_df, _ = excel_normalizer.normalize(args.yardi, cfg)
        normalized_df.to_excel(staging_dir / "Driver_clean.xlsx", index=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Excel normalization failed: %s", exc)
        raise

    manifest = None
    try:
        logger.info("Segmenting PDF invoices")
        manifest = pdf_segmenter.segment(args.pdf, cfg)
        (staging_dir / "invoice_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF segmentation failed: %s", exc)
        if normalized_df is not None:
            normalized_df.to_excel(staging_dir / "Driver_clean.xlsx", index=False)
        raise

    try:
        logger.info("Packaging deliverables")
        summary = file_packager.package(
            normalized_df,
            manifest,
            args.excel,
            args.pdf,
            args.output,
            cfg,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Packaging deliverables failed: %s", exc)
        if normalized_df is not None:
            normalized_df.to_excel(staging_dir / "Driver_clean.xlsx", index=False)
        if manifest is not None:
            (staging_dir / "invoice_manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
        raise

    logger.info("Pipeline completed successfully")
    return summary
