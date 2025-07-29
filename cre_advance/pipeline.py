from __future__ import annotations

"""Pipeline orchestrator for the CRE advance request process."""

from argparse import Namespace
from typing import Any, Dict

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

    logger.info("Normalizing Excel workbooks")
    normalized_df, _ = excel_normalizer.normalize(args.yardi, cfg)

    logger.info("Segmenting PDF invoices")
    manifest = pdf_segmenter.segment(args.pdf, cfg)

    logger.info("Packaging deliverables")
    summary = file_packager.package(
        normalized_df,
        manifest,
        args.excel,
        args.pdf,
        args.output,
        cfg,
    )

    logger.info("Pipeline completed successfully")
    return summary
