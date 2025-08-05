"""PDF utility functions for the CRE Advance package."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pypdf import PdfMerger, PdfReader, PdfWriter
from pypdf.errors import DeprecationError

from .logging import get_logger

logger = get_logger(__name__)


def merge_pdfs(pdf_paths: list[str | Path], output_path: str | Path) -> Path:
    """Merge multiple PDFs into a single file.

    Args:
        pdf_paths: Sequence of PDF file paths in the order to merge.
        output_path: Destination path for the merged PDF.

    Returns:
        Path to the merged PDF.

    Raises:
        ValueError: If ``pdf_paths`` is empty.
        FileNotFoundError: If any source PDF is missing or unreadable.
    """
    if not pdf_paths:
        raise ValueError("No PDFs provided")

    paths = [Path(p) for p in pdf_paths]
    dest = Path(output_path)

    for path in paths:
        if not path.is_file() or not os.access(path, os.R_OK):
            raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Merging %d PDFs into %s", len(paths), dest)

    if len(paths) == 1:
        src = paths[0]
        if src.resolve() == dest.resolve():
            logger.info("Merged %d PDFs into %s", len(paths), dest)
            return src
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        logger.info("Merged %d PDFs into %s", len(paths), dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        merger = PdfMerger()
        try:
            for path in paths:
                merger.append(str(path))
            with dest.open("wb") as fh:
                merger.write(fh)
        finally:
            merger.close()
    except DeprecationError:
        writer = PdfWriter()
        for path in paths:
            reader = PdfReader(str(path))
            for page in reader.pages:
                writer.add_page(page)
        with dest.open("wb") as fh:
            writer.write(fh)
    logger.info("Merged %d PDFs into %s", len(paths), dest)
    return dest
