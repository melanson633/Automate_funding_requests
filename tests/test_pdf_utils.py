from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cre_advance.utils.pdf_utils import merge_pdfs  # noqa: E402


def _create_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as fh:
        writer.write(fh)


def test_merge_pdfs_empty_list(tmp_path: Path) -> None:
    output = tmp_path / "out.pdf"
    with pytest.raises(ValueError, match="No PDFs provided"):
        merge_pdfs([], output)


def test_merge_pdfs_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    output = tmp_path / "out.pdf"
    with pytest.raises(FileNotFoundError):
        merge_pdfs([missing], output)


def test_merge_single_pdf_copies(tmp_path: Path) -> None:
    src = tmp_path / "single.pdf"
    _create_pdf(src)
    dest = tmp_path / "copy.pdf"
    result = merge_pdfs([src], dest)
    assert result == dest
    assert dest.read_bytes() == src.read_bytes()


def test_merge_multiple_pdfs(tmp_path: Path) -> None:
    pdf1 = tmp_path / "a.pdf"
    pdf2 = tmp_path / "b.pdf"
    _create_pdf(pdf1)
    _create_pdf(pdf2)
    output = tmp_path / "merged.pdf"
    result = merge_pdfs([pdf1, pdf2], output)
    assert result == output
    reader = PdfReader(str(output))
    assert len(reader.pages) == 2
