from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Ensure repository root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mock google.genai modules for offline tests
try:  # pragma: no cover - best effort import
    import google as google_pkg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled in tests
    google_pkg = types.ModuleType("google")
    sys.modules["google"] = google_pkg

genai_mock = types.ModuleType("genai")
genai_types_mock = types.ModuleType("types")
genai_mock.types = genai_types_mock
google_pkg.genai = genai_mock
sys.modules.setdefault("google.genai", genai_mock)
sys.modules.setdefault("google.genai.types", genai_types_mock)


class _FakeReader:
    def __init__(self, pages):
        self.pages = pages


@pytest.fixture
def patch_pdf_reader(monkeypatch):
    """Patch PdfReader to return provided pages."""

    def _patch(pages):
        from cre_advance import pdf_segmenter

        monkeypatch.setattr(pdf_segmenter, "PdfReader", lambda p: _FakeReader(pages))

    return _patch


@pytest.fixture
def fake_page():
    """Factory for fake PDF pages."""

    def _page(text: str = "", has_image: bool = False):
        import types

        from PIL import Image

        images = []
        if has_image:
            img = types.SimpleNamespace(image=Image.new("RGB", (10, 10), "white"))
            images = [img]
        return types.SimpleNamespace(extract_text=lambda: text, images=images)

    return _page
