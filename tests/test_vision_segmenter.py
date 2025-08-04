from __future__ import annotations

import sys
import types
from pathlib import Path

import google.api_core  # noqa: F401
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ensure google.genai.types module exists for import
google_mock = types.ModuleType("google")
genai_mock = types.ModuleType("genai")
genai_types_mock = types.ModuleType("types")

genai_mock.types = genai_types_mock
google_mock.genai = genai_mock

sys.modules.setdefault("google", google_mock)
sys.modules.setdefault("google.genai", genai_mock)
sys.modules.setdefault("google.genai.types", genai_types_mock)

from cre_advance import vision_segmenter  # noqa: E402


class FakePix:
    def tobytes(self, fmt: str) -> bytes:
        assert fmt == "png"
        return b"pix"


def test_segment_basic(monkeypatch):
    class FakePage:
        def get_pixmap(self, dpi: int):
            assert dpi == 150
            return FakePix()

    class FakeDoc(list):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open(path):
        return FakeDoc([FakePage(), FakePage()])

    class FakePart:
        @classmethod
        def from_bytes(cls, data: bytes, mime_type: str):
            return {"data": data, "mime_type": mime_type}

    monkeypatch.setattr(vision_segmenter, "fitz", types.SimpleNamespace(open=fake_open))
    monkeypatch.setattr(vision_segmenter.types, "Part", FakePart, raising=False)

    def fake_invoke(contents, cfg):
        assert len(contents) == 3  # prompt + two parts
        return (
            '[{"start_page":1,"vendor":"A","invoice_number":"1",'
            '"date":"2024-01-01","amount":"10","confidence":0.9}]'
        )

    monkeypatch.setattr(vision_segmenter.ai_gemini, "invoke_multimodal", fake_invoke)

    metrics: dict = {}
    manifest = vision_segmenter.segment("dummy.pdf", cfg={}, metrics=metrics)

    assert manifest[0]["start_page"] == 1
    assert manifest[0]["end_page"] == 2
    assert metrics["vision_pages"] == 2
    assert metrics["vision_seconds"] >= 0.0
