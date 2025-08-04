from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest import mock

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Create minimal google.genai.types.Part for import
google_mod = types.ModuleType("google")
genai_mod = types.ModuleType("genai")
types_mod = types.ModuleType("types")


class DummyPart:
    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str):
        return {"data": data, "mime_type": mime_type}


types_mod.Part = DummyPart
genai_mod.types = types_mod
google_mod.genai = genai_mod

sys.modules["google"] = google_mod
sys.modules["google.genai"] = genai_mod
sys.modules["google.genai.types"] = types_mod

from cre_advance import vision_segmenter  # noqa: E402


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(path)
    doc.close()


def test_segment_uses_mocked_gemini(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)

    manifest_json = json.dumps(
        [
            {
                "start_page": 1,
                "vendor": "Vendor",
                "invoice_number": "1",
                "date": "2024-01-01",
                "amount": "10.00",
                "confidence": 0.9,
            }
        ]
    )

    with mock.patch("google.genai.Client", create=True), mock.patch(
        "cre_advance.ai_gemini.invoke_multimodal", return_value=manifest_json
    ):
        manifest = vision_segmenter.segment(
            pdf_path, {"pdf": {"use_vision": True}}, {}
        )

    assert manifest == [
        {
            "start_page": 1,
            "end_page": 2,
            "vendor": "Vendor",
            "invoice_number": "1",
            "date": "2024-01-01",
            "amount": "10.00",
            "confidence": 0.9,
        }
    ]

