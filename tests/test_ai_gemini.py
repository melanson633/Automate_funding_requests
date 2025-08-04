from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import google.api_core  # noqa: F401
from google.api_core import exceptions as google_exceptions

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mock the genai module structure
google_mock = types.ModuleType("google")
genai_mock = types.ModuleType("genai")
genai_types_mock = types.ModuleType("types")
genai_mock.types = genai_types_mock
google_mock.genai = genai_mock

sys.modules.setdefault("google", google_mock)
sys.modules.setdefault("google.genai", genai_mock)
sys.modules.setdefault("google.genai.types", genai_types_mock)

from cre_advance import ai_gemini  # noqa: E402


def test_classify_pages(monkeypatch):
    pages = [
        "Invoice Register\nWorkflow Approval",
        "Invoice #123\nBill To",
        "From: Craig\nSent: yesterday",
    ]
    fake_resp = [
        {
            "page_number": 1,
            "category": "invoice_register",
            "keep": False,
            "confidence": 0.9,
        },
        {"page_number": 2, "category": "invoice", "keep": True, "confidence": 0.95},
        {
            "page_number": 3,
            "category": "email_approval",
            "keep": False,
            "confidence": 0.8,
        },
    ]
    monkeypatch.setattr(
        ai_gemini,
        "_request_json",
        lambda prompt, schema, cfg, temperature=None: fake_resp,
    )

    out = ai_gemini.classify_pages(pages, {})
    assert out == fake_resp


def test_request_json_raises_non_retryable(monkeypatch):
    class FakeRetryOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenConfig:
        def __init__(self, **kwargs):
            self.retry_options = kwargs.get("retry_options")

    monkeypatch.setattr(
        ai_gemini.types, "RetryOptions", FakeRetryOptions, raising=False
    )
    monkeypatch.setattr(
        ai_gemini.types, "GenerateContentConfig", FakeGenConfig, raising=False
    )

    class FakeModels:
        def generate_content(self, model, contents, config):
            assert isinstance(config.retry_options, FakeRetryOptions)
            raise google_exceptions.BadRequest("bad request")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(ai_gemini, "_get_client", lambda cfg: FakeClient())

    with pytest.raises(google_exceptions.BadRequest):
        ai_gemini._request_json("prompt", schema={}, cfg={})
