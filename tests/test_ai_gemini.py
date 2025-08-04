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


def test_classify_page_and_detect_starts():
    pages = [
        "Invoice Register\nWorkflow Approval",
        "Invoice #123\nBill To",
        "From: Craig\nSent: yesterday",
        "Invoice #999\nBill To",
    ]

    assert ai_gemini.classify_page(pages[0]) is False
    assert ai_gemini.classify_page(pages[1]) is True
    assert ai_gemini.classify_page(pages[2]) is False

    starts = ai_gemini.detect_invoice_starts(pages)
    assert starts == [1, 3]


def test_map_headers_basic():
    headers = ["Date", "Amt"]
    mapping = ai_gemini.map_headers(headers, [], ["Date", "Amount"])
    assert mapping["Date"] == "Date"
    assert mapping["Amt"] == "Amount"


def test_invoke_model_raises_non_retryable(monkeypatch):
    class FakeRetryOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenConfig:
        def __init__(self, **kwargs):
            self.retry_options = kwargs.get("retry_options")

    monkeypatch.setattr(ai_gemini.types, "RetryOptions", FakeRetryOptions, raising=False)
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
        ai_gemini._invoke_model("prompt", cfg={})

