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


def test_classify_page_cached():
    ai_gemini.classify_page.cache_clear()
    text = "Invoice #123\nBill To"
    ai_gemini.classify_page(text)
    first = ai_gemini.classify_page.cache_info()
    ai_gemini.classify_page(text)
    second = ai_gemini.classify_page.cache_info()
    assert second.hits == first.hits + 1


def test_map_headers_basic():
    headers = ["Date", "Amt"]
    mapping = ai_gemini.map_headers(headers, [], ["Date", "Amount"])
    assert mapping["Date"] == "Date"
    assert mapping["Amt"] == "Amount"


def test_map_headers_cached():
    ai_gemini.map_headers.cache_clear()
    headers = ["Date", "Amt"]
    targets = ["Date", "Amount"]
    ai_gemini.map_headers(headers, [], targets)
    first = ai_gemini.map_headers.cache_info()
    ai_gemini.map_headers(headers, [], targets)
    second = ai_gemini.map_headers.cache_info()
    assert second.hits == first.hits + 1


def test_invoke_model_raises_non_retryable(monkeypatch):
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
        ai_gemini._invoke_model("prompt", cfg={})


def test_invoke_multimodal_text(monkeypatch):
    class FakeResp:
        text = "result"
        candidates = []

    class FakeModels:
        def generate_content(self, model, contents, stream):
            assert model == "gemini-2.5-pro"
            assert contents == ["a"]
            assert stream is False
            return FakeResp()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(ai_gemini, "_get_client", lambda cfg: FakeClient())

    result = ai_gemini.invoke_multimodal(["a"], cfg={})
    assert result == "result"


def test_invoke_multimodal_candidate(monkeypatch):
    class FakeResp:
        text = None
        candidates = [types.SimpleNamespace(text="alt")]

    class FakeModels:
        def generate_content(self, model, contents, stream):
            return FakeResp()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(ai_gemini, "_get_client", lambda cfg: FakeClient())

    result = ai_gemini.invoke_multimodal([], cfg={})
    assert result == "alt"


def test_parse_manifest_response_success():
    raw = (
        '[{"start_page":1,"vendor":"V","invoice_number":"INV",'
        '"date":"2024-01-01","amount":10.0,"confidence":0.9}]'
    )
    parsed = ai_gemini.parse_manifest_response(raw)
    assert parsed[0]["vendor"] == "V"


def test_parse_manifest_response_invalid():
    with pytest.raises(ValueError):
        ai_gemini.parse_manifest_response("not json")
