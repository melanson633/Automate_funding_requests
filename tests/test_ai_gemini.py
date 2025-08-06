from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import google.api_core  # noqa: F401
import pytest
import pandas as pd
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


def test_invoke_model_configures_retry_and_tools(monkeypatch):
    class FakeRetryOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenConfig:
        def __init__(self, **kwargs):
            self.retry_options = kwargs.get("retry_options")
            self.tools = kwargs.get("tools")

    monkeypatch.setattr(
        ai_gemini.types, "RetryOptions", FakeRetryOptions, raising=False
    )
    monkeypatch.setattr(
        ai_gemini.types, "GenerateContentConfig", FakeGenConfig, raising=False
    )

    class FakeModels:
        def generate_content(self, model, contents, config):
            assert isinstance(config.retry_options, FakeRetryOptions)
            assert config.tools == [
                ai_gemini.map_headers,
                ai_gemini.classify_page,
                ai_gemini.detect_invoice_starts,
            ]
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


def test_stream_generate_content(monkeypatch):
    class Chunk:
        def __init__(self, text):
            self.text = text

    class FakeGenConfig:
        def __init__(self, **kwargs):
            self.tools = kwargs.get("tools")

    monkeypatch.setattr(
        ai_gemini.types, "GenerateContentConfig", FakeGenConfig, raising=False
    )

    class FakeModels:
        def generate_content(self, model, contents, config, stream):
            assert stream is True
            assert config.tools == [
                ai_gemini.map_headers,
                ai_gemini.classify_page,
                ai_gemini.detect_invoice_starts,
            ]
            yield Chunk("a")
            yield Chunk("b")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(ai_gemini, "_get_client", lambda cfg: FakeClient())

    result = "".join(ai_gemini.stream_generate_content("p", {}))
    assert result == "ab"


@pytest.mark.asyncio
async def test_async_generate_content(monkeypatch):
    called = []

    def fake_invoke(prompt, cfg, temperature, tools):
        called.append(prompt)
        return f"resp:{prompt}"

    counter = {"cur": 0, "max": 0}

    async def fake_to_thread(func, *args, **kwargs):
        counter["cur"] += 1
        counter["max"] = max(counter["max"], counter["cur"])
        await asyncio.sleep(0)
        try:
            return func(*args, **kwargs)
        finally:
            counter["cur"] -= 1

    monkeypatch.setattr(ai_gemini, "_invoke_model", fake_invoke)
    monkeypatch.setattr(ai_gemini.asyncio, "to_thread", fake_to_thread)

    res = await ai_gemini.async_generate_content(
        ["a", "b", "c"], {}, concurrency_limit=2
    )
    assert res == ["resp:a", "resp:b", "resp:c"]
    assert counter["max"] <= 2


def test_analyze_excel_content():
    sheets = {"Sheet1": [["A", "B"], ["1", "2"]]}
    prompt = ai_gemini._analyze_excel_content(sheets)
    assert "Sheet: Sheet1" in prompt
    assert "sheet_name" in prompt and "header_row" in prompt


def test_detect_excel_structure_cached(tmp_path, monkeypatch):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    path = tmp_path / "wb.xlsx"
    df.to_excel(path, index=False, sheet_name="Data")

    calls = []

    def fake_invoke(prompt):
        calls.append(prompt)
        return {
            "sheet_name": "Data",
            "header_row": 1,
            "confidence": 0.9,
            "reasoning": "rows",
        }

    monkeypatch.setattr(ai_gemini, "_invoke_model", fake_invoke)
    ai_gemini.detect_excel_structure.cache_clear()

    first = ai_gemini.detect_excel_structure(path)
    second = ai_gemini.detect_excel_structure(path)

    assert first["sheet_name"] == "Data"
    assert second == first
    assert len(calls) == 1
