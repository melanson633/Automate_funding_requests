from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cre_advance.utils.env import get_config  # noqa: E402


def test_model_tier_derives_gemini_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.delenv("MODEL_TIER", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    cfg = get_config("example_lender")
    assert cfg["gemini_model"] == "gemini-2.5-flash"


def test_fallback_to_gemini_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.delenv("MODEL_TIER", raising=False)
    lender_name = "temp_model_lender"
    lenders_dir = Path(__file__).resolve().parents[1] / "configs" / "lenders"
    cfg_path = lenders_dir / f"{lender_name}.yaml"
    cfg_path.write_text("gemini_model: gemini-2.5-pro\n")
    try:
        cfg = get_config(lender_name)
        assert cfg["gemini_model"] == "gemini-2.5-pro"
    finally:
        cfg_path.unlink()


def test_env_model_tier_overrides_yaml(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("MODEL_TIER", "pro")
    cfg = get_config("example_lender")
    assert cfg["gemini_model"] == "gemini-2.5-pro"


def test_default_gemini_model_when_missing(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.delenv("MODEL_TIER", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    lender_name = "temp_default_lender"
    lenders_dir = Path(__file__).resolve().parents[1] / "configs" / "lenders"
    cfg_path = lenders_dir / f"{lender_name}.yaml"
    cfg_path.write_text("model_tier: null\n")
    try:
        cfg = get_config(lender_name)
        assert cfg["gemini_model"] == "gemini-2.5-pro"
    finally:
        cfg_path.unlink()


def test_ocr_defaults_loaded(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    cfg = get_config("example_lender")
    ocr = cfg["ocr"]
    assert ocr["langs"] == ["eng"]
    assert ocr["psm"] == 6
    assert ocr["oem"] == 1


def test_pdf_defaults_loaded(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    cfg = get_config("example_lender")
    pdf = cfg["pdf"]
    assert pdf["use_vision"] is False
    assert pdf["vision_model"] == "gemini-2.5-pro"
    assert pdf["max_pages_per_request"] == 3000


def test_pdf_lender_overrides(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    lender_name = "temp_pdf_lender"
    lenders_dir = Path(__file__).resolve().parents[1] / "configs" / "lenders"
    cfg_path = lenders_dir / f"{lender_name}.yaml"
    cfg_path.write_text(
        "pdf:\n"
        "  use_vision: true\n"
        "  vision_model: custom-model\n"
        "  max_pages_per_request: 10\n"
    )
    try:
        cfg = get_config(lender_name)
        pdf = cfg["pdf"]
        assert pdf["use_vision"] is True
        assert pdf["vision_model"] == "custom-model"
        assert pdf["max_pages_per_request"] == 10
    finally:
        cfg_path.unlink()
