from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cre_advance.utils.env import get_config


def test_model_tier_derives_gemini_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    cfg = get_config("example_lender")
    assert cfg["gemini_model"] == "gemini-2.5-flash"


def test_fallback_to_gemini_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    lender_name = "temp_model_lender"
    lenders_dir = Path(__file__).resolve().parents[1] / "configs" / "lenders"
    cfg_path = lenders_dir / f"{lender_name}.yaml"
    cfg_path.write_text("gemini_model: gemini-2.5-pro\n")
    try:
        cfg = get_config(lender_name)
        assert cfg["gemini_model"] == "gemini-2.5-pro"
    finally:
        cfg_path.unlink()
