import sys
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cre_advance import ai_gemini
from cre_advance.utils.env import get_config


def test_load_prompt_messages(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    cfg = get_config("example_lender")
    messages = ai_gemini.load_prompt("classify_pages", cfg, pages=["a page"])
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "a page" in messages[1]["parts"][0]


def test_lender_prompt_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    lenders_dir = Path(__file__).resolve().parents[1] / "configs" / "lenders"
    lender_name = "override_lender"
    prompt_path = tmp_path / "custom.yaml"
    prompt_path.write_text("template: '{{value}}'\n")
    cfg_path = lenders_dir / f"{lender_name}.yaml"
    cfg_path.write_text(f"prompts:\n  classify_pages: {prompt_path}\n")
    try:
        cfg = get_config(lender_name)
        messages = ai_gemini.load_prompt("classify_pages", cfg, value="zzz")
        assert "zzz" in messages[1]["parts"][0]
    finally:
        cfg_path.unlink()
