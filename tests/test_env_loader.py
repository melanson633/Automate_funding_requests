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
    assert ocr["deskew"] is False


def test_pdf_defaults_loaded(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    cfg = get_config("example_lender")
    pdf = cfg["pdf"]
    assert pdf["use_vision"] is False
    assert pdf["vision_model"] == "gemini-2.5-pro"
    assert pdf["max_pages_per_request"] == 3000
    assert pdf["classification_confidence_threshold"] == 0.5
    assert pdf["min_confidence"] == 0.0
    assert pdf["unmatched_threshold"] == 0.4


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
        "  classification_confidence_threshold: 0.7\n"
        "  min_confidence: 0.3\n"
        "  unmatched_threshold: 0.2\n"
    )
    try:
        cfg = get_config(lender_name)
        pdf = cfg["pdf"]
        assert pdf["use_vision"] is True
        assert pdf["vision_model"] == "custom-model"
        assert pdf["max_pages_per_request"] == 10
        assert pdf["classification_confidence_threshold"] == 0.7
        assert pdf["min_confidence"] == 0.3
        assert pdf["unmatched_threshold"] == 0.2
    finally:
        cfg_path.unlink()


def test_ocr_lender_overrides(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    lender_name = "temp_ocr_lender"
    lenders_dir = Path(__file__).resolve().parents[1] / "configs" / "lenders"
    cfg_path = lenders_dir / f"{lender_name}.yaml"
    cfg_path.write_text(
        "ocr:\n"
        "  langs: spa,eng\n"
        "  psm: 7\n"
        "  oem: 3\n"
        "  deskew: true\n"
    )
    try:
        cfg = get_config(lender_name)
        ocr = cfg["ocr"]
        assert ocr["langs"] == ["spa", "eng"]
        assert ocr["psm"] == 7
        assert ocr["oem"] == 3
        assert ocr["deskew"] is True
    finally:
        cfg_path.unlink()


def test_packager_lender_overrides(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    lender_name = "temp_pkg_lender"
    lenders_dir = Path(__file__).resolve().parents[1] / "configs" / "lenders"
    cfg_path = lenders_dir / f"{lender_name}.yaml"
    cfg_path.write_text(
        "packager:\n"
        "  vendor_ratio_threshold: 0.5\n"
        "  amount_tolerance: 0.1\n"
        "  score_threshold: 1.0\n"
    )
    try:
        cfg = get_config(lender_name)
        pkg = cfg["packager"]
        assert pkg["vendor_ratio_threshold"] == 0.5
        assert pkg["amount_tolerance"] == 0.1
        assert pkg["score_threshold"] == 1.0
    finally:
        cfg_path.unlink()


def test_prompt_and_scoring_overrides(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    cfg = get_config("example_lender")
    prompts = cfg["prompts"]
    assert prompts["classify_pages"].endswith("classify_pages_override.yaml")
    scoring = cfg["scoring"]
    assert scoring["classification_weight"] == 0.7


def test_prompt_and_scoring_defaults(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    lender_name = "temp_prompt_defaults"
    lenders_dir = Path(__file__).resolve().parents[1] / "configs" / "lenders"
    cfg_path = lenders_dir / f"{lender_name}.yaml"
    cfg_path.write_text("\n")
    try:
        cfg = get_config(lender_name)
        prompts = cfg["prompts"]
        assert prompts["classify_pages"].endswith("classify_pages_prompt.yaml")
        scoring = cfg["scoring"]
        assert scoring["classification_weight"] == 1.0
    finally:
        cfg_path.unlink()


def test_report_types_merged(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    lender_name = "temp_report_lender"
    lenders_dir = Path(__file__).resolve().parents[1] / "configs" / "lenders"
    cfg_path = lenders_dir / f"{lender_name}.yaml"
    cfg_path.write_text(
        "report_types:\n"
        "  general_ledger:\n"
        "    header_row: 10\n"
        "  custom_type:\n"
        "    sheet_name: Custom\n"
        "    header_row: 1\n"
    )
    try:
        cfg = get_config(lender_name)
        rtypes = cfg["report_types"]
        assert rtypes["general_ledger"]["sheet_name"] == "Report1"
        assert rtypes["general_ledger"]["header_row"] == 10
        assert "expense_distribution" in rtypes
        assert rtypes["custom_type"]["sheet_name"] == "Custom"
        assert rtypes["custom_type"]["header_row"] == 1
    finally:
        cfg_path.unlink()
