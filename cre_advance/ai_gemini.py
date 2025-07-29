"""Gemini 2.5 Pro wrapper utilities.

This module exposes helper functions for interacting with Google's
Gemini models. Prompts are crafted to return JSON responses which are
validated before being returned to the caller. All requests include
retry logic with exponential backoff using the SDK's ``RetryOptions``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from google import generativeai as genai
from google.generativeai import types

from .utils.logging import get_logger


logger = get_logger(__name__)
_MODEL_NAME = "gemini-2.5-pro"
_MAX_RETRIES = 3

_client: genai.Client | None = None


def _init_client() -> genai.Client:
    """Instantiate and return a Gemini client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Create a .env file with your API key and retry."
        )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            api_version="v1", retry_options=types.RetryOptions(max_retries=_MAX_RETRIES)
        ),
    )


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = _init_client()
    return _client


def _request_json(prompt: str, schema: dict, temperature: float = 0.2) -> Any:
    """Send a prompt to Gemini and return parsed JSON or ``None``."""
    client = _get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )

    delay = 1.0
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.debug("Gemini prompt: %s", prompt)
            resp = client.models.generate_content(
                model=_MODEL_NAME, contents=prompt, config=config
            )
            logger.debug("Gemini raw response: %s", getattr(resp, "text", resp))
            data = getattr(resp, "parsed", None)
            if data is None:
                data = json.loads(getattr(resp, "text", ""))
            return data
        except Exception as exc:  # broad exception from SDK
            logger.warning("Gemini attempt %s failed: %s", attempt, exc)
            if attempt == _MAX_RETRIES:
                raise RuntimeError("Gemini request failed after retries") from exc
            time.sleep(delay)
            delay *= 2
    return None


def map_schema(
    headers: List[str],
    sample_rows: List[Dict[str, Any]],
    target_fields: List[str],
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    """Map raw headers to canonical target fields."""
    cfg = cfg or {}
    prompt = (
        "Map the following spreadsheet column headers to canonical Invoice Log fields. "
        "Only use one of the provided target fields for each header and return a JSON "
        "object where keys are the raw headers and values are the chosen target fields."\
    )
    prompt += f"\nTarget fields: {', '.join(target_fields)}"
    prompt += f"\nHeaders: {', '.join(headers)}"
    prompt += "\nSample rows:" + json.dumps(sample_rows, indent=2)

    schema = {"type": "object", "additionalProperties": {"type": "string"}}
    result = _request_json(prompt, schema, temperature=cfg.get("temperature", 0.1))
    return result or {}


def segment_pdf(pages: List[str], cfg: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Detect invoice boundaries within PDF pages."""
    cfg = cfg or {}
    joined_pages = "\n---\n".join(pages)
    prompt = (
        "Given the OCR text for each page of a PDF, identify the start page of each "
        "invoice and return metadata. Provide a JSON array of objects with fields "
        "'start_page', 'vendor', 'invoice_number', 'date', and 'amount'. Page numbers "
        "are 1-indexed.\n" + joined_pages
    )

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "start_page": {"type": "integer"},
                "vendor": {"type": "string"},
                "invoice_number": {"type": "string"},
                "date": {"type": "string"},
                "amount": {"type": "string"},
            },
            "required": ["start_page"],
        },
    }
    result = _request_json(prompt, schema, temperature=cfg.get("temperature", 0.2))
    return result or []


def extract_metadata(text: str, cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Extract structured invoice metadata from text."""
    cfg = cfg or {}
    prompt = (
        "Extract invoice metadata from the following text and return a JSON object "
        "with fields 'vendor', 'invoice_number', 'date', and 'amount'.\n" + text
    )
    schema = {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "amount": {"type": "string"},
        },
    }
    result = _request_json(prompt, schema, temperature=cfg.get("temperature", 0.1))
    return result or {}


def build_schema(
    headers: List[str], sample_rows: List[Dict[str, Any]], cfg: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Propose a canonical schema and mapping for new lenders."""
    cfg = cfg or {}
    prompt = (
        "You are designing a normalized Invoice Log schema. Based on the raw "
        "headers and sample rows, propose canonical field names and provide a JSON "
        "object with two keys: 'fields' (an array of canonical names) and "
        "'mapping' (object mapping each raw header to a canonical name)."
    )
    prompt += f"\nHeaders: {', '.join(headers)}"
    prompt += "\nSample rows:" + json.dumps(sample_rows, indent=2)

    schema = {
        "type": "object",
        "properties": {
            "fields": {"type": "array", "items": {"type": "string"}},
            "mapping": {"type": "object", "additionalProperties": {"type": "string"}},
        },
    }
    result = _request_json(prompt, schema, temperature=cfg.get("temperature", 0.1))
    return result or {}
