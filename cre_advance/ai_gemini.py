"""Gemini 2.5 Pro helper utilities.

This module exposes simple Python functions that can be surfaced as tools
to Google's Gemini models. Functions rely on lightweight heuristics so
they can be executed locally or via the model's automatic function
calling.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

from .utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "gemini-2.5-pro"
_MAX_RETRIES = 3

_client = None


def _get_client(cfg: Dict[str, Any] | None = None):
    """Get or create Gemini client instance."""
    global _client
    cfg = cfg or {}

    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Create a .env file with your API key and retry."
            )
        _client = genai.Client(api_key=api_key)

    return _client


def map_headers(
    headers: List[str], sample_rows: List[Dict[str, Any]], target_fields: List[str]
) -> Dict[str, str]:
    """Map raw headers to canonical target fields.

    The implementation uses basic case-insensitive and substring matching
    so it can run locally when invoked by the model.

    Args:
        headers: Raw column headers from an incoming spreadsheet.
        sample_rows: Example rows providing context (currently unused).
        target_fields: Canonical field names.

    Returns:
        Mapping from each raw header to the best matching target field. If
        no reasonable match is found an empty string is used.
    """
    mapping: Dict[str, str] = {}
    target_lookup = {f.lower(): f for f in target_fields}
    for hdr in headers:
        normalized = hdr.strip().lower()
        match = ""
        if normalized in target_lookup:
            match = target_lookup[normalized]
        else:
            for key, value in target_lookup.items():
                if key in normalized or normalized in key:
                    match = value
                    break
            if not match and normalized.startswith("amt") and "amount" in target_lookup:
                match = target_lookup["amount"]
        mapping[hdr] = match
    return mapping


def classify_page(text: str) -> bool:
    """Determine whether a page of text is an invoice.

    Heuristics look for the word ``invoice`` alongside common invoice
    components like ``bill to`` or ``invoice #``.

    Args:
        text: OCR or extracted text for a single PDF page.

    Returns:
        ``True`` if the page appears to be an invoice page, otherwise
        ``False``.
    """
    lowered = text.lower()
    if "invoice" not in lowered:
        return False
    indicators = ["bill to", "invoice #", "balance due", "total", "amount due"]
    return any(token in lowered for token in indicators)


def detect_invoice_starts(pages: List[str]) -> List[int]:
    """Detect start indices of invoices within a sequence of pages.

    Args:
        pages: Text for each page in reading order.

    Returns:
        A list of zero-indexed page numbers marking the start of each
        invoice.
    """
    starts: List[int] = []
    prev_is_invoice = False
    for idx, text in enumerate(pages):
        is_invoice = classify_page(text)
        if is_invoice and not prev_is_invoice:
            starts.append(idx)
        prev_is_invoice = is_invoice
    return starts


def build_schema(
    headers: List[str], sample_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create a very small schema proposal.

    This is a light-weight helper used in tests. It leverages
    ``map_headers`` to propose a mapping using the headers themselves as
    target fields.
    """
    mapping = map_headers(headers, sample_rows, headers)
    return {"fields": list(mapping.values()), "mapping": mapping}


def _invoke_model(
    prompt: str,
    cfg: Dict[str, Any] | None = None,
    temperature: float | None = None,
    tools: List[Any] | None = None,
) -> Any:
    """Send a prompt to Gemini using automatic function calling.

    Args:
        prompt: Prompt text to send to the model.
        cfg: Optional configuration dictionary.
        temperature: Optional temperature override.
        tools: Optional list of tool functions to expose to the model.

    Returns:
        The ``parsed`` field from the SDK response if available, otherwise
        ``None``.
    """
    cfg = cfg or {}
    client = _get_client(cfg)
    model_name = cfg.get("gemini_model", _MODEL_NAME)

    temp = temperature if temperature is not None else cfg.get("gemini_temperature", 0.1)

    retry_options = types.RetryOptions(
        max_retries=int(cfg.get("gemini_max_retries", _MAX_RETRIES)),
        initial_delay=1.0,
        backoff_multiplier=2.0,
        max_delay=60.0,
        retryable_errors=[
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
        ],
    )

    generation_config = types.GenerateContentConfig(
        temperature=temp,
        max_output_tokens=2048,
        tools=tools or [map_headers, classify_page, detect_invoice_starts],
        retry_options=retry_options,
    )

    try:
        logger.debug("Gemini prompt: %s", prompt)
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=generation_config,
        )
        logger.debug("Gemini response: %s", resp)
        return getattr(resp, "parsed", None)
    except google_exceptions.BadRequest as exc:
        logger.error("Gemini bad request: %s", exc)
        logger.error("Prompt: %s", prompt)
        logger.error("Temperature: %s", temp)
        raise
    except google_exceptions.GoogleAPIError as exc:
        logger.error("Gemini API error: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected Gemini error: %s", exc)
        raise

