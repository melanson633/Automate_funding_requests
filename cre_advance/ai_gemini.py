"""Gemini 2.5 Pro helper utilities.

This module exposes simple Python functions that can be surfaced as tools
to Google's Gemini models. Functions rely on lightweight heuristics so
they can be executed locally or via the model's automatic function
calling.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

from .utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "gemini-2.5-pro"
_MAX_RETRIES = 3
_CACHE_MAXSIZE = int(os.getenv("AI_CACHE_MAXSIZE", 128))

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


def _serialize_rows(rows: List[Dict[str, Any]]) -> tuple[str, ...]:
    """Return a hashable representation of ``rows`` for caching."""

    return tuple(json.dumps(r, sort_keys=True) for r in rows)


@lru_cache(maxsize=_CACHE_MAXSIZE)
def _map_headers_cached(
    headers_key: tuple[str, ...],
    sample_rows_key: tuple[str, ...],
    target_fields_key: tuple[str, ...],
) -> Dict[str, str]:
    headers = list(headers_key)
    sample_rows = [json.loads(r) for r in sample_rows_key]
    target_fields = list(target_fields_key)
    # ``sample_rows`` is currently unused but included to key the cache
    # correctly should usage change in the future.
    _ = sample_rows

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


def map_headers(
    headers: List[str], sample_rows: List[Dict[str, Any]], target_fields: List[str]
) -> Dict[str, str]:
    """Map raw headers to canonical target fields.

    Results are cached to avoid recomputing mappings for identical input
    sets. The cache size is controlled via the ``AI_CACHE_MAXSIZE``
    environment variable. The underlying :func:`functools.lru_cache`
    implementation employs a thread lock, so the function is safe for
    concurrent use. Callers should treat the returned mapping as
    immutable.

    Args:
        headers: Raw column headers from an incoming spreadsheet.
        sample_rows: Example rows providing context (currently unused).
        target_fields: Canonical field names.

    Returns:
        Mapping from each raw header to the best matching target field. If
        no reasonable match is found an empty string is used.
    """
    headers_key = tuple(headers)
    sample_rows_key = _serialize_rows(sample_rows)
    target_fields_key = tuple(target_fields)
    return dict(_map_headers_cached(headers_key, sample_rows_key, target_fields_key))


map_headers.cache_info = _map_headers_cached.cache_info  # type: ignore[attr-defined]
map_headers.cache_clear = _map_headers_cached.cache_clear  # type: ignore[attr-defined]


@lru_cache(maxsize=_CACHE_MAXSIZE)
def classify_page(text: str) -> bool:
    """Determine whether a page of text is an invoice.

    Heuristics look for the word ``invoice`` alongside common invoice
    components like ``bill to`` or ``invoice #``. Results are cached to
    minimise repeated classification of the same text. The cache size is
    controlled by the ``AI_CACHE_MAXSIZE`` environment variable.

    ``functools.lru_cache`` uses an internal lock making this function
    thread-safe; the returned boolean is immutable.

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


def classify_pages(pages: List[str], cfg: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Classify a batch of pages.

    Args:
        pages: Text for each page in order.
        cfg: Configuration dictionary (unused).

    Returns:
        List of classification dictionaries with ``page_number``, ``category``,
        ``keep`` and ``confidence`` keys.
    """
    results: List[Dict[str, Any]] = []
    for idx, text in enumerate(pages, start=1):
        keep = classify_page(text)
        results.append(
            {
                "page_number": idx,
                "category": "invoice" if keep else "unknown",
                "keep": keep,
                "confidence": 1.0,
            }
        )
    return results


@lru_cache(maxsize=_CACHE_MAXSIZE)
def extract_metadata(text: str) -> Dict[str, str]:
    """Extract basic invoice metadata from ``text``.

    This heuristic function enables local execution and serves as a
    fallback when the Gemini model is unavailable. Results are cached so
    identical invoices are parsed only once. Cache size is governed by
    the ``AI_CACHE_MAXSIZE`` environment variable. The cache is
    thread-safe, but the returned dictionary should be treated as
    read-only.

    Args:
        text: Full text of an invoice, potentially spanning multiple pages.

    Returns:
        Dictionary containing ``vendor``, ``invoice_number``, ``date`` and
        ``amount`` keys. Missing values are returned as empty strings.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    vendor = ""
    if lines and "invoice" not in lines[0].lower():
        vendor = lines[0]

    inv_match = re.search(
        r"invoice\s*(?:number|no\.|#)?\s*[:#]?\s*([A-Za-z0-9-]+)",
        text,
        re.IGNORECASE,
    )
    date_match = re.search(
        r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        text,
    )
    amt_match = re.search(r"\$?\s*([0-9][0-9,]*\.\d{2})", text)

    date = ""
    if date_match:
        raw = date_match.group(1)
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
            try:
                date = datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
        else:
            date = raw

    amount = ""
    if amt_match:
        raw_amt = amt_match.group(1).replace(",", "")
        try:
            amount = f"{float(raw_amt):.2f}"
        except ValueError:
            amount = raw_amt

    return {
        "vendor": vendor,
        "invoice_number": inv_match.group(1) if inv_match else "",
        "date": date,
        "amount": amount,
    }


@lru_cache(maxsize=_CACHE_MAXSIZE)
def _detect_invoice_starts_cached(pages_key: tuple[str, ...]) -> List[int]:
    pages = list(pages_key)
    starts: List[int] = []
    prev_is_invoice = False
    for idx, text in enumerate(pages):
        is_invoice = classify_page(text)
        if is_invoice and not prev_is_invoice:
            starts.append(idx)
        prev_is_invoice = is_invoice
    return starts


def detect_invoice_starts(pages: List[str]) -> List[int]:
    """Detect start indices of invoices within a sequence of pages.

    Results are cached based on the sequence of page text to reduce
    repeated classification. Configure cache size with
    ``AI_CACHE_MAXSIZE``. The underlying implementation is thread-safe;
    callers must not mutate the returned list.

    Args:
        pages: Text for each page in reading order.

    Returns:
        A list of zero-indexed page numbers marking the start of each
        invoice.
    """
    return list(_detect_invoice_starts_cached(tuple(pages)))


detect_invoice_starts.cache_info = (  # type: ignore[attr-defined]
    _detect_invoice_starts_cached.cache_info
)
detect_invoice_starts.cache_clear = (  # type: ignore[attr-defined]
    _detect_invoice_starts_cached.cache_clear
)


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

    temp = (
        temperature if temperature is not None else cfg.get("gemini_temperature", 0.1)
    )

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


def invoke_multimodal(contents: List[Any], cfg: Dict[str, Any]) -> Any:
    """Invoke Gemini with multimodal (text + image) content.

    Args:
        contents: List of text strings and ``Part`` objects comprising the
            request.
        cfg: Configuration dictionary.

    Returns:
        Text extracted from the Gemini response when available.

    Raises:
        google_exceptions.BadRequest: If the request is malformed.
        google_exceptions.GoogleAPIError: For underlying API errors.
        Exception: For any other unexpected errors.
    """
    cfg = cfg or {}
    client = _get_client(cfg)
    model_name = cfg.get("pdf", {}).get("vision_model", "gemini-2.5-pro")

    try:
        logger.debug("Gemini multimodal contents: %s", contents)
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            stream=False,
        )
        logger.debug("Gemini multimodal response: %s", response)
        if getattr(response, "text", None):
            return response.text
        candidates = getattr(response, "candidates", None)
        if candidates:
            first = candidates[0]
            if getattr(first, "text", None):
                return first.text
        return response
    except google_exceptions.BadRequest as exc:
        logger.error("Gemini bad request: %s", exc)
        raise
    except google_exceptions.GoogleAPIError as exc:
        logger.error("Gemini API error: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected Gemini error: %s", exc)
        raise


def parse_manifest_response(response_text: str) -> List[dict]:
    """Parse manifest extraction response from Gemini.

    Args:
        response_text: Raw JSON string returned by Gemini.

    Returns:
        A list of manifest dictionaries with required keys.

    Raises:
        ValueError: If the response is not valid JSON or missing keys.
    """
    logger.debug("Parsing manifest response: %s", response_text)
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid manifest JSON: %s", exc)
        raise ValueError("Manifest response is not valid JSON") from exc

    if not isinstance(data, list):
        logger.error("Manifest JSON is not an array: %s", data)
        raise ValueError("Manifest response must be a JSON array")

    required = {
        "start_page",
        "vendor",
        "invoice_number",
        "date",
        "amount",
        "confidence",
    }
    for item in data:
        if not isinstance(item, dict) or required - item.keys():
            logger.error("Manifest item missing keys: %s", item)
            raise ValueError(
                "Each manifest item must contain keys: start_page, vendor, invoice_number, date, amount, confidence"
            )
    return data
