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

import google.generativeai as genai
import requests
from google.api_core import exceptions as google_exceptions

from .utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "gemini-1.5-flash"
_MAX_RETRIES = 3

_client = None
_client_model = None


def _init_client(cfg: Dict[str, Any] | None = None):
    """Initialize Gemini client with API key."""
    cfg = cfg or {}
    model_name = cfg.get("gemini_model", _MODEL_NAME)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Create a .env file with your API key and retry."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _get_client(cfg: Dict[str, Any] | None = None):
    global _client, _client_model
    cfg = cfg or {}
    model_name = cfg.get("gemini_model", _MODEL_NAME)
    if _client is None or _client_model != model_name:
        _client = _init_client(cfg)
        _client_model = model_name
    return _client


def _request_json(
    prompt: str,
    schema: dict,
    cfg: Dict[str, Any] | None = None,
    temperature: float | None = None,
) -> Any:
    """Send a prompt to Gemini and return parsed JSON or ``None``.

    Network related errors are retried with exponential backoff. Prompt
    validation or parsing failures are logged and raised without retrying.
    """
    cfg = cfg or {}
    model = _get_client(cfg)

    temp = (
        temperature if temperature is not None else cfg.get("gemini_temperature", 0.2)
    )

    # Add JSON instruction to prompt for older SDK
    json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."

    # Configure generation parameters for older SDK
    generation_config = genai.types.GenerationConfig(
        temperature=temp,
        max_output_tokens=2048,
    )

    delay = 1.0
    max_retries = int(cfg.get("gemini_max_retries", _MAX_RETRIES))
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("Gemini prompt: %s", json_prompt)
            resp = model.generate_content(
                json_prompt,
                generation_config=generation_config,
            )
            logger.debug("Gemini raw response: %s", resp.text)

            if not resp.text:
                logger.warning("Empty response from Gemini")
                return None

            text = resp.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            return json.loads(text)

        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON: %s", exc)
            logger.error("Raw response text: %s", getattr(resp, "text", "<empty>"))
            raise
        except google_exceptions.BadRequest as exc:
            logger.error("Gemini bad request: %s", exc)
            logger.error("Prompt: %s", json_prompt)
            logger.error("Temperature: %s", temp)
            raise
        except (
            google_exceptions.DeadlineExceeded,
            google_exceptions.ServiceUnavailable,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            last_exc = exc
            logger.warning(
                "Transient Gemini error on attempt %s/%s: %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
        except google_exceptions.GoogleAPIError as exc:
            logger.error("Gemini API error: %s", exc)
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Unexpected Gemini error on attempt %s/%s: %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
        break

    logger.error(
        "Gemini request failed after %s attempts. Prompt: %s | temperature=%s | model=%s",
        max_retries,
        json_prompt,
        temp,
        cfg.get("gemini_model", _MODEL_NAME),
    )
    raise RuntimeError("Gemini request failed") from last_exc


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
        "object where keys are the raw headers and values are the chosen target fields."
    )
    prompt += f"\nTarget fields: {', '.join(target_fields)}"
    prompt += f"\nHeaders: {', '.join(headers)}"
    prompt += "\nSample rows:" + json.dumps(sample_rows, indent=2)

    schema = {"type": "object", "additionalProperties": {"type": "string"}}
    result = _request_json(
        prompt,
        schema,
        cfg,
        temperature=cfg.get("gemini_temperature", 0.1),
    )
    return result or {}


def classify_pages(
    pages: List[str], cfg: Dict[str, Any] | None = None
) -> List[Dict[str, Any]]:
    """Classify PDF pages and decide which to keep.

    Args:
        pages: OCR or extracted text for each PDF page.
        cfg: Optional configuration dictionary.

    Returns:
        A list of dictionaries describing each page with keys
        ``page_number`` (1-indexed), ``category`` (``invoice``,
        ``invoice_register``, ``email_approval`` or ``unknown``), ``keep``
        (``True`` for invoice pages only) and ``confidence`` (0–1).
    """

    cfg = cfg or {}
    joined_pages = "\n---\n".join(pages)
    prompt = (
        "For each page of a PDF invoice backup, decide whether the page should be "
        "kept. A 'keep' page is an actual invoice. An 'invoice' page has: vendor "
        "logo/name; words like 'Invoice' or 'Invoice #'; date; 'Bill To' sections; "
        "line-item tables; and totals such as 'Total' or 'Balance Due'. A page "
        "labeled 'Invoice Register' with batch/contract fields and a 'Workflow "
        "Approval' table should be removed. A page containing email headers ('From:', "
        "'Sent:', 'To:', 'Subject:', 'Date:') followed by conversational text, "
        "signature blocks and confidentiality notices should also be removed. "
        "Return a JSON array of objects with keys: page_number (integer, 1-indexed), "
        "category ('invoice', 'invoice_register', 'email_approval' or 'unknown'), "
        "keep (boolean), and confidence (number). Page texts (separated by ---):\n"
        + joined_pages
    )

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "page_number": {"type": "integer"},
                "category": {
                    "type": "string",
                    "enum": [
                        "invoice",
                        "invoice_register",
                        "email_approval",
                        "unknown",
                    ],
                },
                "keep": {"type": "boolean"},
                "confidence": {"type": "number"},
            },
            "required": ["page_number", "category", "keep"],
        },
    }

    result = _request_json(
        prompt,
        schema,
        cfg,
        temperature=cfg.get("gemini_temperature", 0.2),
    )
    return result or []


def segment_pdf(
    pages: List[str], cfg: Dict[str, Any] | None = None
) -> List[Dict[str, Any]]:
    """Detect invoice boundaries within PDF pages."""
    cfg = cfg or {}
    joined_pages = "\n---\n".join(pages)
    prompt = (
        "Given the OCR text for each page of a PDF, identify the start page of each "
        "invoice and return metadata. Provide a JSON array of objects with fields "
        "'start_page', 'vendor', 'invoice_number', 'date', 'amount', and "
        "'confidence'. Confidence is a float between 0 and 1 expressing how sure "
        "you are that the invoice starts on the given page. Page numbers are 1-indexed.\n"
        + joined_pages
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
                "confidence": {"type": "number"},
            },
            "required": ["start_page"],
        },
    }
    result = _request_json(
        prompt,
        schema,
        cfg,
        temperature=cfg.get("gemini_temperature", 0.2),
    )
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
    result = _request_json(
        prompt,
        schema,
        cfg,
        temperature=cfg.get("gemini_temperature", 0.1),
    )
    return result or {}


def build_schema(
    headers: List[str],
    sample_rows: List[Dict[str, Any]],
    cfg: Dict[str, Any] | None = None,
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
    result = _request_json(
        prompt,
        schema,
        cfg,
        temperature=cfg.get("gemini_temperature", 0.1),
    )
    return result or {}
