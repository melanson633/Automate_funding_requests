from __future__ import annotations

"""Invoice segmentation and metadata extraction utilities."""

import asyncio
import re
from datetime import datetime
from typing import Dict, List

from . import ai_gemini
from .utils.logging import get_logger

logger = get_logger(__name__)


class InvoiceSegmenter:
    """Segment invoice pages and extract metadata."""

    def segment_invoices(self, texts: List[str], cfg: dict) -> List[dict]:
        """Return invoice manifest for ``texts``.

        Args:
            texts: List of page texts after classification and filtering.
            cfg: Configuration dictionary.

        Returns:
            List of manifest dictionaries with start/end pages and metadata.
        """
        starts = ai_gemini.detect_invoice_starts(texts)
        manifest: List[dict] = []
        if not starts:
            return manifest

        for idx, start in enumerate(starts):
            end = starts[idx + 1] - 1 if idx + 1 < len(starts) else len(texts) - 1
            invoice_pages = texts[start : end + 1]
            invoice_text = "\n".join(invoice_pages)
            metadata: Dict[str, str] = {}
            try:
                metadata = ai_gemini.extract_metadata(invoice_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "extract_metadata failed: %s", exc, extra={"context": "segment"}
                )
                metadata = {}
            fallback = self._regex_extract_metadata(invoice_text)
            combined = {**fallback, **{k: v for k, v in metadata.items() if v}}
            normalized = self._normalize_metadata(combined)
            manifest.append(
                {
                    "start_page": start + 1,
                    "end_page": end + 1,
                    "vendor": normalized["vendor"],
                    "invoice_number": normalized["invoice_number"],
                    "date": normalized["date"],
                    "amount": normalized["amount"],
                    "confidence": 1.0,
                }
            )

        self._reconcile_with_log(manifest, cfg)
        return manifest

    async def segment_invoices_async(
        self, texts_list: List[List[str]], cfg: dict
    ) -> List[List[dict]]:
        tasks = [
            asyncio.to_thread(self.segment_invoices, texts, cfg) for texts in texts_list
        ]
        return await asyncio.gather(*tasks)

    def _regex_extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata using simple regex heuristics."""
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

        return {
            "vendor": vendor,
            "invoice_number": inv_match.group(1) if inv_match else "",
            "date": date_match.group(1) if date_match else "",
            "amount": amt_match.group(1) if amt_match else "",
        }

    def _normalize_metadata(self, meta: Dict[str, str]) -> Dict[str, str]:
        """Normalize metadata fields to canonical forms."""
        vendor = str(meta.get("vendor", "")).strip()
        invoice_number = str(meta.get("invoice_number", "")).strip()
        date_raw = str(meta.get("date", "")).strip()
        date_norm = ""
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
            try:
                date_norm = datetime.strptime(date_raw, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
        else:
            date_norm = date_raw

        amt_raw = str(meta.get("amount", "")).replace(",", "").replace("$", "").strip()
        amount = ""
        try:
            amount = f"{float(amt_raw):.2f}" if amt_raw else ""
        except ValueError:
            amount = amt_raw

        return {
            "vendor": vendor,
            "invoice_number": invoice_number,
            "date": date_norm,
            "amount": amount,
        }

    def _reconcile_with_log(self, manifest: List[dict], cfg: dict) -> None:
        """Fill missing metadata using an Excel log if available."""
        log = cfg.get("excel_log") or []
        for item in manifest:
            for row in log:
                if (
                    str(row.get("invoice_number", "")).strip().lower()
                    == item["invoice_number"].lower()
                ):
                    item["vendor"] = (
                        item["vendor"] or str(row.get("vendor", "")).strip()
                    )
                    item["date"] = item["date"] or str(row.get("date", "")).strip()
                    amount = str(row.get("amount", "")).strip()
                    if not item["amount"] and amount:
                        item["amount"] = amount
                    break
