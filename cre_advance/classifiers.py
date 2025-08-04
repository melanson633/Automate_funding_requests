from __future__ import annotations

"""Page classification utilities."""

import re
from abc import ABC, abstractmethod
from typing import List

from .utils.logging import get_logger

logger = get_logger(__name__)


class PageClassifier(ABC):
    """Abstract base class for page classifiers."""

    @abstractmethod
    def classify(self, pages: list[str], cfg: dict) -> list[dict]:
        """Return classification dict for each page in ``pages``.

        Args:
            pages: Text content of pages.
            cfg: Config dictionary.

        Returns:
            List of classification dictionaries containing ``page_number``,
            ``category``, ``keep`` and ``confidence``.
        """


class HeuristicClassifier(PageClassifier):
    """Regex based page classifier.

    The classifier identifies invoice registers, email approvals, blank cover
    sheets and invoice pages using lender supplied vendor names.
    """

    _COVER_PAT = re.compile(r"(cover\s*sheet|invoice\s+packet)", re.I)

    def classify(self, pages: List[str], cfg: dict) -> List[dict]:
        vendors = [v.lower() for v in cfg.get("vendors", [])]
        results: List[dict] = []

        for idx, text in enumerate(pages, start=1):
            lower = text.lower()
            has_invoice_register = re.search(r"invoice\s+register", lower)
            has_workflow = re.search(r"workflow", lower)
            has_approval = re.search(r"approval", lower)

            has_from = re.search(r"^from:\s", lower, re.M)
            has_sent = re.search(r"^sent:\s", lower, re.M)
            has_subject = re.search(r"^subject:\s", lower, re.M)

            if has_invoice_register and has_workflow and has_approval:
                results.append(
                    {
                        "page_number": idx,
                        "category": "invoice_register",
                        "keep": False,
                        "confidence": 1.0,
                    }
                )
            elif has_from and (has_sent or has_subject):
                results.append(
                    {
                        "page_number": idx,
                        "category": "email_approval",
                        "keep": False,
                        "confidence": 1.0,
                    }
                )
            elif not lower.strip() or self._COVER_PAT.search(lower):
                results.append(
                    {
                        "page_number": idx,
                        "category": "blank_cover",
                        "keep": False,
                        "confidence": 1.0,
                    }
                )
            elif vendors and any(v in lower for v in vendors):
                results.append(
                    {
                        "page_number": idx,
                        "category": "invoice",
                        "keep": True,
                        "confidence": 1.0,
                    }
                )
            else:
                results.append(
                    {
                        "page_number": idx,
                        "category": "unknown",
                        "keep": True,
                        "confidence": 1.0,
                    }
                )

        return results
