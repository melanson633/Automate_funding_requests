#!/usr/bin/env python3
from __future__ import annotations

"""CLI entry point for the CRE advance request pipeline."""

import argparse
import sys
from pathlib import Path

from cre_advance import pipeline
from cre_advance.utils import get_logger


class _Colour:
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    RESET = "\033[0m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a funding request")
    parser.add_argument(
        "--excel",
        required=True,
        help="Funding request template",
    )
    parser.add_argument(
        "--yardi",
        nargs="+",
        required=True,
        help="Yardi Excel exports",
    )
    parser.add_argument("--pdf", required=True, help="PDF of invoices")
    parser.add_argument("--lender", required=True, help="Lender config key")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume using the latest staging files",
    )
    return parser.parse_args()


def _print_summary(summary: dict) -> None:
    print(f"{_Colour.GREEN}Excel{_Colour.RESET}:  {summary.get('excel')}")
    print(f"{_Colour.GREEN}PDF{_Colour.RESET}:    {summary.get('pdf')}")
    print(f"{_Colour.GREEN}Report{_Colour.RESET}: {summary.get('report')}")
    if summary.get("unmatched_rows"):
        print(
            f"{_Colour.RED}Unmatched rows: {summary['unmatched_rows']}"
            f"{_Colour.RESET}"
        )
    else:
        print(f"{_Colour.CYAN}All rows matched successfully{_Colour.RESET}")
    metrics = summary.get("metrics")
    if metrics:
        print("--- Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")


def main() -> None:
    args = _parse_args()
    logger = get_logger(__name__)
    if args.resume:
        staging = Path("data/staging")
        norm_files = sorted(staging.glob("normalized_*.xlsx"))
        manifest_files = sorted(staging.glob("manifest_*.json"))
        if not norm_files or not manifest_files:
            logger.error("No staging files found to resume")
            sys.exit(1)
        args.normalized = str(norm_files[-1])
        args.manifest = str(manifest_files[-1])
    else:
        args.normalized = None
        args.manifest = None
    try:
        summary = pipeline.run(args)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)

    _print_summary(summary)


if __name__ == "__main__":  # pragma: no cover
    main()
