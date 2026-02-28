# Repo Explorer → Claude Code Skill Miner Report

## Pass 1: Structural Survey (Broad Discovery)

### Repository fit for CRE finance/accounting
This repository is **strongly aligned** with CRE finance/accounting operations, especially construction draw workflows: Yardi export normalization, invoice packet segmentation, lender template output, and exception reporting.

## Structural Inventory

### Root
- `README.md` — Product overview, CLI usage, architecture summary, and QA commands.
- `implementation_roadmap.md` — End-to-end design intent, module responsibilities, phased implementation plan, and pipeline walkthrough.
- `gemini_sdk_integration.md` — Gemini SDK usage notes and integration patterns for JSON-constrained responses.
- `requirements.txt` — Core runtime dependencies (`pandas`, `openpyxl`, `pypdf`, `pytesseract`, `google-genai`, etc.).
- `scaffold.sh` — Bootstrap/helper scripting for project setup.
- `AGENTS.md` — Repo-level agent coding workflow and guardrails.

### `bin/`
- `bin/process_advance.py` — CLI entrypoint for funding-request processing, argument parsing, and summary output.

### `cre_advance/` (primary application package)
- `cre_advance/pipeline.py` — Orchestration layer (Excel normalize → PDF segment → package outputs), resume mode, staging artifacts, metrics timings.
- `cre_advance/excel_normalizer.py` — Yardi workbook ingestion, report-type detection, schema/header mapping (AI + fuzzy fallback), type casting, normalization outputs.
- `cre_advance/pdf_segmenter.py` — PDF page text extraction, OCR fallback, page filtering/classification, manifest assembly, fallback strategies.
- `cre_advance/vision_segmenter.py` — Vision/multimodal PDF segmentation using Gemini with page-image+text prompting.
- `cre_advance/segmenters.py` — Invoice boundary detection and metadata extraction orchestration across page text blocks.
- `cre_advance/file_packager.py` — Invoice row ↔ PDF manifest matching, reordered invoice PDF build, Excel output writing, unmatched/exclusion reporting.
- `cre_advance/classifiers.py` — Page classification interfaces + implementations (Gemini classifier and rule-based heuristic classifier).
- `cre_advance/ai_gemini.py` — Gemini client wrapper, retries, prompt loading, schema mapping, page classification, structure detection, multimodal invocation utilities.
- `cre_advance/pdf_parser.py` — OCR-oriented PDF text extraction helpers, image preprocessing (deskew), parser abstraction.
- `cre_advance/metrics.py` — SQLite-based metric + feedback logging utilities.
- `cre_advance/secret_manager.py` — Secret retrieval helper from environment.
- `cre_advance/utils/env.py` — Config loader/merger (`defaults.yaml` + lender overrides + env model tiering).
- `cre_advance/utils/logging.py` — Context-aware logger formatting and adapter utilities.
- `cre_advance/utils/errors.py` — Domain exception classes (`ConfigError`, `NormalizationError`, `PDFSegmentationError`, `PackagingError`).
- `cre_advance/utils/pdf_utils.py` — Generic PDF merge utility for multi-file packet preparation.

### `configs/`
- `configs/defaults.yaml` — Global defaults for model settings, OCR behavior, PDF segmentation, scoring, and packaging thresholds.
- `configs/lenders/*.yaml` — Lender- or report-specific overrides (header rows, schema maps, prompt overrides, thresholds, vendor lists).
- `configs/prompts/*.yaml` + `system_instruction.txt` — Prompt templates for page classification and segmentation workflows.
- `configs/schema_versions/*.yaml` — Versioned schema artifacts for reusable mapping definitions.

### `tests/`
- Unit/integration-like tests across normalization, segmentation, Gemini wrapper behavior, config merging, packaging rules, PDF utils, and pipeline orchestration.
- `tests/test_*` files document expected business rules (e.g., matching thresholds, fallback behavior, and warning/abort criteria).

### CI and agent/meta config
- `.github/workflows/ci.yml` — CI pipeline (lint, pre-commit, pytest, coverage, secret scan).
- `.cursor/rules/automatefunding.mdc` — IDE-agent coding constraints and architecture context.

## Pass 1 Highlights by requested lens
- **Languages/frameworks/libs:** Python 3.12 + pandas/openpyxl/pypdf/pytesseract/google-genai/pytest.
- **Data models/schemas/configs:** YAML-driven lender config model, prompt model, versioned schema files, manifest dictionaries, metrics schema (SQLite table).
- **Templates/prompts/instructions:** prompt YAML + system instruction text + roadmap and AGENTS docs.
- **Automation/ETL logic:** Excel normalization, PDF parsing/OCR/classification/segmentation, packaging/export workflow.
- **Financial/business logic:** Invoice matching scores, tolerance thresholds, confidence/unmatched thresholds, report-type detection and mapping.
- **External integrations:** Google Gemini API, local OCR engine (Tesseract), filesystem artifact pipeline.
- **Document processing:** heavy PDF parsing, OCR fallback, page-level classification, per-invoice packet assembly.
- **Reusable utilities:** config merge, logging context adapter, PDF merge helper, custom exceptions.

---

## Pass 2: Pattern Analysis (Targeted Evaluation)

### 1) `cre_advance/excel_normalizer.py`
- **What it does:** Reads Yardi-like workbooks, detects report structure, maps raw headers to canonical fields (AI + fuzzy fallback), casts values, and returns normalized tabular output.
- **CRE applicability:** Directly valuable for AP invoice normalization, lender draw package prep, and repeated vendor file standardization.
- **Modularity potential:** **HIGH**.
- **Dependencies:** `ai_gemini`, `pandas`, lender config YAML, metrics logger.
- **Valuable domain knowledge:** Header-row inference, schema auto-save/versioning behavior, mapping fallback logic.

### 2) `cre_advance/file_packager.py`
- **What it does:** Matches normalized invoice rows to PDF manifest entries using weighted scoring (invoice number/vendor/date/amount), builds reordered invoice PDF, writes deliverable workbook/report.
- **CRE applicability:** High-value for lender packet QA and AP control workflows where source docs must match register ordering.
- **Modularity potential:** **HIGH**.
- **Dependencies:** `pandas`, `openpyxl`, `pypdf`, config scoring thresholds.
- **Valuable domain knowledge:** Matching thresholds/tolerances, duplicate detection, unmatched handling/exclusion reporting.

### 3) `cre_advance/pdf_segmenter.py`
- **What it does:** End-to-end segmentation pipeline: text extraction, OCR fallback, page filtering (register/email removal), invoice grouping, confidence validation, multiple fallbacks.
- **CRE applicability:** Critical for construction draw backup disaggregation and monthly AP packet ingestion.
- **Modularity potential:** **MEDIUM-HIGH**.
- **Dependencies:** parser, classifier, segmenter, OCR stack, Gemini wrapper, config thresholds.
- **Valuable domain knowledge:** practical fallback ladder and confidence/unmatched gating.

### 4) `cre_advance/segmenters.py`
- **What it does:** Converts classified page text into invoice boundaries and metadata, supports async metadata enrichment and range derivation.
- **CRE applicability:** Reusable for invoice packet decomposition across lenders and JV properties.
- **Modularity potential:** **HIGH**.
- **Dependencies:** `ai_gemini`, datetime parsing assumptions, manifest schema.
- **Valuable domain knowledge:** boundary derivation pattern (`next.start_page - 1`), metadata fill/reconcile routines.

### 5) `cre_advance/classifiers.py`
- **What it does:** Provides abstraction for page classification and two implementations: Gemini and rules/regex heuristic fallback.
- **CRE applicability:** Useful for filtering non-invoice pages (approvals/register/cover sheets) in lender backup docs.
- **Modularity potential:** **HIGH**.
- **Dependencies:** `ai_gemini` optional for LLM mode; regex-only mode runs standalone.
- **Valuable domain knowledge:** classification taxonomy and deterministic fallbacks.

### 6) `cre_advance/ai_gemini.py`
- **What it does:** Centralized Gemini client + retries, prompt loading, JSON parsing, cached helper functions for schema mapping/classification/manifest extraction.
- **CRE applicability:** Strong for standardizing AI integrations in recurring accounting ingestion tasks.
- **Modularity potential:** **MEDIUM** (tied to prompt/config conventions and model assumptions).
- **Dependencies:** `google-genai`, YAML prompts, config keys, cache strategy.
- **Valuable domain knowledge:** JSON-enforced prompting patterns, multimodal invoke wrapper, deterministic parser guards.

### 7) `cre_advance/utils/env.py`
- **What it does:** Merges global defaults, lender-specific YAML, and env overrides into runtime configuration.
- **CRE applicability:** Excellent for multi-lender/multi-property configuration governance.
- **Modularity potential:** **HIGH**.
- **Dependencies:** YAML files + env vars.
- **Valuable domain knowledge:** model-tier fallback, nested override merge semantics, prompt/scoring override support.

### 8) `cre_advance/pdf_parser.py`
- **What it does:** OCR-focused parser with preprocessing (deskew) and Tesseract integration when PDF text extraction is weak.
- **CRE applicability:** High for scanned invoices/leases/statements across property accounting workflows.
- **Modularity potential:** **MEDIUM-HIGH**.
- **Dependencies:** PIL, pytesseract, fitz/PDF interfaces.
- **Valuable domain knowledge:** OCR decisioning and image preprocessing knobs.

### 9) `cre_advance/vision_segmenter.py`
- **What it does:** Multimodal segmentation path using page imagery + prompt templates to derive invoice manifests.
- **CRE applicability:** Useful where text-layer PDFs are poor and scanned backups dominate.
- **Modularity potential:** **MEDIUM** (depends on Gemini multimodal access + specific prompt contract).
- **Dependencies:** Gemini multimodal wrapper, PDF page image extraction.
- **Valuable domain knowledge:** robust fallback when OCR-only segmentation is brittle.

### 10) `configs/lenders/*.yaml` + `configs/defaults.yaml`
- **What it does:** Encodes lender/report-specific data contracts, mapping assumptions, scoring tolerances, and workflow toggles.
- **CRE applicability:** Very high for governance, repeatability, and onboarding new JV lenders.
- **Modularity potential:** **HIGH**.
- **Dependencies:** loader merge semantics (`utils/env.py`).
- **Valuable domain knowledge:** field mapping, thresholds, prompt overrides, vendor hints.

### 11) `configs/prompts/*.yaml` + `system_instruction.txt`
- **What it does:** Prompt templates for page classification and segmentation tasks, including output structure expectations.
- **CRE applicability:** Useful as reusable instruction assets for AI-assisted document ops.
- **Modularity potential:** **HIGH**.
- **Dependencies:** prompt loader + AI invocation wrapper.
- **Valuable domain knowledge:** task framing and response formatting conventions.

### 12) `cre_advance/pipeline.py` + `bin/process_advance.py`
- **What it does:** Operational orchestration, staging artifact persistence, resume flow, and CLI operational interface.
- **CRE applicability:** Strong for batch draw processing and repeatable monthly runs.
- **Modularity potential:** **MEDIUM** (thin orchestrator but coupled to internal module contracts).
- **Dependencies:** all core modules.
- **Valuable domain knowledge:** failure recovery via staged artifacts and resumability.

### 13) `cre_advance/metrics.py`
- **What it does:** Persists runtime metrics and feedback in SQLite for quality/performance monitoring.
- **CRE applicability:** Useful for auditability, SLA monitoring, and model-quality feedback loops.
- **Modularity potential:** **HIGH**.
- **Dependencies:** SQLite only.
- **Valuable domain knowledge:** measurable KPI pattern for doc-ops pipelines.

### 14) `tests/` suite as executable spec
- **What it does:** Captures expected matching logic, fallback behavior, config merge behavior, and integration boundaries.
- **CRE applicability:** Gives reliable acceptance criteria for future CRE automation skills.
- **Modularity potential:** **MEDIUM-HIGH**.
- **Dependencies:** pytest + existing modules.
- **Valuable domain knowledge:** codified edge cases (OCR, unmatched tolerances, heuristic fallbacks).

---

## Pass 3: Skill Synthesis (Ranked Recommendations)

### 1. CRE Invoice Packet Reconciliation Skill
**Value Proposition:** This Skill would automate one of the highest-friction draw tasks: proving every invoice listed in an Excel register ties to supporting PDF pages in the exact lender-required order. It reduces manual packet QA and rework by applying deterministic row-to-document scoring, mismatch detection, and exception reporting.

**Source Components:**
- `cre_advance/file_packager.py`: Weighted row↔invoice matching, duplicate handling, reordered PDF generation, report writing.
- `cre_advance/utils/pdf_utils.py`: Multi-PDF merge support for packet prep.
- `tests/test_file_packager.py`: Edge-case behavior and scoring threshold examples.
- `configs/defaults.yaml`: Default matching tolerances and unmatched thresholds.

**Replication Targets:**
- **Copy:** `_match_invoices` scoring logic, duplicate checks, unmatched summary structure.
- **Modify:** canonical fields (support draw-specific columns like cost code, draw number).
- **Build new:** lightweight adapter for non-Yardi ledgers and a configurable exception dashboard template.

**Modularity Assessment:** Mostly self-contained matching+packaging logic with clear inputs/outputs (tabular ledger + manifest + PDF). Primary boundary is dependency on manifest schema and openpyxl/pypdf I/O. **Rating: HIGH**.

**Proposed SKILL.md Structure:**
```yaml
name: cre-invoice-packet-reconciliation
description: Reconcile invoice logs to PDF backups, reorder pages, and emit mismatch reports for lender-ready draw packages.
```
- Trigger conditions and required inputs (ledger columns + manifest schema)
- Stepwise reconciliation workflow with adjustable scoring weights
- Validation checklist (duplicates, unmatched ratio, ordering correctness)
- Output artifacts template (reordered PDF, exception JSON, summary table)
- Troubleshooting fallbacks for weak invoice metadata

**Confidence Level:** HIGH

### 2. Yardi-to-Lender Excel Normalization Skill
**Value Proposition:** This Skill accelerates monthly draw prep by standardizing messy Yardi exports into lender-ready schemas with repeatable rules and fallback matching. It directly reduces spreadsheet manipulation time and human mapping errors across properties.

**Source Components:**
- `cre_advance/excel_normalizer.py`: Header detection, report detection, schema mapping, fuzzy fallback, type casting.
- `cre_advance/ai_gemini.py`: Header mapping and Excel structure detection helpers.
- `configs/lenders/*.yaml`: Report-specific schema maps/header-row definitions.
- `tests/test_excel_normalizer.py`: Behavioral examples for fallback and schema autosave.

**Replication Targets:**
- **Copy:** header-row inference + map/cast flow.
- **Modify:** canonical output schema for each lender template and internal accounting model.
- **Build new:** optional mapping QA sheet highlighting confidence and unmapped fields.

**Modularity Assessment:** Clear transformation pipeline with config-driven behavior and test coverage. Coupling to Gemini is optional due to fuzzy fallback patterns. **Rating: HIGH**.

**Proposed SKILL.md Structure:**
```yaml
name: yardi-ledger-normalization
description: Normalize Yardi or similar AP exports into lender-specific invoice-log schemas with validation and fallback mapping.
```
- Input contract and supported report signatures
- Mapping strategy hierarchy (explicit map → AI map → fuzzy)
- Data type and quality rules (date, amount, vendor normalization)
- Schema versioning and override strategy
- Verification workflow (row counts, null checks, unmapped threshold)

**Confidence Level:** HIGH

### 3. Invoice PDF Segmentation + OCR Fallback Skill
**Value Proposition:** This Skill extracts individual invoice ranges from bundled backup PDFs, including scans and mixed-quality packets. It eliminates manual page splitting and improves throughput for draw processing and AP audits.

**Source Components:**
- `cre_advance/pdf_segmenter.py`: End-to-end segmentation orchestration and fallback ladder.
- `cre_advance/segmenters.py`: Invoice start detection and metadata reconciliation.
- `cre_advance/classifiers.py`: Gemini + heuristic page classification.
- `cre_advance/pdf_parser.py`: OCR + deskew extraction path.
- `tests/test_pdf_segmenter.py`, `tests/test_segmenters.py`: Fallback/edge-case definitions.

**Replication Targets:**
- **Copy:** classify/filter/segment flow, confidence validation, fallback routing.
- **Modify:** classification taxonomy and thresholds for firm-specific document types.
- **Build new:** configurable post-segmentation review queue for low-confidence invoices.

**Modularity Assessment:** Technically modular but integrates several services (OCR, LLM, parser, config). Works best packaged as a composite skill with pluggable classifiers. **Rating: MEDIUM-HIGH**.

**Proposed SKILL.md Structure:**
```yaml
name: invoice-pdf-segmentation-ocr
description: Split mixed invoice packet PDFs into invoice-level page ranges using OCR, classification, and confidence-aware fallbacks.
```
- Supported PDF quality profiles and prerequisites
- Classification and exclusion rules (register/email/cover)
- Segmentation algorithm and fallback decision tree
- Confidence gating + escalation workflow
- Output manifest schema and QA checklist

**Confidence Level:** HIGH

### 4. Multi-Lender Config Governance Skill
**Value Proposition:** CRE finance teams often support many lender templates and evolving reporting rules. This Skill provides a disciplined pattern to manage per-lender YAML overrides, prompt variants, and threshold tuning without code rewrites.

**Source Components:**
- `cre_advance/utils/env.py`: Deep merge and env override behavior.
- `configs/defaults.yaml`: Baseline operational settings.
- `configs/lenders/*.yaml`: Lender/report-specific overrides.
- `configs/schema_versions/*.yaml`: Versioned schema capture.
- `tests/test_env_loader.py`: Merge and override expectations.

**Replication Targets:**
- **Copy:** layered config merge strategy and override precedence.
- **Modify:** lender naming taxonomy + firm-specific governance metadata (owner, approval date, effective period).
- **Build new:** config lint/check command and change-control template.

**Modularity Assessment:** Very self-contained: pure configuration patterns + merge helper with minimal runtime dependencies. **Rating: HIGH**.

**Proposed SKILL.md Structure:**
```yaml
name: cre-lender-config-governance
description: Create and maintain layered lender configs (defaults + overrides) for repeatable CRE finance automation.
```
- Config layer model and precedence rules
- New-lender onboarding template
- Validation rules and regression tests
- Prompt/scoring override examples
- Versioning and rollout checklist

**Confidence Level:** HIGH

### 5. Gemini JSON-Contract Document AI Skill
**Value Proposition:** This Skill standardizes how finance document tasks call Gemini with strict JSON contracts, retries, and parse guards. It is high leverage across lease abstraction, covenant extraction, vendor invoice metadata, and other structured extraction tasks.

**Source Components:**
- `cre_advance/ai_gemini.py`: Prompt loading, retries, cache, multimodal invoke wrappers, response parsing.
- `configs/prompts/classify_pages_prompt.yaml`: Structured page classification prompt format.
- `configs/prompts/segment_pdf_prompt.yaml`: Segmentation prompt structure.
- `gemini_sdk_integration.md`: SDK invocation patterns and config notes.
- `tests/test_ai_gemini.py`, `tests/test_prompt_loader.py`: Contract validation examples.

**Replication Targets:**
- **Copy:** JSON-only prompt/parse pattern, retry logic, cached request wrappers.
- **Modify:** schema outputs for other CRE use cases (lease terms, covenants, expense categories).
- **Build new:** reusable schema registry and response validator snippets per task.

**Modularity Assessment:** Reusable but moderately coupled to current prompt loader conventions and response schema assumptions; still extractable as a toolkit-style Skill. **Rating: MEDIUM**.

**Proposed SKILL.md Structure:**
```yaml
name: gemini-json-contract-extraction
description: Use Gemini with strict JSON contracts, retries, and validators for CRE finance document extraction workflows.
```
- Trigger criteria and selection of extraction schema
- Prompt construction rules (system + task + examples)
- Retry/backoff and parsing guardrails
- Validation pipeline and confidence flags
- Example adapters for invoices, leases, covenants

**Confidence Level:** MEDIUM-HIGH
