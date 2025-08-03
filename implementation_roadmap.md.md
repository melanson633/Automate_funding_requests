# CRE Advance Request Automation – Implementation Roadmap & End‑to‑End Walk‑Through

*Last updated: 2025‑07‑29*

---

## 1  Repository Structure (target)

```text
Automate_funding_requests/
├── bin/
│   └── process_advance.py          # CLI entry‑point
├── cre_advance/
│   ├── __init__.py
│   ├── pipeline.py                 # Orchestrator
│   ├── excel_normalizer.py         # Yardi → Driver / INVOICES
│   ├── pdf_segmenter.py            # Multi‑invoice PDF → per‑invoice metadata
│   ├── ai_gemini.py                # Google Gemini 2.5 Pro wrapper
│   ├── file_packager.py            # Match, reorder, build outputs
│   ├── future_development/
│   │   └── orchestrator.py         # Placeholder for future orchestration features
│   └── utils/
│       ├── __init__.py
│       ├── env.py                  # .env + yaml config loader
│       ├── logging.py              # Central logging
│       └── errors.py               # Custom exceptions
├── configs/
│   ├── defaults.yaml
│   └── lenders/
│       └── example_lender.yaml
├── data/
│   ├── input/                      # raw files (one sub‑dir per draw)
│   ├── staging/                    # temp artefacts
│   └── output/                     # final deliverables
├── tests/
│   └── test_excel_normalizer.py
├── implementation_updates.txt      # Running log of roadmap & repo changes
└── requirements.txt
```

---

## 2  Module Responsibilities & Key Functions

| Module                                 | Core responsibilities                                                                                              | Key public functions                                             |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| **utils/env.py**                       | Read `.env`, merge global + lender configs                                                                         | `get_config(lender)`                                             |
| **utils/logging.py**                   | Console + file logging, log‑level control                                                                           | `get_logger(name)`                                               |
| **utils/errors.py**                    | Custom error classes                                                                                                | `ConfigError`, `PDFSegmentationError`, etc.                     |
| **ai_gemini.py**                       | Thin wrapper over Gemini SDK; prompt templates; retries                                                            | `map_schema`, `segment_pdf`, `extract_metadata`, `build_schema`  |
| **excel_normalizer.py**                | Detect header row (defaulting to row 4 of the **Driver** tab), call `map_schema`, clean + cast data, write tabs    | `normalize(path, config)`                                       |
| **pdf_segmenter.py**                   | Extract text/OCR, classify pages, call `segment_pdf`, derive page ranges                                                            | `segment(pdf_path, config)`                                     |
| **file_packager.py**                   | Match invoices, rebuild PDF, write Invoice Log, output Excel & JSON report                                         | `package(normalized_df, manifest, template_xlsx)`               |
| **pipeline.py**                        | Wire everything; enforce error‑rate threshold                                                                        | `run(args)`                                                     |
| **bin/process_advance.py**             | CLI argument parsing & pretty output                                                                                | –                                                                |
| **future_development/orchestrator.py**  | Placeholder for future orchestration capabilities (e.g., Gemini‑powered assistants)                                 | `suggest_next_action` (stub)                                     |

> **Update:** `pdf_segmenter` now drops email approvals and "Invoice Register" pages before segmentation.
*Note:* The canonical column schema for the Invoice Log is derived by default from row 4 of the **Driver** tab in the Funding Request template (Driver sheet). This can be overridden via lender‑specific YAML files in `configs/lenders/` fileciteturn0file0.

---

## 3  Build Order & Parallel Tracks

1. **Phase 0 – Utilities & Config loader**  
2. **Phase 1 – `ai_gemini` wrapper** (unblocks everything)  
3. **Phase 2 – Excel normalizer** (parallel with Phase 3)  
4. **Phase 3 – PDF segmenter**  
5. **Phase 4 – File packager**  
6. **Phase 5 – Pipeline + CLI**  
7. **Phase 6 – Tests & performance tuning**  

---

## 4  Function Sketches (inputs → outputs)

```python
# ai_gemini.py
def map_schema(headers: list[str], sample_rows: list[dict], target_fields: list[str], cfg: dict) -> dict[str,str]: ...
def segment_pdf(pages: list[str]) -> list[dict]: ...
def extract_metadata(text: str) -> dict: ...
def build_schema(headers: list[str], sample_rows: list[dict], cfg: dict) -> dict: ...

# excel_normalizer.py
def normalize(path: str, cfg: dict) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    # returns (normalized_df, raw_df)
    ...

# pdf_segmenter.py
def segment(pdf_path: str, cfg: dict) -> list[dict]:
    # returns manifest of invoices incl. page ranges & confidence
    ...

# file_packager.py
def package(df: pandas.DataFrame,
            manifest: list[dict],
            template_path: str,
            output_dir: str,
            cfg: dict) -> dict:
    # builds final Excel, PDF, JSON report; returns paths & stats
    ...
```

---

## 5  Edge‑Case & Error Handling

- Scanned pages → OCR only if `pypdf` text empty  
- Multi‑page invoices → derive `end_page = next.start_page – 1`  
- Unmatched invoices or low confidence → flag in JSON; abort only if error‑rate > 40 %  
- Per‑lender overrides via `configs/lenders/*.yaml`

---

## 6  LLM Collaboration Points

| Task               | Gemini prompt (high‑level)                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------|
| **Schema mapping** | “Map these raw headers & sample rows to the canonical fields … return JSON”                         |
| **Schema builder** | “Given raw headers & sample rows for a new lender, propose canonical fields and mappings; return JSON schema definition” |
| **PDF segmentation** | “For each page text, output vendor, invoice #, date, amount, *start_page* …”                       |
| **Metadata refinement** | “Refine metadata from a single invoice’s text to normalized values.”                           |
| **Fuzzy matching aid** | “Provide Excel row + candidate PDF metadata; ask which is best match.”                         |

---

## 7  Execution Sprints

| Sprint | Deliverables                                                       |
|--------|--------------------------------------------------------------------|
| 1      | Utilities + AI wrapper + basic tests                              |
| 2      | Excel normalizer working on sample Yardi reports                  |
| 3      | PDF segmenter with Gemini & OCR fallback                          |
| 4      | File packager producing reordered PDF & Invoice Log               |
| 5      | Pipeline + CLI, integration tests                                 |
| 6      | Polishing, performance tuning, documentation                      |

---

## 8  End‑to‑End Example Walk‑Through

### 8.1  Drop raw files

```
data/input/2025-07-MallRoad_Draw_10/
├── Expense Distribution Report.xlsx
├── DataGridExport.xlsx
├── Funding_Request_Template.xlsx   # has Driver & Invoice Log tabs
└── MallRoad_Invoices.pdf
```

### 8.2  Run command

```bash
python bin/process_advance.py \
  --excel data/input/2025-07-MallRoad_Draw_10/Funding_Request_Template.xlsx \
  --yardi data/input/2025-07-MallRoad_Draw_10/Expense\ Distribution\ Report.xlsx \
         data/input/2025-07-MallRoad_Draw_10/DataGridExport.xlsx \
  --pdf data/input/2025-07-MallRoad_Draw_10/MallRoad_Invoices.pdf \
  --lender example_lender \
  --output data/output
```

### 8.3  Pipeline flow

| Phase            | What happens                      | Key artefacts                            |
|------------------|-----------------------------------|-------------------------------------------|
| Normalize Excel  | Yardi → Driver tab (clean)        | `staging/Driver_clean.xlsx`              |
| Segment PDF      | AI boundary detection             | `staging/invoice_manifest.json`           |
| Match & reorder  | Align rows ↔ invoices, build PDF  | `staging/reordered_pages.pdf`            |
| Create Invoice Log | Copy selected rows, add totals   | –                                         |
| Write deliverables | Excel, PDF, report JSON         | `data/output/2025-07-MallRoad_Draw_10/…` |

### 8.4  Outputs ready to send

```
data/output/2025-07-MallRoad_Draw_10/
├── MallRoad_Draw_10_request.xlsx
├── MallRoad_Draw_10_invoices.pdf
└── MallRoad_Draw_10_report.json
```

- **MallRoad_Draw_10_request.xlsx** → Driver (hidden) + Invoice Log (visible)  
- **MallRoad_Draw_10_invoices.pdf** → invoices in Invoice‑Log order  
- **MallRoad_Draw_10_report.json** → unmatched / low confidence items

---

## 9  Next Steps

1. Approve this updated roadmap.  
2. Begin Phase 0 + Phase 1 implementation.  
3. Develop and integrate the new **schema builder** and **orchestrator** modules.  
4. Iterate on prompts and performance as real data reveals edge cases.

*End of document.*
