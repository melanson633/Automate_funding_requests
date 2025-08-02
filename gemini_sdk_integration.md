# Repository Insights

## Purpose & Scope
`Automate_funding_requests` automates credit‑advance packages for CRE lenders:

1. Ingest CSV + bundled PDFs  
2. Segment invoices (Gemini Vision)  
3. Extract metadata (Gemini 2.5 Pro)  
4. Map to `InvoiceV2` schema  
5. Emit ZIP with `manifest.json`

### Directory Map

| Path | Responsibility |
|------|----------------|
| src/ingest.py | CSV validation |
| src/pdf_segmenter.py | Vision page‑split |
| src/metadata_extractor.py | Text → JSON |
| src/schema_mapper.py | Schema enforcement |
| src/integration/gemini_client.py | Unified SDK adapter |
| src/cli.py | CLI |
| config/*.yaml | Model settings |

# Page classification

`ai_gemini.classify_pages()` sends each page's OCR text to Gemini and returns
labels such as `invoice`, `invoice_register`, or `email_approval`. Only pages
with `keep: true` are fed into downstream segmentation.

# Gemini SDK Cheat‑Sheet (v 1.27.0, retrieved 2025-07-29)

| Area | Minimal Call | Key Args / Patterns | Notes |
|------|--------------|---------------------|-------|
| **Client** | `genai.Client(api_key="…")` | `vertexai=True` for GCP; `http_options` for retries | One client per app |
| **Generate** | `client.models.generate_content(...)` | `contents`, `types.GenerateContentConfig` (temp, top_p/k, max_tokens, tools) | `.text` raw, `.parsed` JSON |
| **Streaming** | `generate_content_stream` | iterate chunks | |
| **Function Calling** | Auto: `tools=[py_fn]`; Manual: declare `types.FunctionDeclaration` | Disable auto via `AutomaticFunctionCallingConfig(disable=True)` | |
| **JSON Enforcement** | `response_mime_type="application/json"` + `response_schema` | Enum: `text/x.enum` | |
| **Count / Embed** | `count_tokens`, `embed_content` | same `contents` arg | |
| **Files API** | `client.files.upload(...)` | Dev API only | |
| **HttpOptions** | `types.HttpOptions(api_version='v1', retry_options=types.RetryOptions(max_retries=3))` | proxy via `client_args` | |
| **Safety** | `types.SafetySetting(...)` | list in config | |
| **Async** | `client.aio.models.generate_content` | mirrors sync | |

> **JSON + Function pattern**
> ```python
> from google import genai
> from google.genai import types
> def total(text: str) -> float: ...
> client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
> resp = client.models.generate_content(
>     model="gemini-2.5-pro",
>     contents="Extract total: " + text,
>     config=types.GenerateContentConfig(
>         response_mime_type="application/json",
>         response_schema={"type":"object","properties":{"total":{"type":"number"}}},
>         tools=[total],
>     ),
> )
> print(resp.parsed)
> ```

# Quick‑Start Commands

```bash
# Install SDK
pipx install google-genai==1.27.0

# Clone & dev‑install project
git clone https://github.com/melanson633/Automate_funding_requests.git
cd Automate_funding_requests
pipx runpip automate-funding-requests install -e .[dev]

# Auth
export GOOGLE_API_KEY="your_key"

# Run pipeline
afr run --input ./incoming --output ./out --model gemini-2.5-pro
```

# References

1. Google Gen AI Python SDK v1.27.0 – github.com/googleapis/python-genai (retrieved 2025-07-29)  
2. API Examples – googleapis.github.io/python-genai/genai.html (retrieved 2025-07-29)  
3. Automate_funding_requests implementation_roadmap.md (retrieved 2025-07-29)