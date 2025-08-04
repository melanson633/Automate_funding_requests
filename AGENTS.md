# AGENTS.md — Working Guidelines for Codex Agents

Welcome 👋 This repository automates CRE bank advance‑request packages.  
Codex‑style agents (ChatGPT **Code** / **Agent** mode or GitHub Copilot‑Agents)
should follow these guidelines to contribute smoothly and safely.

---

## 1  Project Primer

| Aspect | Detail |
|--------|--------|
| **Goal** | Normalise Yardi Excel + segment multi‑invoice PDFs → lender‑ready Excel + PDF packet |
| **Language** | Python 3.12 |
| **Core Libs** | `pandas`, `openpyxl`, `pypdf`, `pytesseract`, `google-genai` |
| **Entry Points** | `cre_advance/pipeline.py`, `bin/process_advance.py` |
| **Phase Status** | ✅ Phase 0–1 complete; 🚧 Phase 2+ in progress |

---

## 2  Agent Workflow

1. **Read the Roadmap**  
   Start with `implementation_roadmap.md`.

2. **Branch Per Phase**  
   Follow the naming convention `phase-N-<slug>`.

3. **Ask Clarifying Questions**  
   When requirements are ambiguous, *pause* and ask in the PR or chat.

4. **Small Commits**  
   Keep PRs <300 LOC where possible; include a concise description.

5. **Run Tests**  
   ```bash
   pytest -q
   ```  
   Ensure green before pushing.

6. **Logging**  
   Use `cre_advance.utils.logging.get_logger(__name__)`.

7. **API Keys & Secrets**  
   Never commit `.env` or API keys.  Access Google Gemini via `GOOGLE_API_KEY`.

8. **Dataset Handling**  
   Raw input files live under `data/input/` and **must not** be committed.

---

## 3  Tool Invocation Patterns

| Tool | When to call | Pattern |
|------|--------------|---------|
| **Gemini (`ai_gemini.*`)** | Schema mapping, PDF segmentation, metadata extraction | Always request `application/json`; use `response_schema` for validation |
| **pytesseract** | `pypdf` text empty | Respect `cfg.ocr.tesseract_cmd` |
| **Levenshtein** | Vendor fuzzy‑matching | ratio ≥ 0.8 |

---

## 4  Error Handling

* Fail early: raise `NormalizationError`, `PDFSegmentationError`, etc.  
* If >40 % of invoices unmatched → abort pipeline and surface a clear message.

---

## 5  Coding Conventions

* **Imports**: standard lib, third‑party, local — separated by blank lines.  
* **Type Hints**: required for all public functions.  
* **Docstrings**: Google style.  
* **Formatting**: `black` (line length = 88), `isort`, `flake8`.  
* **Logging**: debug‑level for data shapes, info‑level for high‑level steps.

---

## 6  PR Checklist 🗒️

- [ ] Unit tests added / updated  
- [ ] `pytest` passes  
- [ ] `pre‑commit run --all-files` passes  
- [ ] Docs updated (README / roadmap)

---

## 7  Contact & Support

*Open an issue → label `question` or tag `@melanson633` in a PR.*

Happy coding! ☕️
