Automate Funding Requests

This project automates the preparation and organization of bank advance requests for commercial real estate (CRE) properties. It intelligently transforms Yardi-exported Excel reports and multi-invoice PDFs into standardized lender-ready funding packages using Python and Google’s Gemini 2.5 Pro AI.

Features
	•	Automated Excel Normalization: Parses and standardizes Excel reports from Yardi Voyager to match the required Invoice Log format.
	•	AI-Driven PDF Segmentation: Splits multi-invoice PDF documents into individual invoices, extracting critical invoice metadata automatically.
	•	Invoice Alignment: Ensures PDF invoice pages precisely match the order of entries in the Excel Invoice Log.
	•	Modular Extensible Pipeline: Clean Python-based pipeline structured for easy adaptation to varying lender and property-specific formats.

Directory Structure

Automate_funding_requests/
├── bin/
│   └── process_advance.py
├── cre_advance/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── excel_normalizer.py
│   ├── pdf_segmenter.py
│   ├── ai_gemini.py
│   ├── file_packager.py
│   └── utils/
│       ├── __init__.py
│       ├── env.py
│       ├── logging.py
│       └── errors.py
├── configs/
│   ├── lenders/
│   │   └── example_lender.yaml
│   └── defaults.yaml
├── data/
│   ├── input/
│   ├── staging/
│   └── output/
├── tests/
│   └── test_excel_normalizer.py
├── .cursor/
│   └── config.json
├── .env.example
├── requirements.txt
├── .gitignore
└── README.md

Getting Started

Installation

Clone the repository:

git clone https://github.com/<your-username>/Automate_funding_requests.git
cd Automate_funding_requests

Install dependencies:

pip install -r requirements.txt

Configure your environment variables by copying .env.example to .env and filling in your credentials:

cp .env.example .env

Running the Pipeline

Place your raw Yardi-exported Excel and PDF files into the ``data/input/`` directory.

Run the automation script with the required arguments:

```bash
python bin/process_advance.py \
  --excel Funding_Request_Template.xlsx \
  --yardi Expense_Report.xlsx DataGridExport.xlsx \
  --pdf Invoices.pdf \
  --lender example_lender \
  --output data/output
```

Processed funding packages will be saved under the specified output directory.

Dependencies
	•	Python
	•	pandas
	•	openpyxl
	•	pypdf
	•	python-dotenv
	•	google-genai (Gemini 2.5 Pro SDK)

Development Notes
	•	Configuration files for lender-specific adjustments should be placed in configs/lenders/.
	•	Utilize Cursor IDE with predefined rules in .cursor/config.json for intelligent development assistance.

Running Tests
    Execute the unit tests and style checks before committing:
    ```bash
    pre-commit run --all-files  # if configured
    pytest -q
    ```

⸻

Happy automating!
