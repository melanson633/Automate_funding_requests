#!/usr/bin/env bash
set -e

# Paths
paths=(
  "bin/process_advance.py"
  "cre_advance/__init__.py"
  "cre_advance/pipeline.py"
  "cre_advance/excel_normalizer.py"
  "cre_advance/pdf_segmenter.py"
  "cre_advance/ai_gemini.py"
  "cre_advance/file_packager.py"
  "cre_advance/utils/__init__.py"
  "cre_advance/utils/env.py"
  "cre_advance/utils/logging.py"
  "cre_advance/utils/errors.py"
  "configs/lenders/example_lender.yaml"
  "configs/defaults.yaml"
  "data/input/.keep"
  "data/staging/.keep"
  "data/output/.keep"
  "tests/test_excel_normalizer.py"
  ".cursor/config.json"
  ".env.example"
  "requirements.txt"
  ".gitignore"
)

# Check existing paths
for p in "${paths[@]}"; do
  if [ -e "$p" ]; then
    echo "$p already exists. Exiting." >&2
    exit 0
  fi
done

# Create directories
mkdir -p bin cre_advance/utils configs/lenders data/input data/staging data/output tests .cursor

# Create python files with minimal content
py_content='"""Module placeholder."""
# TODO
pass
'
for f in \
  bin/process_advance.py \
  cre_advance/__init__.py \
  cre_advance/pipeline.py \
  cre_advance/excel_normalizer.py \
  cre_advance/pdf_segmenter.py \
  cre_advance/ai_gemini.py \
  cre_advance/file_packager.py \
  cre_advance/utils/__init__.py \
  cre_advance/utils/env.py \
  cre_advance/utils/logging.py \
  cre_advance/utils/errors.py \
  tests/test_excel_normalizer.py
  do
  printf '%s' "$py_content" > "$f"
 done

# Placeholder yaml files
printf '# Example lender config\n' > configs/lenders/example_lender.yaml
printf '# Default configs\n' > configs/defaults.yaml

# .env.example
printf 'GOOGLE_API_KEY=your_key_here\n' > .env.example

# requirements.txt
cat <<'REQ' > requirements.txt
pandas
openpyxl
pypdf
python-dotenv
google-genai
REQ

# .gitignore
cat <<'GI' > .gitignore
data/*
*.pyc
__pycache__/
.env
GI

# .cursor config
cat <<'CUR' > .cursor/config.json
{
  "entryPoints": [],
  "links": [],
  "ignores": []
}
CUR

# touch .keep files to preserve dirs
for d in data/input data/staging data/output; do
  touch "$d/.keep"
 done

# Remove .keep from list of tracked paths for check earlier

# Done
 echo "✅ Scaffold complete."
