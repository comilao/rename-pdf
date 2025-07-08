# PDF Renamer

## Overview

Every month I receive many PDFs from my banks, insurance/investment companies, utility companies, etc. I want to rename these PDFs based on their content and some rules. Hence I use LLM to analyze the PDFs and rename and move them to a specific directory.

## How to Use

The script let local Ollama LLM analyzes PDFs in the project root directory, generates a new name for each PDF, and moves them to a destination directory. The destination directory consists of many subdirectories to let LLM determine where to store the PDF.

```bash
uv run main.py
```

output of `uv run main.py`

```text

Original: DBS_bank_statement_2025_May_uuid_etc.pdf -> Suggested: 202505_DBS.pdf
Processing files interactively...

Analyzing: 202505_DBS.pdf
LLM suggests moving to: ~Documents/Bank eStatement/2025 Bank/DBS
Move 202505_DBS.pdf to ~Documents/Bank eStatement/2025 Bank/DBS? (y/n): y

Processing complete!
```
