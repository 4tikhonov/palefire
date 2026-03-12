---
name: excel expert
description: Read Google Spreadsheets, load them into pandas dataframes, and extract structured metadata and CDIF variables.
---

When the user asks you to act as an "Excel expert" or provides a Google Spreadsheet URL (e.g., `https://docs.google.com/spreadsheets/d/...`), you must instantly perform the following:

1. **Google Spreadsheet Extraction**: Identify the Spreadsheet ID and GID from the URL. Run a bash command to download the data into a pandas dataframe and save it as a local CSV in the `cache/` directory.

   ```bash
   python3 ../palefire/backend/process_google_sheets.py "<SPREADSHEET_URL>"
   ```

2. **Data Analysis & Inventory**:
    - **A) Load & Peek**: Once the script runs, it will print the first 5 rows and summary statistics.
    - **B) Metadata Package**: The script automatically generates a `Dataset` JSON-LD file (schema.org) in the `cache/` folder listing all columns.

3. **Exhaustive Variable Extraction (CDIF)**:
    - Use the standard `python3 ../palefire/backend/write_csv.py cache/cdif_variables_TIMESTAMP.csv "..."` pattern, replacing row values with the exact quantitative or qualitative data from the sheet.
    - Run the `export_cdif.py` script immediately after to ensure the JSON-LD document is created.

4. **Output Format**:
    - Confirm the spreadsheet has been loaded.
    - Present the row/column count and the first few rows (as printed by the script).
    - State that the pandas dataframe is now available and stored as a CSV in the `cache/` directory for further processing.

Do not ask for permission; proceed immediately to execute the `process_google_sheets.py` script.
