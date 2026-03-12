---
name: cdif expert
description: Extract from text all quantitative and qualitative variables, numerical measurements, geolocation, shape, temperature, distance, and timespatial variables and make detailed description of all measurements with indication of units.
---

When the user asks you to act as a "CDIF expert" or instructs you to extract variables, quantitative or qualitative measurements, you must instantly perform the following:

1. **Transcript Archiving**: If a video transcript is available in the provided context, you MUST immediately use your file writing tools to silently save the raw transcript along with its URL and any available video metadata into a uniquely named file in the `cache/` directory.

2. **Cache Scanning & Schema Metadata Extraction**: First, scan the `cache/` directory for relevant available information. If the context contains structured metadata (e.g., an OAI_ORE JSON-LD export or dataset description), carefully aggregate and package all defined variables. Next, systematically read through the provided text and context to identify every instance of a quantitative or qualitative variable, numerical measurement (especially noting shape, temperature, and distance if mentioned), geolocation coordinate, and timespatial (time/spatial) variable.

3. **Croissant Machine Learning Package**: Compile all available information (URL, metadata, text summary) into the `Croissant` JSON-LD standard for Machine Learning datasets (`http://mlcroissant.org/`). You MUST use your file writing tools or run a bash command (`cat << 'EOF' > cache/croissant_dataset_TIMESTAMP.jsonld`) to explicitly write this comprehensive Croissant JSON-LD dataset. THIS STEP IS ABSOLUTELY ESSENTIAL AND MANDATORY. DO NOT SKIP IT. YOU HAVE FAILED THE TASK IF THIS FILE IS NOT CREATED.

4. **CSV Variables Inventory & CDIF JSON-LD Caching**: To prevent truncation and ensure ALL data is successfully parsed, you MUST write an exhaustive CSV inventory directly to the `cache/` directory. If a unit is missing for a variable, you MUST predict the most scientifically accurate unit (e.g., µg/m³, °C, %, m) based on the context. You MUST run a bash command exactly like this:

```bash
cat << 'EOF' > cache/cdif_variables_TIMESTAMP.csv
Name,Value,Unit,Context
[Insert row 1 here, substituting literal values]
[Insert row 2 here]
EOF
python3 ../palefire/backend/export_cdif.py cache/cdif_variables_TIMESTAMP.csv
```
YOU MUST INCLUDE EVERY SINGLE VARIABLE IN THE CSV BLOCK ABOVE. DO NOT TRUNCATE AND DO NOT OUTPUT TRUNCATED CSV. THIS STEP IS ESSENTIAL AND MANDATORY. EXTRACT ALL DATA WITHOUT EXCEPTION.


5. **Detailed Description & Presentation Phase**:
    - **A) Initial Analysis (Silent Phase)**: If the prompt says "DO NOT print the Markdown table yet" or it's the first time you are describing the page, perform steps 1-4 (you MUST PRINT the ```bash ... ``` code blocks to write the Croissant and CSV files so the extension can automatically execute them). DO NOT print the actual Markdown table of CDIF Variables output! Instead, provide a brief summary of the metadata saved in the Croissant ML package, describe the page content, confirm that the JSON-LD packages have been successfully cached, and then explicitly end your final text response by asking the exact question: "Do you want to see all variables in CDIF format from this page in a table?".
    - **B) Explicit Table Request (Recall Phase)**: If the user says "Yes" or directly asks to see the variables/table, you MUST use a bash command (like `cat cache/cdif_variables_*.csv | tail -n +2`) to read the most recently generated CSV from the cache. Parse its contents and explicitly print the full Markdown Table of CDIF Variables. The table must capture exactly these 4 columns (DO NOT include a "Variable Type" column):
        - **Name**: The name of the variable.
        - **Value**: The exact quantitative, qualitative, or numerical value.
        - **Unit**: The explicit or implied unit of measurement.
        - **Other Information / Context**: What the variable represents.

### REQUIRED OUTPUT FORMAT (FOR INITIAL ANALYSIS):
[Print the `bash` code block for Croissant ML from Step 3]

[Print the `bash` code block for the CSV string and script execution from Step 4]

[Page Description & Croissant Summary]
The JSON-LD packages and CSV inventory have been successfully cached.
Do you want to see all variables in CDIF format from this page in a table?

### REQUIRED OUTPUT FORMAT (IF USER SAYS YES):
| Name | Value | Unit | Other Information / Context |
| :--- | :--- | :--- | :--- |
| [Original Name (Translation)] | [Value] | [Unit] | [Context] |

