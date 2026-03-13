---
name: dataverse_expert
description: Convert Croissant metadata to Dataverse format and ingest datasets into Dataverse collections. Use when the user needs to publish datasets or metadata to a Dataverse instance.
---

# Dataverse Expert Skill

The Dataverse Expert skill allows the agent to convert and publish scientific metadata into Dataverse repositories.

## Tools

### 1. Ingest to Dataverse
Ingests one or more Croissant JSON metadata files into a specified Dataverse collection. It can also perform translations of keywords during ingestion.

**Usage:**
```bash
python3 dataverse_expert/scripts/ingest.py --file <PATH_TO_JSON> --dataverse-url <URL> --dataverse-id <ID> --api-token <TOKEN>
```

**Arguments:**
- `--input-dir`: Directory containing `croissant_*.json` files (default: `output`).
- `--file`: Path to a single Croissant JSON file (overrides `--input-dir`).
- `--dataverse-url`: Base URL of the Dataverse instance.
- `--dataverse-id`: Target collection alias or ID.
- `--api-token`: API Token for authentication.
- `--translate-keywords`: Comma-separated languages (e.g., `fr,es`) to translate keywords into.
- `--translation-model`: Ollama model to use for translations (default: `gpt-oss:latest`).

### 2. Croissant to Dataverse Converter
Converts a Croissant JSON file to Dataverse-compliant JSON-LD format for preview or manual upload.

**Usage:**
```bash
python3 dataverse_expert/scripts/converter.py <PATH_TO_JSON>
```

## Workflow
1. **Prepare:** Ensure your metadata is in Croissant JSON format.
2. **Configure:** Set environment variables `DATAVERSE_URL`, `DATAVERSE_ID`, and `DATAVERSE_API_TOKEN` for easier use.
3. **Execute:** Run the ingest tool to push metadata to Dataverse.
4. **Verify:** Check the Dataverse collection to ensure the dataset was created and enriched correctly.
