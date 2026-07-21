# Datamaps

A modular Python library designed to extract, process, and compile spatial data points from map images and PDFs, outputting semantic JSON-LD structures and clean CSV reports.

## Architecture

The `datamaps` library converts legacy procedural scripts into a structured module:
- **`datamaps.extractor`**: Handles spatial coordinate transformations (geo to pixel affine math), map bounds extraction, and pixel-color classification.
- **`datamaps.compiler`**: Manages the conversion of raw points into Linked Data (JSON-LD) and the compilation of final structural CSV spreadsheets.
- **`datamaps.cli`**: Provides a clean command-line interface to string these tools together into a reproducible pipeline.

## Installation

### Standard Local Install
To install the library locally (preferably within a virtual environment):
```bash
git clone <repository_url>
cd datamaps
pip install -e .
```

### Docker
Alternatively, run everything completely encapsulated via Docker:
```bash
cd datamaps
docker compose run --rm app bash
```

## Usage Examples

The package is driven by a single CLI entrypoint: `python -m datamaps <command>`

### 1. Extract Map Data
Extract geographic heatmaps from images using precise affine matrix projections. This generates a raw dataset (`data/pdf_full_dataset.json`).
```bash
python -m datamaps extract
```

### 2. Split to JSON-LD Datasets
Split the raw dataset into decoupled, semantic JSON-LD observations mapped to a central URI location dictionary. The output is placed in `data_points/`.
```bash
python -m datamaps split
```

### 3. Compile CSV Report
Parse the semantic JSON-LD definitions and dynamically construct a master CSV report (`report/final_heatmap_report.csv`). Column headers are reliably structured by corresponding variable and page metadata.
```bash
python -m datamaps compile
```

### 4. Computer Vision Automated Extraction (Palefire LLM)
The most powerful way to use `datamaps` is through the automated LLM-assisted computer vision script located in the parent directory (`extract_computer_vision.py`). This script uses an Ollama vision model to classify maps, extract colors, handle 2x2 grid splitting (forcing 2x1 splits to avoid axis clipping), and automatically runs the `datamaps` spatial extractor.

**Single Image Extraction:**
```bash
cd ..
export OLLAMA_HOST=http://localhost:11434
python extract_computer_vision.py --image datamaps/data/pdf-extraction/page_18.jpg
```

**Local PDF Extraction:**
```bash
cd ..
export OLLAMA_HOST=http://localhost:11434
python extract_computer_vision.py --pdf datamaps/data/pronostico.pdf
```

**PDF URL Direct Download & Extraction:**
```bash
cd ..
export OLLAMA_HOST=http://localhost:11434
python extract_computer_vision.py --pdf-url "https://example.com/report.pdf"
```

**Batch Directory Extraction:**
```bash
cd ..
export OLLAMA_HOST=http://localhost:11434
python extract_computer_vision.py --image-dir datamaps/data/pdf-extraction/
```
The output JSON-LD data and the `final_heatmap_report.csv` will be saved in `../data/session_<timestamp>/<image_name>_datamaps/`.

## Development and Testing

Install with development dependencies to run tests:
```bash
pip install -e ".[dev]"
pytest tests/
```

*(Note: Older experimental scripts and legacy tools have been safely archived into the `scripts/` directory for reference.)*
