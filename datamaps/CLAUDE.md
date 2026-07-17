# Datamaps - Map Data Extraction Pipeline

A toolset and pipeline for extracting structured data, labels, and spatial points from map-based images and PDF documents.

## Project Structure

- `datamaps/`: Core library
  - `cli.py`: Entry point for the command line interface.
  - `extractor.py`: Logic for identifying landmarks and features on maps.
  - `ocr.py`: Text recognition module.
  - `compiler.py`: Aggregator for multi-source extraction results.
- `scripts/`: Research, experimentation, and utility scripts (e.g., Gemini integration, coordinate processing, etc.).
- `data/` & `data_points/`: Repository of extracted JSON data points.
- `report/`: Visual debug reports and final generated maps.

## Development Commands

### Installation
```bash
pip install .
# Install with dev dependencies
pip install -e .[dev]
```

### Running the Application
The app can be run via the `datamaps` command (from `datamaps.cli:main`).
The pipeline can also be executed using the shell scripts:
```bash
./run_pipeline.sh
```

### Environment & Infrastructure
- **Docker:** Use `docker-compose up` to start the standard environment.
- **Dependencies:** Managed via `pyproject.toml` (NumPy, OpenCV, Geopandas, Pandas).

## Coding Standards

- **Language:** Python 3.9+
- **Style:** Follow PEP 8 guidelines.
- **Testing:** Run tests using `pytest`.
  ```bash
  pytest tests/
  ```

## Key Dependencies
- `opencv-python-headless`: For image processing and computer vision tasks.
- `geopandas` & `pandas`: For spatial data manipulation and analysis.
