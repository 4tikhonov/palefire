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
The app can be run via the `datamaps` command (from `datamaps.cli:main`). The pipeline can also be executed using the shell scripts:
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

## Docker Build and Run

The datamaps subproject provides a `Dockerfile` and a minimal `docker-compose.yml`.  
To build the image locally run:

```bash
# From the datamaps directory
cd datamaps
docker compose build   # or: docker build -t palefire-datamaps .
```

If you prefer to use Docker directly without Compose, you can also build with:

```bash
docker build -t palefire-datamaps .
```

### Running the Pipeline

1. **Set any required environment variables** (e.g., a Gemini API key):

   ```bash
   export GEMINI_API_KEY=YOUR_KEY
   ```

2. **Run the full extraction pipeline in a single step**:

   ```bash
   ./run_pipeline.sh <PDF_URL>
   ```

   The script downloads the PDF, extracts PNGs, runs OCR, splits to JSON‑LD, and compiles the final CSV report—all inside the container.

3. **Or execute individual steps manually inside the container** (useful for debugging):

   ```bash
   # Start an interactive shell in the image
   docker compose run --rm -it app bash

   # Inside the container:
   python -m datamaps extract     # extracts images from PDF
   python -m datamaps ocr         # runs OCR on extracted images
   python -m datamaps split       # splits raw data into JSON‑LD files
   python -m datamaps compile     # compiles the CSV report
   ```

4. **For GPU-accelerated workflows** (if you have an NVIDIA GPU), use the provided `docker-compose-nvidia-dev.yaml`:

   ```bash
   docker compose -f docker-compose-nvidia-dev.yaml up --build
   ```

5. **Cleaning up**: When finished, stop any running containers with `docker compose down`.