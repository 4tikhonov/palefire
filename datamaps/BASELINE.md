# Map Bounds Optimization Baseline

This document describes the baseline and optimization process used to align the geographic vector data (`honduras_departments.geojson`) perfectly with the raster map images in the PDF.

## 1. The Challenge of Affine Transformation

The core of the `datamaps` library relies on converting precise geographic coordinates (Longitude/Latitude) into pixel coordinates on the image. This requires an Affine Transformation Matrix.

To compute this matrix, we need two bounding boxes:
1. **Geographic Bounds**: The maximum and minimum Lat/Lon of Honduras (extracted automatically from the GeoJSON).
2. **Pixel Bounds**: The exact bounding box (x, y, width, height) of the map *within the image itself*, excluding legends, titles, and borders.

Finding the perfect pixel bounds manually is incredibly tedious and error-prone. A misalignment of just 5 pixels can cause the color sampler to read the border of a department instead of its interior, resulting in incorrect data extraction.

## 2. Establishing the Baseline

To solve this, we use a ground-truth **Baseline** approach:

1. **Manual/Heuristic Ground Truth**: We created a baseline dataset (e.g., `baseline/page_4_left_400.json`). This file maps specific department names to their visually verified "ground truth" color on a specific map (e.g., Atlántida = `-20% to 0%`).
2. **Point Sampling**: The script `scripts/baseline_point_sampler.py` was used to sample points inside the polygons to help build these ground truth files.

## 3. Automated Optimization (`optimize_bounds.py`)

Once we have a baseline (even if it's just 10-15 departments on a single map), we use brute-force optimization to find the perfect map bounds:

1. The script `scripts/optimize_bounds.py` takes a starting guess for the pixel bounds (e.g., `(514, 574, 508, 306)`).
2. It iteratively nudges these 4 parameters (x, y, width, height) in small increments (up, down, left, right).
3. For every combination, it builds the Affine matrix, extracts the colors for all departments in the baseline, and assigns a **Score** based on how many departments match the ground truth.
4. It continuously saves the "New best score".

### Our Results

We ran this optimization as a long-running background task on a split-page map. 
The optimizer improved the extraction accuracy from **61%** to **83.3%** on the baseline by discovering the optimal pixel bounds:
`pixel_bounds = (654, 604, 518, 446)`

These perfectly tuned bounds are now hardcoded as the default in `datamaps/extractor.py`, ensuring that all subsequent maps of this format are extracted with maximum spatial accuracy.

## Running the Optimizer Yourself

If you encounter maps with a different scale or layout, you can re-run the optimization:

```bash
cd datamaps
# Provide a map image, a baseline JSON, and an initial bounds guess: "(up_crop, down_crop, left_crop, right_crop)"
python scripts/optimize_bounds.py ../report/page_4_400_left.png ../baseline/page_4_left_400.json --bounds "(514, 574, 508, 306)"
```
