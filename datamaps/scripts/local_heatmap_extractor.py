import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from unidecode import unidecode

def normalize_name(name):
    return unidecode(name).strip().lower()

# BGR Colors from the legend
LEGEND_COLORS = {
    "dark_green": {"bgr": (50, 150, 50), "range": ">= 20%"},
    "light_green": {"bgr": (150, 250, 150), "range": "0% to 20%"},
    "yellow": {"bgr": (0, 255, 255), "range": "-20% to 0%"},
    "orange": {"bgr": (0, 165, 255), "range": "-40% to -20%"},
    "dark_reddish_brown": {"bgr": (0, 50, 150), "range": "-65% to -40%"},
}

def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2))

def match_legend(bgr_color):
    best_match = None
    min_dist = float('inf')
    
    for name, data in LEGEND_COLORS.items():
        dist = color_distance(bgr_color, data["bgr"])
        if dist < min_dist:
            min_dist = dist
            best_match = {"color": name, "value_range": data["range"]}
            
    return best_match

def get_affine_transform(geo_bounds, pixel_bounds):
    min_lon, min_lat, max_lon, max_lat = geo_bounds
    x, y, w, h = pixel_bounds
    
    # We use 3 control points to define the Affine Transform:
    # 1. Top-Left (Min Longitude, Max Latitude)
    # 2. Bottom-Right (Max Longitude, Min Latitude)
    # 3. Bottom-Left (Min Longitude, Min Latitude)
    
    # Source points (Longitude, Latitude)
    src_pts = np.float32([
        [min_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat]
    ])
    
    # Destination points (Pixel X, Pixel Y)
    # Note: Pixel Y increases downwards, so Max Latitude corresponds to Y=y (top)
    dst_pts = np.float32([
        [x, y],          # Top-Left
        [x + w, y + h],  # Bottom-Right
        [x, y + h]       # Bottom-Left
    ])
    
    # Calculate the affine transformation matrix
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    return matrix

def geo_to_pixel_affine(lon, lat, matrix):
    # Apply the affine transformation matrix: [x, y] = M * [lon, lat, 1]^T
    pt = np.array([lon, lat, 1.0])
    res = np.dot(matrix, pt)
    return int(res[0]), int(res[1])

def get_polygon_mask(geom, matrix, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        return mask
        
    for poly in polygons:
        # Exterior
        ext_coords = np.array([geo_to_pixel_affine(x, y, matrix) for x, y in poly.exterior.coords])
        cv2.fillPoly(mask, [ext_coords], 255)
        
        # Interiors (holes)
        for interior in poly.interiors:
            int_coords = np.array([geo_to_pixel_affine(x, y, matrix) for x, y in interior.coords])
            cv2.fillPoly(mask, [int_coords], 0)
            
    return mask

def extract_heatmap_local(image_path: str, geojson_path: str, output_path: str):
    print(f"Loading image {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        sys.exit(1)
        
    print(f"Loading GeoJSON {geojson_path}...")
    gdf = gpd.read_file(geojson_path)
    
    # Honduras bounding box
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    geo_bounds = (min_lon, min_lat, max_lon, max_lat)
    
    # Precise pixel control points for Honduras map on report/page_4.png
    # Coordinates provided by the automated bounds optimizer for 1920x1080: (307, 327, 229, 153)
    # Scaled 2x for the 3840x2160 400% zoomed image:
    pixel_bounds = (614, 654, 458, 306)  
    
    # Get the Affine Transformation Matrix
    matrix = get_affine_transform(geo_bounds, pixel_bounds)
    print("Computed Affine Transformation Matrix.")
    
    results = []
    
    for idx, row in gdf.iterrows():
        dept_name = row.get("shapeName", f"Department_{idx}")
        geom = row.geometry
        
        # Get mask for this department using the Affine matrix
        mask = get_polygon_mask(geom, matrix, img.shape)
        
        # Extract pixels
        dept_pixels = img[mask == 255]
        
        if len(dept_pixels) == 0:
            print(f"Warning: No pixels found for {dept_name}")
            continue
            
        # Calculate median color to ignore borders and text
        median_color = np.median(dept_pixels, axis=0)
        
        match = match_legend(median_color)
        
        results.append({
            "location": dept_name,
            "value_range": match["value_range"],
            "matched_color": match["color"]
        })
        
        print(f"Extracted {dept_name}: {match['value_range']}")
        
    # Build JSON output
    output_data = {
        "legend": [{"color": k, "value_range": v["range"]} for k, v in LEGEND_COLORS.items()],
        "data": results
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved local extraction to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract heatmap data locally using OpenCV.")
    parser.add_argument("image", help="Path to heatmap image")
    parser.add_argument("--geojson", default="honduras_departments.geojson", help="Path to GeoJSON")
    parser.add_argument("--output", "-o", default="data/page_4/local_heatmap_data.json", help="Output JSON path")

    args = parser.parse_args()
    extract_heatmap_local(args.image, args.geojson, args.output)
