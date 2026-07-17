import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
import geopandas as gpd
from unidecode import unidecode

def normalize_name(name):
    return unidecode(name).strip().lower()

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

def geo_to_pixel_affine(lon, lat, matrix):
    pt = np.array([lon, lat, 1.0])
    res = np.dot(matrix, pt)
    return int(res[0]), int(res[1])

def extract_heatmap_exact(image_path: str, geojson_path: str, output_path: str):
    print(f"Loading image {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        sys.exit(1)
        
    print(f"Loading GeoJSON {geojson_path}...")
    gdf = gpd.read_file(geojson_path)
    
    # Mathematical True Coordinates of the Extremes
    g_w = (-89.3568207, 14.4668343)
    g_e = (-83.1349418, 15.0047509)
    g_s = (-87.0500161, 12.9808485)
    
    # Pixel Coordinates found by Gemini for report/page_4_400_left.png
    p_w = (291, 592)
    p_e = (673, 566)
    p_s = (478, 737)
    
    src_pts = np.float32([g_w, g_e, g_s])
    dst_pts = np.float32([p_w, p_e, p_s])
    
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    print("Computed EXACT Affine Transformation Matrix from 3 anchor points.")
    
    results = []
    
    for idx, row in gdf.iterrows():
        dept_name = row.get("shapeName", f"Department_{idx}")
        geom = row.geometry
        
        centroid = geom.centroid
        px, py = geo_to_pixel_affine(centroid.x, centroid.y, matrix)
        
        h, w = img.shape[:2]
        r = 2
        y1, y2 = max(0, py - r), min(h, py + r + 1)
        x1, x2 = max(0, px - r), min(w, px + r + 1)
        
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            print(f"Warning: Centroid for {dept_name} is outside image bounds")
            continue
            
        pixels = patch.reshape(-1, 3)
        median_color = np.median(pixels, axis=0)
        
        cv2.circle(img, (px, py), 10, (0, 0, 255), -1)
        cv2.putText(img, dept_name, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        match = match_legend(median_color)
        
        results.append({
            "location": dept_name,
            "centroid": {"lon": centroid.x, "lat": centroid.y},
            "pixel": {"x": px, "y": py},
            "value_range": match["value_range"],
            "matched_color": match["color"]
        })
        
        print(f"Extracted {dept_name} at ({px}, {py}): {match['value_range']}")
        
    cv2.imwrite("report/exact_centroid_debug.png", img)
    print("Saved visual debug map to report/exact_centroid_debug.png")
    
    output_data = {
        "legend": [{"color": k, "value_range": v["range"]} for k, v in LEGEND_COLORS.items()],
        "data": results
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved local extraction to {output_path}")

if __name__ == "__main__":
    extract_heatmap_exact("report/page_4_400_left.png", "honduras_departments.geojson", "data/page_4/exact_centroid_left.json")
