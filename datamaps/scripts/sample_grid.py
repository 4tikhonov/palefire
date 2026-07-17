import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
import geopandas as gpd
import ast

LEGEND_COLORS = {
    "dark_green": {"bgr": (50, 150, 50), "range": ">= 20%"},
    "light_green": {"bgr": (150, 250, 150), "range": "0% to 20%"},
    "yellow": {"bgr": (0, 255, 255), "range": "-20% to 0%"},
    "orange": {"bgr": (0, 165, 255), "range": "-40% to -20%"},
    "dark_reddish_brown": {"bgr": (0, 50, 150), "range": "-65% to -40%"},
}

def color_distance(c1, c2):
    return sum((int(a) - int(b)) ** 2 for a, b in zip(c1, c2))

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
    
    src_pts = np.float32([[min_lon, max_lat], [max_lon, min_lat], [min_lon, min_lat]])
    dst_pts = np.float32([[x, y], [x + w, y + h], [x, y + h]])
    return cv2.getAffineTransform(src_pts, dst_pts)

def sample_grid(image_path, geojson_path, output_path, pixel_bounds, resolution=200):
    print(f"Loading image {image_path}...")
    img = cv2.imread(image_path)
    if img is None: sys.exit(1)
        
    print(f"Loading GeoJSON {geojson_path}...")
    gdf = gpd.read_file(geojson_path)
    geo_bounds = gdf.total_bounds
    min_lon, min_lat, max_lon, max_lat = geo_bounds
    
    matrix = get_affine_transform(geo_bounds, pixel_bounds)
    print(f"Affine Matrix bounds: {pixel_bounds}")
    
    lon_grid = np.linspace(min_lon, max_lon, resolution)
    lat_grid = np.linspace(min_lat, max_lat, resolution)
    
    results = []
    
    print(f"Sampling {resolution}x{resolution} grid...")
    for lon in lon_grid:
        for lat in lat_grid:
            pt = np.array([lon, lat, 1.0])
            res = np.dot(matrix, pt)
            px, py = int(res[0]), int(res[1])
            
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                bgr = img[py, px]
                
                # Filter out pure white (background) or pure black (text/lines)
                if np.all(bgr > 200) or np.all(bgr < 50):
                    continue
                    
                match = match_legend(bgr)
                results.append({
                    "lon": float(lon),
                    "lat": float(lat),
                    "color": match["color"],
                    "value_range": match["value_range"]
                })
                
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved {len(results)} valid spatial points to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--geojson", default="honduras_departments.geojson")
    parser.add_argument("--output", "-o", default="data/lon_lat_grid.json")
    parser.add_argument("--bounds", default="(514, 574, 508, 306)")
    parser.add_argument("--res", type=int, default=200)
    args = parser.parse_args()
    
    bounds = ast.literal_eval(args.bounds)
    sample_grid(args.image, args.geojson, args.output, bounds, args.res)
