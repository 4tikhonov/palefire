import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
import geopandas as gpd
from unidecode import unidecode
import ast

LEGEND_COLORS = {
    "dark_green": {"bgr": (50, 150, 50), "range": ">= 20%"},
    "light_green": {"bgr": (150, 250, 150), "range": "0% to 20%"},
    "yellow": {"bgr": (0, 255, 255), "range": "-20% to 0%"},
    "orange": {"bgr": (0, 165, 255), "range": "-40% to -20%"},
    "dark_reddish_brown": {"bgr": (0, 50, 150), "range": "-65% to -40%"},
}

def normalize_name(name):
    return unidecode(name).strip().lower()

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

def geo_to_pixel_affine(lon, lat, matrix):
    pt = np.array([lon, lat, 1.0])
    res = np.dot(matrix, pt)
    return int(res[0]), int(res[1])

def sample_baseline_points(image_path, baseline_path, geojson_path, pixel_bounds, args):
    print(f"Loading image {image_path}...")
    img = cv2.imread(image_path)
    if img is None: sys.exit(1)
        
    print(f"Loading baseline {baseline_path}...")
    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)["data"]
        
    print(f"Loading GeoJSON {geojson_path}...")
    gdf = gpd.read_file(geojson_path)
    geo_bounds = gdf.total_bounds
    
    matrix = get_affine_transform(geo_bounds, pixel_bounds)
    print(f"Affine bounds: {pixel_bounds}")
    
    results = []
    
    # Draw GeoJSON boundaries to make the shift visual
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None: continue
        
        def draw_poly(polygon):
            pts = []
            for coord in polygon.exterior.coords:
                px, py = geo_to_pixel_affine(coord[0], coord[1], matrix)
                pts.append([px, py])
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 0, 255), 3) # Draw RED boundaries
            
        if geom.geom_type == 'Polygon':
            draw_poly(geom)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                draw_poly(poly)
                
    # 1. Get geocoordinates of all locations from baseline
    for item in baseline_data:
        loc_name = item["location"]
        # Find in GeoJSON
        norm_name = normalize_name(loc_name)
        geom = None
        for idx, row in gdf.iterrows():
            if normalize_name(row.get("shapeName", "")) == norm_name:
                geom = row.geometry
                break
                
        if geom is None:
            # Handle special cases like 'Bay Islands'
            if norm_name == "islas de la bahia" or "bay islands" in norm_name:
                for idx, row in gdf.iterrows():
                    if "bahia" in normalize_name(row.get("shapeName", "")):
                        geom = row.geometry
                        break
            if geom is None:
                print(f"Could not find GeoJSON geometry for baseline location: {loc_name}")
                continue
                
        centroid = geom.centroid
        lon, lat = centroid.x, centroid.y
        
        # 2. Put on the map first
        px, py = geo_to_pixel_affine(lon, lat, matrix)
        
        # 3. Collect the color
        h, w = img.shape[:2]
        r = 2
        y1, y2 = max(0, py - r), min(h, py + r + 1)
        x1, x2 = max(0, px - r), min(w, px + r + 1)
        
        patch = img[y1:y2, x1:x2]
        if patch.size > 0:
            pixels = patch.reshape(-1, 3)
            median_color = np.median(pixels, axis=0)
            
            match = match_legend(median_color)
            status = "MATCH" if match["value_range"] == item["value_range"] else "MISMATCH"
            
            # Draw point on the map holding the sampled color
            b, g, r_color = int(median_color[0]), int(median_color[1]), int(median_color[2])
            cv2.circle(img, (px, py), 12, (b, g, r_color), -1)
            cv2.circle(img, (px, py), 12, (0, 0, 0), 2) # Black border
            
            results.append({
                "location": loc_name,
                "lon": lon,
                "lat": lat,
                "pixel_x": px,
                "pixel_y": py,
                "sampled_color_range": match["value_range"],
                "baseline_expected": item["value_range"]
            })
            
            print(f"[{status}] {loc_name}: Sampled {match['value_range']}, Expected {item['value_range']}")
            
    cv2.imwrite("report/baseline_points_debug.png", img)
    print("Saved visual map to report/baseline_points_debug.png")
    
    dataset = {
        "legend": [{"color": k, "value_range": v["range"]} for k, v in LEGEND_COLORS.items()],
        "data": []
    }
    
    for r in results:
        # Find matched color name
        color_name = next(k for k, v in LEGEND_COLORS.items() if v["range"] == r["sampled_color_range"])
        dataset["data"].append({
            "location": r["location"],
            "value_range": r["sampled_color_range"],
            "matched_color": color_name
        })
        
    out_path = args.output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Dataset successfully saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("baseline")
    parser.add_argument("--geojson", default="honduras_departments.geojson")
    parser.add_argument("--bounds", default="(614, 654, 458, 306)")
    parser.add_argument("--output", "-o", default="dataset_output.json")
    args = parser.parse_args()
    
    bounds = ast.literal_eval(args.bounds)
    sample_baseline_points(args.image, args.baseline, args.geojson, bounds, args)
