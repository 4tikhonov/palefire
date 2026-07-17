import argparse
import ast
import cv2
import numpy as np
import json
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
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

def get_polygon_mask(geom, matrix, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    polygons = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    for poly in polygons:
        ext_coords = np.array([geo_to_pixel_affine(x, y, matrix) for x, y in poly.exterior.coords])
        cv2.fillPoly(mask, [ext_coords], 255)
        for interior in poly.interiors:
            int_coords = np.array([geo_to_pixel_affine(x, y, matrix) for x, y in interior.coords])
            cv2.fillPoly(mask, [int_coords], 0)
    return mask

from collections import Counter

def score_bounds(pixel_bounds, img, gdf, geo_bounds, baseline_dict):
    matrix = get_affine_transform(geo_bounds, pixel_bounds)
    matched = 0
    total = len(baseline_dict)
    
    for idx, row in gdf.iterrows():
        dept_name = row.get("shapeName", f"Department_{idx}")
        geom = row.geometry
        
        all_coords = []
        if geom.geom_type == 'Polygon':
            all_coords.extend(list(geom.exterior.coords))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                all_coords.extend(list(poly.exterior.coords))
                
        if not all_coords:
            continue
            
        points = [
            max(all_coords, key=lambda p: p[1]), # North
            min(all_coords, key=lambda p: p[1]), # South
            max(all_coords, key=lambda p: p[0]), # East
            min(all_coords, key=lambda p: p[0])  # West
        ]
        
        votes = []
        for pt in points:
            px, py = geo_to_pixel_affine(pt[0], pt[1], matrix)
            h, w = img.shape[:2]
            r = 2
            y1, y2 = max(0, py - r), min(h, py + r + 1)
            x1, x2 = max(0, px - r), min(w, px + r + 1)
            
            patch = img[y1:y2, x1:x2]
            if patch.size > 0:
                pixels = patch.reshape(-1, 3)
                median_color = np.median(pixels, axis=0)
                match = match_legend(median_color)
                votes.append(match["value_range"])
                
        if not votes:
            continue
            
        counter = Counter(votes)
        final_value = counter.most_common(1)[0][0]
        
        norm_name = normalize_name(dept_name)
        if norm_name in baseline_dict:
            if baseline_dict[norm_name]["value_range"] == final_value:
                matched += 1
                
    return matched / total

def optimize(image_path, baseline_path, start_bounds):
    img = cv2.imread(image_path)
    gdf = gpd.read_file("honduras_departments.geojson")
    geo_bounds = gdf.total_bounds
    
    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)
    baseline_dict = {normalize_name(item["location"]): item for item in baseline_data["data"]}
    
    best_bounds = start_bounds
    best_score = score_bounds(best_bounds, img, gdf, geo_bounds, baseline_dict)
    print(f"Initial score: {best_score}")
    
    # Grid search around the scaled guess
    for x_offset in range(-100, 101, 10):
        for y_offset in range(-100, 101, 10):
            for w_offset in range(-50, 51, 10):
                for h_offset in range(-50, 51, 10):
                    bounds = (
                        best_bounds[0] + x_offset,
                        best_bounds[1] + y_offset,
                        best_bounds[2] + w_offset,
                        best_bounds[3] + h_offset
                    )
                    score = score_bounds(bounds, img, gdf, geo_bounds, baseline_dict)
                    if score > best_score:
                        best_score = score
                        best_bounds = bounds
                        print(f"New best score: {best_score} with bounds {best_bounds}")
                        if best_score == 1.0:
                            print("Found 100% match!")
                            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image path")
    parser.add_argument("baseline", help="Baseline JSON path")
    parser.add_argument("--bounds", default="(614, 654, 458, 306)", help="Starting bounds tuple")
    args = parser.parse_args()
    
    start_bounds = ast.literal_eval(args.bounds)
    optimize(args.image, args.baseline, start_bounds)
