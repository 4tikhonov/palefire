import json
import cv2
import numpy as np
import geopandas as gpd
from collections import defaultdict
from unidecode import unidecode

def normalize_name(name):
    return unidecode(name).strip().lower()

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

def calibrate(image_path, baseline_path, gdf, geo_bounds, pixel_bounds):
    print(f"\nCalibrating {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return
        
    with open(baseline_path, "r") as f:
        baseline = json.load(f)
        
    matrix = get_affine_transform(geo_bounds, pixel_bounds)
    h, w = img.shape[:2]
    
    # Map normalized names to value ranges
    ground_truth = {normalize_name(d["location"]): d["value_range"] for d in baseline.get("data", [])}
    
    # To collect all BGRs for each value range
    range_bgrs = defaultdict(list)
    
    for idx, row in gdf.iterrows():
        loc_name = row.get("shapeName", "")
        norm_name = normalize_name(loc_name)
        if norm_name not in ground_truth:
            continue
            
        expected_range = ground_truth[norm_name]
        
        centroid = row.geometry.centroid
        px, py = geo_to_pixel_affine(centroid.x, centroid.y, matrix)
        
        r = 2
        y1, y2 = max(0, py - r), min(h, py + r + 1)
        x1, x2 = max(0, px - r), min(w, px + r + 1)
        
        patch = img[y1:y2, x1:x2]
        if patch.size > 0:
            pixels = patch.reshape(-1, 3)
            median_color = np.median(pixels, axis=0)
            b, g, red = int(median_color[0]), int(median_color[1]), int(median_color[2])
            range_bgrs[expected_range].append((b, g, red))
            
    print(f"Discovered BGR centroids:")
    for value_range, bgrs in range_bgrs.items():
        avg_b = int(np.mean([c[0] for c in bgrs]))
        avg_g = int(np.mean([c[1] for c in bgrs]))
        avg_r = int(np.mean([c[2] for c in bgrs]))
        print(f"        ({avg_b}, {avg_g}, {avg_r}): \"{value_range}\",")

if __name__ == "__main__":
    gdf = gpd.read_file("honduras_departments.geojson")
    geo_bounds = gdf.total_bounds
    pixel_bounds = (654, 604, 518, 446)
    
    calibrate("report/debug_full_page-05_left.png", "baseline/page_5_ground_truth_400.json", gdf, geo_bounds, pixel_bounds)
    calibrate("report/debug_full_page-05_right.png", "baseline/page_5_ground_truth_400.json", gdf, geo_bounds, pixel_bounds)
