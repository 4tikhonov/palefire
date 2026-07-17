import cv2
import numpy as np
import json
import os
import glob
from pathlib import Path
import geopandas as gpd
import sys

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

def extract_map(image_path, gdf, output_path):
    print(f"Processing {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return
        
    h, w = img.shape[:2]
    
    # Determine the pixel bounds dynamically based on the image size
    if w == 1920 and h == 2160:
        pixel_bounds = (614, 654, 458, 306) # 4K left/right crop bounds
    elif w == 1920 and h == 1080:
        # 2K Full page width. A left map would occupy the left half (0-960)
        # So we scale the 4K bounds (which were on a 1920 width) by 0.5
        pixel_bounds = (int(614*0.5), int(654*0.5), int(458*0.5), int(306*0.5))
    elif w == 3840 and h == 2160:
        print(f"Skipping {image_path} because it's a full 4K dual map. Use the split versions.")
        return
    else:
        print(f"Unknown resolution {w}x{h} for {image_path}. Using default 4K bounds.")
        pixel_bounds = (614, 654, 458, 306)
        
    geo_bounds = gdf.total_bounds
    matrix = get_affine_transform(geo_bounds, pixel_bounds)
    
    results = []
    
    for idx, row in gdf.iterrows():
        loc_name = row.get("shapeName", f"Department_{idx}")
        geom = row.geometry
        if geom is None: continue
        
        centroid = geom.centroid
        lon, lat = centroid.x, centroid.y
        px, py = geo_to_pixel_affine(lon, lat, matrix)
        
        r = 2
        y1, y2 = max(0, py - r), min(h, py + r + 1)
        x1, x2 = max(0, px - r), min(w, px + r + 1)
        
        patch = img[y1:y2, x1:x2]
        if patch.size > 0:
            pixels = patch.reshape(-1, 3)
            median_color = np.median(pixels, axis=0)
            b, g, r_color = int(median_color[0]), int(median_color[1]), int(median_color[2])
            
            results.append({
                "source": Path(image_path).name,
                "location": loc_name,
                "lon": lon,
                "lat": lat,
                "pixel_x": px,
                "pixel_y": py,
                "color_bgr": [b, g, r_color]
            })
            
            # Draw point
            cv2.circle(img, (px, py), int(12 * (h/2160)), (b, g, r_color), -1)
            cv2.circle(img, (px, py), int(12 * (h/2160)), (0, 0, 0), 2)
            
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"data": results}, f, indent=2, ensure_ascii=False)
        
    debug_path = f"report/debug_process_{Path(image_path).name}"
    cv2.imwrite(debug_path, img)
    print(f"Saved dataset to {output_path} and visual debug to {debug_path}\n")

def main():
    geojson_path = "honduras_departments.geojson"
    print("Loading GeoJSON...")
    gdf = gpd.read_file(geojson_path)
    
    files = [
        "report/page_1.png",
        "report/page_2.png",
        "report/page_3.png",
        "report/page_4_400_left.png",
        "report/page_4_400_right.png",
        "report/page_5_left.png",
        "report/page_5_right.png",
    ]
    
    all_data = []
    
    for f in files:
        if os.path.exists(f):
            out_json = f"data/automated_{Path(f).stem}.json"
            extract_map(f, gdf, out_json)
            
            # Load the generated json and append to massive collection
            with open(out_json, "r") as json_file:
                all_data.extend(json.load(json_file)["data"])
                
    # Save the consolidated massive dataset
    with open("data/full_dataset.json", "w", encoding="utf-8") as f:
        json.dump({"data": all_data}, f, indent=2, ensure_ascii=False)
    print("Saved consolidated massive dataset to data/full_dataset.json")

if __name__ == "__main__":
    main()
