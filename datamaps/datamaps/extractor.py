import cv2
import numpy as np
import geopandas as gpd
from pathlib import Path

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

def extract_map(img, h, w, pixel_bounds, gdf, geo_bounds, source_name):
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
                "source": source_name,
                "location": loc_name,
                "lon": lon,
                "lat": lat,
                "pixel_x": px,
                "pixel_y": py,
                "color_bgr": [b, g, r_color]
            })
            
            # Optional: mutate image to draw circles
            cv2.circle(img, (px, py), 12, (b, g, r_color), -1)
            cv2.circle(img, (px, py), 12, (0, 0, 0), 2)
            
    return results

def process_image(image_path, gdf, geo_bounds, output_dir="report"):
    print(f"Processing {image_path}...")
    img = cv2.imread(str(image_path))
    if img is None: return []
    
    h, w = img.shape[:2]
    all_results = []
    
    if w == 3840 and h == 2160:
        mid = w // 2
        left_img = img[:, :mid].copy()
        right_img = img[:, mid:].copy()
        
        pixel_bounds = (654, 604, 518, 446)
        
        res_left = extract_map(left_img, h, mid, pixel_bounds, gdf, geo_bounds, Path(image_path).name + "_left")
        all_results.extend(res_left)
        cv2.imwrite(f"{output_dir}/debug_{Path(image_path).stem}_left.png", left_img)
        
        res_right = extract_map(right_img, h, mid, pixel_bounds, gdf, geo_bounds, Path(image_path).name + "_right")
        all_results.extend(res_right)
        cv2.imwrite(f"{output_dir}/debug_{Path(image_path).stem}_right.png", right_img)
        
    else:
        print(f"Skipping {image_path} due to unexpected dimensions {w}x{h}")
        
    return all_results
