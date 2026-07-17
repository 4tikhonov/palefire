import cv2
import numpy as np
import json
import sys

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

def test_exact_points(image_path, exact_json_path, baseline_path):
    print(f"Loading image {image_path}...")
    img = cv2.imread(image_path)
    
    with open(exact_json_path, "r") as f:
        exact_data = json.load(f)["data"]
        
    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)["data"]
        
    baseline_dict = {item["location"].lower(): item["value_range"] for item in baseline_data}
    
    matches = 0
    total = 0
    
    for item in exact_data:
        loc_name = item["location"]
        lon = item["centroid"]["lon"]
        lat = item["centroid"]["lat"]
        
        # User requested to use the points from exact_centroid_left.json
        # The file already contains the projected pixel x, y
        px = item["pixel"]["x"]
        py = item["pixel"]["y"]
        
        # Collect the color
        h, w = img.shape[:2]
        r = 2
        y1, y2 = max(0, py - r), min(h, py + r + 1)
        x1, x2 = max(0, px - r), min(w, px + r + 1)
        
        patch = img[y1:y2, x1:x2]
        if patch.size > 0:
            pixels = patch.reshape(-1, 3)
            median_color = np.median(pixels, axis=0)
            
            # Put on heatmap
            cv2.circle(img, (px, py), 8, (255, 0, 0), -1)
            cv2.putText(img, loc_name, (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            match = match_legend(median_color)
            
            expected = baseline_dict.get(loc_name.lower())
            if expected:
                total += 1
                if match["value_range"] == expected:
                    status = "MATCH"
                    matches += 1
                else:
                    status = "MISMATCH"
                print(f"[{status}] {loc_name} (Lon: {lon:.2f}, Lat: {lat:.2f}) -> Pixel ({px}, {py}): Sampled {match['value_range']}, Expected {expected}")
            
    print(f"\nAccuracy: {matches}/{total} ({(matches/total)*100:.1f}%)")
    cv2.imwrite("report/exact_json_points_debug.png", img)
    print("Saved visual map to report/exact_json_points_debug.png")

if __name__ == "__main__":
    test_exact_points("report/page_4_400_left.png", "data/page_4/exact_centroid_left.json", "baseline/page_4_left_400.json")
