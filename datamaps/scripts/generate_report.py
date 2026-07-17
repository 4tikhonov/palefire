import json
import csv
import math

def color_dist(c1, c2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(c1, c2)))

PRECIPITATION_LEGEND = [
    {"bgr": (50, 150, 50), "range": ">= 20%"},
    {"bgr": (150, 250, 150), "range": "0% to 20%"},
    {"bgr": (0, 255, 255), "range": "-20% to 0%"},
    {"bgr": (0, 165, 255), "range": "-40% to -20%"},
    {"bgr": (0, 50, 150), "range": "-65% to -40%"},
]

TEMPERATURE_LEGEND = [
    {"bgr": (50, 220, 230), "range": "0.5 to < 1.0 °C"},
    {"bgr": (45, 175, 230), "range": "1.0 to < 1.5 °C"},
    {"bgr": (36, 76, 223),  "range": "1.5 to < 2.0 °C"},
    {"bgr": (0, 50, 150),   "range": ">= 2.0 °C"},
]

def match_color(bgr, is_right_map):
    if bgr[0] == bgr[1] and bgr[1] == bgr[2]: return "No Map"
    variance = sum((x - sum(bgr)/3)**2 for x in bgr) / 3
    if variance < 100: return "No Map"
    
    legend = TEMPERATURE_LEGEND if is_right_map else PRECIPITATION_LEGEND
    min_dist = float('inf')
    best_match = None
    
    for item in legend:
        d = color_dist(bgr, item["bgr"])
        if d < min_dist:
            min_dist = d
            best_match = item["range"]
            
    if min_dist > 150: return "Unknown Color"
    return best_match

def main():
    with open("data/pdf_full_dataset.json", "r") as f:
        data = json.load(f)["data"]
        
    with open("titles.json", "r") as f:
        titles = json.load(f)
        
    # We want columns ordered by page number, then left/right
    ordered_sources = []
    for page in range(1, 29):
        ordered_sources.append(f"full_page-{page:02d}.png_left")
        ordered_sources.append(f"full_page-{page:02d}.png_right")
        
    matrix = {}
    
    for row in data:
        loc = row["location"]
        src = row["source"]
        bgr = row["color_bgr"]
        
        is_right = src.endswith("_right")
        val = match_color(bgr, is_right)
        
        if loc not in matrix:
            matrix[loc] = {}
            
        matrix[loc][src] = val
        
    with open("report/final_heatmap_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header uses the OCR extracted titles
        header = ["Location"]
        for src in ordered_sources:
            title = titles.get(src, f"Unknown ({src})")
            header.append(title)
            
        writer.writerow(header)
        
        for loc in sorted(matrix.keys()):
            row_out = [loc]
            for src in ordered_sources:
                row_out.append(matrix[loc].get(src, "No Map"))
            writer.writerow(row_out)
                
    print("Generated report/final_heatmap_report.csv using OCR recognized variables in column names.")

if __name__ == "__main__":
    main()
