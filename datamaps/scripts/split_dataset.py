import json
import os
import re
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

def clean_filename(title):
    # Keep only alphanumeric chars and spaces/underscores
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', title)
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')

def main():
    if not os.path.exists("data_points"):
        os.makedirs("data_points")

    with open("data/pdf_full_dataset.json", "r") as f:
        data = json.load(f)["data"]
        
    with open("titles.json", "r") as f:
        titles = json.load(f)
        
    # Group data by source
    grouped = {}
    for row in data:
        src = row["source"]
        loc = row["location"]
        bgr = row["color_bgr"]
        
        is_right = src.endswith("_right")
        val = match_color(bgr, is_right)
        
        if src not in grouped:
            grouped[src] = {}
        grouped[src][loc] = val
        
    # Write separate files
    for src, points in grouped.items():
        # Extracted title from OCR (e.g. "Pronostico de anomalias de Lluvia (Page 4)")
        raw_title = titles.get(src, f"Unknown ({src})")
        
        # Build JSON-LD graph instead of a flat dict
        graph = []
        for loc_name, val in points.items():
            safe_id = loc_name.replace(" ", "_")
            # Minimal url encode
            safe_id = safe_id.replace("á", "%C3%A1").replace("í", "%C3%AD").replace("ó", "%C3%B3").replace("é", "%C3%A9").replace("ú", "%C3%BA").replace("ñ", "%C3%B1")
            
            graph.append({
                "@type": "Observation",
                "location": f"http://maps2ai.org/locations/{safe_id}",
                "value": val
            })
            
        output_data = {
            "@context": {
                "schema": "http://schema.org/",
                "Observation": "schema:Observation",
                "location": {"@id": "schema:location", "@type": "@id"},
                "value": "schema:value",
                "variable": "schema:variableMeasured",
                "source": "schema:publisher"
            },
            "source": src,
            "variable_page_title": raw_title,
            "@graph": graph
        }
        
        # Clean filename safely
        safe_name = clean_filename(raw_title)
        out_file = f"data_points/{safe_name}.json"
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"Created {out_file}")

if __name__ == "__main__":
    main()
