import json
import argparse
import sys
from pathlib import Path
from unidecode import unidecode

def normalize_name(name):
    """Normalize names to handle accents and casing differences."""
    return unidecode(name).strip().lower()

def merge_geojson(geojson_path: str, heatmap_path: str, output_path: str):
    if not Path(geojson_path).exists():
        print(f"Error: GeoJSON file {geojson_path} not found.")
        sys.exit(1)
        
    if not Path(heatmap_path).exists():
        print(f"Error: Heatmap data file {heatmap_path} not found.")
        sys.exit(1)
        
    with open(geojson_path, "r", encoding="utf-8") as f:
        geo_data = json.load(f)
        
    with open(heatmap_path, "r", encoding="utf-8") as f:
        heatmap_data = json.load(f)
        
    # Create a mapping of normalized location names to their value and color
    legend = heatmap_data.get("legend", [])
    color_map = {item["value_range"]: item["color"] for item in legend}
    
    data_points = heatmap_data.get("data", [])
    data_dict = {}
    for item in data_points:
        norm_name = normalize_name(item["location"])
        val_range = item.get("value_range", "N/A")
        color = color_map.get(val_range, "#cccccc")
        data_dict[norm_name] = {
            "value_range": val_range,
            "color": color
        }
        
    print(f"Loaded {len(data_dict)} data points from heatmap.")
    
    # Merge properties into GeoJSON features
    matched_count = 0
    for feature in geo_data.get("features", []):
        props = feature.get("properties", {})
        
        # In geoBoundaries, the ADM1 name is usually under shapeName
        shape_name = props.get("shapeName", "")
        norm_shape_name = normalize_name(shape_name)
        
        if norm_shape_name in data_dict:
            props["heatmap_value_range"] = data_dict[norm_shape_name]["value_range"]
            props["heatmap_color"] = data_dict[norm_shape_name]["color"]
            matched_count += 1
        else:
            props["heatmap_value_range"] = "No Data"
            props["heatmap_color"] = "#cccccc"
            
    print(f"Matched {matched_count} geographic boundaries with heatmap data.")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geo_data, f, ensure_ascii=False)
        
    print(f"Successfully saved merged GeoJSON to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge heatmap data with GeoJSON.")
    parser.add_argument("--geojson", default="honduras_departments.geojson", help="Base GeoJSON file")
    parser.add_argument("--data", default="data/page_4/heatmap_data.json", help="Extracted heatmap data JSON")
    parser.add_argument("--output", "-o", default="data/page_4/honduras_heatmap.geojson", help="Output merged GeoJSON file")

    args = parser.parse_args()
    import os
    merge_geojson(args.geojson, args.data, args.output)
