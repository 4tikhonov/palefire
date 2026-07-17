import os
import json
import csv
import urllib.parse

def main():
    data_dir = "data_points"
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    
    # Store all loaded data
    loaded_data = {}
    all_locations = set()
    
    # Read all files
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)
            
        src = content["source"]
        title = content["variable_page_title"]
        
        points = {}
        for obs in content.get("@graph", []):
            loc_url = obs.get("location", "")
            loc_name = urllib.parse.unquote(loc_url.split("/")[-1]).replace("_", " ")
            points[loc_name] = obs.get("value", "No Map")
        
        loaded_data[src] = {
            "title": title,
            "points": points
        }
        
        for loc in points.keys():
            all_locations.add(loc)
            
    # Order sources: full_page-01.png_left, full_page-01.png_right, etc.
    ordered_sources = []
    for page in range(1, 29):
        ordered_sources.append(f"full_page-{page:02d}.png_left")
        ordered_sources.append(f"full_page-{page:02d}.png_right")
        
    # Build header and data rows
    header = ["Location"]
    for src in ordered_sources:
        if src in loaded_data:
            header.append(loaded_data[src]["title"])
            
    with open("report/final_heatmap_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for loc in sorted(all_locations):
            row_out = [loc]
            for src in ordered_sources:
                if src in loaded_data:
                    row_out.append(loaded_data[src]["points"].get(loc, "No Map"))
            writer.writerow(row_out)
            
    print("Successfully compiled final_heatmap_report.csv from data_points/ directory!")

if __name__ == "__main__":
    main()
