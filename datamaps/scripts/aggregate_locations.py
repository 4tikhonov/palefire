import json
import argparse
import sys
from pathlib import Path
import geopandas as gpd
from collections import Counter

LEGEND_COLORS = {
    "dark_green": {"bgr": (50, 150, 50), "range": ">= 20%"},
    "light_green": {"bgr": (150, 250, 150), "range": "0% to 20%"},
    "yellow": {"bgr": (0, 255, 255), "range": "-20% to 0%"},
    "orange": {"bgr": (0, 165, 255), "range": "-40% to -20%"},
    "dark_reddish_brown": {"bgr": (0, 50, 150), "range": "-65% to -40%"},
}

def aggregate_locations(grid_path, geojson_path, output_path):
    print(f"Loading Grid Data from {grid_path}...")
    with open(grid_path, "r") as f:
        grid_data = json.load(f)
        
    print(f"Loading GeoJSON {geojson_path}...")
    gdf = gpd.read_file(geojson_path)
    
    lons = [item["lon"] for item in grid_data]
    lats = [item["lat"] for item in grid_data]
    colors = [item["color"] for item in grid_data]
    ranges = [item["value_range"] for item in grid_data]
    
    grid_gdf = gpd.GeoDataFrame(
        {'color': colors, 'value_range': ranges},
        geometry=gpd.points_from_xy(lons, lats)
    )
    
    grid_gdf.set_crs(epsg=4326, inplace=True)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
        
    print("Performing Spatial Join...")
    joined = gpd.sjoin(grid_gdf, gdf, how="inner", predicate="intersects")
    
    results = []
    
    for idx, row in gdf.iterrows():
        dept_name = row.get("shapeName", f"Department_{idx}")
        
        dept_pts = joined[joined.index_right == idx]
        
        if len(dept_pts) == 0:
            print(f"Warning: No valid grid points found inside {dept_name}")
            continue
            
        counter = Counter(dept_pts["value_range"])
        final_value = counter.most_common(1)[0][0]
        final_color = dept_pts[dept_pts["value_range"] == final_value].iloc[0]["color"]
        
        results.append({
            "location": dept_name,
            "value_range": final_value,
            "matched_color": final_color,
            "sampled_points": len(dept_pts),
            "vote_distribution": dict(counter)
        })
        
        print(f"Aggregated {dept_name}: {final_value} (from {len(dept_pts)} points)")
        
    output_data = {
        "legend": [{"color": k, "value_range": v["range"]} for k, v in LEGEND_COLORS.items()],
        "data": results
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved aggregated baseline to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="data/page_4/lon_lat_grid.json")
    parser.add_argument("--geojson", default="honduras_departments.geojson")
    parser.add_argument("--output", "-o", default="data/page_4/local_heatmap_data_grid_left.json")
    args = parser.parse_args()
    
    aggregate_locations(args.grid, args.geojson, args.output)
