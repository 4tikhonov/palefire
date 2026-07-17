import json
from datamaps.compiler import get_color_name

def test():
    # Atlántida on Page 5 Left BGR is [50, 220, 230]
    bgr = [50, 220, 230]
    color_name = get_color_name(bgr)
    print(f"Mapped BGR {bgr} to color_name: {color_name}")
    
    with open("data/legends.json", "r", encoding="utf-8") as f:
        legends = json.load(f)
        
    legend = legends["debug_full_page-05_left.png"]
    legend_lookup = { item["color"]: item for item in legend }
    
    mapping = legend_lookup.get(color_name)
    if not mapping:
        print(f"{color_name} not found in exact keys: {list(legend_lookup.keys())}")
        for k in legend_lookup.keys():
            if color_name in k or k in color_name:
                print(f"Fuzzy match found: {k}")
                mapping = legend_lookup[k]
                break
                
    print(mapping)
    
if __name__ == "__main__":
    test()
