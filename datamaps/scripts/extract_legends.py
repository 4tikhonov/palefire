import os
import glob
import json
import time
from pathlib import Path
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

PROMPT = """You are analyzing a cropped heatmap from a meteorological report for Honduras.

YOUR TASK:
1. Locate and read the map's LEGEND.
2. For each color patch in the legend, extract:
   - "color": A textual description of the color. YOU MUST strictly map the visual color to one of the following predefined standard names:
     ["Grey", "White", "Light Green", "Green", "Dark Green", "Light Yellow", "Yellow", "Light Orange", "Orange", "Dark Orange", "Red", "Dark Red", "Light Blue", "Blue", "Dark Blue"]
   - "value_range": The relative anomaly percentage or temperature range (e.g., "0% to 20%", ">= 2 °C", "-40% to -20%")
   - "absolute_range": Any absolute numerical values or thresholds shown next to the color (e.g., "0 to 50 mm", "50 to 100 mm"). If none exist, output "N/A".
   - "label": The categorical text label describing the anomaly (e.g., "Normal to Above Normal", "Far Below Normal", "Warmer"). If the text is spread across lines, consolidate it.

Output ONLY raw JSON (no markdown block, no backticks, just the raw braces) in this format:
{
  "legend": [
    {"color": "Yellow", "value_range": "-20% to 0%", "absolute_range": "50 to 100 mm", "label": "Below Normal"},
    ...
  ]
}
"""

def process_map(client, image_path, retries=3):
    img = Image.open(image_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    
    image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
    
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_part, PROMPT],
            )
            
            output = response.text.strip()
            if output.startswith("```json"):
                output = output[7:]
            elif output.startswith("```"):
                output = output[3:]
            if output.endswith("```"):
                output = output[:-3]
                
            data = json.loads(output.strip())
            return data.get("legend", [])
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                print(f"Rate limit hit processing {image_path}. Retrying in 30 seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(30)
            else:
                print(f"Error processing {image_path}: {e}")
                # Don't return [], return None so we don't save a false empty legend
                return None
    
    print(f"Failed to process {image_path} after {retries} retries due to rate limits.")
    return None

def main():
    print("Extracting legends for all maps...")
    os.makedirs("data", exist_ok=True)
    
    legends = {}
    if os.path.exists("data/legends.json"):
        try:
            with open("data/legends.json", "r", encoding="utf-8") as f:
                legends = json.load(f)
        except Exception:
            pass
            
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMIN_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return
        
    client = genai.Client(api_key=api_key)
    maps = glob.glob("report/debug_full_page-*_*.png")
    maps.sort()
    
    for i, map_path in enumerate(maps):
        map_name = os.path.basename(map_path)
        if map_name in legends:
            continue
            
        print(f"[{i+1}/{len(maps)}] Processing {map_name}...")
        
        legend_data = process_map(client, map_path)
        if legend_data is None:
            print("API error encountered. Stopping to prevent data corruption.")
            break
            
        legends[map_name] = legend_data
        
        with open("data/legends.json", "w", encoding="utf-8") as f:
            json.dump(legends, f, indent=2, ensure_ascii=False)
            
        # Rate limiting to stay under 15 RPM for the free tier
        time.sleep(4.1)
        
    with open("data/legends.json", "w", encoding="utf-8") as f:
        json.dump(legends, f, indent=2, ensure_ascii=False)
        
    print("Legends extraction complete.")

if __name__ == "__main__":
    main()
