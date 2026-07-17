import json
import sys
import os
from pathlib import Path
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

PROMPT = """You are analyzing a cropped heatmap from a meteorological report for Honduras.

YOUR TASK:
1. Locate and read the map's LEGEND.
2. For each color in the legend, extract:
   - "color": A textual description of the color (e.g., "Dark Blue", "Red", "Yellow")
   - "value_range": The relative anomaly percentage or temperature range (e.g., "0% to 20%", ">= 2 °C")
   - "absolute_range": Any absolute numerical values or thresholds shown next to the color (e.g., "0 to 50 mm"). If none exist, output "N/A".
   - "value": A categorical or descriptive value representing what this range means (e.g., "Above Normal", "Very High").
3. Then, for EVERY department in Honduras visible on the map, determine which color it is painted with, and output the EXACT "value_range" and "absolute_range" it corresponds to based on the legend.

Output ONLY raw JSON (no markdown block, just the raw braces) in this format:
{
  "legend": [
    {"color": "Green", "value_range": "0% to 20%", "absolute_range": "50 to 100 mm", "value": "Normal"},
    ...
  ],
  "data": [
    {"location": "Atlántida", "value_range": "0% to 20%", "absolute_range": "50 to 100 mm"},
    ...
  ]
}
"""

def extract_baseline_with_gemini(image_path: str, output_path: str):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return

    print(f"Sending {image_path} to Gemini...")
    img = Image.open(image_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    client = genai.Client()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    
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
        
    output = output.strip()
    
    # Verify it is valid JSON
    try:
        data = json.loads(output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved baseline to {output_path}")
    except Exception as e:
        print("Failed to parse JSON response:")
        print(output)
        print(e)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gemini_baseline.py <input_image> <output_json>")
        sys.exit(1)
    extract_baseline_with_gemini(sys.argv[1], sys.argv[2])
