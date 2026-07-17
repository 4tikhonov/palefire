import json
import argparse
import sys
from pathlib import Path
from io import BytesIO

from PIL import Image
from google import genai
from google.genai import types

HEATMAP_PROMPT = """You are analyzing a heatmap of Honduras containing geographical boundaries (Departments) and colored regions.

1. First, identify the legend mapping colors to percentage values (e.g. ">= 20%", "0% to 20%", "-20% to 0%", etc.).
2. Second, visually analyze the map and identify the geographical regions (Departments) shown.
3. For each identified region, determine its color and map it to the corresponding value range from the legend.

CRITICAL INSTRUCTION: You MUST output data individually for exactly these 18 Departments. DO NOT group them into regions (like "Northern Coastal Region").
List of departments:
Atlántida, Choluteca, Colón, Comayagua, Copán, Cortés, El Paraíso, Francisco Morazán, Gracias a Dios, Intibucá, Islas de la Bahía, La Paz, Lempira, Ocotepeque, Olancho, Santa Bárbara, Valle, Yoro.

Output ONLY raw JSON (no markdown) with this structure:
{
  "legend": [
    {"color": "string", "value_range": "string"}
  ],
  "data": [
    {"location": "Department/Region Name", "value_range": "percentage range string"}
  ]
}
"""

def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def extract_heatmap(image_path: str, model_name: str = "gemini-2.5-flash", output_path: str = "heatmap_data.json"):
    if not Path(image_path).exists():
        print(f"Error: Image {image_path} not found.")
        sys.exit(1)

    print(f"Loading image {image_path}...")
    img = Image.open(image_path)
    image_bytes = image_to_bytes(img)
    mime_type = "image/png"

    client = genai.Client()

    print("Sending heatmap to Gemini for analysis...")
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = client.models.generate_content(
        model=model_name,
        contents=[image_part, HEATMAP_PROMPT],
    )
    
    output = response.text.strip()
    if output.startswith("```json"):
        output = output[7:]
    elif output.startswith("```"):
        output = output[3:]
    if output.endswith("```"):
        output = output[:-3]
    output = output.strip()

    try:
        data = json.loads(output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Extraction successful. Data saved to {output_path}")
        print(f"Found {len(data.get('data', []))} locations.")
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from response. Raw text:\n{output[:200]}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a heatmap image and extract locations and values.")
    parser.add_argument("image", help="Path to the heatmap image (e.g., report/page_4.png)")
    parser.add_argument("--output", "-o", help="Output JSON file name", default="heatmap_data.json")
    parser.add_argument("--model", "-m", help="Gemini model to use", default="gemini-2.5-flash")

    args = parser.parse_args()
    extract_heatmap(args.image, args.model, args.output)
