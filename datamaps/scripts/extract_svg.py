import json
import argparse
import sys
import os
from pathlib import Path
from io import BytesIO

from PIL import Image
from google import genai
from google.genai import types

SVG_PROMPT = """You are analyzing a map of Honduras with geographical boundaries.
Please extract the geographical boundaries (Departments) from the image and generate a valid SVG (Scalable Vector Graphics) representation of these boundaries.
The SVG should:
- Use <path> or <polygon> elements to trace the outline of each department.
- Include a viewBox that matches the proportions of the map in the image.
- Only output the raw SVG code, without any markdown formatting, explanation, or HTML wrapping.
- Just the raw <svg>...</svg> content.
"""

def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def extract_svg(image_path: str, model_name: str = "gemini-2.5-pro", output_path: str = "mapboundaries/boundaries.svg"):
    if not Path(image_path).exists():
        print(f"Error: Image {image_path} not found.")
        sys.exit(1)

    print(f"Loading image {image_path}...")
    img = Image.open(image_path)
    image_bytes = image_to_bytes(img)
    mime_type = "image/png"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    client = genai.Client()

    print("Sending image to Gemini for SVG boundary extraction...")
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = client.models.generate_content(
        model=model_name,
        contents=[image_part, SVG_PROMPT],
    )
    
    output = response.text.strip()
    
    # Clean up markdown formatting if present
    if output.startswith("```xml"):
        output = output[6:]
    elif output.startswith("```svg"):
        output = output[6:]
    elif output.startswith("```"):
        output = output[3:]
    if output.endswith("```"):
        output = output[:-3]
    output = output.strip()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"SVG Extraction successful. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SVG boundaries from a map image.")
    parser.add_argument("image", help="Path to the heatmap image (e.g., report/page_4.png)")
    parser.add_argument("--output", "-o", help="Output SVG file path", default="mapboundaries/boundaries.svg")
    # Use Flash model
    parser.add_argument("--model", "-m", help="Gemini model to use", default="gemini-2.5-flash")

    args = parser.parse_args()
    extract_svg(args.image, args.model, args.output)
