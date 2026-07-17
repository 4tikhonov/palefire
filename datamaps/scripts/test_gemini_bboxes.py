import os
import sys
import json
from PIL import Image
from io import BytesIO
from google import genai
from google.genai import types

PROMPT = """You are analyzing a meteorological map of Honduras.
Find the LEGEND on the map.
For each color patch in the legend, please return:
1. The color name
2. The text (value range) written right next to it
3. A 2D bounding box that tightly encompasses ONLY the colored square/rectangle in the legend.

Use the standard bounding box format: [ymin, xmin, ymax, xmax] scaled to 0-1000.

Output strictly as a JSON array of objects:
[
  {
    "color": "Green",
    "value": "0% to 20%",
    "box": [ymin, xmin, ymax, xmax]
  }
]
"""

def test_bboxes(image_path):
    print(f"Testing {image_path}...")
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
        
    print(output.strip())

if __name__ == "__main__":
    test_bboxes("report/debug_full_page-04_left.png")
