import json
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

def ask_points(image_path):
    img = Image.open(image_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    
    prompt = """You are analyzing a map of Honduras. Look at the colored landmass of Honduras.
Find the EXACT pixel coordinates for these three extreme tips of the landmass:
1. The westernmost tip (furthest left point of the land)
2. The easternmost tip (furthest right point of the land, usually Gracias a Dios)
3. The southernmost tip (furthest bottom point of the land, usually Choluteca)

The image dimensions are """ + f"{img.width}x{img.height}" + """.

Output ONLY valid JSON with no markdown formatting:
{
  "west": {"x": 123, "y": 456},
  "east": {"x": 789, "y": 123},
  "south": {"x": 456, "y": 789}
}"""

    client = genai.Client()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image_part, prompt],
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
    ask_points(sys.argv[1] if len(sys.argv) > 1 else "report/page_4_400_left.png")
