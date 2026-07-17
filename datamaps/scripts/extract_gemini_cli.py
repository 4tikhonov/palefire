import json
import argparse
import sys
import os
import urllib.request
from pathlib import Path
from io import BytesIO

from PIL import Image
import fitz  # PyMuPDF
from google import genai
from google.genai import types


GENERAL_PROMPT = """You are analyzing a page from a meteorological report. 
Please extract all key information from this page, including:
1. "text": A summary or the full relevant text presented on the page.
2. "images": A list of descriptions for any photos, graphs, or charts found on the page.
3. "maps": A list of descriptions for any maps found on the page.

Output ONLY raw JSON (no markdown) with this structure:
{
  "text": "Extracted text content...",
  "images": ["Description of image 1", "Description of image 2"],
  "maps": ["Description of map 1"]
}
"""

PROMPT = """You are analyzing a CROPPED SECTION of a rainfall map of Honduras ("REGISTRO DE LLUVIA ACUMULADA"). 

Each weather station is marked with a small blue dot and has a label with:
- The station NAME in uppercase (e.g. "JUTICALPA", "CHOLUTECA")  
- A rainfall VALUE in millimeters displayed as "X mm" or "X.X mm" right next to the station name

YOUR TASK:
1. For EVERY labeled station visible in this section, read the EXACT rainfall value in mm shown next to its name.
   - Look very carefully at each number. The values are small text next to blue dots.
   - Read the ACTUAL digits printed on the image. Do NOT guess or estimate from the background color.
   - Pay special attention to digits that may be hard to distinguish (e.g., 1 vs 7, 3 vs 8, 5 vs 6).
   - If a value has a decimal point, include it precisely.

2. Output ONLY raw JSON (no markdown) with this structure:
   {"data": [{"location": string, "value": number}, ...]}

Only include stations that are clearly visible in this image section.
Double-check every value before finalizing."""

LEGEND_PROMPT = """You are analyzing a rainfall map of Honduras ("REGISTRO DE LLUVIA ACUMULADA").

1. Read the title/header of the map which contains the measurement period, e.g. "12 JULIO 06 AM - 13 JULIO 06 AM". Extract the start and end date/time.
2. Extract the legend (the "LEYENDA LLUVIA mm" box) with each color and its value range.

Output ONLY raw JSON (no markdown) with this structure:
{"measurement_period": {"start": "YYYY-MM-DD HH:MM", "end": "YYYY-MM-DD HH:MM"}, "legend": [{"color_description": string, "value_range": string}, ...]}

For the dates, use the current year and convert Spanish month names to numbers (ENERO=01, FEBRERO=02, MARZO=03, ABRIL=04, MAYO=05, JUNIO=06, JULIO=07, AGOSTO=08, SEPTIEMBRE=09, OCTUBRE=10, NOVIEMBRE=11, DICIEMBRE=12). Use 24-hour format (06 AM = 06:00, 06 PM = 18:00)."""


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def call_gemini(client, model_name: str, image_bytes: bytes, mime_type: str, prompt: str) -> dict:
    import time
    print("      Sleeping for 15s to respect rate limit...")
    time.sleep(15)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = client.models.generate_content(
        model=model_name,
        contents=[image_part, prompt],
    )
    output = response.text.strip()

    # Strip markdown code fences if present
    if output.startswith("```json"):
        output = output[7:]
    elif output.startswith("```"):
        output = output[3:]
    if output.endswith("```"):
        output = output[:-3]
    output = output.strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from response. Raw text:\n{output[:200]}", file=sys.stderr)
        return {}


def load_pdf_pages(source: str):
    """
    Load a PDF from a URL or local file path, and return a list of PIL Images (one per page).
    """
    if source.startswith("http://") or source.startswith("https://"):
        print(f"Downloading PDF from URL: {source}")
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_bytes = resp.read()
        print(f"  Downloaded: {len(raw_bytes)} bytes")
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    else:
        pdf_file = Path(source)
        if not pdf_file.exists():
            print(f"Error: PDF file '{source}' not found.", file=sys.stderr)
            sys.exit(1)
        doc = fitz.open(pdf_file)
        print(f"  Loaded local PDF: {pdf_file}")

    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Increase zoom for better resolution
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages


def load_image(source: str):
    """
    Load an image from a URL or local file path.
    Returns (PIL.Image, raw_bytes, mime_type).
    """
    if source.startswith("http://") or source.startswith("https://"):
        print(f"Downloading image from URL: {source}")
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_bytes = resp.read()
            content_type = resp.headers.get("Content-Type", "image/png")
        mime_type = content_type.split(";")[0].strip()
        img = Image.open(BytesIO(raw_bytes))
        print(f"  Downloaded: {len(raw_bytes)} bytes, {img.size[0]}x{img.size[1]}, {mime_type}")
    else:
        image_file = Path(source)
        if not image_file.exists():
            print(f"Error: Image file '{source}' not found.", file=sys.stderr)
            sys.exit(1)
        raw_bytes = image_file.read_bytes()
        suffix = image_file.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webp": "image/webp", ".gif": "image/gif"}
        mime_type = mime_map.get(suffix, "image/png")
        img = Image.open(image_file)
        print(f"  Loaded local file: {len(raw_bytes)} bytes, {img.size[0]}x{img.size[1]}")

    return img, raw_bytes, mime_type


def extract_map_data(img: Image.Image, client, model_name: str):
    """
    Extracts legend and map data from a single image using quadrant approach.
    """
    w, h = img.size
    full_bytes = image_to_bytes(img)
    mime_type = "image/png"

    overlap_x = int(w * 0.15)
    overlap_y = int(h * 0.15)
    mid_x = w // 2
    mid_y = h // 2

    quadrants = {
        "top-left":     img.crop((0, 0, mid_x + overlap_x, mid_y + overlap_y)),
        "top-right":    img.crop((mid_x - overlap_x, 0, w, mid_y + overlap_y)),
        "bottom-left":  img.crop((0, mid_y - overlap_y, mid_x + overlap_x, h)),
        "bottom-right": img.crop((mid_x - overlap_x, mid_y - overlap_y, w, h)),
    }

    # 1. Extract legend and measurement period from the full image
    print(f"    Extracting legend and measurement period from full image...")
    legend_result = call_gemini(client, model_name, full_bytes, mime_type, LEGEND_PROMPT)
    legend = legend_result.get("legend", [])
    period = legend_result.get("measurement_period", {})
    period_start = period.get("start", "")
    period_end = period.get("end", "")
    print(f"      Found {len(legend)} legend entries.")

    # 2. Extract data from each quadrant
    all_data = {}
    for name, quad_img in quadrants.items():
        print(f"    Processing quadrant: {name}...")
        quad_bytes = image_to_bytes(quad_img)
        result = call_gemini(client, model_name, quad_bytes, mime_type, PROMPT)
        entries = result.get("data", [])
        for entry in entries:
            loc = entry.get("location", "").strip()
            val = entry.get("value")
            if loc:
                all_data[loc] = val

    data_list = [
        {"location": k, "value": v,
         "measurement_start": period_start, "measurement_end": period_end}
        for k, v in sorted(all_data.items())
    ]
    print(f"    Total unique stations on map: {len(data_list)}")
    return {"measurement_period": period, "legend": legend, "data": data_list}


def process_source(source: str, model_name: str = "gemini-2.5-flash"):
    client = genai.Client()
    is_pdf = source.lower().endswith(".pdf")

    report_dir = Path("report")
    report_dir.mkdir(exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if is_pdf:
        pages = load_pdf_pages(source)
    else:
        img, _, _ = load_image(source)
        pages = [img]

    for i, page_img in enumerate(pages):
        page_num = i + 1
        print(f"\n--- Processing Page {page_num} ---")
        
        # Save page image
        page_image_path = report_dir / f"page_{page_num}.png"
        page_img.save(page_image_path, "PNG")
        print(f"  Saved page image to {page_image_path}")

        page_data_dir = data_dir / f"page_{page_num}"
        page_data_dir.mkdir(parents=True, exist_ok=True)

        page_bytes = image_to_bytes(page_img)
        mime_type = "image/png"

        # General extraction (text, images, maps)
        print("  Extracting general page info (text, images, maps)...")
        general_info = call_gemini(client, model_name, page_bytes, mime_type, GENERAL_PROMPT)
        
        # Determine if there's a map that looks like a rainfall map
        # For simplicity, we apply the map extraction if we detected maps in general info,
        # but since we don't know for sure, we'll run it and see if it finds stations.
        has_maps = len(general_info.get("maps", [])) > 0
        if has_maps:
            print("  Map detected. Extracting map data points...")
            map_data = extract_map_data(page_img, client, model_name)
            general_info["map_data"] = map_data
        else:
            print("  No maps detected. Skipping map data point extraction.")

        output_json_path = page_data_dir / "data.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(general_info, f, indent=2, ensure_ascii=False)
        print(f"  Saved extracted data to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract legend and data values from map or PDF using Gemini API")
    parser.add_argument("source", help="Path or URL to the map image or PDF", nargs='?',
                        default="https://cenaos.copeco.gob.hn/productos/pronostico_estacional/pronosticoestacional.pdf")
    parser.add_argument("--model", "-m", help="Gemini model to use", default="gemini-2.5-flash")

    args = parser.parse_args()
    process_source(args.source, args.model)
