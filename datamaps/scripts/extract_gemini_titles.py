import os
import json
import base64
import cv2
import requests
import glob

API_KEY = os.environ.get("GEMINI_API_KEY")

from google import genai
from PIL import Image

def get_gemini_title(image_path):
    img = cv2.imread(image_path)
    if img is None: return "Unknown"
    h, w = img.shape[:2]
    crop = img[0:int(h*0.15), 0:w]
    cv2.imwrite("/tmp/temp.jpg", crop)
    
    pil_img = Image.open("/tmp/temp.jpg")

    prompt = """You are looking at the top crop of a presentation slide or map.
    Extract the main title or variable description. 
    Examples of good outputs: 'Pronostico de anomalias de Lluvia', 'Pronostico de anomalias de Temperatura', 'REPUBLICA DE HONDURAS', 'LLUVIA ESTIMADA POR SATELITE'.
    Respond ONLY with the exact extracted title text. Do not add any extra text."""

    client = genai.Client()
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_img]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error for {image_path}: {e}")
        return "Unknown"

def main():
    # Load sources from the dataset
    with open("data/pdf_full_dataset.json", "r") as f:
        data = json.load(f)["data"]
        
    sources = set(row["source"] for row in data)
    title_map = {}
    
    for src in sorted(list(sources)):
        # src looks like full_page-04.png_left
        parts = src.split(".png_")
        page_name = parts[0] + ".png" # full_page-04.png
        side = parts[1] # left
        
        orig_img = f"report/{page_name}"
        if not os.path.exists(orig_img):
            continue
            
        img = cv2.imread(orig_img)
        h, w = img.shape[:2]
        
        # Split original image
        mid = w // 2
        if side == "left":
            split_img = img[:, :mid]
        else:
            split_img = img[:, mid:]
            
        cv2.imwrite("/tmp/temp.jpg", split_img)
        
        title = get_gemini_title("/tmp/temp.jpg")
        # Format the title with the page number
        page_num = int(page_name.split("-")[1].split(".")[0])
        
        # Clean up title
        title = title.replace("\n", " ").strip()
        final_col_name = f"{title} (Page {page_num})"
        title_map[src] = final_col_name
        print(f"{src} -> {final_col_name}")
        
    with open("gemini_titles.json", "w") as f:
        json.dump(title_map, f, indent=2)

if __name__ == "__main__":
    main()
