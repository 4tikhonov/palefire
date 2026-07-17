import subprocess
import os
import json
import cv2

def clean_title(text):
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
    clean_lines = []
    for line in lines:
        if "GrADS" in line or "CENAOS" in line or "MET" in line or "22N" in line or "21N" in line or "20N" in line or "19N" in line or "18N" in line or "17N" in line: continue
        if len(line) < 8: continue
        clean_lines.append(line)
        
    if not clean_lines: return "Unknown Title"
    
    longest = max(clean_lines, key=len)
    return longest.replace('"', '').replace('|', '').strip()

def main():
    with open("data/pdf_full_dataset.json", "r") as f:
        data = json.load(f)["data"]
        
    sources = set(row["source"] for row in data)
    title_map = {}
    
    for src in sorted(list(sources)):
        # src: full_page-03.png_left
        parts = src.split(".png_")
        if len(parts) != 2: continue
        img_name = f"report/debug_{parts[0]}_{parts[1]}.png"
        # Read and crop using cv2 to guarantee dimensions
        img = cv2.imread(img_name)
        if img is not None:
            h, w = img.shape[:2]
            crop = img[0:int(h*0.15), int(w*0.25):w]
            cv2.imwrite("/tmp/temp_crop.png", crop)
        else:
            continue
            
        # OCR
        res = subprocess.run(["tesseract", "/tmp/temp_crop.png", "stdout"], capture_output=True, text=True)
        title = clean_title(res.stdout)
        
        # If it couldn't find a good title, fall back to "Pronostico de anomalias..."
        if title == "Unknown Title":
            if parts[1] == "left":
                title = "Pronostico de anomalias de Lluvia"
            else:
                title = "Pronostico de anomalias de Temperatura"
                
        # Format the title
        page_num = parts[0].split("-")[1]
        title_map[src] = f"{title} (Page {int(page_num)})"
        print(f"{src} -> {title_map[src]}")
        
    with open("titles.json", "w") as f:
        json.dump(title_map, f, indent=2)

if __name__ == "__main__":
    main()
