import cv2
import sys
import os
import subprocess

def extract_title(image_path):
    img = cv2.imread(image_path)
    if img is None: return "Unknown Title"
    
    # Crop top 15%
    h, w = img.shape[:2]
    crop = img[0:int(h*0.15), 0:w]
    
    # Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite("temp_crop.png", gray)
    
    # OCR via subprocess
    try:
        result = subprocess.run(['tesseract', 'temp_crop.png', 'stdout'], capture_output=True, text=True)
        text = result.stdout
    except Exception as e:
        return f"Error: {e}"
    
    # Clean lines
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
    
    # Filter out junk like "GrADS/COLA" or coordinates
    clean_lines = []
    for line in lines:
        if "GrADS" in line or "CENAOS" in line: continue
        if len(line) < 10: continue
        clean_lines.append(line)
        
    if not clean_lines:
        return "Unknown Title"
        
    # Return the longest string (most likely the title)
    longest = max(clean_lines, key=len)
    return longest.replace('"', '').strip()

print(extract_title("report/page_4_400_left.png"))
print(extract_title("report/page_4_400_right.png"))
