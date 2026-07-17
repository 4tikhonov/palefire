#!/usr/bin/env python3
"""
Extract all text labels from a PNG and give you their coordinates.
Dependencies: Pillow, pytesseract
"""

import os
from PIL import Image
import pytesseract
import json

IMG = "HONDURAS12MllM.png"
IMG = "upscale.png"

# 1. Load image
im = Image.open(IMG)

# 2. Run OCR – full page mode (psm 6) is usually best for maps
data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)

# 3. Build a list of labels with text + bbox
labels = []
n_boxes = len(data['level'])
for i in range(n_boxes):
    text = data['text'][i].strip()
    if not text:          # skip empty strings
        continue
    x, y, w, h = (data['left'][i], data['top'][i],
                  data['width'][i], data['height'][i])
    labels.append({
        "text": text,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "confidence": data['conf'][i]
    })

# 4. Output
out_file = "labels.json"
with open(out_file, "w") as f:
    json.dump(labels, f, indent=2)

print(f"Found {len(labels)} labels → {out_file}")

