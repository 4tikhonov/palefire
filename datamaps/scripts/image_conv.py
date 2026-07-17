#!/usr/bin/env python3
import os, sys, subprocess, tempfile
from PIL import Image

IMG = "HONDURAS12MllM.png"
OUT_SVG = "map_coloured.svg"

# 1. Load image and get unique colours (as RGB tuples)
im = Image.open(IMG).convert("RGB")
pixels = list(im.getdata())
unique_colors = sorted(set(pixels))          # e.g., [(0,0,0), (255,0,0), ...]

print(f"Found {len(unique_colors)} distinct colours")

# 2. Create a temporary directory to hold masks
tmpdir = tempfile.mkdtemp()
paths = []

for idx, color in enumerate(unique_colors):
    mask_name = os.path.join(tmpdir, f"mask_{idx}.pbm")
    # Build PBM: black for pixels that match the colour, white otherwise
    with open(mask_name, "wb") as f:
        # PBM header (plain PBM is easier to write)
        f.write(b"P1\n")
        f.write(f"{im.width} {im.height}\n".encode())

        for y in range(im.height):
            row = []
            for x in range(im.width):
                if im.getpixel((x, y)) == color:
                    row.append("0")   # black
                else:
                    row.append("1")   # white
            f.write((" ".join(row) + "\n").encode())

    # 3. Run potrace on the mask to get SVG fragment
    svg_fragment = os.path.join(tmpdir, f"frag_{idx}.svg")
    subprocess.run(
        ["potrace", "-s", "-o", svg_fragment, mask_name],
        check=True,
    )

    # Read the fragment and strip outer <svg> tags
    with open(svg_fragment) as f:
        data = f.read()
        start = data.find("<g")          # Potrace outputs a <g> block
        end   = data.rfind("</g>")
        if start != -1 and end != -1:
            fragment = data[start:end+4]
            # Add the colour as a fill attribute
            fragment = fragment.replace(
                "fill=\"#000000\"",
                f"fill=\"rgb{color}\""
            )
            paths.append(fragment)

# 4. Assemble final SVG
svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{im.width}" height="{im.height}"
     viewBox="0 0 {im.width} {im.height}">
'''
with open(OUT_SVG, "w") as out:
    out.write(svg_header)
    for p in paths:
        out.write(p + "\n")
    out.write("</svg>")

print(f"Vectorised SVG written to {OUT_SVG}")

