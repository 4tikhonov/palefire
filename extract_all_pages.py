import fitz # PyMuPDF
from PIL import Image
from pathlib import Path

pdf_file = "/tmp/pronostico.pdf"
out_dir = Path("datamaps/data/pdf-extraction")
out_dir.mkdir(parents=True, exist_ok=True)

# Clear old images
for f in out_dir.glob("*.jpg"):
    f.unlink()

print(f"Extracting all pages from {pdf_file}...")
doc = fitz.open(pdf_file)
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    out_path = out_dir / f"page_{page_num + 1:02d}.jpg"
    img.save(out_path, "JPEG", quality=90)
    print(f"Saved {out_path}")

print(f"Total pages extracted: {len(doc)}")
