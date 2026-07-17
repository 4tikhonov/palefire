#!/bin/bash
set -euo pipefail

URL="$1"
if [ -z "$URL" ]; then
    echo "Usage: $0 <pdf-url>"
    exit 1
fi

echo "1. Downloading PDF..."
mkdir -p report raw

file_size() {
    # Portable size in bytes (GNU and BSD/macOS)
    if stat -c%s "$1" >/dev/null 2>&1; then
        stat -c%s "$1"
    else
        stat -f%z "$1"
    fi
}

need_download=1
if [ -f "report/source.pdf" ]; then
    local_size=$(file_size report/source.pdf)
    remote_content_len=$(curl --write-out '%{http_code} %{size_download}' -sI --max-time 10 "$URL" 2>/dev/null \
        | awk 'BEGIN{IGNORECASE=1} /^Content-Length:/{print $2}' | tr -d '\r' | head -1)
    if [ -n "$remote_content_len" ] && [ "${local_size:-0}" = "$remote_content_len" ]; then
        echo "Using cached report/source.pdf (${local_size} bytes)"
        need_download=0
    fi
fi

if [ "$need_download" = "1" ]; then
    curl -L --fail --progress-bar -o report/source.pdf "$URL"
fi

echo "2. Extracting pages to PNG..."
rm -f report/full_page-*.png

if [ -n "${PAGES:-}" ]; then
    echo "Filtering to pages: $PAGES" | tee -a .pipeline_cache.log
    page_num=$(echo "$PAGES" | grep -oE '[0-9]+' | head -1)
    if [ -z "${page_num:-}" ]; then
        echo "WARNING: Invalid PAGES format: $PAGES" | tee -a .pipeline_cache.log
        exit 1
    fi
    # pdftoppm uses the real page number in the filename (e.g. full_page-2.png)
    pdftoppm -png -r 288 -f "$page_num" -l "$page_num" report/source.pdf report/full_page
else
    pdftoppm -png -r 288 report/source.pdf report/full_page
fi

# Normalize to zero-padded names (full_page-02.png) to match the rest of the pipeline
for f in report/full_page-*.png; do
    [ -e "$f" ] || continue
    base=$(basename "$f" .png)          # full_page-2 or full_page-02
    num=${base#full_page-}
    padded=$(printf "%02d" "$((10#$num))")
    target="report/full_page-${padded}.png"
    if [ "$f" != "$target" ]; then
        mv "$f" "$target"
    fi
done

echo "PDF conversion complete. Running datamaps extract pipeline..."

docker compose run --rm app python -m datamaps extract || {
    echo "# WARNING: Extract step failed" | tee -a .pipeline_cache.log
}

echo "3. Running OCR on extracted maps (raw Tesseract -> raw/page-NN/)..."

docker compose run --rm app python -m datamaps ocr || {
    echo "# WARNING: OCR step failed" | tee -a .pipeline_cache.log
}

echo "4. Extracting legends..."

docker compose run --rm app python scripts/extract_legends.py || {
    echo "# WARNING: Legend extraction skipped" | tee -a .pipeline_cache.log
}

echo "5. Splitting to JSON-LD datasets..."

docker compose run --rm app python -m datamaps split || {
    echo "# WARNING: Split step failed" | tee -a .pipeline_cache.log
}

echo "6. Compiling final CSV report..."

if compgen -G "data_points/*.json" > /dev/null; then
    docker compose run --rm app python -m datamaps compile
else
    echo "# No JSON-LD files to compile yet" | tee -a .pipeline_cache.log
fi

echo "Pipeline complete! Check raw/, data_points/, and report/final_heatmap_report.csv"
