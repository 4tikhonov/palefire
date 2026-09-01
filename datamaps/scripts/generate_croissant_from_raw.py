import os
import re
import csv
import json
from pathlib import Path

def extract_metadata_from_page01(raw_dir):
    page01_file = os.path.join(raw_dir, "page-01", "fullpage.txt")
    if not os.path.exists(page01_file):
        return "Unknown Title", "Unknown Author"

    with open(page01_file, "r", encoding="utf-8") as f:
        content = f.read()

    lines = [line.strip() for line in content.split("\n") if line.strip() and "~" not in line]
    
    title = "Unknown Title"
    author = "Unknown Author"
    
    # Simple heuristics based on the provided sample
    for i, line in enumerate(lines):
        if "PERSPECTIVA" in line.upper() or "CLIMATICA" in line.upper():
            title = " ".join(lines[i:i+3]) # grab next 2 lines as well
            break
            
    author_lines = []
    for line in lines:
        if "Secretaria" in line or "CENAOS" in line or "COPECO" in line or "CENTRO" in line or "SISMICOS" in line:
            author_lines.append(line)
            
    if author_lines:
        author = " / ".join(author_lines)

    return title, author

def process_data_points(raw_dir, data_points_dir, output_csv):
    pages = sorted([d for d in os.listdir(raw_dir) if d.startswith("page-") and d != "page-01"])
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    fieldnames = ["page_num", "fullpage_txt_file", "legend_txt_file", "left_txt_file", "right_txt_file", "left_data_point_file", "right_data_point_file"]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for page in pages:
            page_dir = os.path.join(raw_dir, page)
            row = {"page_num": page}
            
            for txt_name in ["fullpage.txt", "legend.txt", "left.txt", "right.txt"]:
                filepath = os.path.join(page_dir, txt_name)
                key = txt_name.replace(".txt", "_txt_file")
                file_rel_path = ""
                if os.path.exists(filepath):
                    file_rel_path = f"raw/{page}/{txt_name}"
                row[key] = file_rel_path
                
            m = re.match(r"page-(\d+)", page)
            page_num = str(int(m.group(1))) if m else "0"
            
            left_dp, right_dp = "", ""
            if os.path.exists(data_points_dir):
                for fname in os.listdir(data_points_dir):
                    if fname.startswith(f"page_{page_num}_left_") and fname.endswith(".json"):
                        left_dp = f"data_points/{fname}"
                    elif fname.startswith(f"page_{page_num}_right_") and fname.endswith(".json"):
                        right_dp = f"data_points/{fname}"
                        
            row["left_data_point_file"] = left_dp
            row["right_data_point_file"] = right_dp
                
            writer.writerow(row)
            
def create_markdown_and_get_title(page_dir, processed_dir, page_id):
    txt_path = os.path.join(page_dir, "fullpage.txt")
    if not os.path.exists(txt_path):
        return "Unknown Page", ""
    
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = [line.strip().replace("~", "") for line in content.split("\n") if line.strip() and "~" not in line]
    if not lines:
        title = "Unknown Page"
    else:
        title = " ".join(lines[:2])
        if len(title) > 100:
            title = title[:97] + "..."
            
    md_dir = os.path.join(processed_dir, "markdowns")
    os.makedirs(md_dir, exist_ok=True)
    md_path = os.path.join(md_dir, f"{page_id}_fullpage.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content}")
        
    return title, md_path

def generate_croissant(title, author, csv_rel_path, raw_dir, data_points_dir, processed_dir, output_jsonld):
    dataset = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "sc": "https://schema.org/"
        },
        "@type": "cr:Dataset",
        "name": title,
        "description": "Dataset extracted from map images/PDF pages.",
        "creator": {
            "@type": "Organization",
            "name": author
        },
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "pages_data_csv",
                "name": "pages_data.csv",
                "contentUrl": csv_rel_path,
                "encodingFormat": "text/csv"
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "page_data",
                "name": "page_data",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "page_data/page_num",
                        "name": "page_num",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "pages_data_csv"},
                            "extract": {"column": "page_num"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "page_data/fullpage_txt_file",
                        "name": "fullpage_txt_file",
                        "dataType": "sc:URL",
                        "source": {
                            "fileObject": {"@id": "pages_data_csv"},
                            "extract": {"column": "fullpage_txt_file"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "page_data/legend_txt_file",
                        "name": "legend_txt_file",
                        "dataType": "sc:URL",
                        "source": {
                            "fileObject": {"@id": "pages_data_csv"},
                            "extract": {"column": "legend_txt_file"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "page_data/left_txt_file",
                        "name": "left_txt_file",
                        "dataType": "sc:URL",
                        "source": {
                            "fileObject": {"@id": "pages_data_csv"},
                            "extract": {"column": "left_txt_file"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "page_data/right_txt_file",
                        "name": "right_txt_file",
                        "dataType": "sc:URL",
                        "source": {
                            "fileObject": {"@id": "pages_data_csv"},
                            "extract": {"column": "right_txt_file"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "page_data/left_data_point_file",
                        "name": "left_data_point_file",
                        "dataType": "sc:URL",
                        "source": {
                            "fileObject": {"@id": "pages_data_csv"},
                            "extract": {"column": "left_data_point_file"}
                        }
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "page_data/right_data_point_file",
                        "name": "right_data_point_file",
                        "dataType": "sc:URL",
                        "source": {
                            "fileObject": {"@id": "pages_data_csv"},
                            "extract": {"column": "right_data_point_file"}
                        }
                    }
                ]
            }
        ]
    }
    
    # Populate distribution with files grouped by page using FileSet
    pages = sorted([d for d in os.listdir(raw_dir) if d.startswith("page-") and d != "page-01"])
    for page in pages:
        m = re.match(r"page-(\d+)", page)
        page_num = str(int(m.group(1))) if m else "0"
        
        page_dir = os.path.join(raw_dir, page)
        page_title, md_path = create_markdown_and_get_title(page_dir, processed_dir, page)
        
        fileset_id = f"page_{page_num}_files"
        dataset["distribution"].append({
            "@type": "cr:FileSet",
            "@id": fileset_id,
            "name": page_title
        })
        
        # Add generated md file
        if md_path:
            md_rel_path = os.path.relpath(md_path, start=os.path.dirname(output_jsonld))
            md_id = md_rel_path.replace("/", "_").replace(".", "_")
            dataset["distribution"].append({
                "@type": "cr:FileObject",
                "@id": md_id,
                "name": f"{page_title} - Markdown",
                "contentUrl": md_rel_path,
                "encodingFormat": "text/markdown",
                "containedIn": {"@id": fileset_id}
            })
        
        # Text files for this page
        for txt_name in ["fullpage.txt", "legend.txt", "left.txt", "right.txt"]:
            if os.path.exists(os.path.join(page_dir, txt_name)):
                file_rel_path = f"raw/{page}/{txt_name}"
                file_id = file_rel_path.replace("/", "_").replace(".", "_")
                dataset["distribution"].append({
                    "@type": "cr:FileObject",
                    "@id": file_id,
                    "name": f"{page_title} - {txt_name}",
                    "contentUrl": file_rel_path,
                    "encodingFormat": "text/plain",
                    "containedIn": {"@id": fileset_id}
                })
                
        # JSON files for this page
        if os.path.exists(data_points_dir):
            for fname in sorted(os.listdir(data_points_dir)):
                if fname.startswith(f"page_{page_num}_left_") or fname.startswith(f"page_{page_num}_right_"):
                    if fname.endswith(".json"):
                        file_rel_path = f"data_points/{fname}"
                        file_id = file_rel_path.replace("/", "_").replace(".", "_")
                        side = "Left" if "_left_" in fname else "Right"
                        dataset["distribution"].append({
                            "@type": "cr:FileObject",
                            "@id": file_id,
                            "name": f"{page_title} - {side} Data",
                            "contentUrl": file_rel_path,
                            "encodingFormat": "application/json",
                            "containedIn": {"@id": fileset_id}
                        })
                        
    with open(output_jsonld, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "raw")
    data_points_dir = os.path.join(base_dir, "data_points")
    processed_dir = os.path.join(base_dir, "processed_data")
    
    csv_file = os.path.join(processed_dir, "pages_data.csv")
    jsonld_file = os.path.join(base_dir, "dataset_croissant.jsonld")
    
    print("Extracting metadata from page-01...")
    title, author = extract_metadata_from_page01(raw_dir)
    print(f"Title: {title}")
    print(f"Author: {author}")
    
    print(f"\nProcessing data points to {csv_file}...")
    process_data_points(raw_dir, data_points_dir, csv_file)
    
    print(f"\nGenerating Croissant JSON-LD to {jsonld_file}...")
    generate_croissant(title, author, "processed_data/pages_data.csv", raw_dir, data_points_dir, processed_dir, jsonld_file)
    print("Done!")

if __name__ == "__main__":
    main()
