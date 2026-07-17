import argparse
import glob
import json
import os
import geopandas as gpd

from .extractor import process_image
from .compiler import split_and_convert_to_jsonld, compile_report


def main():
    parser = argparse.ArgumentParser(description="datamaps pipeline")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("extract", help="Extract map data from PDF images")
    subparsers.add_parser("ocr", help="Run OCR on extracted maps to get titles")
    subparsers.add_parser("split", help="Split dataset into semantic JSON-LD files")
    subparsers.add_parser("compile", help="Compile all data points into single CSV report")
    translate_parser = subparsers.add_parser("translate", help="Translate Spanish content to English")
    translate_parser.add_argument("--titles", default="report/titles.json",
                                  help="Path to titles JSON (default: report/titles.json)")
    translate_parser.add_argument("--data-dir", default="data_points",
                                  help="Path to data points directory (default: data_points)")
    translate_parser.add_argument("--raw-dir", default="raw",
                                  help="Path to raw OCR directory (default: raw)")
    translate_parser.add_argument("--output-dir", default="translations",
                                  help="Output directory for translations (default: translations)")

    args = parser.parse_args()

    if args.command == "extract":
        print("Extracting...")
        geojson_path = "honduras_departments.geojson"
        gdf = gpd.read_file(geojson_path)
        geo_bounds = gdf.total_bounds

        files = sorted(glob.glob("report/full_page-*.png"))
        dataset = []
        for f in files:
            dataset.extend(process_image(f, gdf, geo_bounds, "report"))

        os.makedirs("data", exist_ok=True)
        with open("data/pdf_full_dataset.json", "w") as f:
            json.dump({"data": dataset}, f, indent=2)

    elif args.command == "ocr":
        print("Running OCR on extracted maps...")
        from .ocr import run_ocr
        run_ocr("data/pdf_full_dataset.json", "report/titles.json", raw_dir="raw")

    elif args.command == "split":
        print("Splitting dataset...")
        split_and_convert_to_jsonld(
            "data/pdf_full_dataset.json",
            "report/titles.json",
            "data_points",
            raw_dir="raw",
        )

    elif args.command == "compile":
        print("Compiling report...")
        compile_report("data_points", "report/final_heatmap_report.csv")

    elif args.command == "translate":
        print("Translating Spanish content to English...")
        from .translator import run_translate
        run_translate(
            titles_path=args.titles,
            data_dir=args.data_dir,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
