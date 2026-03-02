import sys
import csv
import json
import os

def convert_csv_to_jsonld(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
        
    json_path = csv_path.replace('.csv', '.jsonld')
    
    variables = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            variables.append({
                "@type": "schema:PropertyValue",
                "schema:name": row.get("Name", "").strip(),
                "schema:value": row.get("Value", "").strip(),
                "schema:unitText": row.get("Unit", "").strip(),
                "schema:description": row.get("Context", "").strip()
            })
            
    json_ld = {
        "@context": {"schema": "https://schema.org/"},
        "@type": "schema:Dataset",
        "schema:name": "CDIF Variable Extraction",
        "schema:variableMeasured": variables
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_ld, f, indent=2)
        
    print(f"Successfully exported {len(variables)} variables to {json_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 export_cdif.py <path_to_csv>")
        sys.exit(1)
        
    convert_csv_to_jsonld(sys.argv[1])
