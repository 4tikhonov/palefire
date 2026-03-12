import json
import sys
import os

def show_fields():
    file_path = 'cache/dataverse/citation_metadata.json'
    if not os.path.exists(file_path):
        print("Error: Metadata schema not found. Please run individual extraction first.")
        return

    with open(file_path, 'r') as f:
        fields = json.load(f)
    
    print("| Field Label | System Name | Description |")
    print("| :--- | :--- | :--- |")
    for f in fields:
        # Avoid showing empty rows or metadata blocks
        if f['name'].startswith('#') or f['label'] == 'nan':
            continue
        print(f"| {f['label']} | {f['name']} | {f['description']} |")

if __name__ == "__main__":
    show_fields()
