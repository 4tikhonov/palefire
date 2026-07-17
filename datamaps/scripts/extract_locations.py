import json
import urllib.parse
import os

def create_jsonld():
    input_file = "data/pdf_full_dataset.json"
    output_file = "data/locations.jsonld"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]
        
    locations_dict = {}
    
    for row in data:
        loc_name = row["location"]
        if loc_name not in locations_dict:
            # Create URL safe ID
            safe_id = urllib.parse.quote(loc_name.replace(" ", "_"))
            locations_dict[loc_name] = {
                "@id": f"http://maps2ai.org/locations/{safe_id}",
                "@type": "Place",
                "name": loc_name,
                "latitude": row["lat"],
                "longitude": row["lon"]
            }
            
    jsonld_doc = {
        "@context": {
            "schema": "http://schema.org/",
            "name": "schema:name",
            "latitude": "schema:latitude",
            "longitude": "schema:longitude",
            "Place": "schema:Place"
        },
        "@graph": list(locations_dict.values())
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(jsonld_doc, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully extracted {len(locations_dict)} unique locations to {output_file}")

if __name__ == "__main__":
    create_jsonld()
