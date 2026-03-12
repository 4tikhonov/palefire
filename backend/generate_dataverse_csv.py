import pandas as pd
import json
import os
import sys

def generate_deposit_csvs(source_csv, mapping_json, output_dir='cache/dataverse/deposits'):
    """
    source_csv: Path to the CSV file (from Google Sheets)
    mapping_json: JSON string or path to JSON file containing mapping:
                  {"Original Column Name": "Dataverse System Name"}
    """
    if not os.path.exists(source_csv):
        print(f"Error: Source CSV not found: {source_csv}")
        return

    if os.path.exists(mapping_json):
        with open(mapping_json, 'r') as f:
            mapping = json.load(f)
    else:
        try:
            mapping = json.loads(mapping_json)
        except:
            print(f"Error: Invalid mapping JSON: {mapping_json}")
            return

    try:
        df = pd.read_csv(source_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter columns that are in the mapping
    available_cols = [col for col in df.columns if col in mapping]
    if not available_cols:
        print(f"Error: None of the columns in the mapping match the source CSV ({df.columns.tolist()})")
        return

    generated_files = []
    for i, row in df.iterrows():
        # Create a dictionary for this row using mapped names
        deposit_data = {}
        for col in available_cols:
            dv_name = mapping[col]
            deposit_data[dv_name] = [row[col]] # pandas needs a list or index to create df
            
        deposit_df = pd.DataFrame(deposit_data)
        
        # Generate filename (use index + title if available)
        title_val = ""
        for k, v in mapping.items():
            if v == 'title':
                title_val = str(row[k]).replace(' ', '_').lower()[:20]
                break
        
        filename = f"deposit_{i}_{title_val}.csv" if title_val else f"deposit_{i}.csv"
        filepath = os.path.join(output_dir, filename)
        
        deposit_df.to_csv(filepath, index=False)
        generated_files.append(filepath)

    print(f"Successfully generated {len(generated_files)} deposit files in {output_dir}")
    for f in generated_files[:5]: # Show first 5
        print(f"- {f}")
    if len(generated_files) > 5:
        print(f"... and {len(generated_files)-5} more.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 generate_dataverse_csv.py <source_csv> <mapping_json_or_file>")
        sys.exit(1)
        
    generate_deposit_csvs(sys.argv[1], sys.argv[2])
