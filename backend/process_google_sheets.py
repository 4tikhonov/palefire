import sys
import pandas as pd
import re
import os
import json

def extract_sheet_info(url):
    # Pattern for spreadsheet ID
    id_match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
    if not id_match:
        return None, None
    
    spreadsheet_id = id_match.group(1)
    
    # Pattern for gid (sheet index)
    gid_match = re.search(r'gid=([0-9]+)', url)
    gid = gid_match.group(1) if gid_match else '0'
    
    return spreadsheet_id, gid

def process_sheets(url, output_dir='cache'):
    spreadsheet_id, gid = extract_sheet_info(url)
    if not spreadsheet_id:
        print(f"Error: Invalid Google Spreadsheet URL: {url}")
        sys.exit(1)
        
    export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
    
    try:
        # Load without headers first to find the best header row
        df_raw = pd.read_csv(export_url, header=None)
        
        header_row = 0
        for i in range(min(5, len(df_raw))):
            row_values = df_raw.iloc[i].astype(str).tolist()
            if any(h in row_values for h in ['Title', 'Field Label', 'Field Label Order']):
                header_row = i
                break
        
        # Re-load with the correct header
        df = pd.read_csv(export_url, skiprows=header_row)
        
        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Save to local cache
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = f"gsheet_{spreadsheet_id}_{gid}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        # Provide summary
        print(f"Successfully loaded Google Spreadsheet: {spreadsheet_id} (gid={gid})")
        print(f"Saved to: {filepath}")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head().to_markdown(index=False))
        
        # Also generate JSON-LD metadata
        jsonld_path = filepath.replace('.csv', '.jsonld')
        metadata = {
            "@context": "https://schema.org/",
            "@type": "Dataset",
            "name": f"Google Spreadsheet Export: {spreadsheet_id}",
            "distribution": {
                "@type": "DataDownload",
                "contentUrl": url,
                "encodingFormat": "text/csv"
            },
            "variableMeasured": [
                {"@type": "PropertyValue", "name": col} for col in df.columns
            ]
        }
        with open(jsonld_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"Error processing Google Spreadsheet: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 process_google_sheets.py <GOOGLE_SHEETS_URL>")
        sys.exit(1)
        
    url = sys.argv[1]
    process_sheets(url)
