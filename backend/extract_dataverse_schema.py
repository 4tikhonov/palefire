import pandas as pd
import json
import os

def extract_dataverse_schema(url, sheet_name):
    spreadsheet_id = url.split('/d/')[1].split('/')[0]
    xlsx_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=xlsx"
    
    try:
        # Load the sheet
        df = pd.read_excel(xlsx_url, sheet_name=sheet_name, header=None)
        
        # Row 2 (index 1) contains headers
        headers = df.iloc[1].tolist()
        
        # Find index of 'Field Label' and 'System Name'
        label_idx = -1
        name_idx = -1
        for i, h in enumerate(headers):
            if str(h).strip() == 'Field Label':
                label_idx = i
            elif str(h).strip() == 'System Name':
                name_idx = i
        
        if label_idx == -1 or name_idx == -1:
            # Fallback if names don't match exactly
            label_idx = 1
            name_idx = 2
            
        # Data starts from Row 3 (index 2)
        fields = []
        for i in range(2, len(df)):
            row = df.iloc[i]
            label = str(row[label_idx]).strip()
            name = str(row[name_idx]).strip()
            
            if name != 'nan' and name != '':
                fields.append({
                    "label": label,
                    "name": name,
                    "description": str(row[3]).strip() if len(row) > 3 else ""
                })
        
        # Save to cache
        cache_dir = 'cache/dataverse'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        output_file = os.path.join(cache_dir, 'citation_metadata.json')
        with open(output_file, 'w') as f:
            json.dump(fields, f, indent=2)
            
        print(f"Successfully extracted {len(fields)} fields to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    url = "https://docs.google.com/spreadsheets/d/1VAhZ83hKURX_4T-bOkn7XCb9h62Gr2gcJI1fWe4Rl4c/edit?gid=0#gid=0"
    extract_dataverse_schema(url, "Citation metadata")
