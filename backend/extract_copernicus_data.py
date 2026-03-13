import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse
import os
import urllib.parse

def extract_copernicus_search(query, content_type="marine_indicator"):
    """
    Extracts dataset titles and URLs from Copernicus Marine search results.
    """
    base_url = "https://marine.copernicus.eu/search"
    params = {
        "search_api_fulltext": query,
        "sort_bef_combine": "relevance_DESC"
    }
    if content_type:
        # f[0]=content_type:marine_indicator
        params["f[0]"] = f"content_type:{content_type}"
    
    encoded_params = urllib.parse.urlencode(params)
    search_url = f"{base_url}?{encoded_params}"
    
    print(f"Searching Copernicus Marine: {search_url}")
    
    # Use headers to avoid being blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        datasets = []
        # The search cards are typically found in articles or div with specific classes
        # Based on analysis, they often have class 'card-product' or similar
        cards = soup.select('.views-row') or soup.select('article')
        
        for card in cards:
            title_tag = card.select_one('h2 a') or card.select_one('h3 a') or card.select_one('.field-name-title a')
            desc_tag = card.select_one('.field-name-body') or card.select_one('.description')
            
            if title_tag:
                title = title_tag.get_text(strip=True)
                path = title_tag.get('href')
                if path.startswith('/'):
                    full_url = f"https://marine.copernicus.eu{path}"
                else:
                    full_url = path
                
                description = desc_tag.get_text(strip=True) if desc_tag else f"Copernicus Marine dataset: {title}"
                
                datasets.append({
                    "title": title,
                    "download_url": full_url,
                    "datasetContactEmail": "servicedesk.cmems@mercator-ocean.eu",
                    "dsDescriptionValue": description
                })
        
        return datasets
    
    except Exception as e:
        print(f"Error scraping Copernicus: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Extract dataset links from Copernicus Marine Search.")
    parser.add_argument("query", help="Search query (e.g., 'Salinity')")
    parser.add_argument("--type", default="marine_indicator", help="Content type filter")
    parser.add_argument("--output", default="cache/copernicus_inventory.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    datasets = extract_copernicus_search(args.query, args.type)
    
    if datasets:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df = pd.DataFrame(datasets)
        df.to_csv(args.output, index=False)
        print(f"Extracted {len(datasets)} datasets to {args.output}")
        print(df[['title', 'download_url']].head())
    else:
        print("No datasets found.")

if __name__ == "__main__":
    main()
