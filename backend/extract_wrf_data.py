import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
import os

def extract_wrf_mandatory():
    url = "https://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # The first table usually holds mandatory fields
    tables = soup.find_all('table')
    mandatory_table = tables[0]
    
    data = []
    rows = mandatory_table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if not cols: continue
        
        # Link is in the first column usually
        link_tag = cols[0].find('a')
        if link_tag:
            name = link_tag.text.strip()
            href = link_tag.get('href')
            if href:
                if not href.startswith('http'):
                    href = "https://www2.mmm.ucar.edu/wrf/src/wps_files/" + href.split('/')[-1]
                data.append({"title": name, "download_url": href, "description": f"Mandatory field {name} for WPS/WRF."})
    
    df = pd.DataFrame(data)
    df.to_csv("cache/wrf_mandatory_inventory.csv", index=False)
    print("Extracted WRF mandatory fields to cache/wrf_mandatory_inventory.csv")

if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    extract_wrf_mandatory()
