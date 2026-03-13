
import os
import sys
import json
import re
import argparse
import requests
import glob

# Ensure we can import converter.py if run from another directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from converter import croissant_to_dataverse

def translate_term(term, target_lang, context=None, model="gpt-oss:latest", ollama_host=None):
    """
    Translates a term using Ollama.
    """
    if not ollama_host:
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    # Handle comma-separated hosts
    if ',' in ollama_host:
        ollama_host = ollama_host.split(',')[0].strip()

    context_str = f" Context: {context}" if context else ""
    # STRONGER prompt to force JSON
    prompt = f"You are a translation engine. Output ONLY valid JSON with key 'translation'. Translate '{term}' to {target_lang}.{context_str} JSON format: {{\"translation\": \"<translated_term>\"}}"
    
    try:
        url = f"{ollama_host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1  # Lower temperature for deterministic output
            }
        }
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            response_text = data.get("response", "")
            
            # PARSING LOGIC
            json_data = None
            
            # 1. Try strict JSON parse
            try:
                json_data = json.loads(response_text)
            except:
                pass
            
            # 2. Try Regex Extraction of JSON Object { ... }
            if not json_data:
                match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if match:
                    try:
                        json_data = json.loads(match.group(1))
                    except:
                        pass
            
            # 3. If we have JSON data, try to navigate it
            if json_data:
                # { "fr": { "translation": "..." } }
                if target_lang in json_data:
                    val = json_data[target_lang]
                    if isinstance(val, dict):
                        return val.get("translation", term)
                    return val # if it's just the string
                
                # Check case-insensitive key
                for k in json_data.keys():
                    if k.lower() == target_lang.lower():
                        val = json_data[k]
                        if isinstance(val, dict):
                             return val.get("translation", term)
                        return val

                # Maybe it is just { "translation": "..." } directly?
                if "translation" in json_data:
                    return json_data["translation"]
                    
            # 4. Fallback Regex for Hallucinations like: 'We need to translate ": "Culmen Length (mm)" }'
            # Look for colon followed by quotes/string at the end of string or before closing brace
            # Pattern: ": "VALUE"
            m_hallucination = re.search(r':\s*["\']([^"\']+)["\']\s*\}', response_text)
            if m_hallucination:
                 return m_hallucination.group(1)

            # Pattern: "translation": "VALUE"
            m_trans = re.search(r'["\']translation["\']\s*:\s*["\'](.*?)["\']', response_text)
            if m_trans:
                return m_trans.group(1)

            print(f"Warning: Could not parse JSON for {term}. Raw: {response_text[:50]}...")
            return term
        else:
            print(f"Translation API error: {resp.status_code}")
            return term
    except Exception as e:
        print(f"Translation error: {e}")
        return term

def ingest_dataverse(input_dir, dataverse_url, dataverse_id, api_token, input_file=None, config=None, translate_langs=None, translation_model="gpt-oss:latest"):
    """
    Ingests Croissant metadata files into Dataverse.
    """
    if input_file:
        if not os.path.exists(input_file):
             print(f"Error: Input file {input_file} does not exist.")
             sys.exit(1)
        files = [input_file]
    else:
        if not os.path.isdir(input_dir):
            print(f"Error: Input directory {input_dir} does not exist.")
            sys.exit(1)
        files = glob.glob(os.path.join(input_dir, "croissant_*.json"))
        
    # Normalize URL
    dataverse_url = dataverse_url.rstrip('/')
    
    # API Endpoint
    api_endpoint = f"{dataverse_url}/api/dataverses/{dataverse_id}/datasets"
    
    headers = {
        "X-Dataverse-key": api_token,
        "Content-Type": "application/ld+json"
    }

    if not files:
        print(f"No metadata files found.")
        return

    print(f"Found {len(files)} metadata file(s) to ingest...")

    for file_path in files:
        print(f"\nProcessing {file_path}...")
        try:
            with open(file_path, 'r') as f:
                croissant_data = json.load(f)
            
            # Fetch Wikidata Link from SparqlMuse EARLY to use in metadata
            sparqlmuse_api_url = config.get("sparqlmuse_api") if config else None
            wikilink_url = None
            wikilink_json = None
            term_name = croissant_data.get("name") # Define term_name here for use later
            
            if sparqlmuse_api_url:
                description = croissant_data.get("description", "")
                
                if term_name:
                    import urllib.parse
                    params = {
                        "term": term_name,
                        "context": description[:500],
                        "language": "en",
                        "format": "json"
                    }
                    query_string = urllib.parse.urlencode(params)
                    full_url = f"{sparqlmuse_api_url}?{query_string}"
                    
                    try:
                        print(f"\nFetching Wikidata link from SparqlMuse: {full_url}")
                        wikilink_resp = requests.get(full_url)
                        
                        if wikilink_resp.status_code == 200:
                            wikilink_str = wikilink_resp.text
                            try:
                                wikilink_data = json.loads(wikilink_str)
                                wikilink_url = wikilink_data.get("url")
                                wikilink_json = wikilink_str # Keep for file upload later
                                if wikilink_url:
                                    print(f"Propagating Wikidata URI: {wikilink_url}")
                            except ValueError:
                                print("Invalid JSON from SparqlMuse")
                    except Exception as e:
                        print(f"Error handling SparqlMuse fetch: {e}")

            # Update config with fetched URI if available
            current_config = config.copy() if config else {}
            if wikilink_url:
                current_config['keyword_vocabulary_uri'] = wikilink_url

            # Convert
            dv_data = croissant_to_dataverse(croissant_data, current_config)
            
            # Search for existing dataset by title to avoid duplicates
            dataset_title = dv_data.get('http://purl.org/dc/terms/title', 'Untitled')
            print(f"Searching for existing dataset with title: '{dataset_title}'...")
            
            search_api = f"{dataverse_url}/api/search"
            search_params = {
                "q": f'title:"{dataset_title}"',
                "type": "dataset",
                "per_page": 1
            }
            
            persistent_id = None
            try:
                search_resp = requests.get(search_api, params=search_params, headers={"X-Dataverse-key": api_token})
                if search_resp.status_code == 200:
                    results = search_resp.json().get('data', {}).get('items', [])
                    for item in results:
                        if item.get('name') == dataset_title:
                            print(f"Found existing dataset: {item.get('global_id')} ({item.get('name')})")
                            persistent_id = item.get('global_id')
                            break
            except Exception as e:
                 print(f"Search exception: {e}")

            if persistent_id:
                print(f"Dataset exists ({persistent_id}). Enriching keywords...")
                # Prepare native keywords from Croissant fields
                native_keywords = []
                try:
                    record_sets = croissant_data.get('recordSet', [])
                    for rs in record_sets:
                        fields = rs.get('field', [])
                        for f in fields:
                            name = f.get('name')
                            desc = f.get('description')
                            if name:
                                # Original keyword
                                entry = {"keywordValue": {"typeName": "keywordValue", "value": name}}
                                if desc:
                                    entry["keywordVocabulary"] = {"typeName": "keywordVocabulary", "value": desc}
                                native_keywords.append(entry)

                                # Translate if requested
                                if translate_langs:
                                    for lang in translate_langs:
                                        lang = lang.strip()
                                        if not lang: continue

                                        print(f"Translating '{name}' to {lang} (Context: {desc[:50] if desc else 'None'})...")
                                        translated_name = translate_term(name, lang, context=desc, model=translation_model)
                                        if translated_name and translated_name.lower() != name.lower():
                                            trans_entry = {"keywordValue": {"typeName": "keywordValue", "value": translated_name}}
                                            trans_entry["keywordVocabulary"] = {"typeName": "keywordVocabulary", "value": f"Translation ({lang})"}
                                            native_keywords.append(trans_entry)

                except Exception as e:
                    print(f"Error preparing keywords: {e}")

                if native_keywords:
                    # Fetch-Merge-Push Pattern using editMetadata
                    api_url_draft = f"{dataverse_url}/api/datasets/:persistentId/versions/:draft"
                    params = {"persistentId": persistent_id}
                    headers = {"X-Dataverse-key": api_token}
                    
                    try:
                        get_resp = requests.get(api_url_draft, params=params, headers=headers)
                        existing_keywords = []
                        if get_resp.status_code == 200:
                            existing_data = get_resp.json().get('data', {})
                            existing_blocks = existing_data.get('metadataBlocks', {})
                            if 'citation' in existing_blocks:
                                citation_fields = existing_blocks['citation']['fields']
                                kw_field = next((f for f in citation_fields if f['typeName'] == 'keyword'), None)
                                if kw_field:
                                    existing_keywords = kw_field['value']
                        
                        # Merge logic
                        existing_vals = {v['keywordValue']['value'] for v in existing_keywords}
                        for new_k in native_keywords:
                            if new_k['keywordValue']['value'] not in existing_vals:
                                existing_keywords.append(new_k)
                        
                        # Prepare editMetadata payload
                        edit_payload = {
                            "fields": [
                                {
                                    "typeName": "keyword",
                                    "multiple": True,
                                    "typeClass": "compound",
                                    "value": existing_keywords
                                }
                            ]
                        }
                        
                        api_url_edit = f"{dataverse_url}/api/datasets/:persistentId/editMetadata"
                        params_edit = {"persistentId": persistent_id, "replace": "true"}
                        
                        print("Sending metadata update request (editMetadata)...")
                        put_resp = requests.put(api_url_edit, params=params_edit, data=json.dumps(edit_payload), headers={"X-Dataverse-key": api_token, "Content-Type": "application/json"})
                        
                        if put_resp.status_code == 200:
                            print(f"Successfully updated metadata for {persistent_id}")
                        else:
                            print(f"Metadata update failed: {put_resp.status_code} {put_resp.text}")
                    except Exception as e:
                         print(f"Update exception: {e}")

            else:
                # Upload Metadata to Create Dataset
                print(f"Creating dataset at {api_endpoint}...")
                response = requests.post(api_endpoint, data=json.dumps(dv_data), headers=headers)
                
                if response.status_code == 201:
                    persistent_id = response.json().get('data', {}).get('persistentId')
                    print(f"Success! Created dataset: {persistent_id}")
                else:
                    print(f"Failed to create dataset: {response.status_code} {response.text}")
                    continue

            # Check if we have an ID to proceed with file uploads
            if persistent_id:
                
                # Initialize API
                from pyDataverse.api import NativeApi
                api = NativeApi(dataverse_url, api_token)
                
                # 1. Upload SKOS (Enrichment)
                cite_as = croissant_data.get("citeAs", "")
                hips_match = re.search(r'(MH\d+)', cite_as, re.IGNORECASE)
                if hips_match:
                    hips_code = hips_match.group(1).upper()
                    print(f"\nFound HIPS Code: {hips_code}")
                    hips_api_url = f"https://www.undrr.org/api/terms/hips/{hips_code}"
                    
                    try:
                        print(f"Fetching SKOS from {hips_api_url}...")
                        skos_resp = requests.get(hips_api_url)
                        
                        if skos_resp.status_code == 200:
                            safe_name = croissant_data.get("name", "hazard").lower().replace(" ", "_").replace("/", "_")
                            skos_filename = f"skos_{safe_name}.json"
                            
                            with open(skos_filename, 'w') as skos_f:
                                skos_f.write(skos_resp.text)
                                
                            print(f"Uploading {skos_filename}...")
                            skos_upload = api.upload_datafile(persistent_id, skos_filename, json.dumps({"description": f"SKOS data for {hips_code}", "categories": ["Data"]}))
                            
                            if skos_upload.status_code == 200:
                                print(f"SKOS upload successful.")
                            else:
                                print(f"SKOS upload failed: {skos_upload.status_code}")

                            if os.path.exists(skos_filename):
                                os.remove(skos_filename)
                    except Exception as e:
                        print(f"Error handling SKOS: {e}")

                # 2. Upload Wikilink (Enrichment)
                if wikilink_json and term_name:
                    safe_name = croissant_data.get("name", "hazard").lower().replace(" ", "_").replace("/", "_")
                    wikilink_filename = f"wikilink_{safe_name}.json"
                    try:
                        with open(wikilink_filename, 'w') as wl_f:
                            wl_f.write(wikilink_json)
                            
                        print(f"Uploading {wikilink_filename}...")
                        wl_upload = api.upload_datafile(persistent_id, wikilink_filename, json.dumps({"description": f"Wikidata Link for {term_name}", "categories": ["Data"]}))
                        
                        if wl_upload.status_code == 200:
                            print(f"Wikilink upload successful.")
                        else:
                            print(f"Wikilink upload failed: {wl_upload.status_code}")
                            
                        if os.path.exists(wikilink_filename):
                            os.remove(wikilink_filename)
                    except Exception as e:
                        print(f"Error uploading Wikilink file: {e}")

                # 3. Upload Translations CSV and Capture ID
                translations_url = None
                translations_csv_path = config.get("translations_csv") if config else None
                if translations_csv_path and os.path.exists(translations_csv_path):
                    if term_name:
                        import csv
                        safe_name = croissant_data.get("name", "hazard").lower().replace(" ", "_").replace("/", "_")
                        translations_out_file = f"translations_{safe_name}.csv"
                        found_translations = False
                        
                        try:
                            with open(translations_csv_path, 'r', encoding='utf-8') as csv_in, \
                                 open(translations_out_file, 'w', encoding='utf-8', newline='') as csv_out:
                                
                                reader = csv.DictReader(csv_in)
                                writer = csv.DictWriter(csv_out, fieldnames=reader.fieldnames)
                                writer.writeheader()
                                
                                for row in reader:
                                    if row.get('term') == term_name:
                                        writer.writerow(row)
                                        found_translations = True
                            
                            if found_translations:
                                print(f"Found translations for '{term_name}', uploading {translations_out_file}...")
                                trans_upload = api.upload_datafile(persistent_id, translations_out_file, json.dumps({"description": f"Translations for {term_name}", "categories": ["Data"]}))
                                
                                if trans_upload.status_code == 200:
                                    print(f"Translations upload successful.")
                                    try:
                                        file_data = trans_upload.json()['data']['files'][0]['dataFile']
                                        file_id = file_data['id']
                                        translations_url = f"{dataverse_url}/api/access/datafile/{file_id}"
                                        print(f"Captured Translations URL: {translations_url}")
                                    except Exception as e:
                                        print(f"Could not extract file ID: {e}")
                                else:
                                    print(f"Translations upload failed: {trans_upload.status_code} {trans_upload.text}")
                            else:
                                print(f"No translations found for '{term_name}' in {translations_csv_path}")

                            # Cleanup
                            if os.path.exists(translations_out_file):
                                os.remove(translations_out_file)

                        except Exception as e:
                            print(f"Error processing translations CSV: {e}")

                # 4. Modify Croissant Data & Upload
                if translations_url:
                    print("Updating Croissant metadata with translations URL...")
                    # Find FileObject named 'multilingual_cv_data'
                    # It might be in distribution list
                    distributions = croissant_data.get('distribution', [])
                    updated = False
                    for dist in distributions:
                        if dist.get('@type') == 'cr:FileObject' and dist.get('name') == 'multilingual_cv_data':
                            dist['contentUrl'] = translations_url
                            dist['sha256'] = "" # Reset hash as content is now remote/dynamic
                            updated = True
                            print("Updated contentUrl for multilingual_cv_data")
                    
                    if not updated:
                         print("Warning: Could not find 'multilingual_cv_data' FileObject to update.")
                
                # Save modified JSON
                safe_name = croissant_data.get("name", "hazard").lower().replace(" ", "_").replace("/", "_")
                final_croissant_file = f"croissant_{safe_name}_final.json"
                with open(final_croissant_file, 'w') as f:
                    json.dump(croissant_data, f, indent=2)
                
                print(f"Uploading final Croissant file {final_croissant_file}...")
                try:
                    df = api.upload_datafile(persistent_id, final_croissant_file, json.dumps({"description": "Croissant Metadata", "categories": ["Metadata"]}))
                    if df.status_code == 200:
                        print(f"Croissant file upload successful.")
                    else:
                        print(f"Croissant file upload failed: {df.status_code} {df.text}")
                except Exception as e:
                    print(f"Croissant upload exception: {e}")
                    
                if os.path.exists(final_croissant_file):
                    os.remove(final_croissant_file)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Croissant metadata into Dataverse.")
    parser.add_argument("--input-dir", default="output", help="Directory containing croissant_*.json files")
    parser.add_argument("--file", help="Path to a single Croissant JSON file to ingest (overrides --input-dir)")
    parser.add_argument("--dataverse-url", default=os.environ.get("DATAVERSE_URL"), help="Dataverse base URL")
    parser.add_argument("--dataverse-id", default=os.environ.get("DATAVERSE_ID"), help="Target Dataverse collection alias or ID")
    parser.add_argument("--api-token", default=os.environ.get("DATAVERSE_API_TOKEN"), help="Dataverse API Token")
    parser.add_argument("--config", default="dataverse/ingest.conf", help="Path to configuration file")
    parser.add_argument("--translate-keywords", help="Comma-separated target languages (e.g., 'fr,es')")
    parser.add_argument("--translation-model", default="gpt-oss:latest", help="Ollama model to use")

    args = parser.parse_args()

    # Add default config path within the skill directory
    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(skill_dir, "scripts", "ingest.conf")
    
    config_path = args.config if args.config != "dataverse/ingest.conf" else default_config

    if not args.dataverse_url or not args.dataverse_id or not args.api_token:
        print("Error: Missing Dataverse configuration. Please provide --dataverse-url, --dataverse-id, and --api-token (or set env vars).")
        sys.exit(1)

    config_data = None
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default values.")

    langs = args.translate_keywords.split(',') if args.translate_keywords else None
    ingest_dataverse(args.input_dir, args.dataverse_url, args.dataverse_id, args.api_token, args.file, config_data, langs, args.translation_model)
