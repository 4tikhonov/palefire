import sys
import json
import urllib.request
import urllib.error
import re
import os

def resolve_did(did):
    url = f"https://dev.uniresolver.io/1.0/identifiers/{did}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                
                # Extracting fields as requested by user: url, name, description
                # Uniresolver returns a DID Document in a specific format
                # The actual metadata is typically inside 'didDocument'
                did_doc = data.get('didDocument', {})
                metadata = data.get('didDocumentMetadata', {})
                
                # User wants url, name, and description.
                # These might be in different places depending on the DID method.
                # We'll try common places.
                
                name = did_doc.get('name', 'N/A')
                description = did_doc.get('description', 'N/A')
                url_resource = did_doc.get('url', 'N/A')
                
                # If they are not at the root, check 'service' entries
                services = did_doc.get('service', [])
                if isinstance(services, list):
                    for s in services:
                        if s.get('type') == 'LinkedDomains' or s.get('type') == 'MetadataService':
                            if url_resource == 'N/A':
                                url_resource = s.get('serviceEndpoint', 'N/A')
                
                return {
                    "did": did,
                    "name": name,
                    "description": description,
                    "url": url_resource,
                    "raw_response": data # In case the LLM needs more details
                }
            else:
                return {"error": f"Failed to resolve DID. Status code: {response.status}"}
    except Exception as e:
        return {"error": str(e)}

def extract_did_from_odrl(content):
    # Regex to find did:oyd:... or did:...: in ODRL text/json
    did_pattern = r'did:[a-zA-Z0-9:]+'
    match = re.search(did_pattern, content)
    if match:
        return match.group(0)
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 resolve_odrl_did.py <ODRL_FILE_PATH_OR_URL_OR_CONTENT>")
        sys.exit(1)

    input_data = sys.argv[1]
    content = ""

    if input_data.startswith("http"):
        # Reading from a URL (e.g., GitHub raw URL)
        try:
            # If it's a normal github.com URL, we try to convert it to raw.githubusercontent.com
            if "github.com" in input_data and "/blob/" in input_data:
                input_data = input_data.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            
            with urllib.request.urlopen(input_data) as response:
                content = response.read().decode('utf-8')
        except Exception as e:
            print(json.dumps({"error": f"Failed to fetch content from URL: {str(e)}"}))
            sys.exit(1)
    elif os.path.exists(input_data):
        # Reading from a local file
        try:
            with open(input_data, 'r') as f:
                content = f.read()
        except Exception as e:
            print(json.dumps({"error": f"Failed to read file: {str(e)}"}))
            sys.exit(1)
    else:
        # Assuming input is the raw content
        content = input_data

    did = extract_did_from_odrl(content)
    if not did:
        print(json.dumps({"error": "No DID found in the ODRL content."}))
        sys.exit(1)

    result = resolve_did(did)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
