import sys
import json
import requests
import re
import os
from urllib.parse import urlparse

def resolve_did(did):
    url = f"https://dev.uniresolver.io/1.0/identifiers/{did}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            did_doc = data.get('didDocument', {})
            
            # Initialize with root values if they exist
            name = did_doc.get('name', 'N/A')
            description = did_doc.get('description', 'N/A')
            url_resource = did_doc.get('url', 'N/A')
            
            # OYD specific and generic service extraction
            services = did_doc.get('service', [])
            if isinstance(services, list):
                for s in services:
                    # check if metadata is inside 'payload' (typical for did:oyd)
                    payload = s.get('payload', {})
                    if isinstance(payload, dict):
                        if name == 'N/A':
                            name = payload.get('name', 'N/A')
                        if description == 'N/A':
                            description = payload.get('description', 'N/A')
                        if url_resource == 'N/A':
                            url_resource = payload.get('url', 'N/A')
                    
                    # check common service endpoints if still N/A
                    if s.get('type') in ['LinkedDomains', 'MetadataService', 'ResourceMetadata', 'Custom']:
                        if url_resource == 'N/A':
                            url_resource = s.get('serviceEndpoint', 'N/A')
                        if name == 'N/A':
                            name = s.get('name', 'N/A')
                        if description == 'N/A':
                            description = s.get('description', 'N/A')

            return {
                "did": did,
                "name": name,
                "description": description,
                "url": url_resource
            }
        else:
            return {"error": f"Failed to resolve DID. Status code: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def extract_did_from_text(content):
    did_pattern = r'did:[a-zA-Z0-9:]+'
    match = re.search(did_pattern, content)
    if match:
        return match.group(0)
    return None

def get_github_raw_url(url):
    """Converts a standard GitHub file URL to a raw URL."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

def find_odrl_files_in_github(repo_url):
    """
    Finds ODRL.md files. If the URL points to a file, returns just that file.
    If it points to a directory, lists ODRL.md files in that directory.
    """
    parsed_url = urlparse(repo_url)
    parts = parsed_url.path.strip("/").split("/")
    
    if "github.com" not in parsed_url.netloc or len(parts) < 2:
        return []

    owner = parts[0]
    repo = parts[1]
    
    # Path processing for tree (directory) or blob (file)
    path = ""
    is_file = False
    if len(parts) > 2:
        if parts[2] in ["tree", "blob"]:
            is_file = (parts[2] == "blob")
            path = "/".join(parts[4:]) if len(parts) > 4 else ""
            if is_file and not path: # Handle repo root blob if that were even a thing
                path = parts[-1]
        else:
            # Maybe it's just owner/repo/path
            path = "/".join(parts[2:])

    # If it's a direct file link to ODRL.md, return it immediately
    if repo_url.endswith("ODRL.md") and "/blob/" in repo_url:
        return [get_github_raw_url(repo_url)]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            odrl_files = []
            
            # API returns a list for directories, a dict for files
            if isinstance(data, list):
                for f in data:
                    if f['type'] == 'file' and f['name'].endswith('ODRL.md'):
                        odrl_files.append(f['download_url'])
            elif isinstance(data, dict):
                if data.get('type') == 'file' and data.get('name', '').endswith('ODRL.md'):
                    odrl_files.append(data.get('download_url'))
            
            return odrl_files
        else:
            return []
    except Exception as e:
        print(f"Error fetching repo contents: {e}", file=sys.stderr)
        return []

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 github_odrl_resolver.py <GITHUB_REPO_OR_FILE_URL>")
        sys.exit(1)

    repo_url = sys.argv[1]
    odrl_files = find_odrl_files_in_github(repo_url)
    
    if not odrl_files:
        # Final fallback - if it looks like a direct URL, just try it
        if repo_url.endswith('ODRL.md'):
            odrl_files = [get_github_raw_url(repo_url)]
        else:
            print(json.dumps({"error": "No ODRL.md files found or matched for the specified location."}))
            sys.exit(1)

    results = []
    for file_url in odrl_files:
        try:
            resp = requests.get(file_url, timeout=10)
            if resp.status_code == 200:
                content = resp.text
                did = extract_did_from_text(content)
                if did:
                    resolution = resolve_did(did)
                    resolution["source_file"] = file_url
                    results.append(resolution)
                else:
                    results.append({"error": f"No DID found in {file_url}", "source_file": file_url})
            else:
                results.append({"error": f"Failed to fetch {file_url}. Status: {resp.status_code}", "source_file": file_url})
        except Exception as e:
            results.append({"error": str(e), "source_file": file_url})

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
