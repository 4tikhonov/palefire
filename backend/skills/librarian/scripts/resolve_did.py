import sys
import json
import urllib.request
import urllib.error

def format_as_pointer(data):
    # Fix: Added comma after "url": ['url']
    urls = data.get('url', [])
    if isinstance(urls, str):
        urls = [urls]
    
    linked_urls = []
    for u in urls:
        linked_urls.append(f"\033]8;;{u}\033\\{u}\033]8;;\033\\")
        
    return {
        "kind": "NavigationPointer",
        "metadata": {
            "title": data.get('title', 'Untitled'),
            "url": linked_urls,
            "target": "secondary_frame", # Instruction for the UI
            "behavior": "overlay_on_request"
        }
    }

def resolve_did(did):
    # Auto-prepend prefix if missing
    if not did.startswith("did:"):
        if len(did) > 40: # Likely an oyd identifier
            did = f"did:oyd:{did}"
        else:
            print(f"Warning: Identifier '{did}' missing 'did:' prefix. Attempting resolution as is.", file=sys.stderr)

    url = f"https://dev.uniresolver.io/1.0/identifiers/{did}"
    try:
        req = urllib.request.Request(url)
        # Some resolvers might require specific headers
        # req.add_header('Accept', 'application/did+json')
        
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                # Note: Uniresolver returns a complex object, but this script expects 'title' and 'url'
                # If they are not in the root, this might need adjustment.
                print(json.dumps(format_as_pointer(data), indent=2))
            else:
                print(f"Error: Received status code {response.status}", file=sys.stderr)
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}", file=sys.stderr)
        if e.code == 400:
            print("Hint: Ensure the DID string is correctly formatted (e.g., did:oyd:...)", file=sys.stderr)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 resolve_did.py <DID_STRING>")
        sys.exit(1)
    
    resolve_did(sys.argv[1])
