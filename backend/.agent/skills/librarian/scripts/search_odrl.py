import sys
import json
import urllib.request
import urllib.error
import urllib.parse

def search_odrl(query, collection):
    params = urllib.parse.urlencode({'q': query, 'collection': collection})
    url = f"https://odrl.dev.codata.org/api/oac/search?{params}"
    
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                urls = []
                # Assuming the response is a list or contains a list of results
                # Based on the user request, we look for "url" fields.
                if isinstance(data, list):
                    for item in data:
                        # Check root and nested json_ld
                        if "url" in item:
                            urls.append(item["url"])
                        elif "json_ld" in item and isinstance(item["json_ld"], dict) and "url" in item["json_ld"]:
                            urls.append(item["json_ld"]["url"])
                elif isinstance(data, dict):
                    # Check common keys like 'results', 'hits', or the root itself
                    results = data.get("results", data.get("hits", []))
                    if isinstance(results, list):
                        for item in results:
                            if "url" in item:
                                urls.append(item["url"])
                    elif "url" in data:
                        urls.append(data["url"])
                
                return urls
            else:
                print(f"Error: Received status code {response.status}", file=sys.stderr)
                return []
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return []

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "api"
    collection = sys.argv[2] if len(sys.argv) > 2 else "dataverse"
    
    results_urls = search_odrl(query, collection)
    if results_urls:
        for u in results_urls:
            print(u)
    else:
        print("No URLs found.")