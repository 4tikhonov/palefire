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
                        urls.append(item)
                elif isinstance(data, dict):
                    results = data.get("results", data.get("hits", []))
                    if isinstance(results, list):
                        for item in results:
                            urls.append(item)
                    else:
                        urls.append(data)
                
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
            # Create a copy for formatting
            display_item = u.copy()
            if "did" in display_item:
                did = display_item["did"]
                # Create a link for the DID pointing to the resolver
                resolver_url = f"https://dev.uniresolver.io/1.0/identifiers/{did}"
                display_item["did"] = f"\033]8;;{resolver_url}\033\\{did}\033]8;;\033\\"
            
            # Check for URL in json_ld or root
            url_to_link = None
            if "url" in display_item:
                url_to_link = display_item["url"]
            elif "json_ld" in display_item and isinstance(display_item["json_ld"], dict) and "url" in display_item["json_ld"]:
                url_to_link = display_item["json_ld"]["url"]
            
            if url_to_link:
                linked_url = f"\033]8;;{url_to_link}\033\\{url_to_link}\033]8;;\033\\"
                if "url" in display_item:
                    display_item["url"] = linked_url
                elif "json_ld" in display_item:
                    display_item["json_ld"]["url"] = linked_url

            print(json.dumps(display_item, indent=2))
    else:
        print("No URLs found.")
