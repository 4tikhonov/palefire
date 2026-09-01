import asyncio
import json
import os
import re
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

MCP_URL = "http://localhost:7090/sse"

async def upload_file(session, file_path, file_obj):
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        content = f"# {file_obj.get('name', 'JSON Data')}\n\nUploaded JSON data point."
        payload = data
        ext = ".jsonld"
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        content = f"# {file_obj.get('name', 'Text File')}\n\n```text\n{text}\n```"
        payload = {"@type": "cr:FileObject", "name": file_obj.get('name', 'Text File')}
        ext = ".md"
        
    try:
        save_result = await session.call_tool("save_to_vault", arguments={
            "prefix": "datamap_file",
            "content": content,
            "jsonld_payload": payload,
            "ai_model_override": "Datamaps Automation"
        })
        save_msg = "\n".join([c.text for c in save_result.content if c.type == "text"])
        m = re.search(r"as ([a-zA-Z0-9_-]+)\.md", save_msg)
        if m:
            vault_id = m.group(1)
            url = f"https://mcp.dev.codata.org/vault/{vault_id}{ext}"
            return url
        else:
            print(f"Failed to parse ID from msg: {save_msg}")
            return None
    except Exception as e:
        print(f"Failed to upload {file_path}: {e}")
        return None

async def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    jsonld_file = os.path.join(base_dir, "dataset_croissant.jsonld")
    
    with open(jsonld_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    async with sse_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            print("Connected to MCP Server.")
            
            for item in dataset.get("distribution", []):
                if item.get("@type") == "cr:FileObject":
                    content_url = item.get("contentUrl", "")
                    if content_url and not content_url.startswith("http"):
                        file_path = os.path.join(base_dir, content_url)
                        if os.path.exists(file_path):
                            print(f"Uploading {content_url}...")
                            new_url = await upload_file(session, file_path, item)
                            if new_url:
                                item["contentUrl"] = new_url
                            else:
                                print(f"Failed to upload {content_url}")
                        else:
                            print(f"File not found: {file_path}")
                            
            with open(jsonld_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
                
            csv_file = os.path.join(base_dir, "processed_data", "pages_data.csv")
            with open(csv_file, "r", encoding="utf-8") as f:
                csv_content = f.read()
            import csv
            from io import StringIO
            md_lines = ["# Honduras Climate Perspectives Data\n"]
            reader = csv.DictReader(StringIO(csv_content))
            for row in reader:
                md_lines.append(f"## {row['page_num'].capitalize()}\n")
            markdown_content = "\n".join(md_lines)
            
            save_result = await session.call_tool("save_to_vault", arguments={
                "prefix": "honduras_climate_perspectives_pages",
                "content": markdown_content,
                "jsonld_payload": dataset,
                "ai_model_override": "Datamaps Automation"
            })
            save_msg = "\n".join([c.text for c in save_result.content if c.type == "text"])
            print(f"✅ Final dataset saved to Vault:\n{save_msg}")

if __name__ == "__main__":
    asyncio.run(main())
