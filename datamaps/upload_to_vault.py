import asyncio
import json
import os
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

MCP_URL = "http://localhost:7090/sse"

async def upload_files():
    print(f"Connecting to MCP Server at {MCP_URL}...")
    
    # Read the files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "processed_data", "pages_data.csv")
    jsonld_file = os.path.join(base_dir, "dataset_croissant.jsonld")
    
    if not os.path.exists(csv_file) or not os.path.exists(jsonld_file):
        print("Required files not found. Make sure pages_data.csv and dataset_croissant.jsonld exist.")
        return

    import csv
    from io import StringIO
    
    with open(csv_file, "r", encoding="utf-8") as f:
        csv_content = f.read()
        
    # Convert CSV to readable Markdown
    md_lines = ["# Honduras Climate Perspectives Data\n"]
    reader = csv.DictReader(StringIO(csv_content))
    for row in reader:
        md_lines.append(f"## {row['page_num'].capitalize()}\n")
        if row.get('fullpage_text'):
            md_lines.append(f"**Full Page Text:**\n```\n{row['fullpage_text']}\n```\n")
        if row.get('legend_text'):
            md_lines.append(f"**Legend Text:**\n```\n{row['legend_text']}\n```\n")
            
    markdown_content = "\n".join(md_lines)
        
    with open(jsonld_file, "r", encoding="utf-8") as f:
        jsonld_content = f.read()

    # Parse JSON-LD payload to a dictionary, as the schema expects an object
    jsonld_payload = json.loads(jsonld_content)

    async with sse_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            print("Connected and initialized session.")
            
            try:
                print("Calling save_to_vault tool...")
                save_result = await session.call_tool("save_to_vault", arguments={
                    "prefix": "honduras_climate_perspectives_pages",
                    "content": markdown_content,
                    "jsonld_payload": jsonld_payload,
                    "ai_model_override": "Datamaps Automation"
                })
                save_msg = "\n".join([c.text for c in save_result.content if c.type == "text"])
                print(f"✅ Saved to Vault:\n{save_msg}")
            except Exception as e:
                print(f"❌ Failed to save to Vault programmatically: {e}")

if __name__ == "__main__":
    asyncio.run(upload_files())
