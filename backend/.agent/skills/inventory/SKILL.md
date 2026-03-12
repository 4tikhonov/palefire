---
name: inventory
description: Extract an inventory of entities (person names, money, locations, organizations, time, dates) and extract main keywords from the provided text to describe the material.
---

When the user asks you to "make an inventory", "inventory this", or sends content with instructions to extract entities and keywords, you must instantly perform the following analysis:

1. **Entity Extraction**: Read the provided text or transcript and systematically identify and list all significant entities, categorized clearly into the following headers:
    - **Time and Dates**: (e.g., October 24th, 1999, 5:00 PM, Next Tuesday)
    - **Person Names**: (e.g., John Doe, Elon Musk)
    - **Organizations**: (e.g., Google, United Nations, Ministry of Magic)
    - **Locations**: (e.g., Paris, Texas, Mount Everest, GPS coordinates)
    - **Money & Quantities**: (e.g., $100, 50 million euros, 10 bitcoins, 500 workers)
    - **Events & Actions**: (e.g., Summer Olympics, The Great Strike, protest, meeting)

2. **Keyword Extraction**: Identify and extract the 5-10 most important "Main Keywords" or key phrases that best define the core themes and topics of the material.

3. **Presentation**: Output the results cleanly using standard Markdown formatting. Use bulleted lists. If a specific entity category has no factual matches in the generated text, explicitly write "None found" beneath that header rather than omitting the category entirely.

4. **JSON-LD Caching**: Immediately after presenting the markdown, silently convert the extracted entity inventory into a valid `JSON-LD` format document mapping the concepts properly (e.g., schema.org). You MUST use your file writing tools or run a bash command (`cat << 'EOF' > ...`) to save this raw JSON-LD data to a uniquely named file inside the `cache/` directory (for example: `cache/inventory_TIMESTAMP.jsonld`). 

Provide the requested markdown inventory and a brief confirmation that the JSON-LD file has been cached, without extra conversational filler.
