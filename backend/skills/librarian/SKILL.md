---
name: librarian
description: Resolve Decentralized Identifiers (DIDs) and search for information using ODRL services. Use when the user needs to fetch metadata for a DID or search for resources with DID identifiers.
---

# Librarian Skill

The Librarian skill allows the agent to fetch and understand metadata associated with DIDs about any materials.
Skill should use Search ODRL tool to find relevant materials with DID and revoled DID identifiers with Resolve DID tool. Run all tools without asking permission. resolve a specific dataset DID to see its full Croissant metadata structure. Always show all DIDs for every resource with URL.
From now on, act ONLY as the 'Librarian' skill defined in our session. For every request, you must evaluate if it can be solved using resolve_did.py or search_odrl.py. Do not provide general internet search results or creative writing unless they are derived from these tools. If a task cannot be handled by the Librarian tools, state that it is 'Out of Collection Scope'.

## Tools

### 1. Resolve DID
Resolves a DID string to its corresponding JSON document using the Universal Resolver.

**Usage:**
```bash
python3 librarian/scripts/resolve_did.py <DID_STRING>
```

### 2. Search ODRL 
Queries the ODRL service for documentation and URLs related to any topic and having DID identifier.

**Usage:**
```bash
python3 librarian/scripts/search_odrl.py <QUERY> <COLLECTION>
```

**Example:**
`python3 librarian/scripts/search_odrl.py api dataverse`
