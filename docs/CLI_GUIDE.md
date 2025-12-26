# Pale Fire CLI Guide

## Overview

Pale Fire now includes a powerful command-line interface for ingesting episodes and querying the knowledge graph.

## Installation

```bash
cd /path/to/palefire

# Install dependencies
pip install -r requirements-ner.txt
python -m spacy download en_core_web_sm

# Configure environment
cp .env.example .env  # Edit with your settings
```

## Quick Start

```bash
# 1. Ingest demo episodes
python palefire-cli.py ingest --demo

# 2. Ask a question
python palefire-cli.py query "Who was the California Attorney General in 2020?"

# 3. Show configuration
python palefire-cli.py config
```

## Commands

### `ingest` - Ingest Episodes

Load episodes into the knowledge graph with optional NER enrichment.

**Usage:**
```bash
python palefire-cli.py ingest [OPTIONS]
```

**Options:**
- `--file, -f PATH` - Path to JSON file containing episodes
- `--demo` - Use built-in demo episodes
- `--no-ner` - Disable NER enrichment (faster but less accurate)

**Examples:**
```bash
# Ingest from file with NER enrichment
python palefire-cli.py ingest --file episodes.json

# Ingest from file without NER (faster)
python palefire-cli.py ingest --file episodes.json --no-ner

# Use built-in demo data
python palefire-cli.py ingest --demo
```

### `query` - Search the Knowledge Graph

Ask questions and get intelligent answers from the knowledge graph.

**Usage:**
```bash
python palefire-cli.py query QUESTION [OPTIONS]
```

**Options:**
- `--method, -m METHOD` - Search method to use
  - `standard` - Basic RRF search
  - `connection` - Connection-based ranking
  - `question-aware` - Intelligent question-type detection (default)
- `--export, -e FILE` - Export results to JSON file

**Examples:**
```bash
# Basic query (uses question-aware by default)
python palefire-cli.py query "Who was the California Attorney General in 2020?"

# Use specific search method
python palefire-cli.py query "Where did Kamala Harris work?" --method standard

# Export results to JSON
python palefire-cli.py query "Who is Gavin Newsom?" --export results.json

# Combine method and export
python palefire-cli.py query "When did Gavin Newsom become governor?" -m question-aware -e output.json

# Short form with export
python palefire-cli.py query "Your question?" -m standard -e results.json
```

### `config` - Show Configuration

Display current configuration settings.

**Usage:**
```bash
python palefire-cli.py config
```

**Output:**
```
⚙️  PALE FIRE CONFIGURATION
Neo4j URI: bolt://10.147.18.253:7687
Neo4j User: neo4j
LLM Model: deepseek-r1:7b
LLM Base URL: http://10.147.18.253:11434/v1
Embedder Model: nomic-embed-text
```

### `clean` - Clean/Clear Database

Clean or clear all data from the Neo4j database.

**Usage:**
```bash
python palefire-cli.py clean [OPTIONS]
```

**Options:**
- `--confirm` - Skip confirmation prompt and clean immediately
- `--nodes-only` - Delete only nodes and relationships (preserve database structure)

**Examples:**
```bash
# Clean with confirmation prompt
python palefire-cli.py clean

# Clean without confirmation (use with caution!)
python palefire-cli.py clean --confirm

# Delete only nodes (keep indexes and constraints)
python palefire-cli.py clean --nodes-only

# Quick clean for testing
python palefire-cli.py clean --confirm
```

**Output:**
```
================================================================================
🗑️  DATABASE CLEANUP
================================================================================
Current database contents:
  Nodes: 1523
  Relationships: 4567

⚠️  WARNING: This will permanently delete all data!
   Mode: Complete cleanup (all nodes, relationships, and data)

Are you sure you want to continue? (yes/no): yes

🔄 Cleaning database...

================================================================================
✅ DATABASE CLEANED SUCCESSFULLY
================================================================================
Deleted:
  Nodes: 1523
  Relationships: 4567

The database is now empty and ready for new data.
================================================================================
```

**⚠️ Warning:** This operation is irreversible! Always backup your data before cleaning.

**Use Cases:**
- Reset database for testing
- Clear old data before new ingestion
- Remove corrupted data
- Start fresh with a clean slate

## Episode File Format

Episodes should be in JSON format:

```json
[
    {
        "content": "Text content here...",
        "type": "text",
        "description": "Description of the episode"
    },
    {
        "content": {
            "name": "John Doe",
            "position": "CEO",
            "company": "Example Corp"
        },
        "type": "json",
        "description": "Structured metadata"
    }
]
```

**Fields:**
- `content` (required) - String for text episodes, object for JSON episodes
- `type` (required) - Either "text" or "json"
- `description` (optional) - Description of the episode

**Example file:** See `example_episodes.json`

## JSON Export Format

When using `--export`, results are saved in a structured JSON format:

```json
{
  "query": "Who was the California Attorney General in 2020?",
  "method": "question-aware",
  "timestamp": "2025-12-26T14:45:00.123456+00:00",
  "total_results": 5,
  "results": [
    {
      "rank": 1,
      "uuid": "abc123-def456-ghi789",
      "name": "Kamala Harris",
      "summary": "Attorney General of California from 2011 to 2017...",
      "labels": ["Person", "PoliticalFigure"],
      "attributes": {
        "position": "Attorney General",
        "state": "California",
        "term_start": "January 3, 2011",
        "term_end": "January 3, 2017"
      },
      "scoring": {
        "final_score": 0.9234,
        "original_score": 0.8456,
        "connection_score": 0.7823,
        "temporal_score": 1.0,
        "query_match_score": 0.9123,
        "entity_type_score": 2.0
      },
      "connections": {
        "count": 15,
        "entities": [
          {
            "name": "California",
            "type": "LOC",
            "labels": ["Entity", "LOC"],
            "uuid": "xyz789-abc123-def456"
          },
          {
            "name": "San Francisco",
            "type": "LOC",
            "labels": ["Entity", "LOC"],
            "uuid": "uvw456-rst789-mno012"
          },
          {
            "name": "District Attorney",
            "type": "ORG",
            "labels": ["Entity", "ORG"],
            "uuid": "pqr345-stu678-vwx901"
          }
        ],
        "relationship_types": ["WORKED_AT", "LOCATED_IN", "HELD_POSITION"]
      },
      "temporal_info": {
        "properties": {
          "term_start": "January 3, 2011",
          "term_end": "January 3, 2017"
        }
      },
      "recognized_entities": {
        "PER": ["Kamala Harris"],
        "LOC": ["California", "San Francisco"],
        "ORG": ["Attorney General"],
        "DATE": ["January 3, 2011", "January 3, 2017", "2020"]
      },
      "all_entities": [
        {"text": "Kamala Harris", "type": "PER", "start": 0, "end": 13},
        {"text": "California", "type": "LOC", "start": 45, "end": 55},
        {"text": "Attorney General", "type": "ORG", "start": 25, "end": 41}
      ]
    }
  ]
}
```

**Fields:**
- `query` - Original search query
- `method` - Search method used
- `timestamp` - When the search was performed (ISO 8601 format)
- `total_results` - Number of results returned
- `results` - Array of result objects with:
  - `rank` - Position in results (1-based)
  - `uuid` - Node UUID in Neo4j
  - `name` - Node name
  - `summary` - Node summary
  - `labels` - Node labels
  - `attributes` - Node attributes
  - `scoring` - Detailed scoring breakdown (question-aware method only)
  - `connections` - Connection information (question-aware/connection methods)
    - `count` - Number of connections
    - `entities` - Array of connected entities with:
      - `name` - Entity name
      - `type` - Entity type (PER, LOC, ORG, etc.) if available
      - `labels` - All Neo4j labels
      - `uuid` - Entity UUID
    - `relationship_types` - Types of relationships
  - `temporal_info` - Temporal data (question-aware method only)
  - `recognized_entities` - NER-extracted entities grouped by type (question-aware method only)
  - `all_entities` - All recognized entities with positions (question-aware method only)

**Note:** The `name_embedding` field is automatically excluded from attributes to reduce file size.

**Example file:** See `example_export.json`

**Use Cases:**
- Post-processing and analysis
- Integration with other tools
- Batch processing pipelines
- Result archiving and comparison
- Machine learning training data

## Search Methods

### Standard Search
- **Method**: `standard`
- **Factors**: RRF (Reciprocal Rank Fusion) only
- **Use case**: Simple queries, fastest
- **Example**: `python palefire-cli.py query "California" -m standard`

### Connection-Based Search
- **Method**: `connection`
- **Factors**: RRF + Graph connectivity
- **Use case**: Finding central/important entities
- **Example**: `python palefire-cli.py query "Who is important?" -m connection`

### Question-Aware Search (Recommended)
- **Method**: `question-aware`
- **Factors**: RRF + Connections + Temporal + Query matching + Entity type intelligence
- **Use case**: Natural language questions (WHO/WHERE/WHEN/etc.)
- **Example**: `python palefire-cli.py query "Who was the AG?" -m question-aware`

## Environment Variables

Create a `.env` file with:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenAI/Ollama Configuration
OPENAI_API_KEY=your_api_key
```

## Common Workflows

### Workflow 1: First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements-ner.txt
python -m spacy download en_core_web_sm

# 2. Configure environment
nano .env  # Add your Neo4j and API credentials

# 3. Ingest demo data
python palefire-cli.py ingest --demo

# 4. Test with a query
python palefire-cli.py query "Who is Gavin Newsom?"
```

### Workflow 2: Custom Data
```bash
# 1. Create your episodes file
cat > my_episodes.json << 'EOF'
[
    {
        "content": "Your content here...",
        "type": "text",
        "description": "Your description"
    }
]
EOF

# 2. Ingest your data
python palefire-cli.py ingest --file my_episodes.json

# 3. Query your data
python palefire-cli.py query "Your question here?"
```

### Workflow 3: Batch Processing
```bash
# Ingest multiple files
for file in data/*.json; do
    python palefire-cli.py ingest --file "$file"
done

# Run multiple queries
python palefire-cli.py query "Question 1?"
python palefire-cli.py query "Question 2?"
python palefire-cli.py query "Question 3?"
```

## Question Types

The question-aware search automatically detects these question types:

| Type | Examples | Boosts |
|------|----------|--------|
| **WHO** | "Who was the AG?" | Person entities (2.0x) |
| **WHERE** | "Where did she work?" | Location entities (2.0x) |
| **WHEN** | "When was he governor?" | Date entities (2.0x) |
| **WHAT** (org) | "What organization?" | Organization entities (2.0x) |
| **WHAT** (position) | "What position?" | Person/Org entities (1.5x) |
| **HOW MANY** | "How many years?" | Number entities (2.0x) |
| **WHY** | "Why did she leave?" | Event entities (1.5x) |
| **WHAT** (event) | "What happened?" | Event entities (2.0x) |

## Performance Tips

### For Faster Ingestion
```bash
# Disable NER enrichment
python palefire-cli.py ingest --file large_file.json --no-ner
```

### For Better Accuracy
```bash
# Use NER enrichment (default)
python palefire-cli.py ingest --file data.json

# Use question-aware search
python palefire-cli.py query "Your question?" -m question-aware
```

### For Large Datasets
```bash
# Split into smaller files
split -l 100 large_episodes.json episodes_part_

# Ingest in batches
for file in episodes_part_*; do
    python palefire-cli.py ingest --file "$file"
done
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'modules'"
```bash
# Make sure you're in the palefire directory
cd /path/to/palefire
python palefire-cli.py query "test"
```

### "NEO4J_URI must be set"
```bash
# Create or update .env file
echo "NEO4J_URI=bolt://localhost:7687" >> .env
echo "NEO4J_USER=neo4j" >> .env
echo "NEO4J_PASSWORD=password" >> .env
```

### "spaCy model not found"
```bash
# Download the spaCy model
python -m spacy download en_core_web_sm
```

### "Connection refused" (Neo4j)
```bash
# Start Neo4j
# Check Neo4j is running on the configured URI
# Verify credentials in .env
```

## Advanced Usage

### Custom LLM Configuration

Edit `palefire-cli.py` to change LLM settings:

```python
llm_config = LLMConfig(
    api_key="your_key",
    model="gpt-4",
    base_url=None,  # Use OpenAI default
)
```

### Programmatic Usage

```python
from palefire_cli import load_episodes_from_file, ingest_episodes

# Load episodes
episodes = load_episodes_from_file("data.json")

# Ingest with custom settings
await ingest_episodes(episodes, graphiti, use_ner=True)
```

### Batch Query Script

```python
import asyncio
from palefire_cli import search_query, create_graphiti

questions = [
    "Who was the AG?",
    "Where did she work?",
    "When was he governor?"
]

async def batch_query():
    graphiti = create_graphiti()
    for q in questions:
        await search_query(q, graphiti, method='question-aware')

asyncio.run(batch_query())
```

## Help

```bash
# General help
python palefire-cli.py --help

# Command-specific help
python palefire-cli.py ingest --help
python palefire-cli.py query --help
python palefire-cli.py config --help
```

## Examples

### Example 1: Political Data
```bash
# Create episodes about politicians
cat > politicians.json << 'EOF'
[
    {
        "content": "Joe Biden became the 46th President of the United States on January 20, 2021.",
        "type": "text",
        "description": "Presidential information"
    }
]
EOF

# Ingest and query
python palefire-cli.py ingest --file politicians.json
python palefire-cli.py query "Who became president in 2021?"
```

### Example 2: Company Data
```bash
# Create episodes about companies
cat > companies.json << 'EOF'
[
    {
        "content": {
            "name": "OpenAI",
            "founded": "2015",
            "location": "San Francisco",
            "ceo": "Sam Altman"
        },
        "type": "json",
        "description": "Company information"
    }
]
EOF

# Ingest and query
python palefire-cli.py ingest --file companies.json
python palefire-cli.py query "Where is OpenAI located?"
```

### Example 3: Historical Events
```bash
# Create episodes about events
cat > events.json << 'EOF'
[
    {
        "content": "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969.",
        "type": "text",
        "description": "Historical event"
    }
]
EOF

# Ingest and query
python palefire-cli.py ingest --file events.json
python palefire-cli.py query "When did humans land on the Moon?"
```

## Integration

### Shell Scripts
```bash
#!/bin/bash
# ingest_and_query.sh

python palefire-cli.py ingest --file "$1"
python palefire-cli.py query "$2"
```

### Python Scripts
```python
#!/usr/bin/env python3
import subprocess
import sys

def ingest(file):
    subprocess.run(['python', 'palefire-cli.py', 'ingest', '--file', file])

def query(question):
    subprocess.run(['python', 'palefire-cli.py', 'query', question])

if __name__ == '__main__':
    ingest(sys.argv[1])
    query(sys.argv[2])
```

### API Wrapper (Future)
```python
from fastapi import FastAPI
from palefire_cli import search_query

app = FastAPI()

@app.post("/query")
async def api_query(question: str):
    result = await search_query(question, graphiti)
    return {"results": result}
```

## Best Practices

1. **Always use NER enrichment** for production (better accuracy)
2. **Use question-aware search** for natural language queries
3. **Batch process** large datasets for better performance
4. **Version your episode files** for reproducibility
5. **Test queries** after ingestion to verify data quality
6. **Monitor logs** for errors and warnings
7. **Backup Neo4j** database regularly

## Support

For issues or questions:
1. Check this guide
2. Review documentation in the palefire directory
3. Check logs for error messages
4. Verify environment configuration

---

**Pale Fire CLI** - Intelligent Knowledge Graph Search Made Easy

