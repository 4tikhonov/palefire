# JSON Export Feature

## Overview

Pale Fire now supports exporting search results to JSON format for post-processing, analysis, and integration with other tools.

## Usage

### Basic Export

```bash
python palefire-cli.py query "Your question?" --export results.json
```

### With Search Method

```bash
python palefire-cli.py query "Who is Gavin Newsom?" -m question-aware -e output.json
```

### Short Form

```bash
python palefire-cli.py query "Your question?" -e results.json
```

## Export Format

The exported JSON file contains:

```json
{
  "query": "Original search query",
  "method": "Search method used",
  "timestamp": "ISO 8601 timestamp",
  "total_results": 5,
  "results": [...]
}
```

### Result Object Structure

Each result in the `results` array contains:

#### Basic Fields (All Methods)
- `rank` - Position in results (1-based)
- `uuid` - Node UUID in Neo4j
- `name` - Node name
- `summary` - Node summary
- `labels` - Array of node labels
- `attributes` - Object with node attributes

#### Enhanced Fields (question-aware/connection methods)

**Scoring Breakdown:**
```json
"scoring": {
  "final_score": 0.9234,
  "original_score": 0.8456,
  "connection_score": 0.7823,
  "temporal_score": 1.0,
  "query_match_score": 0.9123,
  "entity_type_score": 2.0
}
```

**Connection Information:**
```json
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
}
```

**Entity Object Fields:**
- `name` - The entity's name
- `type` - Entity type (PER, LOC, ORG, DATE, etc.) if recognized
- `labels` - All Neo4j labels assigned to the entity
- `uuid` - Unique identifier in the knowledge graph

**Temporal Information:**
```json
"temporal_info": {
  "created_at": "2025-12-26T12:00:00Z",
  "valid_at": "2011-01-03T00:00:00Z",
  "invalid_at": "2017-01-03T00:00:00Z",
  "properties": {
    "term_start": "January 3, 2011",
    "term_end": "January 3, 2017"
  }
}
```

**Recognized Entities (NER):**
```json
"recognized_entities": {
  "PER": ["Kamala Harris"],
  "LOC": ["California", "San Francisco"],
  "ORG": ["Attorney General"],
  "DATE": ["January 3, 2011", "January 3, 2017", "2020"]
},
"all_entities": [
  {"text": "Kamala Harris", "type": "PER", "start": 0, "end": 13},
  {"text": "California", "type": "LOC", "start": 45, "end": 55},
  {"text": "Attorney General", "type": "ORG", "start": 25, "end": 41},
  {"text": "January 3, 2011", "type": "DATE", "start": 102, "end": 117}
]
```

**Note:** 
- `recognized_entities` groups entities by type for easy filtering
- `all_entities` includes position information for each entity
- The `name_embedding` attribute is automatically excluded from exports to reduce file size

## Use Cases

### 1. Post-Processing and Analysis

```python
import json

# Load results
with open('results.json') as f:
    data = json.load(f)

# Analyze scoring
for result in data['results']:
    print(f"{result['name']}: {result['scoring']['final_score']:.4f}")

# Analyze connected entities with types
for result in data['results']:
    print(f"\n{result['name']} connections:")
    for entity in result['connections']['entities']:
        entity_type = entity.get('type', 'Unknown')
        print(f"  - {entity['name']} ({entity_type})")
```

### 2. Batch Processing

```bash
# Run multiple queries and export
queries=(
    "Who was the California Attorney General in 2020?"
    "Where did Kamala Harris work?"
    "When did Gavin Newsom become governor?"
)

for i in "${!queries[@]}"; do
    python palefire-cli.py query "${queries[$i]}" -e "result_$i.json"
done
```

### 3. Result Comparison

```python
import json

# Compare results from different methods
methods = ['standard', 'connection', 'question-aware']
query = "Who is Gavin Newsom?"

for method in methods:
    # Run query with each method
    # python palefire-cli.py query "$query" -m $method -e "${method}_results.json"
    
    with open(f'{method}_results.json') as f:
        data = json.load(f)
        print(f"{method}: {data['total_results']} results")
```

### 4. Integration with Other Tools

```python
import json
import pandas as pd

# Convert to DataFrame for analysis
with open('results.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['results'])
print(df[['rank', 'name', 'scoring.final_score']])
```

### 5. Machine Learning Training Data

```python
import json

# Extract features for ML
with open('results.json') as f:
    data = json.load(f)

features = []
for result in data['results']:
    features.append({
        'query': data['query'],
        'name': result['name'],
        'final_score': result['scoring']['final_score'],
        'connection_count': result['connections']['count'],
        'entity_types': list(result['recognized_entities'].keys()),
        'entity_count': len(result['all_entities'])
    })
```

## Examples

### Example 1: Export Question-Aware Search

```bash
python palefire-cli.py query "Who was the California Attorney General in 2020?" \
    --method question-aware \
    --export ag_2020.json
```

**Output:**
```
💾 Results exported to: ag_2020.json
   Total results: 5
```

### Example 2: Export Standard Search

```bash
python palefire-cli.py query "California" -m standard -e california.json
```

### Example 3: Multiple Exports

```bash
# Compare different methods
python palefire-cli.py query "Who is Gavin Newsom?" -m standard -e standard.json
python palefire-cli.py query "Who is Gavin Newsom?" -m connection -e connection.json
python palefire-cli.py query "Who is Gavin Newsom?" -m question-aware -e question_aware.json
```

## File Naming Conventions

### Recommended Patterns

**By Query:**
```bash
--export "query_$(date +%Y%m%d_%H%M%S).json"
```

**By Method:**
```bash
--export "results_${method}_$(date +%Y%m%d).json"
```

**By Topic:**
```bash
--export "california_ag_results.json"
--export "political_figures.json"
```

## Error Handling

If export fails, you'll see:
```
❌ Failed to export results: [error message]
```

Common issues:
- **Permission denied**: Check write permissions for the output directory
- **Invalid path**: Ensure the directory exists
- **Disk full**: Check available disk space

## Tips

1. **Use descriptive filenames** - Include query topic and date
2. **Organize by project** - Create subdirectories for different projects
3. **Archive regularly** - Compress old exports to save space
4. **Version control** - Track important results in git (if appropriate)
5. **Backup** - Keep backups of critical search results

## Advanced Usage

### Programmatic Export

```python
import asyncio
from palefire_cli import search_query, export_results_to_json

async def batch_search():
    queries = [
        "Who was the California Attorney General in 2020?",
        "Where did Kamala Harris work?",
        "When did Gavin Newsom become governor?"
    ]
    
    for i, query in enumerate(queries):
        results = await search_query(query, graphiti, method='question-aware')
        export_results_to_json(results, f'result_{i}.json', query, 'question-aware')

asyncio.run(batch_search())
```

### Custom Processing

```python
import json

def process_export(filepath):
    """Process exported results."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Extract top result
    top_result = data['results'][0]
    
    # Create summary
    summary = {
        'query': data['query'],
        'top_answer': top_result['name'],
        'confidence': top_result['scoring']['final_score'],
        'method': data['method']
    }
    
    return summary

# Usage
summary = process_export('results.json')
print(f"Best answer: {summary['top_answer']} (confidence: {summary['confidence']:.2f})")
```

## See Also

- [CLI Guide](CLI_GUIDE.md) - Complete CLI documentation
- [Quick Reference](QUICK_REFERENCE.md) - Quick command reference
- [Configuration](CONFIGURATION.md) - Configuration options
- `example_export.json` - Example export file

---

**JSON Export Feature v1.0** - Export, Analyze, Integrate! 📊

