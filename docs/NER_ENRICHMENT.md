# NER (Named Entity Recognition) Enrichment System

## Overview

The NER Enrichment System automatically extracts and tags entities from episode content before ingesting them into the Graphiti knowledge graph. This enhances search accuracy and provides structured metadata about entities mentioned in the content.

## Supported Entity Types

The system recognizes and tags the following entity types:

| Entity Type | Tag | Description | Examples |
|-------------|-----|-------------|----------|
| **Person** | PER | People, including fictional | Kamala Harris, Gavin Newsom |
| **Location** | LOC | Countries, cities, states, geographic features | California, San Francisco, United States |
| **Organization** | ORG | Companies, agencies, institutions | FBI, Google, Stanford University |
| **Date** | DATE | Absolute or relative dates | January 3, 2011, 2020 |
| **Time** | TIME | Times smaller than a day | 3:00 PM, morning |
| **Money** | MONEY | Monetary values | $1 million, €500 |
| **Percent** | PERCENT | Percentage values | 50%, 3.5% |
| **Facility** | FAC | Buildings, airports, highways, bridges | Golden Gate Bridge |
| **Product** | PRODUCT | Objects, vehicles, foods, etc. | iPhone, Tesla Model 3 |
| **Event** | EVENT | Named hurricanes, battles, wars, sports events | World War II |
| **Law** | LAW | Named documents made into laws | Constitution, Civil Rights Act |
| **Language** | LANGUAGE | Any named language | English, Spanish |
| **NORP** | NORP | Nationalities, religious/political groups | American, Republican |

## How It Works

### 1. Entity Extraction

The system uses two methods for entity extraction:

#### **Method A: spaCy (Recommended)**
- Uses the `en_core_web_sm` model
- High accuracy for entity recognition
- Supports all entity types listed above
- Install: `pip install spacy && python -m spacy download en_core_web_sm`

#### **Method B: Pattern-Based (Fallback)**
- Regex-based extraction
- Works without external dependencies
- Limited to:
  - Capitalized names (PER/LOC)
  - Date patterns (DATE)
  - Year patterns (DATE)
- Lower accuracy but always available

### 2. Entity Enrichment Process

```python
# Original episode
episode = {
    'content': 'Kamala Harris is the Attorney General of California. She was previously the district attorney for San Francisco.',
    'type': EpisodeType.text,
    'description': 'podcast transcript'
}

# After enrichment
enriched_episode = {
    'content': '...',  # Original content
    'type': EpisodeType.text,
    'description': 'podcast transcript',
    'entities': [
        {'text': 'Kamala Harris', 'type': 'PER', 'start': 0, 'end': 13},
        {'text': 'Attorney General', 'type': 'ORG', 'start': 21, 'end': 37},
        {'text': 'California', 'type': 'LOC', 'start': 41, 'end': 51},
        {'text': 'San Francisco', 'type': 'LOC', 'start': 103, 'end': 116}
    ],
    'entities_by_type': {
        'PER': ['Kamala Harris'],
        'ORG': ['Attorney General'],
        'LOC': ['California', 'San Francisco']
    },
    'entity_count': 4
}
```

### 3. Content Annotation

The enriched content is annotated with entity information:

```
Original: "Kamala Harris is the Attorney General of California..."

Enriched: "Kamala Harris is the Attorney General of California...

[ENTITIES: PER: Kamala Harris; ORG: Attorney General; LOC: California, San Francisco]"
```

This annotation helps the LLM in Graphiti better understand the entities and their types when building the knowledge graph.

## Usage

### Basic Usage

```python
from maintest import EntityEnricher

# Initialize enricher
enricher = EntityEnricher(use_spacy=True)

# Enrich a single episode
episode = {
    'content': 'Kamala Harris worked in California.',
    'type': EpisodeType.text,
    'description': 'example'
}

enriched = enricher.enrich_episode(episode)

# Access extracted entities
print(f"Found {enriched['entity_count']} entities")
for entity_type, entities in enriched['entities_by_type'].items():
    print(f"{entity_type}: {', '.join(entities)}")

# Get enriched content for ingestion
enriched_content = enricher.create_enriched_content(enriched)
```

### Ingestion with NER

```python
# Set ADD = True in maintest.py
ADD = True

# The system will automatically:
# 1. Initialize EntityEnricher
# 2. Extract entities from each episode
# 3. Display extracted entities
# 4. Add enriched content to Graphiti
```

## Benefits

### 1. **Improved Search Accuracy**
- Entities are explicitly tagged with their types
- Helps distinguish between "California" (location) vs "California" (organization name)
- Better semantic understanding of content

### 2. **Structured Metadata**
- Each episode has structured entity information
- Can filter/search by entity type
- Enables entity-centric queries

### 3. **Enhanced Knowledge Graph**
- LLM receives entity type hints when building the graph
- More accurate node creation and relationship extraction
- Better entity disambiguation

### 4. **Query Term Matching Boost**
- Extracted entities can be used to boost query matching scores
- Proper nouns are identified and weighted appropriately
- Location/person/organization queries are more accurate

## Example Output

### Console Output During Ingestion

```
================================================================================
📝 EPISODE INGESTION WITH NER ENRICHMENT
================================================================================

[Episode 0] Processing...
  ✓ Extracted 4 entities:
    - PER: Kamala Harris
    - ORG: Attorney General
    - LOC: California, San Francisco
  ✓ Added to graph: Freakonomics Radio 0 (text)

[Episode 1] Processing...
  ✓ Extracted 3 entities:
    - PER: Harris
    - DATE: January 3, 2011, January 3, 2017
  ✓ Added to graph: Freakonomics Radio 1 (text)

================================================================================
✅ INGESTION COMPLETE
================================================================================
```

## Configuration

### Using spaCy (Recommended)

```python
# Install spaCy and model
pip install spacy
python -m spacy download en_core_web_sm

# Use in code
enricher = EntityEnricher(use_spacy=True)
```

### Using Pattern-Based Fallback

```python
# No installation needed
# Automatically used if spaCy is not available
enricher = EntityEnricher(use_spacy=False)
```

## Integration with Multi-Factor Ranking

The NER system enhances the query term matching component of the multi-factor ranking:

1. **Entity Extraction**: Identifies proper nouns and their types
2. **Query Analysis**: Extracts entities from the query
3. **Type-Aware Matching**: Matches query entities with content entities
4. **Weighted Scoring**: Entities get higher weights in query matching

Example:
```
Query: "Who was the California Attorney General?"

Without NER:
- Simple text matching
- "California" matches as generic term

With NER:
- "California" identified as LOC
- "Attorney General" identified as ORG/role
- Nodes with matching LOC and ORG entities rank higher
- More precise results
```

## Limitations

### spaCy Method
- Requires installation and model download (~12 MB)
- Slightly slower processing
- May require more memory

### Pattern-Based Method
- Limited entity types (PER, LOC, DATE only)
- Lower accuracy
- May miss complex entity patterns
- No context-aware disambiguation

## Future Enhancements

Potential improvements:

1. **Custom Entity Types**: Add domain-specific entity types (e.g., "POLITICAL_POSITION", "GOVERNMENT_AGENCY")
2. **Entity Linking**: Link entities to knowledge bases (Wikipedia, Wikidata)
3. **Coreference Resolution**: Resolve pronouns to entities ("She" → "Kamala Harris")
4. **Relationship Extraction**: Extract relationships between entities
5. **Multi-language Support**: Support for non-English content
6. **Entity Confidence Scores**: Provide confidence scores for each entity
7. **Entity Normalization**: Normalize entity mentions ("AG" → "Attorney General")

## Troubleshooting

### spaCy Model Not Found

```
Error: Can't find model 'en_core_web_sm'
Solution: python -m spacy download en_core_web_sm
```

### Memory Issues

```
Problem: High memory usage with spaCy
Solution: Use pattern-based method or process episodes in smaller batches
```

### Low Accuracy

```
Problem: Pattern-based method missing entities
Solution: Install spaCy for better accuracy
```

## Performance

### spaCy Method
- **Speed**: ~100-500 ms per episode (depends on length)
- **Accuracy**: ~90-95% for common entity types
- **Memory**: ~200-500 MB

### Pattern-Based Method
- **Speed**: ~10-50 ms per episode
- **Accuracy**: ~60-70% for simple patterns
- **Memory**: ~10-20 MB

## Best Practices

1. **Use spaCy for Production**: Higher accuracy and more entity types
2. **Batch Processing**: Process episodes in batches for better performance
3. **Cache Results**: Cache enriched episodes to avoid re-processing
4. **Monitor Entity Counts**: Track entity extraction metrics
5. **Validate Entities**: Spot-check extracted entities for quality
6. **Update Models**: Keep spaCy models updated for best performance

