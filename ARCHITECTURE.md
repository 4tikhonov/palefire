# Pale Fire - Architecture Overview

## Project Structure

```
palefire/
├── palefire-cli.py              # Main CLI application
├── modules/                     # Core modules
│   ├── __init__.py             # Module exports
│   └── PaleFireCore.py         # Entity enrichment & question detection
├── requirements-ner.txt         # NER dependencies
├── PALEFIRE_SETUP.md           # Setup guide
├── QUICK_REFERENCE.md          # Quick reference
├── RANKING_SYSTEM.md           # Ranking documentation
├── NER_ENRICHMENT.md           # NER documentation
├── QUESTION_TYPE_DETECTION.md  # Question-type guide
└── revisions/                  # Backup files
    └── maintest.py             # Original version
```

## Module Organization

### `modules/PaleFireCore.py`

Contains the core classes that power Pale Fire's intelligent search:

#### **EntityEnricher**
- Extracts named entities from text (PER, LOC, ORG, DATE, etc.)
- Supports both spaCy (recommended) and pattern-based extraction
- Enriches episodes with entity metadata before ingestion

**Key Methods:**
- `extract_entities(text)` - Extract entities from text
- `enrich_episode(episode)` - Add entity metadata to episode
- `create_enriched_content(episode)` - Create annotated content

#### **QuestionTypeDetector**
- Detects question type (WHO/WHERE/WHEN/WHAT/WHY/HOW)
- Maps question types to entity type weights
- Provides confidence scores for detection

**Key Methods:**
- `detect_question_type(query)` - Detect question type from query
- `apply_entity_type_weights(node, enriched_episode, entity_weights)` - Calculate weighted score

### `palefire-cli.py`

The main CLI application that orchestrates everything:

**Components:**
1. **Configuration** - Neo4j, LLM, episode data
2. **Helper Functions** - Graph operations, temporal analysis, query matching
3. **Search Functions** - 5 different ranking approaches
4. **Main Function** - Ingestion and search workflows

## Import Structure

```python
# In palefire-cli.py
from modules import EntityEnricher, QuestionTypeDetector

# Usage
enricher = EntityEnricher(use_spacy=True)
detector = QuestionTypeDetector()
```

## Data Flow

### Episode Ingestion
```
1. Load episodes from data
   ↓
2. EntityEnricher.enrich_episode()
   ├─ Extract entities (PER, LOC, ORG, etc.)
   ├─ Group by type
   └─ Create enriched content
   ↓
3. Add to Graphiti with annotations
   ↓
4. Build knowledge graph
```

### Search Query
```
1. User query input
   ↓
2. QuestionTypeDetector.detect_question_type()
   ├─ Identify question type (WHO/WHERE/etc.)
   ├─ Get entity type weights
   └─ Calculate confidence
   ↓
3. Execute hybrid search (RRF)
   ↓
4. For each result:
   ├─ Get connection count
   ├─ Extract temporal info
   ├─ Calculate query match score
   ├─ Extract entities from node
   └─ Calculate entity type score
   ↓
5. Combine scores with weights
   ↓
6. Rank and return top results
```

## Search Methods

### 1. Standard Search
- **File**: `search_episodes()`
- **Factors**: RRF only
- **Use**: Simple queries

### 2. Connection-Based
- **File**: `search_episodes_with_custom_ranking()`
- **Factors**: RRF + Connections
- **Use**: Find central entities

### 3. Temporal-Aware
- **File**: `search_episodes_with_temporal_ranking()`
- **Factors**: RRF + Connections + Temporal
- **Use**: Date-specific queries

### 4. Multi-Factor
- **File**: `search_episodes_with_multi_factor_ranking()`
- **Factors**: RRF + Connections + Temporal + Query Match
- **Use**: Complex queries

### 5. Question-Aware (Recommended)
- **File**: `search_episodes_with_question_aware_ranking()`
- **Factors**: All 4 + Entity Type Intelligence
- **Use**: Natural language questions

## Configuration

### Environment Variables (.env)
```bash
NEO4J_URI=bolt://10.147.18.253:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=your_key
```

### LLM Configuration (in code)
```python
llm_config = LLMConfig(
    api_key="ollama",
    model="deepseek-r1:7b",
    base_url="http://10.147.18.253:11434/v1"
)
```

## Dependencies

### Core
- `graphiti-core` - Knowledge graph framework
- `python-dotenv` - Environment configuration

### NER (Optional but Recommended)
- `spacy` - Industrial NLP library
- `en_core_web_sm` - English language model

## Extending the System

### Adding Custom Entity Types

Edit `modules/PaleFireCore.py`:
```python
class EntityEnricher:
    ENTITY_TYPES = {
        # Add your custom types
        'CUSTOM_TYPE': 'CUSTOM',
        ...
    }
```

### Adding Custom Question Types

Edit `modules/PaleFireCore.py`:
```python
class QuestionTypeDetector:
    QUESTION_PATTERNS = {
        'CUSTOM_QUESTION': {
            'patterns': [r'\byour pattern\b'],
            'entity_weights': {'PER': 1.5, 'LOC': 1.2},
            'description': 'Your custom question type'
        },
        ...
    }
```

### Adding Custom Search Methods

Add to `palefire-cli.py`:
```python
async def search_episodes_with_custom_method(graphiti, query, ...):
    # Your custom ranking logic
    pass
```

## Performance Considerations

### Memory Usage
- **spaCy**: ~200-500 MB
- **Pattern-based**: ~10-20 MB
- **Neo4j driver**: ~50-100 MB

### Speed
- **Question detection**: 1-5ms
- **Entity extraction (spaCy)**: 50-500ms per node
- **Entity extraction (pattern)**: 10-50ms per node
- **Standard search**: 100-300ms
- **Question-aware search**: 500-2000ms

### Optimization Tips
1. Use spaCy for better accuracy
2. Reduce `node_search_config.limit` for faster searches
3. Cache enriched episodes
4. Batch process large datasets
5. Use connection pooling for Neo4j

## Testing

### Unit Tests
```python
# Test entity extraction
enricher = EntityEnricher(use_spacy=True)
episode = {'content': 'Test text', 'type': 'text'}
result = enricher.enrich_episode(episode)
assert 'entities' in result

# Test question detection
detector = QuestionTypeDetector()
info = detector.detect_question_type("Who is John?")
assert info['type'] == 'WHO'
```

### Integration Tests
```bash
# Test full pipeline
python palefire-cli.py  # with ADD=True for ingestion
python palefire-cli.py  # with ADD=False for search
```

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'modules'
```
**Solution**: Ensure you're running from the palefire directory

### spaCy Model Not Found
```
OSError: Can't find model 'en_core_web_sm'
```
**Solution**: `python -m spacy download en_core_web_sm`

### Neo4j Connection Error
```
ServiceUnavailable: Failed to establish connection
```
**Solution**: Check Neo4j is running and credentials are correct

## Best Practices

1. **Modular Design**: Keep classes in modules/, functions in CLI
2. **Type Hints**: Use type annotations for better IDE support
3. **Logging**: Use logger for debugging, not print()
4. **Error Handling**: Wrap external calls in try/except
5. **Documentation**: Update docs when adding features
6. **Testing**: Test new features before deployment
7. **Version Control**: Keep revisions/ for rollback capability

## Future Enhancements

### Planned Features
- [ ] REST API wrapper
- [ ] Web UI
- [ ] Batch processing API
- [ ] Result caching
- [ ] Multi-language support
- [ ] Custom entity types per domain
- [ ] ML-based question detection
- [ ] Entity linking to knowledge bases

### API Design (Future)
```python
# Future API structure
from palefire import PaleFire

pf = PaleFire(neo4j_uri, neo4j_user, neo4j_password)
await pf.ingest_episodes(episodes)
results = await pf.search("Who was the AG?")
```

## Contributing

When adding features:
1. Add classes to `modules/PaleFireCore.py`
2. Add functions to `palefire-cli.py`
3. Update documentation
4. Test thoroughly
5. Update ARCHITECTURE.md

## License

Inherits license from parent Open WebUI project.

