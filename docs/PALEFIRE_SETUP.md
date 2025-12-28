# Pale Fire - Intelligent Knowledge Graph Search System

## Overview

Pale Fire is an advanced knowledge graph search system built on Graphiti, featuring:
- **5-Factor Ranking System**: Semantic, connectivity, temporal, query matching, and entity-type intelligence
- **Question-Type Detection**: Automatically understands WHO/WHERE/WHEN/WHAT/WHY/HOW questions
- **NER Enrichment**: Extracts and tags entities (PER, LOC, ORG, DATE, etc.)
- **Multi-Factor Search**: Combines multiple relevance signals for optimal results

## Quick Start

### 1. Install Dependencies

```bash
cd /path/to/palefire

# Install base dependencies
pip install graphiti-core python-dotenv

# Install NER dependencies (optional but recommended)
pip install -r requirements-ner.txt
python -m spacy download en_core_web_sm

# Install keyword extraction dependencies
pip install gensim>=4.3.0
# Optional: For better stemming support
pip install nltk
```

### 2. Configure Environment

Copy the example configuration file and customize it:

```bash
cp env.example .env
# Edit .env with your settings
```

Key configuration options in `.env`:

```bash
# Neo4j Configuration (REQUIRED)
NEO4J_URI=bolt://10.147.18.253:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Provider ('ollama' or 'openai')
LLM_PROVIDER=ollama
OPENAI_API_KEY=your_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://10.147.18.253:11434/v1
OLLAMA_MODEL=deepseek-r1:7b
OLLAMA_VERIFICATION_MODEL=gpt-oss:latest  # Optional: separate model for NER verification

# Search Configuration
DEFAULT_SEARCH_METHOD=question-aware
SEARCH_RESULT_LIMIT=20
SEARCH_TOP_K=5

# Ranking Weights (must sum to <= 1.0)
WEIGHT_CONNECTION=0.15
WEIGHT_TEMPORAL=0.20
WEIGHT_QUERY_MATCH=0.20
WEIGHT_ENTITY_TYPE=0.15
```

All configuration is centralized in `config.py`, which reads from environment variables with sensible defaults.

### 3. View Configuration

Check your current configuration:

```bash
python palefire-cli.py config
```

This will display all settings including Neo4j connection, LLM configuration, search parameters, and ranking weights.

### 4. Customize Your Data

Create a JSON file with your episodes (see `example_episodes.json`):

```json
[
    {
        "content": "Your content here...",
        "type": "text",
        "description": "your description"
    },
    # Add more episodes...
]
```

### 4. Run the System

```bash
# First time: Ingest episodes with NER enrichment
# Set ADD = True in palefire-cli.py
python palefire-cli.py

# After ingestion: Run search queries
# Set ADD = False in palefire-cli.py
python palefire-cli.py
```

## Project Structure

```
palefire/
├── palefire-cli.py                # Main CLI application
├── RANKING_SYSTEM.md              # Ranking system documentation
├── NER_ENRICHMENT.md              # NER system documentation
├── QUESTION_TYPE_DETECTION.md     # Question-type detection guide
├── QUERY_MATCH_SCORING.md         # Query matching documentation
├── requirements-ner.txt           # NER dependencies
└── PALEFIRE_SETUP.md             # This file
```

## Features

### 1. Episode Ingestion with NER
- Automatic entity extraction (persons, locations, organizations, dates)
- Entity-enriched content for better graph understanding
- Visual feedback during ingestion

### 2. Question-Type Aware Search
- Detects 8 question types (WHO, WHERE, WHEN, etc.)
- Automatically adjusts entity type weights
- Example: WHO questions boost person entities 2.0x

### 3. Multi-Factor Ranking
- **Semantic** (30%): RRF hybrid search
- **Connectivity** (15%): Graph connections
- **Temporal** (20%): Time period matching
- **Query Match** (20%): Term matching
- **Entity Type** (15%): Question-type alignment

### 4. Comparison Mode
Run all 5 search approaches side-by-side:
1. Standard RRF
2. Connection-based
3. Temporal-aware
4. Multi-factor
5. Question-aware (recommended)

## Customization Guide

### Change Neo4j Connection

Edit in `palefire-cli.py`:
```python
neo4j_uri = "bolt://your-server:7687"
neo4j_user = "your-username"
neo4j_password = "your-password"
```

### Change LLM Provider

Currently configured for Ollama. To use OpenAI:
```python
llm_config = LLMConfig(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model="gpt-4",
    small_model="gpt-3.5-turbo",
    base_url=None,  # Use OpenAI default
)
```

### Adjust Ranking Weights

For question-aware search:
```python
await search_episodes_with_question_aware_ranking(
    graphiti, query,
    connection_weight=0.15,      # Graph connectivity
    temporal_weight=0.20,        # Time matching
    query_match_weight=0.20,     # Term matching
    entity_type_weight=0.15      # Entity type intelligence
    # Remaining 30% = semantic relevance
)
```

### Add Custom Question Types

In `palefire-cli.py`, extend `QuestionTypeDetector.QUESTION_PATTERNS`:
```python
'CUSTOM_TYPE': {
    'patterns': [r'\byour pattern\b'],
    'entity_weights': {'PER': 1.5, 'LOC': 1.2},
    'description': 'Your custom query type'
}
```

## Example Queries

```python
# WHO questions (boosts person entities)
"Who was the California Attorney General in 2020?"
"Who is Gavin Newsom?"

# WHERE questions (boosts location entities)
"Where did Kamala Harris work as district attorney?"
"Where is the Attorney General's office located?"

# WHEN questions (boosts date entities)
"When did Gavin Newsom become governor?"
"When was Kamala Harris Attorney General?"

# WHAT questions (boosts organization/role entities)
"What position did Harris hold in 2015?"
"What organization did she lead?"
```

## Performance Tips

1. **Use spaCy for NER**: Much better accuracy than pattern-based fallback
2. **Batch Ingestion**: Process episodes in batches for large datasets
3. **Index Optimization**: Ensure Neo4j indices are built (done automatically)
4. **Adjust Weights**: Tune ranking weights based on your use case
5. **Cache Results**: Consider caching frequently accessed results

## Troubleshooting

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### Neo4j Connection Error
- Verify Neo4j is running
- Check connection details in .env
- Test connection: `neo4j://localhost:7687`

### Low Search Accuracy
- Ensure NER enrichment is enabled (spaCy installed)
- Check entity extraction during ingestion
- Adjust ranking weights
- Try question-aware search instead of standard

### Memory Issues
- Reduce `node_search_config.limit` (default: 20)
- Process episodes in smaller batches
- Use pattern-based NER instead of spaCy

## Documentation

- **RANKING_SYSTEM.md**: Complete ranking system guide
- **NER_ENRICHMENT.md**: NER system documentation
- **QUESTION_TYPE_DETECTION.md**: Question-type detection guide
- **QUERY_MATCH_SCORING.md**: Query matching details

## Next Steps

1. **Add Your Data**: Replace example episodes with your content
2. **Run Ingestion**: Set `ADD = True` and run `python palefire-cli.py`
3. **Test Queries**: Set `ADD = False` and test different query types
4. **Tune Weights**: Adjust ranking weights for your use case
5. **Monitor Performance**: Track search accuracy and speed

## Advanced Features

### Custom Entity Types
Extend `EntityEnricher.ENTITY_TYPES` for domain-specific entities

### Multi-Language Support
Add language-specific patterns to `QuestionTypeDetector`

### API Integration
Wrap search functions in FastAPI/Flask for REST API access

### Batch Processing
Process large datasets with async batch operations

### Result Caching
Implement Redis caching for frequently accessed queries

## Support

For issues or questions:
1. Check documentation files in this directory
2. Review example queries in `palefire-cli.py`
3. Examine console output for debugging information

## License

Inherits license from parent Open WebUI project.

---

**Pale Fire** - Named after Vladimir Nabokov's novel, where a poem becomes the subject of extensive commentary and interpretation, much like how this system builds a rich knowledge graph from text and enables intelligent exploration through questions.

