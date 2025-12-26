# Pale Fire - Intelligent Knowledge Graph Search System

> Named after Vladimir Nabokov's novel, where a poem becomes the subject of extensive commentary and interpretation—just like how this system builds a rich knowledge graph from text and enables intelligent exploration through questions.

## Overview

Pale Fire is an advanced knowledge graph search system featuring:

- **🧠 Question-Type Detection** - Automatically understands WHO/WHERE/WHEN/WHAT/WHY/HOW questions
- **🏷️ NER Enrichment** - Extracts and tags 18+ entity types (PER, LOC, ORG, DATE, etc.)
- **📊 5-Factor Ranking** - Combines semantic, connectivity, temporal, query matching, and entity-type intelligence
- **⚡ CLI Interface** - Easy-to-use command-line interface for ingestion and queries
- **🔧 Modular Architecture** - Clean separation of concerns for maintainability

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements-ner.txt
python -m spacy download en_core_web_sm

# 2. Configure environment
cp env.example .env  # Edit with your settings

# 3. View configuration
python palefire-cli.py config

# 4. Ingest demo data
python palefire-cli.py ingest --demo

# 5. Ask a question
python palefire-cli.py query "Who was the California Attorney General in 2020?"
```

## Features

### Intelligent Question Detection

Automatically detects 8 question types and adjusts entity weights:

```bash
# WHO questions → boost person entities 2.0x
python palefire-cli.py query "Who was the Attorney General?"

# WHERE questions → boost location entities 2.0x
python palefire-cli.py query "Where did Kamala Harris work?"

# WHEN questions → boost date entities 2.0x
python palefire-cli.py query "When did Gavin Newsom become governor?"
```

### NER-Enriched Ingestion

Extract entities automatically during ingestion:

```bash
# With NER enrichment (recommended)
python palefire-cli.py ingest --file episodes.json

# Without NER (faster)
python palefire-cli.py ingest --file episodes.json --no-ner
```

### Multiple Search Methods

Choose the best method for your query:

```bash
# Question-aware (recommended for natural questions)
python palefire-cli.py query "Who is Gavin Newsom?" -m question-aware

# Connection-based (for finding central entities)
python palefire-cli.py query "Important people" -m connection

# Standard (fastest, basic RRF)
python palefire-cli.py query "California" -m standard
```

## CLI Commands

### Ingest Episodes

```bash
# From file
python palefire-cli.py ingest --file episodes.json

# Demo data
python palefire-cli.py ingest --demo

# Without NER
python palefire-cli.py ingest --file episodes.json --no-ner
```

### Query Knowledge Graph

```bash
# Basic query
python palefire-cli.py query "Your question here?"

# With specific method
python palefire-cli.py query "Your question?" --method question-aware

# Export results to JSON
python palefire-cli.py query "Your question?" --export results.json

# Combine method and export
python palefire-cli.py query "Who is X?" -m standard -e output.json

# Short form
python palefire-cli.py query "Who is X?" -m standard
```

### Show Configuration

```bash
python palefire-cli.py config
```

### Clean Database

```bash
# Clean database (with confirmation prompt)
python palefire-cli.py clean

# Clean without confirmation
python palefire-cli.py clean --confirm

# Delete only nodes (keep database structure)
python palefire-cli.py clean --nodes-only
```

### Get Help

```bash
python palefire-cli.py --help
python palefire-cli.py ingest --help
python palefire-cli.py query --help
```

## Episode File Format

Create a JSON file with your episodes:

```json
[
    {
        "content": "Kamala Harris is the Attorney General of California.",
        "type": "text",
        "description": "Biography"
    },
    {
        "content": {
            "name": "Gavin Newsom",
            "position": "Governor",
            "state": "California"
        },
        "type": "json",
        "description": "Structured data"
    }
]
```

See `example_episodes.json` for a complete example.

## Architecture

```
palefire/
├── palefire-cli.py              # Main CLI application
├── modules/                     # Core modules
│   ├── __init__.py
│   └── PaleFireCore.py         # EntityEnricher + QuestionTypeDetector
├── example_episodes.json        # Example data
├── CLI_GUIDE.md                # Complete CLI documentation
├── QUICK_REFERENCE.md          # Quick reference card
├── ARCHITECTURE.md             # Architecture details
└── [other documentation]
```

## 5-Factor Ranking System

Pale Fire combines 5 independent factors for optimal search results:

1. **Semantic Relevance** (30%) - RRF hybrid search (vector + keyword)
2. **Connectivity** (15%) - How well-connected in the knowledge graph
3. **Temporal Match** (20%) - Active during query time period
4. **Query Term Match** (20%) - Explicit matches of query terms
5. **Entity Type Match** (15%) - Entity types relevant to question type

## Question Types

| Type | Pattern | Boosts | Example |
|------|---------|--------|---------|
| WHO | who, whom, whose | PER (2.0x) | "Who was the AG?" |
| WHERE | where, which place | LOC (2.0x) | "Where did she work?" |
| WHEN | when, what year | DATE (2.0x) | "When was he governor?" |
| WHAT (org) | what organization | ORG (2.0x) | "What organization?" |
| WHAT (position) | what position | PER/ORG (1.5x) | "What position?" |
| HOW MANY | how many | CARDINAL (2.0x) | "How many years?" |
| WHY | why | EVENT (1.5x) | "Why did she leave?" |
| WHAT (event) | what happened | EVENT (2.0x) | "What happened?" |

## Entity Types

Automatically extracted with NER:

- **PER** - Persons (Kamala Harris, Gavin Newsom)
- **LOC** - Locations (California, San Francisco)
- **ORG** - Organizations (Attorney General, FBI)
- **DATE** - Dates (January 3, 2011, 2020)
- **TIME** - Times (3:00 PM, morning)
- **MONEY** - Money ($1 million)
- **PERCENT** - Percentages (50%)
- **EVENT** - Events (World War II)
- Plus 10 more types

## Configuration

All configuration is centralized in `config.py` and loaded from `.env`:

```bash
# Copy example configuration
cp env.example .env

# Edit with your settings
nano .env

# View current configuration
python palefire-cli.py config
```

Key settings:

```bash
# Neo4j (required)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Provider
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=deepseek-r1:7b

# Search Configuration
DEFAULT_SEARCH_METHOD=question-aware
SEARCH_RESULT_LIMIT=20
SEARCH_TOP_K=5

# Ranking Weights (must sum to ≤ 1.0)
WEIGHT_CONNECTION=0.15
WEIGHT_TEMPORAL=0.20
WEIGHT_QUERY_MATCH=0.20
WEIGHT_ENTITY_TYPE=0.15
```

See [CONFIGURATION.md](CONFIGURATION.md) for complete documentation.

## Examples

### Example 1: Political Queries

```bash
# Ingest
python palefire-cli.py ingest --demo

# Query
python palefire-cli.py query "Who was the California Attorney General in 2020?"
python palefire-cli.py query "Where did Kamala Harris work as DA?"
python palefire-cli.py query "When did Gavin Newsom become governor?"
```

### Example 2: Custom Data

```bash
# Create your data file
cat > my_data.json << 'EOF'
[
    {
        "content": "Your content here...",
        "type": "text",
        "description": "Your description"
    }
]
EOF

# Ingest and query
python palefire-cli.py ingest --file my_data.json
python palefire-cli.py query "Your question?"
```

### Example 3: Batch Processing

```bash
# Ingest multiple files
for file in data/*.json; do
    python palefire-cli.py ingest --file "$file"
done

# Run multiple queries
python palefire-cli.py query "Question 1?"
python palefire-cli.py query "Question 2?"
```

## Documentation

- **[CONFIGURATION.md](CONFIGURATION.md)** - Complete configuration guide
- **[CLI_GUIDE.md](CLI_GUIDE.md)** - Complete CLI documentation
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference card
- **[PALEFIRE_SETUP.md](PALEFIRE_SETUP.md)** - Setup instructions
- **ARCHITECTURE.md** - Architecture details (if exists)
- **RANKING_SYSTEM.md** - Ranking system documentation (if exists)
- **NER_ENRICHMENT.md** - NER system guide (if exists)
- **QUESTION_TYPE_DETECTION.md** - Question-type detection guide (if exists)
- **QUERY_MATCH_SCORING.md** - Query matching details (if exists)

## Requirements

### Core
- Python 3.8+
- graphiti-core
- python-dotenv
- Neo4j database

### NER (Optional but Recommended)
- spacy
- en_core_web_sm model

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Question detection | 1-5ms | Regex-based |
| Entity extraction (spaCy) | 50-500ms | Per node |
| Entity extraction (pattern) | 10-50ms | Per node |
| Standard search | 100-300ms | RRF only |
| Question-aware search | 500-2000ms | All factors |

## Troubleshooting

### Module not found
```bash
cd /path/to/palefire
python palefire-cli.py --help
```

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Neo4j connection error
```bash
# Check Neo4j is running
# Verify credentials in .env
```

## Best Practices

1. ✅ Use NER enrichment for production
2. ✅ Use question-aware search for natural questions
3. ✅ Batch process large datasets
4. ✅ Monitor logs for errors
5. ✅ Backup Neo4j database regularly

## Future Enhancements

- [ ] REST API wrapper
- [ ] Web UI
- [ ] Result caching
- [ ] Multi-language support
- [ ] Custom entity types
- [ ] ML-based question detection

## Contributing

When adding features:
1. Add classes to `modules/PaleFireCore.py`
2. Add functions to `palefire-cli.py`
3. Update documentation
4. Test thoroughly

## License

Inherits license from parent Open WebUI project.

## Support

For issues or questions:
1. Check documentation files
2. Review CLI_GUIDE.md
3. Check logs for error messages
4. Verify environment configuration

---

**Pale Fire** - Intelligent Knowledge Graph Search Made Easy 🚀
