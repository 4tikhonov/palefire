# Pale Fire Configuration Guide

## Overview

Pale Fire uses a centralized configuration system managed through `config.py`. All settings can be customized via environment variables in a `.env` file.

## Configuration File

All configuration is defined in `config.py`, which:
- Loads environment variables from `.env` file
- Provides sensible defaults for all settings
- Validates configuration on startup
- Offers helper functions for accessing configuration

## Environment Variables

### Neo4j Configuration (REQUIRED)

```bash
NEO4J_URI=bolt://10.147.18.253:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

These settings are **required** for Pale Fire to connect to your Neo4j database.

### LLM Configuration

#### Provider Selection

```bash
# Choose 'ollama' or 'openai'
LLM_PROVIDER=ollama
```

#### OpenAI API Key

```bash
# Required by Graphiti (can be placeholder for Ollama)
OPENAI_API_KEY=your-api-key-here
```

#### Ollama Settings

```bash
OLLAMA_BASE_URL=http://10.147.18.253:11434/v1
OLLAMA_MODEL=deepseek-r1:7b
OLLAMA_SMALL_MODEL=deepseek-r1:7b
OLLAMA_API_KEY=ollama  # Placeholder
```

#### OpenAI Settings

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
OPENAI_SMALL_MODEL=gpt-3.5-turbo
```

### Embedder Configuration

#### Provider Selection

```bash
# Choose 'ollama' or 'openai'
EMBEDDER_PROVIDER=ollama
```

#### Ollama Embedder

```bash
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_DIM=768
OLLAMA_EMBEDDING_BASE_URL=http://10.147.18.253:11434/v1
```

#### OpenAI Embedder

```bash
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_EMBEDDING_DIM=1536
```

### Search Configuration

#### Search Method

```bash
# Default search method: 'standard', 'connection', 'question-aware'
DEFAULT_SEARCH_METHOD=question-aware
```

Available methods:
- **standard**: Basic semantic search
- **connection**: Adds connection-based ranking
- **question-aware**: Full 5-factor ranking with question-type detection (recommended)

#### Search Limits

```bash
# Number of results to retrieve before reranking
SEARCH_RESULT_LIMIT=20

# Number of top results to return to user
SEARCH_TOP_K=5
```

### Ranking Weights

Pale Fire uses a multi-factor ranking system. Configure the weight of each factor:

```bash
WEIGHT_CONNECTION=0.15    # Node connectivity importance
WEIGHT_TEMPORAL=0.20      # Time-based relevance
WEIGHT_QUERY_MATCH=0.20   # Query term matching
WEIGHT_ENTITY_TYPE=0.15   # Entity type relevance
```

**Important**: The sum of all weights must be ≤ 1.0. The remaining weight is automatically assigned to semantic similarity.

Example calculation:
```
Semantic weight = 1.0 - (0.15 + 0.20 + 0.20 + 0.15) = 0.30
```

#### Weight Tuning Guidelines

- **WEIGHT_CONNECTION** (0.0-0.3): Higher values favor well-connected entities (hubs)
- **WEIGHT_TEMPORAL** (0.0-0.3): Higher values favor entities with matching time periods
- **WEIGHT_QUERY_MATCH** (0.0-0.4): Higher values favor exact term matches
- **WEIGHT_ENTITY_TYPE** (0.0-0.3): Higher values favor entities matching question type
- **Semantic** (remaining): Automatically calculated, represents embedding similarity

### NER Configuration

```bash
# Enable/disable NER enrichment
NER_ENABLED=true

# Use spaCy (more accurate) or pattern-based (faster) NER
NER_USE_SPACY=true

# spaCy model to use
SPACY_MODEL=en_core_web_sm
```

NER (Named Entity Recognition) enriches episodes with entity types:
- **PER**: Person names
- **LOC**: Locations
- **ORG**: Organizations
- **DATE**: Dates and time periods
- **GPE**: Geopolitical entities
- **POSITION**: Job titles/roles

### Logging Configuration

```bash
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_DATE_FORMAT=%Y-%m-%d %H:%M:%S
```

Available log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Application Configuration

```bash
# Prefix for episode names
EPISODE_NAME_PREFIX=Episode

# Use current time for episode timestamps
USE_CURRENT_TIME=true
```

## Configuration Functions

The `config.py` module provides helper functions:

### `validate_config()`

Validates all required settings and raises `ValueError` if any are missing or invalid.

```python
try:
    config.validate_config()
except ValueError as e:
    print(f"Configuration error: {e}")
```

### `get_llm_config()`

Returns LLM configuration based on the selected provider:

```python
llm_cfg = config.get_llm_config()
# Returns: {'api_key': '...', 'model': '...', 'small_model': '...', 'base_url': '...'}
```

### `get_embedder_config()`

Returns embedder configuration based on the selected provider:

```python
emb_cfg = config.get_embedder_config()
# Returns: {'api_key': '...', 'embedding_model': '...', 'embedding_dim': 768, 'base_url': '...'}
```

### `print_config()`

Displays the current configuration in a formatted table:

```bash
python palefire-cli.py config
```

Output:
```
================================================================================
⚙️  PALE FIRE CONFIGURATION
================================================================================
Neo4j URI: bolt://10.147.18.253:7687
Neo4j User: neo4j
LLM Provider: ollama
LLM Model: deepseek-r1:7b
LLM Base URL: http://10.147.18.253:11434/v1
Embedder Provider: ollama
Embedder Model: nomic-embed-text
Embedder Dimensions: 768

Search Configuration:
  Default Method: question-aware
  Result Limit: 20
  Top K: 5

Ranking Weights:
  Connection: 0.15
  Temporal: 0.20
  Query Match: 0.20
  Entity Type: 0.15
  Semantic: 0.30
================================================================================
```

## Setup Instructions

### 1. Copy Example Configuration

```bash
cd /path/to/palefire
cp env.example .env
```

### 2. Edit Configuration

```bash
nano .env  # or your preferred editor
```

### 3. Verify Configuration

```bash
python palefire-cli.py config
```

### 4. Test with Query

```bash
python palefire-cli.py query "Who was the California Attorney General in 2020?"
```

## Configuration Best Practices

### For Production

```bash
# Use specific, stable models
OLLAMA_MODEL=llama2:13b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Conservative search limits
SEARCH_RESULT_LIMIT=30
SEARCH_TOP_K=10

# Balanced weights
WEIGHT_CONNECTION=0.15
WEIGHT_TEMPORAL=0.20
WEIGHT_QUERY_MATCH=0.20
WEIGHT_ENTITY_TYPE=0.15

# Enable NER for better accuracy
NER_ENABLED=true
NER_USE_SPACY=true

# Production logging
LOG_LEVEL=INFO
```

### For Development

```bash
# Faster models
OLLAMA_MODEL=deepseek-r1:7b
OLLAMA_SMALL_MODEL=deepseek-r1:7b

# Smaller search limits for speed
SEARCH_RESULT_LIMIT=10
SEARCH_TOP_K=3

# Debug logging
LOG_LEVEL=DEBUG

# Disable NER for faster iteration
NER_ENABLED=false
```

### For Research/Experimentation

```bash
# Larger search space
SEARCH_RESULT_LIMIT=50
SEARCH_TOP_K=20

# Experiment with weights
WEIGHT_CONNECTION=0.10
WEIGHT_TEMPORAL=0.15
WEIGHT_QUERY_MATCH=0.25
WEIGHT_ENTITY_TYPE=0.10
# Semantic = 0.40 (remaining)

# Enable detailed logging
LOG_LEVEL=DEBUG
```

## Troubleshooting

### Configuration Not Loading

If your `.env` file isn't being read:

1. Check file location (must be in palefire directory)
2. Verify file name is exactly `.env` (not `env.txt` or `.env.example`)
3. Check file permissions: `chmod 600 .env`

### Invalid Configuration

If you get configuration errors:

```bash
# Check current config
python palefire-cli.py config

# Validate weights sum
python -c "import config; config.validate_config()"
```

### Connection Issues

If Neo4j connection fails:

1. Verify Neo4j is running: `neo4j status`
2. Check URI format: `bolt://host:port` or `neo4j://host:port`
3. Test credentials: `cypher-shell -u neo4j -p password`

### LLM/Embedder Issues

If LLM or embedder fails:

1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check model is downloaded: `ollama list`
3. Test API endpoint: `curl http://localhost:11434/v1/models`

## Advanced Configuration

### Custom Configuration Module

You can extend `config.py` with your own settings:

```python
# In config.py
CUSTOM_SETTING = os.environ.get('CUSTOM_SETTING', 'default_value')

def get_custom_config():
    return {
        'setting1': CUSTOM_SETTING,
        'setting2': 'value2'
    }
```

### Runtime Configuration Override

Override configuration at runtime:

```python
import config

# Override for this session
config.SEARCH_TOP_K = 10
config.WEIGHT_CONNECTION = 0.25
```

### Multiple Environments

Use different `.env` files for different environments:

```bash
# Development
cp env.example .env.dev
# Edit .env.dev

# Production
cp env.example .env.prod
# Edit .env.prod

# Load specific environment
ln -sf .env.dev .env  # or .env.prod
```

## See Also

- [CLI Guide](CLI_GUIDE.md) - Command-line interface documentation
- [Setup Guide](PALEFIRE_SETUP.md) - Installation and setup instructions
- [Quick Reference](QUICK_REFERENCE.md) - Quick command reference

