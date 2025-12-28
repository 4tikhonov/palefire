# Pale Fire AI Agent - Usage Guide

Complete guide on how to start, stop, and use the Pale Fire AI Agent for keyword extraction and entity recognition.

## Table of Contents

1. [Starting the Agent](#starting-the-agent)
2. [Stopping the Agent](#stopping-the-agent)
3. [Checking Status](#checking-status)
4. [Sending Queries](#sending-queries)
5. [Examples](#examples)

## Starting the Agent

### Method 1: CLI (Recommended for Development)

```bash
# Start daemon in background
python palefire-cli.py agent start --daemon

# Start in foreground (for debugging, shows logs)
python palefire-cli.py agent start

# Start without spaCy (faster startup, less accurate NER)
python palefire-cli.py agent start --no-spacy --daemon
```

### Method 2: Docker

```bash
# Standalone agent container
docker-compose -f agents/docker-compose.agent.yml up -d

# View logs
docker-compose -f agents/docker-compose.agent.yml logs -f

# Stop
docker-compose -f agents/docker-compose.agent.yml down
```

### Method 3: System Service (Production)

**Linux (systemd):**
```bash
# Copy and configure service file
sudo cp agents/palefire-agent.service /etc/systemd/system/
sudo nano /etc/systemd/system/palefire-agent.service  # Edit paths

# Enable and start
sudo systemctl enable palefire-agent
sudo systemctl start palefire-agent
```

**macOS (launchd):**
```bash
# Copy and configure plist
cp agents/palefire-agent.plist ~/Library/LaunchAgents/
nano ~/Library/LaunchAgents/palefire-agent.plist  # Edit paths

# Load service
launchctl load ~/Library/LaunchAgents/palefire-agent.plist
```

## Stopping the Agent

### CLI Method

```bash
# Stop the daemon
python palefire-cli.py agent stop

# Restart the daemon
python palefire-cli.py agent restart --daemon
```

### Docker Method

```bash
# Stop container
docker-compose -f agents/docker-compose.agent.yml stop

# Stop and remove container
docker-compose -f agents/docker-compose.agent.yml down
```

### System Service Method

**Linux:**
```bash
sudo systemctl stop palefire-agent
sudo systemctl disable palefire-agent  # Optional: disable auto-start
```

**macOS:**
```bash
launchctl unload ~/Library/LaunchAgents/palefire-agent.plist
```

## Checking Status

```bash
# Check if agent is running
python palefire-cli.py agent status

# Check Docker container status
docker ps | grep palefire-ai-agent

# Check systemd service status (Linux)
sudo systemctl status palefire-agent
```

## Sending Queries

### Method 1: Programmatic Usage (Python)

The agent keeps models loaded in memory, so queries are fast after initial startup.

#### Basic Usage

```python
from agents import get_daemon

# Get daemon instance (singleton pattern)
daemon = get_daemon(use_spacy=True)

# Initialize models (only done once, models stay loaded)
daemon.model_manager.initialize(use_spacy=True)

# Extract keywords
text = "Artificial intelligence and machine learning are transforming technology."
keywords = daemon.extract_keywords(
    text,
    num_keywords=10,
    method='tfidf'  # or 'textrank', 'word_freq', 'combined'
)

# Print results
for kw in keywords:
    print(f"{kw['keyword']}: {kw['score']:.4f}")

# Extract entities
entities = daemon.extract_entities(
    "Kamala Harris was the Attorney General of California."
)

# Print entity results
print(f"Found {entities['entity_count']} entities")
for entity_type, entity_list in entities['entities_by_type'].items():
    print(f"{entity_type}: {', '.join(entity_list)}")
```

#### Advanced Keyword Extraction

```python
from agents import get_daemon

daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)

# Extract keywords with custom parameters
keywords = daemon.extract_keywords(
    "Your text here",
    num_keywords=20,
    method='combined',  # Combines TF-IDF, TextRank, and word frequency
    enable_ngrams=True,  # Include 2-4 word phrases
    min_ngram=2,
    max_ngram=4,
    tfidf_weight=0.4,
    textrank_weight=0.3,
    word_freq_weight=0.3
)

# Results include both unigrams and n-grams
for kw in keywords:
    kw_type = kw.get('type', 'unigram')
    print(f"{kw['keyword']} ({kw_type}): {kw['score']:.4f}")
```

#### Direct Model Access

```python
from agents import ModelManager

# Create model manager directly
manager = ModelManager()
manager.initialize(use_spacy=True)

# Access models directly
extractor = manager.keyword_extractor
enricher = manager.entity_enricher

# Use extractor
keywords = extractor.extract("Your text here", num_keywords=10)

# Use enricher
episode = {
    'content': "Your text here",
    'type': 'text',
    'description': 'Entity extraction'
}
entities = enricher.enrich_episode(episode)
```

### Method 2: CLI Commands

The CLI can also use the agent for faster keyword extraction:

```bash
# Extract keywords using agent (if running)
python palefire-cli.py keywords "Your text here" \
    --method combined \
    --num-keywords 10 \
    --enable-ngrams \
    --min-ngram 2 \
    --max-ngram 4

# Extract with custom weights
python palefire-cli.py keywords "Your text here" \
    --method combined \
    --tfidf-weight 0.5 \
    --textrank-weight 0.3 \
    --word-freq-weight 0.2 \
    --position-weight 0.1
```

### Method 3: Using the Example Script

```bash
# Run the example usage script
python agents/agent_usage.py
```

This demonstrates:
- Direct daemon usage
- Multiple requests (benefiting from loaded models)
- Keyword and entity extraction

## Examples

### Example 1: Simple Keyword Extraction

```python
from agents import get_daemon

# Initialize
daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)

# Extract keywords
text = "Machine learning algorithms process data efficiently."
keywords = daemon.extract_keywords(text, num_keywords=5)

# Output
for kw in keywords:
    print(f"{kw['keyword']}: {kw['score']:.4f}")
```

**Output:**
```
machine: 0.8234
learning: 0.7654
algorithms: 0.7123
process: 0.6543
data: 0.6123
```

### Example 2: Entity Extraction

```python
from agents import get_daemon

daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)

text = "Kamala Harris was the Attorney General of California before becoming Vice President."
entities = daemon.extract_entities(text)

print(f"Found {entities['entity_count']} entities:")
for entity_type, entity_list in entities['entities_by_type'].items():
    print(f"\n{entity_type}:")
    for entity in entity_list:
        print(f"  - {entity}")
```

**Output:**
```
Found 3 entities:

PERSON:
  - Kamala Harris

ORG:
  - Attorney General

GPE:
  - California
  - Vice President
```

### Example 3: Batch Processing

```python
from agents import get_daemon

daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)

texts = [
    "Machine learning algorithms process data efficiently.",
    "Natural language processing helps computers understand human language.",
    "Deep learning neural networks can recognize patterns in images."
]

# Process multiple texts (models loaded once, very fast)
for i, text in enumerate(texts, 1):
    keywords = daemon.extract_keywords(text, num_keywords=5)
    print(f"\nText {i}:")
    print(f"  Keywords: {', '.join([kw['keyword'] for kw in keywords[:3]])}")
```

### Example 4: N-gram Extraction

```python
from agents import get_daemon

daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)

text = "Artificial intelligence and machine learning are transforming technology."

# Extract with n-grams (2-4 word phrases)
keywords = daemon.extract_keywords(
    text,
    num_keywords=15,
    method='combined',
    enable_ngrams=True,
    min_ngram=2,
    max_ngram=4,
    ngram_weight=0.3  # Weight for n-grams vs unigrams
)

# Separate unigrams and n-grams
unigrams = [kw for kw in keywords if kw.get('type') == 'unigram']
ngrams = [kw for kw in keywords if kw.get('type') != 'unigram']

print("Unigrams:")
for kw in unigrams[:5]:
    print(f"  {kw['keyword']}: {kw['score']:.4f}")

print("\nN-grams:")
for kw in ngrams[:5]:
    print(f"  {kw['keyword']} ({kw.get('type')}): {kw['score']:.4f}")
```

**Output:**
```
Unigrams:
  artificial: 0.8234
  intelligence: 0.7654
  machine: 0.7123
  learning: 0.6543
  transforming: 0.6123

N-grams:
  artificial intelligence (2-gram): 0.9123
  machine learning (2-gram): 0.8765
  transforming technology (2-gram): 0.7432
```

### Example 5: Custom Configuration

```python
from agents import get_daemon

daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)

# Extract with custom weights
keywords = daemon.extract_keywords(
    "Your text here",
    num_keywords=20,
    method='combined',
    tfidf_weight=0.5,      # Higher weight for TF-IDF
    textrank_weight=0.3,  # Medium weight for TextRank
    word_freq_weight=0.2, # Lower weight for word frequency
    position_weight=0.1,  # Boost keywords in title/first sentence
    title_weight=0.2,
    first_sentence_weight=0.15,
    enable_ngrams=True,
    min_ngram=2,
    max_ngram=3,
    ngram_weight=0.25
)
```

## Performance Benefits

The agent keeps models loaded in memory, providing:

1. **Fast First Query**: Models load once on startup
2. **Instant Subsequent Queries**: No model loading delay
3. **Efficient Batch Processing**: Process hundreds of texts quickly
4. **Memory Efficiency**: Single model instance shared across requests

### Performance Comparison

**Without Agent (models load each time):**
- First query: ~5-10 seconds (model loading)
- Subsequent queries: ~0.5-1 second

**With Agent (models stay loaded):**
- First query: ~5-10 seconds (one-time model loading)
- Subsequent queries: ~0.01-0.1 second (10-100x faster!)

## Troubleshooting

### Agent Not Starting

```bash
# Check logs
python palefire-cli.py agent status

# Check if PID file exists
ls -la /tmp/palefire_ai_agent.pid

# Check for port conflicts (if using HTTP API in future)
netstat -an | grep 8000
```

### Models Not Loading

```bash
# Verify spaCy model is installed
python -m spacy info en_core_web_sm

# Reinstall if needed
python -m spacy download en_core_web_sm

# Check gensim installation
python -c "import gensim; print(gensim.__version__)"
```

### Permission Issues

```bash
# Check PID file permissions
ls -la /tmp/palefire_ai_agent.pid

# Fix permissions if needed
chmod 666 /tmp/palefire_ai_agent.pid
```

## Best Practices

1. **Start Once**: Initialize the agent once at application startup
2. **Reuse Instance**: Use `get_daemon()` to get the singleton instance
3. **Initialize Early**: Call `initialize()` before handling requests
4. **Handle Errors**: Wrap queries in try-except blocks
5. **Monitor Memory**: Keep an eye on memory usage with loaded models

## Integration with API

The agent can be integrated into FastAPI or other web frameworks:

```python
from fastapi import FastAPI
from agents import get_daemon

app = FastAPI()

# Initialize agent on startup
daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)

@app.post("/keywords")
async def extract_keywords(text: str, num_keywords: int = 10):
    keywords = daemon.extract_keywords(text, num_keywords=num_keywords)
    return {"keywords": keywords}

@app.post("/entities")
async def extract_entities(text: str):
    entities = daemon.extract_entities(text)
    return entities
```

## See Also

- [AI Agent Daemon Documentation](AI_AGENT_DAEMON.md) - Detailed architecture and system integration
- [Docker Deployment Guide](DOCKER.md) - Running agent in Docker
- [Example Usage Script](agent_usage.py) - Working examples

