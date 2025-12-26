# Pale Fire - Quick Reference Card

## 🚀 Quick Commands

```bash
# Install dependencies
pip install -r requirements-ner.txt
python -m spacy download en_core_web_sm

# Ingest episodes (NEW CLI)
python palefire-cli.py ingest --demo
python palefire-cli.py ingest --file episodes.json

# Ask questions (NEW CLI)
python palefire-cli.py query "Who was the California Attorney General in 2020?"
python palefire-cli.py query "Where did Kamala Harris work?" --method question-aware

# Export results to JSON
python palefire-cli.py query "Your question?" --export results.json
python palefire-cli.py query "Your question?" -m standard -e output.json

# Show configuration
python palefire-cli.py config

# Clean database
python palefire-cli.py clean
python palefire-cli.py clean --confirm  # Skip confirmation

# Get help
python palefire-cli.py --help
python palefire-cli.py ingest --help
python palefire-cli.py query --help
python palefire-cli.py clean --help
```

## 📊 Search Methods Comparison

| Method | Use When | Accuracy | Speed |
|--------|----------|----------|-------|
| **Standard** | Simple queries | ⭐⭐ | ⚡⚡⚡ |
| **Connection-based** | Find central entities | ⭐⭐⭐ | ⚡⚡ |
| **Temporal-aware** | Date-specific queries | ⭐⭐⭐⭐ | ⚡⚡ |
| **Multi-factor** | Complex queries | ⭐⭐⭐⭐ | ⚡ |
| **Question-aware** | Natural questions | ⭐⭐⭐⭐⭐ | ⚡ |

## 🎯 Question Types

| Question Word | Boosts | Example |
|---------------|--------|---------|
| **WHO** | PER (2.0x) | "Who was the AG?" |
| **WHERE** | LOC (2.0x) | "Where did she work?" |
| **WHEN** | DATE (2.0x) | "When was he governor?" |
| **WHAT** (org) | ORG (2.0x) | "What organization?" |
| **WHAT** (position) | PER/ORG (1.5x) | "What position?" |
| **HOW MANY** | CARDINAL (2.0x) | "How many years?" |
| **WHY** | EVENT (1.5x) | "Why did she leave?" |
| **WHAT** (event) | EVENT (2.0x) | "What happened?" |

## 🏷️ Entity Types

| Type | Tag | Examples |
|------|-----|----------|
| Person | PER | Kamala Harris, Gavin Newsom |
| Location | LOC | California, San Francisco |
| Organization | ORG | Attorney General, FBI |
| Date | DATE | January 3, 2011, 2020 |
| Time | TIME | 3:00 PM, morning |
| Money | MONEY | $1 million |
| Percent | PERCENT | 50% |
| Event | EVENT | World War II |

## ⚙️ Weight Tuning

### Recommended Presets

**Balanced (Default)**
```python
connection_weight=0.15
temporal_weight=0.20
query_match_weight=0.20
entity_type_weight=0.15
# Semantic: 30%
```

**Temporal Focus** (for date-heavy queries)
```python
connection_weight=0.10
temporal_weight=0.30
query_match_weight=0.20
entity_type_weight=0.15
# Semantic: 25%
```

**Entity Focus** (for WHO/WHERE queries)
```python
connection_weight=0.15
temporal_weight=0.15
query_match_weight=0.20
entity_type_weight=0.25
# Semantic: 25%
```

**Connection Focus** (for relationship queries)
```python
connection_weight=0.25
temporal_weight=0.15
query_match_weight=0.20
entity_type_weight=0.10
# Semantic: 30%
```

## 📝 Code Snippets

### Basic Search
```python
await search_episodes_with_question_aware_ranking(
    graphiti,
    "Who was the California Attorney General in 2020?"
)
```

### Custom Weights
```python
await search_episodes_with_question_aware_ranking(
    graphiti,
    query,
    connection_weight=0.20,
    temporal_weight=0.25,
    query_match_weight=0.15,
    entity_type_weight=0.20
)
```

### Detect Question Type
```python
detector = QuestionTypeDetector()
info = detector.detect_question_type(query)
print(f"Type: {info['type']}")
print(f"Weights: {info['entity_weights']}")
```

### Extract Entities
```python
enricher = EntityEnricher(use_spacy=True)
enriched = enricher.enrich_episode(episode)
print(f"Entities: {enriched['entities_by_type']}")
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
NEO4J_URI=bolt://10.147.18.253:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=your_key_here
```

### Ollama Configuration (in code)
```python
llm_config = LLMConfig(
    api_key="ollama",
    model="deepseek-r1:7b",
    base_url="http://10.147.18.253:11434/v1"
)
```

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| spaCy not found | `pip install spacy` |
| Model not found | `python -m spacy download en_core_web_sm` |
| Neo4j connection error | Check Neo4j is running, verify .env |
| Low accuracy | Enable spaCy, use question-aware search |
| Slow performance | Reduce `node_search_config.limit` |

## 📈 Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Question detection | 1-5ms | Regex-based |
| Entity extraction (spaCy) | 50-500ms | Per node |
| Entity extraction (pattern) | 10-50ms | Per node |
| Standard search | 100-300ms | RRF only |
| Question-aware search | 500-2000ms | All factors |

## 💡 Best Practices

1. ✅ **Use spaCy** for production (better accuracy)
2. ✅ **Enable NER enrichment** during ingestion
3. ✅ **Use question-aware search** for natural queries
4. ✅ **Tune weights** based on your domain
5. ✅ **Monitor entity extraction** quality
6. ✅ **Test with diverse queries** before deployment

## 📚 Documentation Files

- `PALEFIRE_SETUP.md` - Complete setup guide
- `RANKING_SYSTEM.md` - Ranking system details
- `NER_ENRICHMENT.md` - NER documentation
- `QUESTION_TYPE_DETECTION.md` - Question-type guide
- `QUERY_MATCH_SCORING.md` - Query matching details

## 🎓 Example Queries by Type

```python
# WHO - Returns people
"Who was the California Attorney General in 2020?"
"Who succeeded Kamala Harris?"

# WHERE - Returns locations
"Where did Kamala Harris work as DA?"
"Where is the AG office located?"

# WHEN - Returns dates
"When did Gavin Newsom become governor?"
"When was Harris Attorney General?"

# WHAT (position) - Returns roles
"What position did Harris hold?"
"What role did Newsom have?"

# WHAT (organization) - Returns orgs
"What organization did she lead?"
"What agency was he part of?"

# HOW MANY - Returns numbers
"How many years was she AG?"
"How many terms did he serve?"
```

## 🔄 Workflow

```
1. Prepare Data
   └─> Edit episodes in palefire-cli.py

2. Ingest with NER
   └─> Set ADD = True
   └─> Run python palefire-cli.py
   └─> Verify entity extraction

3. Test Queries
   └─> Set ADD = False
   └─> Edit query in palefire-cli.py
   └─> Run python palefire-cli.py
   └─> Compare 5 search methods

4. Tune & Deploy
   └─> Adjust weights
   └─> Choose best method
   └─> Integrate into application
```

---

**Quick Tip**: Start with question-aware search (method 5) - it automatically handles most query types intelligently!

