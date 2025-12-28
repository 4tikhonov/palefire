# Pale Fire - Architecture Overview

## Project Structure

```
palefire/
├── palefire-cli.py              # Main CLI application
├── modules/                     # Core modules
│   ├── __init__.py             # Module exports
│   ├── PaleFireCore.py         # Entity enrichment & question detection
│   ├── KeywordBase.py          # Keyword extraction (Gensim)
│   └── api_models.py           # Pydantic models for API
├── agents/                      # AI Agent daemon and parsers
│   ├── __init__.py             # Agent module exports
│   ├── AIAgent.py              # ModelManager, AIAgentDaemon
│   ├── palefire-agent-service.py  # Service script
│   ├── parsers/                 # File parsers
│   │   ├── __init__.py         # Parser registry
│   │   ├── base_parser.py      # Base parser class
│   │   ├── txt_parser.py       # Text file parser
│   │   ├── csv_parser.py       # CSV parser
│   │   ├── pdf_parser.py       # PDF parser
│   │   └── spreadsheet_parser.py  # Excel/ODS parser
│   ├── docker-compose.agent.yml  # Docker compose for agent
│   ├── Dockerfile.agent        # Dockerfile for agent
│   └── DOCKER.md               # Docker documentation
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
5. **Agent Management** - Start/stop/status daemon commands
6. **File Parsing** - Parse various file formats (TXT, CSV, PDF, Spreadsheet)
7. **Keyword Extraction** - Extract keywords with optional daemon integration

### `agents/AIAgent.py`

Contains the AI Agent daemon system for keeping models loaded in memory:

#### **ModelManager**
- Thread-safe manager for loaded models
- Keeps KeywordExtractor (Gensim) and EntityEnricher (spaCy) in memory
- Provides singleton access to models
- Handles model initialization and reloading

**Key Methods:**
- `initialize(use_spacy=True)` - Load all models into memory
- `keyword_extractor` - Get KeywordExtractor instance
- `entity_enricher` - Get EntityEnricher instance
- `is_initialized()` - Check if models are loaded
- `reload()` - Reload all models

#### **AIAgentDaemon**
- Long-running daemon service
- Keeps models loaded to avoid startup delays
- Provides fast keyword and entity extraction
- Handles graceful shutdown with signal handlers

**Key Methods:**
- `start(daemon=False)` - Start daemon (foreground or background)
- `stop()` - Stop daemon gracefully
- `extract_keywords(text, **kwargs)` - Extract keywords using loaded models
- `extract_entities(text)` - Extract entities using loaded models
- `parse_file(file_path, **kwargs)` - Parse file and extract text
- `get_status()` - Get daemon status and capabilities

#### **get_daemon()**
- Singleton function to get or create daemon instance
- Ensures only one daemon instance exists
- Thread-safe access

### `agents/parsers/`

File parsing system for extracting text from various formats:

#### **BaseParser** (Abstract Base Class)
- Defines interface for all parsers
- Provides file validation utilities
- Standardizes ParseResult format

**Key Methods:**
- `parse(file_path, **kwargs)` - Parse file and extract text
- `get_supported_extensions()` - List supported file types
- `validate_file(file_path)` - Check if file is valid

#### **TXTParser**
- Parses plain text files (.txt)
- Supports custom encoding
- Splits text into pages/chunks
- Extracts metadata (line count, file size)

**Supported:** `.txt`, `.text`

#### **CSVParser**
- Parses CSV files (.csv)
- Auto-detects delimiter
- Extracts table structure
- Supports header row handling

**Supported:** `.csv`

#### **PDFParser**
- Parses PDF files (.pdf)
- Supports PyPDF2 and pdfplumber
- Extracts text page-by-page
- Extracts tables (with pdfplumber)
- Preserves document metadata

**Supported:** `.pdf`

#### **SpreadsheetParser**
- Parses spreadsheet files (.xlsx, .xls, .ods)
- Supports multiple sheets
- Extracts table structure
- Handles headers and data rows

**Supported:** `.xlsx`, `.xls`, `.xlsm`, `.ods`

#### **Parser Registry**
- Automatic parser selection based on file extension
- Factory function `get_parser(file_path)`
- Extensible for new file types

## Import Structure

```python
# Core modules
from modules import EntityEnricher, QuestionTypeDetector, KeywordExtractor

# AI Agent
from agents import ModelManager, AIAgentDaemon, get_daemon

# File parsers
from agents.parsers import TXTParser, CSVParser, PDFParser, SpreadsheetParser, get_parser

# Usage
enricher = EntityEnricher(use_spacy=True)
detector = QuestionTypeDetector()

# Agent usage
daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)
keywords = daemon.extract_keywords("text")
entities = daemon.extract_entities("text")

# Parser usage
parser = get_parser('document.pdf')
result = parser.parse('document.pdf')
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

### AI Agent Daemon Lifecycle
```
1. Start daemon
   ├─ Create AIAgentDaemon instance
   ├─ Initialize ModelManager
   │   ├─ Load KeywordExtractor (Gensim)
   │   └─ Load EntityEnricher (spaCy)
   └─ Fork to background (if daemon=True)
   ↓
2. Daemon running
   ├─ Models stay loaded in memory
   ├─ Health check loop (every 60s)
   └─ Ready for requests
   ↓
3. Process requests
   ├─ extract_keywords() - Fast (models loaded)
   ├─ extract_entities() - Fast (models loaded)
   └─ parse_file() - Uses appropriate parser
   ↓
4. Stop daemon
   ├─ Receive SIGTERM/SIGINT
   ├─ Cleanup models
   └─ Remove PID file
```

### File Parsing Flow
```
1. User provides file path
   ↓
2. Parser registry selects parser
   ├─ Check file extension
   └─ Instantiate appropriate parser
   ↓
3. Parser validates file
   ├─ Check file exists
   ├─ Check file is readable
   └─ Check file size > 0
   ↓
4. Parse file
   ├─ Extract text content
   ├─ Extract metadata
   ├─ Extract tables (if applicable)
   └─ Split into pages (if multi-page)
   ↓
5. Return ParseResult
   ├─ text: Full extracted text
   ├─ metadata: File information
   ├─ pages: Page-by-page text (optional)
   ├─ tables: Extracted tables (optional)
   └─ success: Boolean status
   ↓
6. Optional: Extract keywords
   ├─ Use daemon (if running)
   └─ Extract keywords from parsed text
```

### Keyword Extraction with Agent
```
1. Check if daemon is running
   ├─ Read PID file
   ├─ Check process exists
   └─ Start daemon if not running
   ↓
2. Get daemon instance
   ├─ Singleton pattern
   └─ Ensure models initialized
   ↓
3. Extract keywords
   ├─ Use loaded KeywordExtractor
   ├─ Apply TF-IDF, TextRank, Word Frequency
   ├─ Extract n-grams (2-4 words)
   └─ Combine and rank results
   ↓
4. Return keywords
   └─ List of {keyword, score, type}
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

### Keyword Extraction
- `gensim>=4.3.0` - TF-IDF, TextRank algorithms
- `nltk` (optional) - Better stemming support

### AI Agent
- `psutil>=5.9.0` - System monitoring and process management

### File Parsing (Optional)
- `PyPDF2>=3.0.0` - PDF parsing (or `pdfplumber>=0.9.0` for better table extraction)
- `openpyxl>=3.1.0` - Excel .xlsx files
- `xlrd>=2.0.0` - Excel .xls files
- `odfpy>=1.4.0` - OpenDocument Spreadsheet (.ods) files

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

### Adding Custom File Parsers

1. Create parser class in `agents/parsers/`:
```python
from .base_parser import BaseParser, ParseResult

class CustomParser(BaseParser):
    def parse(self, file_path: str, **kwargs) -> ParseResult:
        # Your parsing logic
        text = extract_text(file_path)
        return ParseResult(text=text, metadata={...})
    
    def get_supported_extensions(self) -> List[str]:
        return ['.custom']
```

2. Register in `agents/parsers/__init__.py`:
```python
from .custom_parser import CustomParser

PARSERS = {
    ...
    '.custom': CustomParser,
}
```

### Extending Agent Functionality

Add methods to `AIAgentDaemon` class:
```python
class AIAgentDaemon:
    def custom_operation(self, data):
        """Custom operation using loaded models."""
        extractor = self.model_manager.keyword_extractor
        # Use extractor for custom logic
        return result
```

## Performance Considerations

### Memory Usage
- **spaCy**: ~200-500 MB
- **Pattern-based**: ~10-20 MB
- **Neo4j driver**: ~50-100 MB
- **Gensim (KeywordExtractor)**: ~100-300 MB
- **AI Agent Daemon**: ~300-800 MB total (with all models loaded)

### Speed

#### Without Agent (models load each time)
- **Model loading**: 5-10 seconds (one-time)
- **Keyword extraction**: 0.5-1 second per request
- **Entity extraction (spaCy)**: 50-500ms per node
- **Entity extraction (pattern)**: 10-50ms per node

#### With Agent (models stay loaded)
- **Model loading**: 5-10 seconds (one-time on startup)
- **Keyword extraction**: 0.01-0.1 second per request (10-100x faster!)
- **Entity extraction**: Same as above (models already loaded)
- **File parsing**: Varies by file type and size

#### Search Operations
- **Question detection**: 1-5ms
- **Standard search**: 100-300ms
- **Question-aware search**: 500-2000ms

### Optimization Tips
1. **Use AI Agent Daemon** for production - eliminates model loading delays
2. Use spaCy for better accuracy
3. Reduce `node_search_config.limit` for faster searches
4. Cache enriched episodes
5. Batch process large datasets
6. Use connection pooling for Neo4j
7. **Keep daemon running** - models stay loaded, requests are instant
8. **Parse files once** - reuse parsed text for multiple operations
9. **Use appropriate parsers** - PDF parsers vary in speed (pdfplumber slower but better)

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

## AI Agent Architecture

### ModelManager

**Purpose**: Thread-safe manager for keeping ML models loaded in memory.

**Architecture**:
```
ModelManager
├── Thread-safe lock (RLock)
├── KeywordExtractor (Gensim)
│   ├── TF-IDF models
│   ├── TextRank models
│   └── Word frequency counters
├── EntityEnricher (spaCy/Pattern)
│   ├── spaCy model (if available)
│   └── Pattern matchers
└── Initialization state
```

**Thread Safety**: Uses `threading.RLock()` to ensure safe concurrent access.

**Lifecycle**:
1. Create instance: `manager = ModelManager()`
2. Initialize: `manager.initialize(use_spacy=True)`
3. Access models: `manager.keyword_extractor`, `manager.entity_enricher`
4. Reload (optional): `manager.reload()`

### AIAgentDaemon

**Purpose**: Long-running service that keeps models loaded for fast access.

**Architecture**:
```
AIAgentDaemon
├── ModelManager (manages loaded models)
├── Process management
│   ├── PID file handling
│   ├── Signal handlers (SIGTERM, SIGINT)
│   └── Health check loop
├── Operations
│   ├── extract_keywords()
│   ├── extract_entities()
│   └── parse_file()
└── Status tracking
    ├── Running state
    └── Model initialization state
```

**Deployment Options**:
1. **CLI**: `python palefire-cli.py agent start --daemon`
2. **Docker**: `docker-compose -f agents/docker-compose.agent.yml up -d`
3. **Systemd**: Service file for Linux
4. **Launchd**: Plist file for macOS

**Communication**:
- Currently: Direct Python import (singleton pattern)
- Future: Socket/HTTP API for remote access

### File Parser Architecture

**Design Pattern**: Strategy pattern with factory

**Architecture**:
```
Parser System
├── BaseParser (Abstract)
│   ├── parse() - Abstract method
│   ├── get_supported_extensions() - Abstract method
│   └── validate_file() - Concrete utility
├── Concrete Parsers
│   ├── TXTParser
│   ├── CSVParser
│   ├── PDFParser
│   └── SpreadsheetParser
└── Parser Registry
    └── get_parser() - Factory function
```

**ParseResult Structure**:
```python
{
    'text': str,              # Full extracted text
    'metadata': dict,         # File metadata
    'pages': List[str],       # Page-by-page text (optional)
    'tables': List[dict],     # Extracted tables (optional)
    'success': bool,          # Parsing status
    'error': str | None       # Error message if failed
}
```

**Parser Selection**:
1. Extract file extension
2. Lookup in `PARSERS` registry
3. Instantiate appropriate parser
4. Return parser instance

**Error Handling**:
- Invalid file: Returns `ParseResult` with `success=False` and error message
- Missing dependencies: Parser logs warning, returns error
- Encoding issues: TXT parser handles with `errors='replace'`

## Integration Points

### CLI Integration

**Agent Commands**:
- `agent start` - Start daemon
- `agent stop` - Stop daemon
- `agent restart` - Restart daemon
- `agent status` - Check daemon status

**Parse Commands**:
- `parse <file>` - Auto-detect and parse file
- `parse-txt <file>` - Parse text file
- `parse-csv <file>` - Parse CSV file
- `parse-pdf <file>` - Parse PDF file
- `parse-spreadsheet <file>` - Parse spreadsheet

**Keywords Command**:
- Automatically checks for daemon
- Starts daemon if not running
- Uses daemon for faster extraction

### API Integration (Future)

```python
# Future API design
from agents import get_daemon

@app.on_event("startup")
async def startup():
    daemon = get_daemon(use_spacy=True)
    daemon.model_manager.initialize(use_spacy=True)

@app.post("/keywords")
async def extract_keywords(request: KeywordRequest):
    daemon = get_daemon()
    return daemon.extract_keywords(request.text)

@app.post("/parse")
async def parse_file(file: UploadFile):
    daemon = get_daemon()
    return daemon.parse_file(file.filename)
```

## Future Enhancements

### Planned Features
- [x] AI Agent daemon for model persistence
- [x] File parsers (TXT, CSV, PDF, Spreadsheet)
- [x] Keyword extraction with n-grams
- [ ] REST API wrapper
- [ ] Web UI
- [ ] Batch processing API
- [ ] Result caching
- [ ] Multi-language support
- [ ] Custom entity types per domain
- [ ] ML-based question detection
- [ ] Entity linking to knowledge bases
- [ ] Socket/HTTP communication for daemon
- [ ] Parser plugins system
- [ ] Additional file formats (DOCX, RTF, etc.)

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

