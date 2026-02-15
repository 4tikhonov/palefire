# Palefire Project Guide

## 1. Project Overview
**Pale Fire** is an intelligent Knowledge Graph Search System that transforms data points into human-readable narratives. It features:
*   **Question-Type Detection**: Automatically understands WHO/WHERE/WHEN/WHAT/WHY/HOW questions.
*   **NER Enrichment**: Extracts and tags 18+ entity types (PER, LOC, ORG, DATE, etc.).
*   **5-Factor Ranking**: Combines semantic, connectivity, temporal, query matching, and entity-type intelligence.
*   **AI Agent Daemon**: Keeps gensim and spacy models loaded for instant access.

## 2. Architecture
*   `palefire-cli.py`: Main CLI application.
*   `api.py`: REST API server.
*   `modules/`: Core functionality (PaleFireCore, KeywordBase).
*   `agents/`: AI Agent daemon and file parsers.
*   `prompts/`: LLM prompts.
*   `docs/`: Extended documentation.

## 3. Setup Instructions

### Docker (Recommended)
```bash
# Start all services
docker-compose up -d

# Initial setup (pull models)
make setup
```

### Manual Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    pip install gensim>=4.3.0
    ```
2.  **Configuration**:
    ```bash
    cp env.example .env
    # Edit .env with Neo4j and Ollama settings
    ```

## 4. Operational Commands

### Docker Commands (via Makefile)
| Command | Description |
| :--- | :--- |
| `make up` | Start all services |
| `make down` | Stop all services |
| `make logs` | View logs |
| `make setup` | Pull necessary models |
| `make ingest-demo` | Ingest demo data |
| `make clean-db` | Clean Neo4j database |

### CLI Commands
*   **Ingest**: `python palefire-cli.py ingest --demo` or `python palefire-cli.py ingest --file <file>`
*   **Query**: `python palefire-cli.py query "Your question?"`
*   **Config**: `python palefire-cli.py config`
*   **Clean**: `python palefire-cli.py clean --confirm`
*   **Keywords**: `python palefire-cli.py keywords "Text" --method combined`

### AI Agent Daemon
*   **Start**: `python palefire-cli.py agent start --daemon`
*   **Status**: `python palefire-cli.py agent status`
*   **Stop**: `python palefire-cli.py agent stop`

## 5. Testing
```bash
# Run all tests
pytest

# Run specific suite
pytest tests/test_ai_agent.py -v

# Run with coverage
./run_tests.sh coverage
```


## 6. Ghostwriter Skill
The Ghostwriter skill allows Palefire to ingest URLs, manage knowledge collections, and perform RAG-based Q&A.

### Dependencies
Ensure you have `qdrant-client` and `sentence-transformers` installed:
```bash
pip install qdrant-client sentence-transformers
```

### Commands
*   **Ingest URL**:
    ```bash
    python palefire-cli.py ghostwriter ingest "https://example.com" --collection "my-collection"
    # Output: Successfully ingested 5 chunks from https://example.com
    ```

*   **Ask Question (RAG)**:
    ```bash
    python palefire-cli.py ghostwriter ask "What is the main topic?" --collection "my-collection"
    # Output: The main topic is... [Source: https://example.com]
    ```

*   **Semantic Search**:
    ```bash
    python palefire-cli.py ghostwriter search "specific keyword" --collection "my-collection"
    # Output: Found content related to...
    ```

*   **List Collections**:
    ```bash
    python palefire-cli.py ghostwriter collections
    # Output:
    # - my-collection
    # - palefire_knowledge (default)
    ```

## 7. Troubleshooting
*   **Daemon not starting**: Check if `palefire_ai_agent.pid` exists in `/tmp/` and remove it if the process is dead.
*   **Neo4j Connection**: Verify credentials in `.env` and that Neo4j is running on port 7687.
*   **Missing Models**: Run `python -m spacy download en_core_web_sm` or `make setup` for Docker.
*   **Ghostwriter Unavailable**: Ensure `qdrant-client` and `sentence-transformers` are installed.
