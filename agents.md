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


## 6. Ghostwriter Skill (RAG & Web Ingestion)

The Ghostwriter functionality enables Palefire to act as an intelligent research assistant by ingesting content from URLs, indexing it in a vector database (Qdrant), and using RAG (Retrieval-Augmented Generation) to answer questions.

### Prerequisites
- **Qdrant**: Vector database for storing content embeddings.
- **Ollama**: Local LLM server for generating responses and embeddings.
- **Dependencies**: `qdrant-client`, `sentence-transformers`.

### Configuration
Run the interactive init command to set up your environment:
```bash
python palefire-cli.py init
```
This will prompt you for:
- `OLLAMA_HOST`: URL of your Ollama instance (default: `http://localhost:11434/v1`)
- `QDRANT_HOST`: Hostname of your Qdrant instance (default: `localhost`)
- `GHOSTWRITER_UI_CONTAINER` / `PORT`: For integration links

### Usage Commands

#### 1. Ingest Content
Download and index content from a URL.
```bash
# Basic ingestion
python palefire-cli.py ghostwriter ingest "https://example.com/article"

# Ingest into a specific collection
python palefire-cli.py ghostwriter ingest "https://example.com/ai-news" --collection ai-knowledge
```

#### 2. Ask Questions (RAG)
Ask questions based on the ingested knowledge.
```bash
# Ask using default collection
python palefire-cli.py ghostwriter ask "Summarize the article."

# Ask from a specific collection
python palefire-cli.py ghostwriter ask "What are the key findings?" --collection ai-knowledge
```

#### 3. Semantic Search
Search for relevant text chunks without generating an answer.
```bash
python palefire-cli.py ghostwriter search "neural networks" --collection ai-knowledge
```

#### 4. Manage Collections
List all available knowledge collections.
```bash
python palefire-cli.py ghostwriter collections
```

### Advanced Features
- **Chunking**: Content is automatically split into manageable chunks with overlap to ensure context preservation.
- **Source Tracking**: Every answer includes citations to the source URL.
- **Error Handling**: Automatically handles SSL verification issues (in test mode) and connection retries.

- **Error Handling**: Automatically handles SSL verification issues (in test mode) and connection retries.

## 7. MCP Server Integration

Palefire provides an MCP server compatible with clients like Claude Desktop, allowing LLMs to directly invoke Ghostwriter tools.

### Running the Server
```bash
python mcp_server.py
```

### Configuration (Claude Desktop with Docker)
Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "palefire": {
      "command": "docker",
      "args": [
        "compose",
        "run",
        "--rm",
        "-T", 
        "mcp-server"
      ],
      "cwd": "/absolute/path/to/palefire",
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
      }
    }
  }
}
```
**Note**: Replace `/absolute/path/to/palefire` with the actual path to your project. The `-T` flag is critical to disable pseudo-TTY allocation, which interferes with MCP communication.

### Available Tools
- `ingest_url(url, collection_name)`
- `ask_question(question, collection_name)`
- `search_content(query, collection_name)`
- `list_collections()`

## 8. Troubleshooting
*   **Daemon not starting**: Check if `palefire_ai_agent.pid` exists in `/tmp/` and remove it if the process is dead.
*   **Neo4j Connection**: Verify credentials in `.env` and that Neo4j is running on port 7687.
*   **Missing Models**: Run `python -m spacy download en_core_web_sm` or `make setup` for Docker.
*   **Ghostwriter Unavailable**: Ensure `qdrant-client` and `sentence-transformers` are installed.
