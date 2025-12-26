# Pale Fire API Guide

## Overview

Pale Fire provides a RESTful API built with FastAPI for integrating knowledge graph search into your applications.

## Quick Start

### 1. Install Dependencies

```bash
cd /path/to/palefire
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp env.example .env
# Edit .env with your settings
```

### 3. Start the API Server

```bash
python api.py
```

Or with uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the API

- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### GET `/` - Root

Get API status.

**Response:**
```json
{
  "status": "ok",
  "message": "Pale Fire API is running"
}
```

### GET `/health` - Health Check

Check API and database health.

**Response:**
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "database_stats": {
    "nodes": 1523,
    "relationships": 4567
  }
}
```

### GET `/config` - Get Configuration

Get current configuration.

**Response:**
```json
{
  "neo4j_uri": "bolt://localhost:7687",
  "llm_provider": "ollama",
  "llm_model": "deepseek-r1:7b",
  "embedder_model": "nomic-embed-text",
  "search_method": "question-aware",
  "search_limit": 20,
  "ner_enabled": true
}
```

### POST `/ingest` - Ingest Episodes

Ingest episodes into the knowledge graph.

**Request Body:**
```json
{
  "episodes": [
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
  ],
  "enable_ner": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully ingested 2 episodes"
}
```

### POST `/search` - Search Knowledge Graph

Search the knowledge graph with intelligent ranking.

**Request Body:**
```json
{
  "query": "Who was the California Attorney General in 2020?",
  "method": "question-aware",
  "limit": 5
}
```

**Parameters:**
- `query` (required): Search query string
- `method` (optional): Search method - `standard`, `connection`, or `question-aware` (default)
- `limit` (optional): Maximum number of results (1-100, default: 5)

**Response:**
```json
{
  "query": "Who was the California Attorney General in 2020?",
  "method": "question-aware",
  "total_results": 5,
  "timestamp": "2025-12-26T14:45:00.123456+00:00",
  "results": [
    {
      "rank": 1,
      "uuid": "abc123-def456-ghi789",
      "name": "Kamala Harris",
      "summary": "Attorney General of California from 2011 to 2017...",
      "labels": ["Person", "PoliticalFigure"],
      "attributes": {
        "position": "Attorney General",
        "state": "California"
      },
      "scoring": {
        "final_score": 0.9234,
        "original_score": 0.8456,
        "connection_score": 0.7823,
        "temporal_score": 1.0,
        "query_match_score": 0.9123,
        "entity_type_score": 2.0
      },
      "connections": {
        "count": 15,
        "entities": [
          {
            "name": "California",
            "type": "LOC",
            "labels": ["Entity", "LOC"],
            "uuid": "xyz789-abc123-def456"
          }
        ],
        "relationship_types": ["WORKED_AT", "LOCATED_IN"]
      },
      "recognized_entities": {
        "PER": ["Kamala Harris"],
        "LOC": ["California"],
        "ORG": ["Attorney General"]
      }
    }
  ]
}
```

### DELETE `/clean` - Clean Database

Clear all data from the Neo4j database.

⚠️ **WARNING**: This permanently deletes all data!

**Response:**
```json
{
  "status": "success",
  "message": "Database cleaned successfully. Deleted 1523 nodes and 4567 relationships."
}
```

## Usage Examples

### Python

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Ingest episodes
episodes_data = {
    "episodes": [
        {
            "content": "Kamala Harris is the Attorney General of California.",
            "type": "text",
            "description": "Biography"
        }
    ],
    "enable_ner": True
}
response = requests.post(f"{BASE_URL}/ingest", json=episodes_data)
print(response.json())

# Search
search_data = {
    "query": "Who is Kamala Harris?",
    "method": "question-aware",
    "limit": 5
}
response = requests.post(f"{BASE_URL}/search", json=search_data)
results = response.json()
print(f"Found {results['total_results']} results")
for result in results['results']:
    print(f"{result['rank']}. {result['name']} (score: {result['scoring']['final_score']:.4f})")
```

### JavaScript/TypeScript

```typescript
const BASE_URL = 'http://localhost:8000';

// Health check
const health = await fetch(`${BASE_URL}/health`);
const healthData = await health.json();
console.log(healthData);

// Ingest episodes
const ingestResponse = await fetch(`${BASE_URL}/ingest`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    episodes: [
      {
        content: 'Kamala Harris is the Attorney General of California.',
        type: 'text',
        description: 'Biography'
      }
    ],
    enable_ner: true
  })
});
const ingestData = await ingestResponse.json();
console.log(ingestData);

// Search
const searchResponse = await fetch(`${BASE_URL}/search`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'Who is Kamala Harris?',
    method: 'question-aware',
    limit: 5
  })
});
const searchData = await searchResponse.json();
console.log(`Found ${searchData.total_results} results`);
searchData.results.forEach(result => {
  console.log(`${result.rank}. ${result.name} (score: ${result.scoring.final_score})`);
});
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Ingest episodes
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": [
      {
        "content": "Kamala Harris is the Attorney General of California.",
        "type": "text",
        "description": "Biography"
      }
    ],
    "enable_ner": true
  }'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who is Kamala Harris?",
    "method": "question-aware",
    "limit": 5
  }'

# Clean database
curl -X DELETE http://localhost:8000/clean
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `500` - Internal Server Error
- `503` - Service Unavailable (health check failed)

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Authentication

Currently, the API does not implement authentication. For production use, consider adding:

- API keys
- OAuth 2.0
- JWT tokens
- Rate limiting

## CORS Configuration

The API currently allows all origins (`*`). For production, configure appropriately:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

## Deployment

### Production Server

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t palefire-api .
docker run -p 8000:8000 --env-file .env palefire-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.13
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data

  palefire-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: password
    depends_on:
      - neo4j

volumes:
  neo4j_data:
```

## Performance

### Response Times

| Endpoint | Typical Response Time |
|----------|----------------------|
| `/health` | 10-50ms |
| `/config` | < 5ms |
| `/ingest` | 100-500ms per episode |
| `/search` (standard) | 100-300ms |
| `/search` (question-aware) | 500-2000ms |
| `/clean` | 100-5000ms (depends on data size) |

### Optimization Tips

1. **Use connection pooling** - Configure Neo4j driver appropriately
2. **Enable caching** - Cache frequent queries
3. **Batch ingestion** - Ingest multiple episodes at once
4. **Async operations** - API is fully async for better concurrency
5. **Load balancing** - Run multiple instances behind a load balancer

## Monitoring

### Health Checks

Use the `/health` endpoint for monitoring:

```bash
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### Logging

The API logs to stdout. Configure log level via environment:

```bash
LOG_LEVEL=DEBUG python api.py
```

### Metrics

Consider adding:
- Prometheus metrics
- Request/response times
- Error rates
- Database query performance

## Troubleshooting

### API Won't Start

**Problem**: `Failed to initialize Pale Fire API`

**Solutions**:
1. Check Neo4j is running
2. Verify `.env` configuration
3. Check network connectivity
4. Review logs for specific error

### Search Returns No Results

**Problem**: Empty results array

**Solutions**:
1. Check database has data: `GET /health`
2. Try simpler query
3. Verify data was ingested successfully
4. Check search method compatibility

### Slow Response Times

**Problem**: API is slow

**Solutions**:
1. Check Neo4j performance
2. Reduce search limit
3. Use simpler search method
4. Enable query caching
5. Scale horizontally

## See Also

- [CLI Guide](CLI_GUIDE.md) - Command-line interface
- [Configuration](CONFIGURATION.md) - Configuration options
- [Quick Reference](QUICK_REFERENCE.md) - Quick commands

---

**Pale Fire API v1.0** - Knowledge Graph Search as a Service! 🚀

