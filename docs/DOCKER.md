# Docker Deployment Guide

Complete guide for running Pale Fire with Docker and Docker Compose.

## Quick Start

### 1. Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- (Optional) NVIDIA Docker for GPU support

### 2. Start All Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 3. Access Services

- **Pale Fire API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474
- **Ollama API**: http://localhost:11434

### 4. Initial Setup

```bash
# Wait for all services to be healthy (30-60 seconds)
docker-compose ps

# Pull Ollama models
docker-compose exec ollama ollama pull deepseek-r1:7b
docker-compose exec ollama ollama pull nomic-embed-text

# Test the API
curl http://localhost:8000/health
```

---

## Architecture

### Services

```
┌─────────────────────────────────────────┐
│         Pale Fire API (Port 8000)       │
│    FastAPI + Knowledge Graph Search     │
└──────────┬──────────────────┬───────────┘
           │                  │
           ▼                  ▼
┌──────────────────┐  ┌──────────────────┐
│  Neo4j (7687)    │  │  Ollama (11434)  │
│  Graph Database  │  │  LLM + Embeddings│
└──────────────────┘  └──────────────────┘
```

### Components

1. **palefire-api** - Main API service
2. **neo4j** - Graph database for knowledge storage
3. **ollama** - Local LLM and embedding service
4. **palefire-cli** - Optional CLI container (profile: cli)

---

## Configuration

### Environment Variables

All configuration is done via environment variables in `docker-compose.yml`:

```yaml
environment:
  # Neo4j
  - NEO4J_URI=bolt://neo4j:7687
  - NEO4J_USER=neo4j
  - NEO4J_PASSWORD=palefire123
  
  # LLM
  - OLLAMA_MODEL=deepseek-r1:7b
  - OLLAMA_BASE_URL=http://ollama:11434/v1
  
  # Search
  - DEFAULT_SEARCH_METHOD=question-aware
  - SEARCH_RESULT_LIMIT=20
```

### Custom Configuration

Create a `.env` file:

```bash
# Copy example
cp env.example .env

# Edit with your values
nano .env
```

Then update `docker-compose.yml`:

```yaml
palefire-api:
  env_file:
    - .env
```

---

## Usage

### Using the API

```bash
# Health check
curl http://localhost:8000/health

# Get configuration
curl http://localhost:8000/config

# Ingest data
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": [
      {
        "content": "Kamala Harris is the Attorney General of California.",
        "type": "text",
        "description": "Biography"
      }
    ]
  }'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who was the California Attorney General?",
    "method": "question-aware",
    "limit": 5
  }'
```

### Using the CLI

```bash
# Start CLI container
docker-compose --profile cli up -d palefire-cli

# Run CLI commands
docker-compose exec palefire-cli python palefire-cli.py config
docker-compose exec palefire-cli python palefire-cli.py ingest --demo
docker-compose exec palefire-cli python palefire-cli.py query "Who is Kamala Harris?"

# Interactive shell
docker-compose exec palefire-cli bash
```

---

## Management

### Start/Stop Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d neo4j

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f palefire-api

# Last 100 lines
docker-compose logs --tail=100 palefire-api
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart palefire-api
```

### Scale Services

```bash
# Run multiple API instances
docker-compose up -d --scale palefire-api=3
```

---

## Data Management

### Volumes

```bash
# List volumes
docker volume ls | grep palefire

# Inspect volume
docker volume inspect palefire_neo4j_data

# Backup Neo4j data
docker run --rm \
  -v palefire_neo4j_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/neo4j-backup.tar.gz /data

# Restore Neo4j data
docker run --rm \
  -v palefire_neo4j_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/neo4j-backup.tar.gz -C /
```

### Database Operations

```bash
# Clean database
docker-compose exec palefire-cli python palefire-cli.py clean --confirm

# Export data
docker-compose exec palefire-cli python palefire-cli.py query "test" --export /app/data/export.json

# Access Neo4j directly
docker-compose exec neo4j cypher-shell -u neo4j -p palefire123
```

---

## GPU Support

### NVIDIA GPU

The Ollama service is configured for GPU support:

```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

### Prerequisites

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Check GPU in Ollama container
docker-compose exec ollama nvidia-smi
```

### CPU-Only Mode

If you don't have a GPU, comment out the GPU configuration:

```yaml
ollama:
  # deploy:
  #   resources:
  #     reservations:
  #       devices:
  #         - driver: nvidia
  #           count: all
  #           capabilities: [gpu]
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs palefire-api

# Check health
docker-compose ps

# Restart service
docker-compose restart palefire-api
```

### Neo4j Connection Issues

```bash
# Check Neo4j is running
docker-compose ps neo4j

# Check Neo4j logs
docker-compose logs neo4j

# Test connection
docker-compose exec neo4j cypher-shell -u neo4j -p palefire123

# Verify network
docker network inspect palefire_palefire-network
```

### Ollama Model Issues

```bash
# Check Ollama is running
docker-compose ps ollama

# List models
docker-compose exec ollama ollama list

# Pull missing models
docker-compose exec ollama ollama pull deepseek-r1:7b
docker-compose exec ollama ollama pull nomic-embed-text

# Test model
docker-compose exec ollama ollama run deepseek-r1:7b "Hello"
```

### API Connection Issues

```bash
# Check API logs
docker-compose logs -f palefire-api

# Check API health
curl http://localhost:8000/health

# Check environment variables
docker-compose exec palefire-api env | grep NEO4J
```

### Out of Memory

```bash
# Increase Neo4j memory in docker-compose.yml
environment:
  - NEO4J_dbms_memory_heap_max__size=4G
  - NEO4J_dbms_memory_pagecache_size=1G

# Restart Neo4j
docker-compose restart neo4j
```

---

## Performance Tuning

### Neo4j Optimization

```yaml
neo4j:
  environment:
    # Increase heap size
    - NEO4J_dbms_memory_heap_max__size=4G
    
    # Increase page cache
    - NEO4J_dbms_memory_pagecache_size=2G
    
    # Enable query logging
    - NEO4J_dbms_logs_query_enabled=true
```

### API Optimization

```yaml
palefire-api:
  environment:
    # Increase result limit
    - SEARCH_RESULT_LIMIT=50
    
    # Disable NER for faster ingestion
    - NER_ENABLED=false
  
  # Add resource limits
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
```

---

## Production Deployment

### Security

1. **Change default passwords**:
```yaml
neo4j:
  environment:
    - NEO4J_AUTH=neo4j/your-secure-password
```

2. **Use secrets**:
```yaml
secrets:
  neo4j_password:
    file: ./secrets/neo4j_password.txt

neo4j:
  environment:
    - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
  secrets:
    - neo4j_password
```

3. **Enable HTTPS** (use reverse proxy like Nginx/Traefik)

### Monitoring

```yaml
# Add Prometheus metrics
palefire-api:
  environment:
    - ENABLE_METRICS=true
  ports:
    - "9090:9090"  # Metrics port
```

### Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker run --rm \
  -v palefire_neo4j_data:/data \
  -v /backups:/backup \
  alpine tar czf /backup/neo4j-$DATE.tar.gz /data

# Keep only last 7 days
find /backups -name "neo4j-*.tar.gz" -mtime +7 -delete
```

### High Availability

For production, consider:
- Neo4j cluster mode
- Load balancer for API
- Separate Ollama instances
- Redis for caching

---

## Development

### Build Custom Image

```bash
# Build image
docker-compose build

# Build with no cache
docker-compose build --no-cache

# Build specific service
docker-compose build palefire-api
```

### Development Mode

```yaml
palefire-api:
  volumes:
    - .:/app  # Mount source code
  environment:
    - LOG_LEVEL=DEBUG
  command: python api.py --reload  # Auto-reload on changes
```

### Run Tests

```bash
# Run tests in container
docker-compose exec palefire-api pytest

# Run with coverage
docker-compose exec palefire-api pytest --cov=. --cov-report=html
```

---

## Quick Commands Reference

```bash
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# View logs
docker-compose logs -f

# Restart API
docker-compose restart palefire-api

# Run CLI command
docker-compose exec palefire-cli python palefire-cli.py query "test"

# Access Neo4j shell
docker-compose exec neo4j cypher-shell -u neo4j -p palefire123

# Pull Ollama model
docker-compose exec ollama ollama pull deepseek-r1:7b

# Backup data
docker run --rm -v palefire_neo4j_data:/data -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz /data

# Clean everything (WARNING: deletes data)
docker-compose down -v
```

---

## See Also

- [API Guide](API_GUIDE.md) - REST API documentation
- [CLI Guide](CLI_GUIDE.md) - Command-line interface
- [Configuration](CONFIGURATION.md) - Configuration options
- [Architecture](ARCHITECTURE.md) - System architecture

---

**Docker Deployment v1.0** - Containerized Pale Fire! 🐳

