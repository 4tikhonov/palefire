# AI Agent Docker Deployment

This guide explains how to run the Pale Fire AI Agent daemon using Docker.

**Note:** All docker-compose commands should be run from the `palefire` root directory (parent of `agents/`).

## Overview

The AI Agent daemon keeps Gensim and spaCy models loaded in memory to eliminate start/stop delays. This is essential for production deployments where models need to be ready instantly for keyword extraction and entity recognition.

## Quick Start

### Standalone Deployment

Run the AI Agent as a standalone service:

```bash
# Build and start the agent (run from palefire root directory)
docker-compose -f agents/docker-compose.agent.yml up -d

# View logs
docker-compose -f agents/docker-compose.agent.yml logs -f

# Stop the agent
docker-compose -f agents/docker-compose.agent.yml down
```

### Integrated Deployment

Run the AI Agent alongside the main Pale Fire services:

```bash
# Start all services including the agent
docker-compose -f docker-compose.yml -f agents/docker-compose.agent.yml up -d

# View agent logs
docker-compose -f agents/docker-compose.agent.yml logs -f palefire-agent

# Stop only the agent
docker-compose -f agents/docker-compose.agent.yml stop palefire-agent
```

## Configuration

### Environment Variables

The following environment variables can be configured:

| Variable | Default | Description |
|----------|---------|-------------|
| `PALEFIRE_PIDFILE` | `/tmp/palefire_ai_agent.pid` | Path to PID file |
| `PALEFIRE_USE_SPACY` | `true` | Enable spaCy for NER |
| `PALEFIRE_DAEMON` | `true` | Run in daemon mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `NLTK_DATA` | `/root/nltk_data` | Path to NLTK data |

### Custom Configuration

Edit `agents/docker-compose.agent.yml` to customize:

```yaml
services:
  palefire-agent:
    environment:
      - PALEFIRE_USE_SPACY=false  # Disable spaCy for faster startup
      - LOG_LEVEL=DEBUG           # Enable debug logging
```

## Building the Image

Build the agent image manually:

```bash
# Build the image (run from palefire root directory)
docker build -f agents/Dockerfile.agent -t palefire-agent:latest .

# Run the container (run from palefire root directory)
docker run -d \
  --name palefire-ai-agent \
  -v $(pwd)/logs:/app/logs \
  -e PALEFIRE_USE_SPACY=true \
  palefire-agent:latest
```

## Health Checks

The container includes a health check that verifies the agent process is running:

```bash
# Check container health
docker ps

# View health check logs
docker inspect palefire-ai-agent | grep -A 10 Health
```

## Logs

View agent logs:

```bash
# Follow logs
docker-compose -f agents/docker-compose.agent.yml logs -f palefire-agent

# View last 100 lines
docker-compose -f agents/docker-compose.agent.yml logs --tail=100 palefire-agent

# View logs from Docker directly
docker logs -f palefire-ai-agent
```

## Volumes

The following volumes are mounted (paths relative to palefire root directory):

- `../logs:/app/logs` - Application logs
- `../data:/app/data` - Data directory (optional)

## Network Configuration

### Standalone Mode

The agent runs on its own network (`palefire-network`).

### Integrated Mode

To connect the agent to the main Pale Fire network:

1. Start the main services first:
   ```bash
   docker-compose up -d
   ```

2. Update `agents/docker-compose.agent.yml`:
   ```yaml
   networks:
     palefire-network:
       external: true
       name: palefire-network
   ```

3. Start the agent:
   ```bash
   docker-compose -f agents/docker-compose.agent.yml up -d
   ```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker-compose -f agents/docker-compose.agent.yml logs palefire-agent
```

### Models Not Loading

Verify spaCy model is installed:
```bash
docker exec palefire-ai-agent python -m spacy info en_core_web_sm
```

### Health Check Failing

Check if the process is running:
```bash
docker exec palefire-ai-agent ps aux | grep palefire-agent
```

### Permission Issues

Ensure the container has write access to mounted volumes:
```bash
docker exec palefire-ai-agent ls -la /app/logs
```

## Production Deployment

### Resource Limits

Add resource limits to `agents/docker-compose.agent.yml`:

```yaml
services:
  palefire-agent:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Restart Policy

The default restart policy is `unless-stopped`. For production:

```yaml
services:
  palefire-agent:
    restart: always
```

### Logging Configuration

Configure log rotation:

```yaml
services:
  palefire-agent:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Integration with Main Services

The AI Agent can be used by the main Pale Fire API service. To enable this:

1. Ensure both services are on the same network
2. The API can connect to the agent via the network
3. Or use the agent's programmatic interface directly

Example API integration:

```python
from agents import get_daemon

# Initialize daemon (models loaded once)
daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)

# Use in API endpoints
@app.post("/keywords")
async def extract_keywords(request: KeywordExtractionRequest):
    keywords = daemon.extract_keywords(
        request.text,
        method=request.method.value,
        num_keywords=request.num_keywords
    )
    return {"keywords": keywords}
```

## Monitoring

### Process Monitoring

Monitor the agent process:
```bash
docker exec palefire-ai-agent ps aux | grep python
```

### Resource Usage

Monitor resource usage:
```bash
docker stats palefire-ai-agent
```

### Log Monitoring

Set up log aggregation:
```bash
# Using docker logs
docker logs -f palefire-ai-agent | grep ERROR

# Using compose
docker-compose -f agents/docker-compose.agent.yml logs -f | grep ERROR
```

## Updating

To update the agent:

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose -f agents/docker-compose.agent.yml up -d --build
```

## Cleanup

Remove the agent container and volumes:

```bash
# Stop and remove container
docker-compose -f agents/docker-compose.agent.yml down
```

# Remove image
docker rmi palefire-agent:latest

# Remove volumes (if needed)
docker volume ls | grep palefire
docker volume rm <volume-name>
```

