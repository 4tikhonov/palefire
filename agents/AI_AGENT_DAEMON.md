# AI Agent Daemon

## Overview

The AI Agent daemon is a long-running service that keeps Gensim and spaCy models loaded in memory to eliminate start/stop delays. This is essential for production deployments where models need to be ready instantly for keyword extraction and entity recognition.

## Architecture

### Components

1. **ModelManager**: Thread-safe manager that keeps models loaded in memory
   - Manages KeywordExtractor (Gensim) instances
   - Manages EntityEnricher (spaCy) instances
   - Provides thread-safe access to models

2. **AIAgentDaemon**: Daemon service that runs continuously
   - Loads models on startup
   - Keeps models in memory
   - Provides API for keyword extraction and NER
   - Handles graceful shutdown

3. **AIAgentClient**: Client for communicating with daemon (future: socket/HTTP)

## Usage

### Command Line

```bash
# Start daemon in background
python palefire-cli.py agent start --daemon

# Start in foreground (for debugging)
python palefire-cli.py agent start

# Check status
python palefire-cli.py agent status

# Stop daemon
python palefire-cli.py agent stop

# Restart daemon
python palefire-cli.py agent restart --daemon

# Start without spaCy (faster, less accurate)
python palefire-cli.py agent start --no-spacy --daemon
```

### Programmatic Usage

```python
from agents import get_daemon

# Get daemon instance (singleton pattern)
daemon = get_daemon(use_spacy=True)

# Initialize models (only done once)
daemon.model_manager.initialize(use_spacy=True)

# Extract keywords (fast - models already loaded)
keywords = daemon.extract_keywords(
    "Artificial intelligence and machine learning",
    num_keywords=10,
    method='tfidf'
)

# Extract entities (fast - models already loaded)
entities = daemon.extract_entities(
    "Kamala Harris was the Attorney General of California."
)

### Direct Model Access

```python
from agents import ModelManager

# Create model manager
manager = ModelManager()
manager.initialize(use_spacy=True)

# Access models directly
extractor = manager.keyword_extractor
enricher = manager.entity_enricher

# Use models
keywords = extractor.extract("Your text here")
entities = enricher.extract_entities("Your text here")
```

## System Integration

### Linux (systemd)

1. Copy service file:
```bash
sudo cp agents/palefire-agent.service /etc/systemd/system/
```

2. Edit service file with correct paths:
```bash
sudo nano /etc/systemd/system/palefire-agent.service
```

Update these lines:
- `WorkingDirectory`: Path to palefire directory
- `ExecStart`: Full path to `agents/palefire-agent-service.py`
- `User`: User to run as

3. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable palefire-agent
sudo systemctl start palefire-agent
```

4. Check status:
```bash
sudo systemctl status palefire-agent
```

### macOS (launchd)

1. Copy plist file:
```bash
cp agents/palefire-agent.plist ~/Library/LaunchAgents/com.palefire.aiagent.plist
```

2. Edit plist with correct paths:
```bash
nano ~/Library/LaunchAgents/com.palefire.aiagent.plist
```

Update:
- Path to `agents/palefire-agent-service.py`
- Environment variables if needed

3. Load service:
```bash
launchctl load ~/Library/LaunchAgents/com.palefire.aiagent.plist
```

4. Check status:
```bash
launchctl list | grep palefire
```

## Configuration

### Environment Variables

Set these in your service file or environment:

- `PALEFIRE_PIDFILE`: Path to PID file (default: `/tmp/palefire_ai_agent.pid`)
- `PALEFIRE_USE_SPACY`: Enable spaCy (default: `true`)
- `PALEFIRE_DAEMON`: Run as daemon (default: `true`)

### PID File

The daemon creates a PID file to track the running process. Default location: `/tmp/palefire_ai_agent.pid`

## Benefits

### Performance

- **No Model Loading Delays**: Models loaded once at startup
- **Instant Response**: Ready for immediate use
- **Reduced CPU**: No repeated initialization overhead

### Memory Efficiency

- **Single Instance**: Models loaded once, shared across requests
- **Persistent Memory**: Models stay in memory between requests
- **Predictable Usage**: Consistent memory footprint

### Production Ready

- **Graceful Shutdown**: Handles SIGTERM/SIGINT properly
- **Health Monitoring**: Periodic health checks
- **Auto-recovery**: Can reload models if they become unavailable
- **Logging**: Comprehensive logging for monitoring

## Monitoring

### Check Status

```bash
# Via CLI
python palefire-cli.py agent status

# Output:
{
  "running": true,
  "models_initialized": true,
  "use_spacy": true,
  "spacy_available": true
}
```

### Logs

Logs are written to:
- stdout/stderr (when run in foreground)
- `/tmp/palefire_ai_agent.log` (when run as service)

### Health Checks

The daemon performs periodic health checks every 60 seconds to ensure models are still loaded. If models become unavailable, it attempts to reload them.

## Troubleshooting

### Daemon Won't Start

1. Check if already running:
```bash
ps aux | grep palefire-agent
```

2. Check PID file:
```bash
cat /tmp/palefire_ai_agent.pid
```

3. Remove stale PID file if process doesn't exist:
```bash
rm /tmp/palefire_ai_agent.pid
```

### Models Not Loading

1. Check dependencies:
```bash
pip install gensim>=4.3.0
pip install spacy
python -m spacy download en_core_web_sm
```

2. Check logs:
```bash
tail -f /tmp/palefire_ai_agent.log
```

3. Test models directly:
```python
from agents import ModelManager
manager = ModelManager()
manager.initialize(use_spacy=True)
```

### High Memory Usage

- Models are kept in memory, which is expected
- Gensim models: ~50-200 MB
- spaCy models: ~100-500 MB
- Total: ~150-700 MB depending on models

### Performance Issues

- Ensure daemon is running (check status)
- Verify models are initialized (check logs)
- Consider disabling spaCy if not needed (`--no-spacy`)

## API Integration

The daemon can be integrated into the FastAPI server:

```python
from agents import get_daemon

# Initialize daemon on startup
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

## Best Practices

1. **Start daemon before API server**: Ensures models are ready
2. **Monitor memory usage**: Models consume significant memory
3. **Use health checks**: Monitor daemon status regularly
4. **Graceful shutdown**: Always stop daemon properly (SIGTERM)
5. **Log rotation**: Configure log rotation for production
6. **Resource limits**: Set appropriate memory limits in service file

## Example Workflow

```bash
# 1. Start daemon
python palefire-cli.py agent start --daemon

# 2. Wait for initialization (check logs)
tail -f /tmp/palefire_ai_agent.log

# 3. Verify status
python palefire-cli.py agent status

# 4. Use models (no loading delay)
python palefire-cli.py keywords "Your text" --num-keywords 10

# 5. Stop when done
python palefire-cli.py agent stop
```

## Future Enhancements

- [ ] HTTP API endpoint for daemon communication
- [ ] Unix socket communication
- [ ] Model versioning and hot-reloading
- [ ] Multiple model instances for load balancing
- [ ] Metrics and monitoring integration
- [ ] Model caching strategies

---

**AI Agent Daemon** - Keep Models Ready, Keep Performance High 🚀

