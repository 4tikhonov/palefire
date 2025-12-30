# Voices Module

Parallel LLM request handling with session-based logging.

## Overview

The `voices` module provides a clean interface for sending parallel requests to multiple Ollama LLM models. It reuses libraries from `AIAgent.py` and `utils/llm_client.py`, uses the same environment variables from `.env`, and organizes all logs by session ID.

## Features

- **Parallel Requests**: Send requests to multiple models simultaneously
- **Session-based Logging**: All logs organized in `logs/sessionid/` subfolders
- **JSON Parsing**: Automatically parses responses into structured JSON format
- **Environment Configuration**: Automatically uses settings from `.env` file
- **Error Handling**: Graceful fallback to sequential processing if parallel fails
- **Request/Response Tracking**: Full logging of all requests and responses

## Structure

```
voices/
├── __init__.py          # Module exports
├── parallel_llm.py      # Main parallel request handler
├── session_logger.py    # Session-based logging
├── response_parser.py  # JSON response parser
└── README.md           # This file
```

## Usage

### Basic Usage

```python
from voices import ParallelLLMRequest

# Initialize with automatic config from .env
request_handler = ParallelLLMRequest()

# Send parallel requests
responses = request_handler.send_parallel(
    prompt="What is the capital of France?",
    request_type="chat",
    temperature=0.7,
    max_tokens=100
)

# Process responses
for response in responses:
    if response.success:
        print(f"{response.model}: {response.response}")
        # Access parsed JSON
        if response.parsed_json:
            print(f"  Parsed format: {response.parsed_json['format']}")
            print(f"  Parsed data: {response.parsed_json['data']}")
            print(f"  JSON saved to: {response.parsed_json_path}")
    else:
        print(f"{response.model}: Error - {response.error}")
```

### Custom Configuration

```python
from voices import ParallelLLMRequest

# Initialize with custom models
request_handler = ParallelLLMRequest(
    models=['model1:latest', 'model2:latest', 'model3:latest'],
    timeout=300,
    parallel=True,
    session_id='my-session-123'
)

# Send requests
responses = request_handler.send_parallel(
    prompt="Explain quantum computing.",
    request_type="explanation"
)
```

### Async Usage

```python
import asyncio
from voices import ParallelLLMRequest

async def main():
    request_handler = ParallelLLMRequest()
    
    responses = await request_handler.send_parallel_async(
        prompt="What is AI?",
        request_type="question"
    )
    
    for response in responses:
        print(f"{response.model}: {response.response}")

asyncio.run(main())
```

## Configuration

The module automatically reads configuration from `.env`:

- `OLLAMA_VERIFICATION_MODEL`: Comma-separated list of models (e.g., `model1:latest,model2:latest`)
- `OLLAMA_MODEL`: Default model if verification models not set
- `OLLAMA_BASE_URL`: Ollama API base URL
- `OLLAMA_API_KEY`: API key (usually 'ollama')
- `OLLAMA_VERIFICATION_TIMEOUT`: Request timeout in seconds
- `OLLAMA_PARALLEL_REQUESTS`: Enable parallel requests (true/false)

## Session Logging

All requests and responses are logged to `logs/sessionid/`:

```
logs/
└── <session-id>/
    ├── request_chat_model1_latest_20251230_120000_abc123.txt
    ├── response_chat_model1_latest_20251230_120001_abc123.txt
    ├── parsed/
    │   ├── 20251230_120001_parsed_chat_model1_latest_abc123.json
    │   └── 20251230_120001_parsed_chat_model2_latest_def456.json
    └── parallel_request_summary.json
```

Each session gets its own directory, making it easy to track all related requests.

### Parsed JSON Structure

Each parsed JSON file includes:

```json
{
  "parsed": true,
  "format": "json|markdown_json|embedded_json|structured_text|plain_text",
  "data": { ... },
  "original_response": "...",
  "response_length": 123,
  "metadata": {
    "question": "What is the capital of France?",
    "model": "model1:latest",
    "date_time": "2025-12-30T12:00:01.123456",
    "session": "session-uuid",
    "request_id": "abc123",
    "request_type": "chat",
    "temperature": 0.7,
    "max_tokens": 100
  }
}
```

The metadata section includes:
- **question**: The original prompt/question sent to the model
- **model**: Which model generated the response
- **date_time**: ISO format timestamp of when the response was received
- **session**: Session ID for grouping related requests
- **request_id**: Unique request identifier
- **request_type**: Type of request (e.g., "chat", "verification")
- **temperature**: Temperature setting used
- **max_tokens**: Max tokens setting used

## API Reference

### ParallelLLMRequest

Main class for handling parallel LLM requests.

#### Methods

- `send_parallel(prompt, request_type, temperature, max_tokens)`: Send parallel requests (synchronous)
- `send_parallel_async(prompt, request_type, temperature, max_tokens)`: Send parallel requests (async)
- `send_sequential(prompt, request_type, temperature, max_tokens)`: Send sequential requests
- `get_session_id()`: Get current session ID
- `get_session_dir()`: Get session logs directory

### ParallelLLMResponse

Response from a single model.

#### Attributes

- `model`: Model name
- `response`: Response text
- `success`: Whether request succeeded
- `error`: Error message if failed
- `request_id`: Request ID for correlation
- `metadata`: Optional metadata
- `parsed_json`: Parsed JSON structure (dict with 'parsed', 'format', 'data')
- `parsed_json_path`: Path to saved parsed JSON file

### SessionLogger

Session-based logging utility.

#### Methods

- `log_request(model, prompt, request_type, metadata)`: Log a request
- `log_response(model, response, request_id, request_type, metadata)`: Log a response
- `log_json(data, filename, subfolder)`: Log JSON data
- `get_session_dir()`: Get session directory

### ResponseParser

Response parser for converting LLM responses to structured JSON.

#### Methods

- `parse(response, response_type)`: Parse response into structured JSON
- `parse_to_json_file(response, output_path, response_type, metadata)`: Parse and save to file

#### Supported Formats

- Direct JSON
- JSON in markdown code blocks (```json ... ```)
- Embedded JSON objects in text
- Structured text/markdown (key-value pairs, lists, sections)
- Plain text (converted to structured format)

## Testing

Run the test script:

```bash
python3 test_voices.py
```

Or with custom environment variables:

```bash
OLLAMA_VERIFICATION_MODEL=model1:latest,model2:latest python3 test_voices.py
```

## Examples

See `test_voices.py` for complete examples.

