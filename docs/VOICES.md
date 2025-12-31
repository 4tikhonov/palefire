# Voices for LLMs

> "A thousand, thousand voices, whispering...
> The time has passed for choices."
> — Ozzy Osbourne, "See You on the Other Side"

Parallel, session-logged requests to multiple LLM models (Ollama) with automatic response parsing and simple integration into Pale Fire.

## Overview

The Voices module enables sending the same prompt to multiple LLM models in parallel and collecting their responses with full session-based logging and structured parsing. It lives in [voices/](../voices) and is designed to be a lightweight, reusable layer for verification, comparison, and ensemble-style workflows.

- Parallel requests to multiple models
- Session-based logging under [logs/](../logs)
- Automatic JSON parsing of model outputs
- Respects global `.env`/configuration via [config.py](../config.py)
- Graceful fallback to sequential execution

Core implementation files:

- [voices/parallel_llm.py](../voices/parallel_llm.py)
- [voices/session_logger.py](../voices/session_logger.py)
- [voices/response_parser.py](../voices/response_parser.py)
- Public exports in [voices/__init__.py](../voices/__init__.py)

## When To Use Voices

- Compare multiple models’ answers for reliability or diversity
- Verify results from an algorithm (e.g., NER) using LLMs
- Run ensembles or majority voting over model responses
- Capture complete request/response traces for audits and evaluation

The Pale Fire agent integrates voices-like functionality directly in [agents/AIAgent.py](../agents/AIAgent.py) for certain flows, but the standalone Voices module is ideal for simple, scriptable verification and experimentation.

## Setup

Voices uses the project’s configuration values (from `.env` via [config.py](../config.py)):

- `verification_models` or `verification_model` → List or single model used for parallel runs
- `model` → Fallback model if no verification models are provided
- `base_url` → Ollama API endpoint (e.g., `http://localhost:11434/v1`)
- `api_key` → Ollama API key (typically `ollama`)
- `verification_timeout` → Per-request timeout (seconds)
- `parallel_requests` → Enable/disable parallel mode

You can override these at construction time.

## Quick Start

```python
from voices import ParallelLLMRequest

# Auto-configured from config/.env
req = ParallelLLMRequest()

responses = req.send_parallel(
    prompt="What is the capital of France?",
    request_type="chat",
    temperature=0.7,
    max_tokens=100,
)

for r in responses:
    if r.success:
        print(f"{r.model}: {r.response[:200]}...")
        if r.parsed_json:
            print("Parsed:", r.parsed_json["format"])  # json / markdown_json / embedded_json / structured_text / plain_text
            print("Saved:", r.parsed_json_path)
    else:
        print(f"{r.model}: ERROR → {r.error}")
```

Async usage:

```python
import asyncio
from voices import ParallelLLMRequest

async def main():
    req = ParallelLLMRequest()
    results = await req.send_parallel_async(
        prompt="Explain quantum computing in one paragraph.",
        request_type="explanation",
    )
    for r in results:
        print(r.model, "→", (r.response or "").strip()[:200])

asyncio.run(main())
```

Custom configuration:

```python
from voices import ParallelLLMRequest

req = ParallelLLMRequest(
    models=["model1:latest", "model2:latest"],
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=300,
    parallel=True,
    session_id="my-session-123",
)

results = req.send_parallel(prompt="List three prime numbers.")
```

## Logging & Sessions

Every run is assigned a session ID and logs are written to a dedicated folder under [logs/](../logs):

```
logs/
└── <session-id>/
    ├── request_chat_model1_latest_YYYYMMDD_HHMMSS_<id>.txt
    ├── response_chat_model1_latest_YYYYMMDD_HHMMSS_<id>.txt
    ├── parsed/
    │   ├── YYYYMMDD_HHMMSS_parsed_chat_model1_latest_<id>.json
    │   └── YYYYMMDD_HHMMSS_parsed_chat_model2_latest_<id>.json
    └── parallel_request_summary.json
```

The logger is implemented by [voices/session_logger.py](../voices/session_logger.py). Use `req.get_session_id()` and `req.get_session_dir()` to retrieve session metadata and paths.

## Response Parsing

Parsing is handled by [voices/response_parser.py](../voices/response_parser.py). It recognizes:

- Direct JSON
- JSON within Markdown code blocks (```json ... ```)
- Embedded JSON fragments within text
- Structured text/Markdown (sections, key-value, lists)
- Plain text (wrapped into a simple structure)

Each saved JSON includes metadata such as model, timestamp, session ID, request type, and generation settings.

## API Summary

- Class `ParallelLLMRequest`
  - `send_parallel(prompt, request_type, temperature, max_tokens)`
  - `send_parallel_async(prompt, request_type, temperature, max_tokens)`
  - `send_sequential(prompt, request_type, temperature, max_tokens)`
  - `get_session_id()` / `get_session_dir()`
- Class `ParallelLLMResponse`
  - `model`, `response`, `success`, `error`, `request_id`
  - `parsed_json`, `parsed_json_path`
- Class `SessionLogger`
  - `log_request()`, `log_response()`, `log_json()`, `get_session_dir()`
- Class `ResponseParser`
  - `parse()`, `parse_to_json_file()`

## Integration Notes

- The agent has voices-like flows in [agents/AIAgent.py](../agents/AIAgent.py); you can call Voices directly in scripts or services where you want simple parallel comparison.
- Ensure Ollama is running and models referenced in configuration are available locally.
- For deterministic comparison runs, set low `$temperature` and constrain `$max_tokens` appropriately.

## Troubleshooting

- No models configured → check `verification_models`/`verification_model` or pass `models=[...]` when constructing `ParallelLLMRequest`.
- Event loop errors in sync mode → install `nest_asyncio` or use `send_parallel_async(...)` directly.
- Empty or unparsable responses → inspect raw response logs in the session folder and adjust prompt or model settings.

---

See module source for details: [voices/parallel_llm.py](../voices/parallel_llm.py), [voices/session_logger.py](../voices/session_logger.py), [voices/response_parser.py](../voices/response_parser.py).
