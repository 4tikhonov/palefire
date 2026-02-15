import ollama
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

class SimpleOllamaClient:
    """
    A simple client for Ollama using the official 'ollama' Python library.
    """
    def __init__(self, model: str, base_url: str, api_key: str = "ollama", timeout: int = 300):
        self.model = model
        # The ollama library expects the host, not necessarily the /v1 suffix
        self.host = base_url.replace('/v1', '')
        self.timeout = timeout
        self.client = ollama.Client(host=self.host, timeout=timeout)

    def complete(self, messages, temperature: float = 0.1, max_tokens: int = 500) -> str:
        """
        Synchronous chat completion request using the ollama library.
        Accepts either a string prompt or a list of message dicts.
        """
        try:
            logger.info(f"[OllamaClient] Sending request to model={self.model}, timeout={self.timeout}s")

            # Handle both string and messages formats
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], str):
                # If it's a list of strings, convert to message format
                messages = [{"role": "user", "content": messages[0]}]

            # Log message preview and options
            try:
                _preview = messages[0].get('content', '') if isinstance(messages, list) else str(messages)
                logger.info("[OllamaClient] message preview: %s", (_preview[:200] + ("..." if len(_preview) > 200 else "")))
            except Exception:
                pass
            _options = {"temperature": temperature, "num_predict": max_tokens}
            logger.info("[OllamaClient] options: %s", _options)

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=_options
            )
            content = response.get('message', {}).get('content', '')
            logger.info("[OllamaClient] response content preview: %s", (content[:200] + ("..." if len(content) > 200 else "")))
            if not content:
                logger.warning(f"Ollama returned empty response for model {self.model}")
                return ""
            return content
        except TimeoutError as e:
            logger.error(f"Ollama request timed out after {self.timeout}s for model {self.model}: {e}")
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if 'timeout' in error_msg or 'timed out' in error_msg:
                logger.error(f"Ollama request timed out after {self.timeout}s for model {self.model}: {e}")
            else:
                logger.error(f"Ollama library request failed (timeout: {self.timeout}s) for model {self.model}: {e}")
            raise


    async def acomplete(self, messages, temperature: float = 0.1, max_tokens: int = 500) -> str:
        """
        Asynchronous chat completion request using httpx for direct HTTP calls.
        This allows truly parallel requests to Ollama's REST API.
        """
        try:
            import httpx

            # Handle both string and messages formats
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], str):
                # If it's a list of strings, convert to message format
                messages = [{"role": "user", "content": messages[0]}]

            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            }
            try:
                _preview = messages[0].get('content', '') if isinstance(messages, list) else str(messages)
                logger.info("[OllamaClient] (async) message preview: %s", (_preview[:200] + ("..." if len(_preview) > 200 else "")))
                logger.info("[OllamaClient] (async) options: %s", payload.get('options'))
            except Exception:
                pass

            # Use httpx for async HTTP request - this allows truly parallel requests
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
                content = result.get('message', {}).get('content', '')
                logger.info("[OllamaClient] (async) response content preview: %s", (content[:200] + ("..." if len(content) > 200 else "")))
                if not content:
                    logger.warning(f"Ollama returned empty response for model {self.model}")
                    return ""
                return content

        except ImportError:
            # Fallback to synchronous method if httpx not available
            logger.warning("httpx not available, falling back to sync method for async call")
            return self.complete(messages, temperature, max_tokens)
        except Exception as e:
            logger.error(f"Async Ollama request failed (timeout: {self.timeout}s): {e}")
            raise
