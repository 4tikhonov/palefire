import ollama
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

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

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 500) -> str:
        """
        Synchronous chat completion request using the ollama library.
        """
        try:
            logger.debug(f"Sending request to Ollama model {self.model} with timeout {self.timeout}s")
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            content = response.get('message', {}).get('content', '')
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

    async def acomplete(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 500) -> str:
        """
        Asynchronous chat completion request using the ollama library.
        """
        try:
            # ollama.AsyncClient can be used for async calls
            async_client = ollama.AsyncClient(host=self.host, timeout=self.timeout)
            response = await async_client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama library async request failed (timeout: {self.timeout}s): {e}")
            raise
