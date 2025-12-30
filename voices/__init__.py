"""
Voices Module

Provides parallel LLM request handling with session-based logging.
Reuses libraries from AIAgent.py and utils/llm_client.py.
"""

from .parallel_llm import ParallelLLMRequest, ParallelLLMResponse
from .session_logger import SessionLogger
from .response_parser import ResponseParser

__all__ = ['ParallelLLMRequest', 'ParallelLLMResponse', 'SessionLogger', 'ResponseParser']

