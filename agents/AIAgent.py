"""
Pale Fire - AI Agent Daemon

Manages long-running AI agents that keep Gensim and spaCy models loaded in memory
to avoid start/stop delays. Provides a daemon service for keyword extraction and NER.
"""

from modules.PaleFireCore import EntityEnricher
from modules.KeywordBase import KeywordExtractor
import logging
import signal
import sys
import os
import time
import threading
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import re

# LLM functionality - implement voices-like functionality directly
# Import required modules for LLM operations
try:
    from ollama import Client as OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama client not available")

import asyncio
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

# LLM functionality - voices-like implementation
@dataclass
class LLMResponse:
    """Response from a single LLM model."""
    model: str
    response: str
    success: bool
    error: Optional[str] = None
    request_id: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    parsed_json_path: Optional[str] = None

class SessionLogger:
    """Session-based logger for LLM requests and responses."""

    def __init__(self, session_id: Optional[str] = None):
        if session_id is None:
            session_id = str(uuid.uuid4())
        self.session_id = session_id
        self.logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        self.session_dir = os.path.join(self.logs_dir, session_id)
        os.makedirs(self.session_dir, exist_ok=True)

    def log_request(self, model: str, prompt: Union[str, List[Dict[str, str]]], request_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        request_id = str(uuid.uuid4())[:8]
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
        filename = f"request_{request_type}_{safe_model}_{timestamp}_{request_id}.txt"
        filepath = os.path.join(self.session_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Request ID: {request_id}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Request Type: {request_type}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                if metadata:
                    f.write("Metadata:\n")
                    for key, value in metadata.items():
                        f.write(f"  {key}: {value}\n")

                f.write("\n" + "="*80 + "\nPROMPT:\n" + "="*80 + "\n")

                if isinstance(prompt, str):
                    f.write(prompt)
                elif isinstance(prompt, list):
                    for msg in prompt:
                        if isinstance(msg, dict):
                            f.write(f"{msg.get('role', 'user')}: {msg.get('content', '')}\n")
                        else:
                            f.write(f"{msg}\n")

                f.write("\n" + "="*80 + "\n")
        except Exception as e:
            logger.warning(f"Failed to log request: {e}")

        return request_id

    def log_response(self, model: str, response: str, request_id: str, request_type: str, metadata: Optional[Dict[str, Any]] = None):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
        filename = f"response_{request_type}_{safe_model}_{timestamp}_{request_id}.txt"
        filepath = os.path.join(self.session_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Request ID: {request_id}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Request Type: {request_type}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Response Length: {len(response) if response else 0}\n")

                if metadata:
                    f.write("Metadata:\n")
                    for key, value in metadata.items():
                        f.write(f"  {key}: {value}\n")

                f.write("\n" + "="*80 + "\nRESPONSE:\n" + "="*80 + "\n")
                f.write(response if response else "(empty response)")
                f.write("\n" + "="*80 + "\n")
        except Exception as e:
            logger.warning(f"Failed to log response: {e}")

    def log_json(self, data: Dict[str, Any], filename: str, subfolder: Optional[str] = None) -> str:
        target_dir = self.session_dir
        if subfolder:
            target_dir = os.path.join(target_dir, subfolder)
            os.makedirs(target_dir, exist_ok=True)

        if not any(c.isdigit() for c in filename[:8]):
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(target_dir, f"{timestamp}_{filename}")
        else:
            filepath = os.path.join(target_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to log JSON: {e}")
            return ""

        return filepath

class LLMRequestHandler:
    """Parallel LLM request handler - voices-like functionality."""

    def __init__(self, agent_instance=None, models: Optional[List[str]] = None, timeout: Optional[int] = None):
        self.agent = agent_instance  # Reference to AIAgent instance for parsing methods
        self.models = models or []
        self.timeout = timeout or 300
        self.session_logger = SessionLogger()

        if not self.models and CONFIG_AVAILABLE:
            llm_cfg = config.get_llm_config()
            verification_models = llm_cfg.get('verification_models', [])
            if verification_models:
                self.models = verification_models
            else:
                self.models = [llm_cfg.get('model', 'deepseek-r1:7b')]

    def _create_client(self, model: str):
        """Create Ollama client for model."""
        if not OLLAMA_AVAILABLE:
            return None

        if CONFIG_AVAILABLE:
            llm_cfg = config.get_llm_config()
            base_url = llm_cfg.get('base_url', 'http://localhost:11434').rstrip('/v1')
            return OllamaClient(host=base_url, timeout=self.timeout)
        else:
            return OllamaClient(timeout=self.timeout)

    async def _send_request_async(self, model: str, prompt: Union[str, List[Dict[str, str]]], request_type: str, temperature: float, max_tokens: int, parse_json: bool = False, original_entities: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        client = self._create_client(model)
        if not client:
            return LLMResponse(model=model, response="", success=False, error="Client creation failed")

        request_id = self.session_logger.log_request(model, prompt, request_type, {
            'temperature': temperature, 'max_tokens': max_tokens
        })

        try:
            # Prepare messages
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt

            response = client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )

            response_text = response['message']['content']

            self.session_logger.log_response(model, response_text, request_id, request_type)

            # Parse response to JSON if requested
            parsed_json = None
            parsed_json_path = None

            if parse_json:
                try:
                    # Use the appropriate parsing method based on request type
                    if request_type.startswith("ner_verification") and original_entities:
                        # For NER verification, use the specialized parser
                        parsed_entities = self.agent._parse_llm_response(response_text, original_entities)
                        parsed_json = {
                            'parsed': True,
                            'format': 'ner_entities',
                            'data': parsed_entities,
                            'original_response': response_text,
                            'response_length': len(response_text)
                        }
                    else:
                        # For general responses, try to parse as structured text
                        parsed_data = self._parse_general_response(response_text)
                        parsed_json = {
                            'parsed': True,
                            'format': 'structured_text',
                            'data': parsed_data,
                            'original_response': response_text,
                            'response_length': len(response_text)
                        }

                    # Save parsed JSON to file
                    safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
                    json_filename = f"parsed_{request_type}_{safe_model}_{request_id}.json"
                    parsed_json_path = self.session_logger.log_json(parsed_json, json_filename, subfolder="parsed")

                except Exception as parse_error:
                    logger.warning(f"Failed to parse JSON response from {model}: {parse_error}")
                    parsed_json = {
                        'parsed': False,
                        'format': 'raw_text',
                        'data': {'text': response_text},
                        'original_response': response_text,
                        'response_length': len(response_text),
                        'parse_error': str(parse_error)
                    }

            return LLMResponse(
                model=model,
                response=response_text,
                success=True,
                request_id=request_id,
                parsed_json=parsed_json,
                parsed_json_path=parsed_json_path
            )

        except Exception as e:
            error_msg = str(e)
            self.session_logger.log_response(model, f"ERROR: {error_msg}", request_id, request_type, {'error': error_msg})
            return LLMResponse(
                model=model,
                response="",
                success=False,
                error=error_msg,
                request_id=request_id
            )

    async def send_parallel_async(self, prompt: Union[str, List[Dict[str, str]]], request_type: str, temperature: float, max_tokens: int, parse_json: bool = False, original_entities: Optional[List[Dict[str, Any]]] = None) -> List[LLMResponse]:
        if not self.models:
            return []

        logger.info(f"Sending parallel requests to {len(self.models)} models: {', '.join(self.models)}")

        tasks = [
            self._send_request_async(model, prompt, request_type, temperature, max_tokens, parse_json, original_entities)
            for model in self.models
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                model = self.models[i]
                logger.error(f"Exception for model {model}: {result}")
                responses.append(LLMResponse(
                    model=model,
                    response="",
                    success=False,
                    error=str(result)
                ))
            else:
                responses.append(result)

        successful = [r for r in responses if r.success]
        failed = [r for r in responses if not r.success]

        logger.info(f"Parallel requests completed: {len(successful)} successful, {len(failed)} failed")

        return responses

    def send_parallel(self, prompt: Union[str, List[Dict[str, str]]], request_type: str, temperature: float, max_tokens: int, parse_json: bool = False, original_entities: Optional[List[Dict[str, Any]]] = None) -> List[LLMResponse]:
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            logger.warning("nest_asyncio not available")

        try:
            return asyncio.run(self.send_parallel_async(prompt, request_type, temperature, max_tokens, parse_json, original_entities))
        except Exception as e:
            logger.warning(f"Parallel execution failed: {e}, falling back to sequential")
            return self.send_sequential(prompt, request_type, temperature, max_tokens, parse_json, original_entities)

    def send_sequential(self, prompt: Union[str, List[Dict[str, str]]], request_type: str, temperature: float, max_tokens: int, parse_json: bool = False, original_entities: Optional[List[Dict[str, Any]]] = None) -> List[LLMResponse]:
        responses = []
        for model in self.models:
            client = self._create_client(model)
            if not client:
                responses.append(LLMResponse(
                    model=model,
                    response="",
                    success=False,
                    error="Client creation failed"
                ))
                continue

            request_id = self.session_logger.log_request(model, prompt, request_type, {
                'temperature': temperature, 'max_tokens': max_tokens
            })

            try:
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages = prompt

                response = client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )

                response_text = response['message']['content']

                self.session_logger.log_response(model, response_text, request_id, request_type)

                # Parse response to JSON if requested
                parsed_json = None
                parsed_json_path = None

                if parse_json:
                    try:
                        # Use the appropriate parsing method based on request type
                        if request_type.startswith("ner_verification") and original_entities:
                            # For NER verification, use the specialized parser
                            parsed_entities = self.agent._parse_llm_response(response_text, original_entities)
                            parsed_json = {
                                'parsed': True,
                                'format': 'ner_entities',
                                'data': parsed_entities,
                                'original_response': response_text,
                                'response_length': len(response_text)
                            }
                        else:
                            # For general responses, try to parse as structured text
                            parsed_data = self.agent._parse_general_response(response_text)
                            parsed_json = {
                                'parsed': True,
                                'format': 'structured_text',
                                'data': parsed_data,
                                'original_response': response_text,
                                'response_length': len(response_text)
                            }

                        # Save parsed JSON to file
                        safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
                        json_filename = f"parsed_{request_type}_{safe_model}_{request_id}.json"
                        parsed_json_path = self.session_logger.log_json(parsed_json, json_filename, subfolder="parsed")

                    except Exception as parse_error:
                        logger.warning(f"Failed to parse JSON response from {model}: {parse_error}")
                        parsed_json = {
                            'parsed': False,
                            'format': 'raw_text',
                            'data': {'text': response_text},
                            'original_response': response_text,
                            'response_length': len(response_text),
                            'parse_error': str(parse_error)
                        }

                responses.append(LLMResponse(
                    model=model,
                    response=response_text,
                    success=True,
                    request_id=request_id,
                    parsed_json=parsed_json,
                    parsed_json_path=parsed_json_path
                ))

            except Exception as e:
                error_msg = str(e)
                self.session_logger.log_response(model, f"ERROR: {error_msg}", request_id, request_type, {'error': error_msg})
                responses.append(LLMResponse(
                    model=model,
                    response="",
                    success=False,
                    error=error_msg,
                    request_id=request_id
                ))

        return responses

    def get_session_id(self) -> str:
        return self.session_logger.session_id

# Mark voices as available since we implemented it directly
VOICES_AVAILABLE = True


# Import models

# Try to import config for LLM access
try:
    import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logger.warning("Config module not available, LLM verification disabled")

# Import parsers (optional)
try:
    from .parsers import get_parser, BaseParser, ParseResult
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False
    logger.warning("File parsers not available")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available")


class ModelManager:
    """
    Manages loaded models to keep them in memory.
    Provides thread-safe access to models.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._keyword_extractor: Optional[KeywordExtractor] = None
        self._entity_enricher: Optional[EntityEnricher] = None
        self._spacy_model = None
        self._initialized = False

    def initialize(self, use_spacy: bool = True):
        """Initialize and load all models."""
        with self._lock:
            if self._initialized:
                logger.info("Models already initialized")
                return

            logger.info("Initializing AI models...")

            # Initialize Keyword Extractor (Gensim)
            try:
                self._keyword_extractor = KeywordExtractor()
                logger.info("✅ KeywordExtractor (Gensim) initialized")
            except Exception as e:
                logger.error(f"Failed to initialize KeywordExtractor: {e}")
                raise

            # Initialize Entity Enricher (spaCy)
            if use_spacy and SPACY_AVAILABLE:
                try:
                    self._entity_enricher = EntityEnricher(use_spacy=True)
                    logger.info("✅ EntityEnricher (spaCy) initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize spaCy enricher: {e}")
                    self._entity_enricher = EntityEnricher(use_spacy=False)
            else:
                self._entity_enricher = EntityEnricher(use_spacy=False)
                logger.info("✅ EntityEnricher (pattern-based) initialized")

            self._initialized = True
            logger.info("✅ All models initialized successfully")

    @property
    def keyword_extractor(self) -> KeywordExtractor:
        """Get keyword extractor instance."""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Models not initialized. Call initialize() first.")
            return self._keyword_extractor

    @property
    def entity_enricher(self) -> EntityEnricher:
        """Get entity enricher instance."""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Models not initialized. Call initialize() first.")
            return self._entity_enricher

    def is_initialized(self) -> bool:
        """Check if models are initialized."""
        with self._lock:
            return self._initialized

    def reload(self):
        """Reload all models."""
        with self._lock:
            logger.info("Reloading models...")
            self._initialized = False
            self._keyword_extractor = None
            self._entity_enricher = None
            self._spacy_model = None
            self.initialize()


class AIAgentDaemon:
    """
    Daemon service that keeps AI models loaded and ready.
    Provides API endpoints for keyword extraction and NER.
    """

    def __init__(self, pidfile: Optional[str] = None, use_spacy: bool = True):
        """
        Initialize the daemon.

        Args:
            pidfile: Path to PID file for daemon management
            use_spacy: Whether to use spaCy for NER
        """
        self.pidfile = pidfile or '/tmp/palefire_ai_agent.pid'
        self.use_spacy = use_spacy
        self.model_manager = ModelManager()
        self.running = False
        self._shutdown_event = threading.Event()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def start(self, daemon: bool = False):
        """
        Start the daemon service.

        Args:
            daemon: If True, run as background daemon
        """
        if self.running:
            logger.warning("Daemon is already running")
            return

        # Write PID file before daemonizing (so parent can read it)
        if daemon:
            # Write PID file with parent PID first (will be updated after fork)
            parent_pid = os.getpid()
            with open(self.pidfile, 'w') as f:
                f.write(str(parent_pid))
            self._daemonize()
            # After daemonizing, update PID file with actual daemon PID
            with open(self.pidfile, 'w') as f:
                f.write(str(os.getpid()))
        else:
            # Write PID file for foreground mode
            with open(self.pidfile, 'w') as f:
                f.write(str(os.getpid()))

        logger.info("Starting AI Agent Daemon...")
        self.running = True

        # Initialize models
        try:
            self.model_manager.initialize(use_spacy=self.use_spacy)
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.stop()
            sys.exit(1)

        # Run main loop
        self._run()

    def _daemonize(self):
        """Fork process to run as daemon."""
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            logger.error(f"Fork failed: {e}")
            sys.exit(1)

        # Decouple from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)

        # Second fork
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            logger.error(f"Second fork failed: {e}")
            sys.exit(1)

        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(os.devnull, 'r')
        so = open(os.devnull, 'a+')
        se = open(os.devnull, 'a+')
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

    def _run(self):
        """Main daemon loop."""
        logger.info("AI Agent Daemon is running...")

        # Health check thread
        health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        health_thread.start()

        try:
            # Keep running until shutdown
            while self.running and not self._shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.stop()

    def _health_check_loop(self):
        """Periodic health check to ensure models are loaded."""
        while self.running and not self._shutdown_event.is_set():
            time.sleep(60)  # Check every minute
            if not self.model_manager.is_initialized():
                logger.warning("Models not initialized, attempting reload...")
                try:
                    self.model_manager.initialize(use_spacy=self.use_spacy)
                except Exception as e:
                    logger.error(f"Failed to reload models: {e}")

    def stop(self):
        """Stop the daemon."""
        if not self.running:
            return

        logger.info("Stopping AI Agent Daemon...")
        self.running = False
        self._shutdown_event.set()

        # Remove PID file
        try:
            if os.path.exists(self.pidfile):
                os.remove(self.pidfile)
        except Exception as e:
            logger.warning(f"Failed to remove PID file: {e}")

        logger.info("AI Agent Daemon stopped")

    def extract_keywords(self, text: str, method: str = 'combined', num_keywords: int = 20, verify_ner: bool = False, deep: bool = False, blocksize: int = 1, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract keywords using the loaded model.

        Args:
            text: Text to extract keywords from
            method: Extraction method ('tfidf', 'textrank', 'word_freq', 'combined', 'ner')
            num_keywords: Number of keywords to extract
            verify_ner: If True and method is 'ner', verify results using LLM
            deep: If True and method is 'ner', process text sentence-by-sentence with ordered index
            blocksize: Number of sentences per block when deep=True (default: 1 = sentence-by-sentence)
            **kwargs: Additional arguments for KeywordExtractor.extract()

        Returns:
            List of keyword dictionaries
        """
        # If NER method is requested, use spaCy NER for keyword extraction
        if method == 'ner':
            return self.extract_keywords_ner(text, num_keywords=num_keywords, verify_ner=verify_ner, deep=deep, blocksize=blocksize)

        # Otherwise, use gensim-based methods
        # Create a new extractor with the desired parameters
        extractor_kwargs = {
            'method': method,
            'num_keywords': num_keywords,
            **kwargs  # Additional parameters
        }
        extractor = KeywordExtractor(**extractor_kwargs)

        return extractor.extract(text)

    def _parse_structured_markdown(self, response: str, original_entities: List[Dict[str, Any]], model_name: str = "LLM") -> List[Dict[str, Any]]:
        """
        Parse simplified structured markdown format response (optimized for gemma3:4b).

        Supported formats:
        **1. Verified Entities:**
        * Entity Name (TYPE) - Brief reason
        * **Entity Name** (TYPE) - Brief reason

        **2. New Entities Found:**
        * New Entity (TYPE) - Brief reason
        * **New Entity** (TYPE) - Brief reason

        **3. Entities to Remove:**
        * Entity Name (TYPE) - Reason to remove
        * **Entity Name** (TYPE) - Reason to remove

        Args:
            response: Markdown-formatted response with simplified structure
            original_entities: Original entities from spaCy for matching
            model_name: Name of the LLM model used (for logging)

        Returns:
            List of parsed entities in standard format
        """
        result = []

        # Split response into lines
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line.startswith('**1. Verified Entities:**'):
                current_section = 'verified'
                continue
            elif line.startswith('**2. New Entities Found:**'):
                current_section = 'new'
                continue
            elif line.startswith('**3. Entities to Remove:**'):
                current_section = 'remove'
                continue

            # Skip other header lines
            if line.startswith('**') and ('Entities' in line or 'Found' in line or 'Remove' in line):
                continue

            # Process bullet points based on current section
            if current_section and re.match(r'^[\*\-\•]\s+', line):
                if current_section == 'remove':
                    # Pattern for removal: * Entity Name (TYPE) - Reason OR * **Entity Name** (TYPE) - Reason
                    remove_match = re.match(
                        r'^[\*\-\•]\s+(?:\*\*)?([^*]+?)(?:\*\*)?\s*(?:\(([A-Z_]+)\))?\s*(?:\-\s*(.*))?$', line)
                    if remove_match:
                        entity_text = remove_match.group(1).strip()
                        entity_type = remove_match.group(2).strip() if remove_match.group(2) else ""
                        reason = remove_match.group(3).strip() if remove_match.group(3) else "Marked for removal"

                        # Look for matching original entity to preserve position info
                        entity_lower = entity_text.lower()
                        matched_original = None
                        for orig in original_entities:
                            if orig.get('text', '').lower() == entity_lower:
                                matched_original = orig
                                break

                        # Create a "remove" entity (always, since LLM says to remove it)
                        entity = {
                            'text': entity_text,
                            'type': 'REMOVE',
                            'confidence': 0.8,
                            'reasoning': f"Marked for removal by {model_name}: {reason}",
                            'source': 'verified'
                        }

                        # Preserve position info if we found a match
                        if matched_original:
                            entity.update({
                                'start': matched_original.get('start', 0),
                                'end': matched_original.get('end', 0)
                            })

                        result.append(entity)

                else:  # verified or new entities
                    # Pattern: * Entity Name (TYPE) - Brief reason OR * **Entity Name** (TYPE) - Brief reason
                    entity_match = re.match(
                        r'^[\*\-\•]\s+(?:\*\*)?([^*]+?)(?:\*\*)?\s*\(([A-Z_]+)\)\s*(?:\-\s*(.*))?$', line)
                    if entity_match:
                        entity_text = entity_match.group(1).strip()
                        entity_type = entity_match.group(2).strip()
                        description = entity_match.group(3).strip() if entity_match.group(3) else ""

                        if entity_text and entity_type:
                            # Try to match with original entity (case-insensitive)
                            entity_lower = entity_text.lower()
                            matched_original = None
                            for orig in original_entities:
                                if orig.get('text', '').lower() == entity_lower:
                                    matched_original = orig
                                    break

                            # Create entity in standard format
                            source = 'verified' if matched_original else 'discovered'

                            entity = {
                                'text': entity_text,
                                'type': entity_type,
                                'confidence': 0.9,  # High confidence from LLM
                                'reasoning': f"Identified by {model_name} as {entity_type}" + (f" - {description}" if description else ""),
                                'source': source
                            }

                            # Preserve position info from original if available
                            if matched_original:
                                entity.update({
                                    'start': matched_original.get('start', 0),
                                    'end': matched_original.get('end', 0)
                                })

                            result.append(entity)

        return result

    def _parse_gemma3_markdown(self, response: str, original_entities: List[Dict[str, Any]], model_name: str = "LLM") -> List[Dict[str, Any]]:
        """
        Parse Gemma3 markdown format response and convert to entity format.

        Gemma3 format example:
        **Organizations (ORG)**
        *   Ask.now.museum
        *   Elastic Search

        **People/Entities (PER)**
        *   John Doe

        Args:
            response: Markdown-formatted response from Gemma3
            original_entities: Original entities from spaCy for matching
            model_name: Name of the LLM model used (for logging)

        Returns:
            List of parsed entities in standard format
        """
        result = []

        # Pattern to match headers like "**Organizations (ORG)**" or "**People/Entities (PER)**"
        header_pattern = r'\*\*([^*]+?)\s*\((\w+)\)\*\*'

        # Split response into lines
        lines = response.split('\n')
        current_type = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a header with entity type
            header_match = re.match(header_pattern, line)
            if header_match:
                current_type = header_match.group(2)  # Extract entity type (ORG, PER, etc.)
                continue

            # Check if this is a bullet point (starts with *, -, or •)
            if re.match(r'^[\*\-\•]\s+', line):
                # Extract entity name (remove bullet, trim, remove parenthetical descriptions)
                entity_text = re.sub(r'^[\*\-\•]\s+', '', line)
                # Remove parenthetical descriptions like "(appears frequently)" or "(as an open-source LLM provider)"
                entity_text = re.sub(r'\s*\([^)]+\)\s*$', '', entity_text).strip()

                if entity_text and current_type:
                    # Try to match with original entity (case-insensitive)
                    entity_lower = entity_text.lower()
                    original = next(
                        (e for e in original_entities
                         if e['text'].strip().lower() == entity_lower or
                            entity_lower in e['text'].strip().lower() or
                            e['text'].strip().lower() in entity_lower),
                        None
                    )

                    if original:
                        result.append({
                            'text': original['text'],  # Use original text for exact match
                            'type': current_type,
                            'verified': True,
                            'start': original.get('start', 0),
                            'end': original.get('end', 0),
                            'confidence': 1.0,
                            'reasoning': f"Verified as {current_type} by LLM model {model_name}"
                        })
                    else:
                        # Entity not found in original list, but LLM identified it
                        result.append({
                            'text': entity_text,
                            'type': current_type,
                            'verified': True,
                            'start': 0,
                            'end': 0,
                            'confidence': 0.8,  # Lower confidence for new entities
                            'reasoning': f"Identified as {current_type} by LLM model {model_name} (not in original spaCy results)"
                        })

        return result


    def verify_ner_with_llm(self, text: str, entities: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
        """
        Verify and correct spaCy NER results using LLM via voices module.

        Args:
            text: Original text
            entities: List of entities from spaCy
            debug: Enable debug output

        Returns:
            List of verified entities (with REMOVE type for false positives)
        """
        if not VOICES_AVAILABLE:
            logger.warning("Voices module not available, skipping LLM verification")
            return entities

        if not CONFIG_AVAILABLE:
            logger.warning("Config not available, skipping LLM verification")
            return entities

        try:
            # Create a verification prompt for the entities
            system_prompt = "You are an expert at Named Entity Recognition (NER). Verify and correct the following spaCy NER entities.\n\n"

            # Add entity type definitions
            entity_type_descriptions = self.model_manager.entity_enricher.ENTITY_TYPES
            system_prompt += "Entity type definitions:\n"
            for entity_type in entity_type_descriptions:
                if entity_type == 'PERSON' or entity_type == 'PER':
                    system_prompt += f"- {entity_type}: Person indicates human beings being mentioned\n"
                elif entity_type == 'ORG':
                    system_prompt += f"- {entity_type}: Organization indicates companies, agencies, institutions\n"
                elif entity_type == 'GPE':
                    system_prompt += f"- {entity_type}: Geopolitical Entity indicates countries, cities, states\n"
                elif entity_type == 'LOC':
                    system_prompt += f"- {entity_type}: Location indicates non-GPE locations, mountain ranges, bodies of water\n"
                else:
                    system_prompt += f"- {entity_type}: {entity_type} entity\n"
            system_prompt += "\n"

            # Add full text context
            if text and text.strip():
                system_prompt += f"Full text context:\n{text.strip()}\n"

            # Add instructions
            system_prompt += "\nReturn verified entities in this simple format:\n\n**1. Verified Entities:**\n* **Entity Name** (TYPE) - Brief reason\n* **Another Entity** (TYPE) - Brief reason\n\n**2. New Entities Found:**\n* **New Entity** (TYPE) - Brief reason\n\n**3. Entities to Remove:**\n* **False Entity** - Reason to remove\n\nUse these entity types: PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT, etc.\n\nBe concise and accurate. Only include entities that actually exist in the text."

            # Prepare entities for verification
            entities_json = json.dumps(entities, separators=(',', ':'))
            verification_prompt = f"{system_prompt}\n\nEntities to verify:\n{entities_json}"

            # Use embedded LLM request handler (voices-like functionality)
            logger.info("Creating LLMRequestHandler for NER verification")
            request_handler = LLMRequestHandler(agent_instance=self)

            # Log what models are configured
            logger.info(f"LLMRequestHandler configured with {len(request_handler.models)} models: {request_handler.models}")

            # Send parallel requests to verification models
            logger.info("Sending parallel NER verification requests")
            responses = request_handler.send_parallel(
                prompt=verification_prompt,
                request_type="ner_verification",
                temperature=0.1,
                max_tokens=500,
                parse_json=True,  # Enable JSON parsing for NER verification
                original_entities=entities  # Pass original entities for parsing
            )

            logger.info(f"Received {len(responses)} responses from LLM request handler")

            # Process responses and extract entities
            all_verified_entities = []
            successful_models = []

            for response in responses:
                if response.success:
                    # Use parsed JSON if available, otherwise parse manually
                    if response.parsed_json and response.parsed_json.get('parsed', False):
                        # Use pre-parsed entities
                        parsed_entities = response.parsed_json.get('data', [])
                        if isinstance(parsed_entities, list):
                            for entity in parsed_entities:
                                if isinstance(entity, dict):
                                    entity_copy = entity.copy()
                                    entity_copy['source_model'] = response.model
                                    all_verified_entities.append(entity_copy)
                            successful_models.append(response.model)
                            logger.debug(f"Successfully used pre-parsed {len(parsed_entities)} entities from {response.model}")
                        else:
                            logger.warning(f"Invalid parsed data format from {response.model}")
                    else:
                        # Fallback to manual parsing
                        try:
                            parsed_entities = self._parse_llm_response(response.response, entities)
                            for entity in parsed_entities:
                                if isinstance(entity, dict):
                                    entity_copy = entity.copy()
                                    entity_copy['source_model'] = response.model
                                    all_verified_entities.append(entity_copy)
                            successful_models.append(response.model)
                            logger.debug(f"Successfully parsed {len(parsed_entities)} entities from {response.model}")
                        except Exception as e:
                            logger.warning(f"Failed to parse response from {response.model}: {e}")
                else:
                    logger.warning(f"Request failed for model {response.model}: {response.error}")

            if not all_verified_entities:
                logger.warning("No successful LLM verification responses, returning original entities")
                return entities

            # Merge results from multiple models
            merged_entities = self._merge_multi_model_results(all_verified_entities, entities, successful_models)
            logger.info(f"LLM verification complete: {len(merged_entities)} entities from {len(successful_models)} models")
            return merged_entities

        except Exception as e:
            logger.warning(f"LLM verification failed in verify_ner_with_llm: {e}, using original entities")
            if debug:
                import traceback
                traceback.print_exc()
            return entities

    def _parse_verification_models(self, verification_model_str: Optional[str], main_model: str) -> List[str]:
        """
        Parse verification model string into a list of model names.

        Args:
            verification_model_str: Comma-separated string of model names or None
            main_model: Fallback main model name

        Returns:
            List of model names to use for verification
        """
        if verification_model_str and verification_model_str.strip():
            # Split by comma and strip whitespace
            models = [model.strip() for model in verification_model_str.split(',') if model.strip()]
            # Remove empty strings and return valid models
            return [model for model in models if model]
        else:
            # No verification models specified, use main model
            return [main_model]

    def _create_ollama_client(self, model_name: str, timeout: int = 300):
        """Create an Ollama client for the given model."""
        if not CONFIG_AVAILABLE:
            return None

        llm_cfg = config.get_llm_config()
        if not llm_cfg.get('api_key'):
            return None

        from utils.llm_client import SimpleOllamaClient

        return SimpleOllamaClient(
            model=model_name,
            base_url=llm_cfg['base_url'],
            api_key=llm_cfg['api_key'],
            timeout=timeout
        )

    async def _verify_ner_with_llm_client_async(self, text: str, entities: List[Dict[str, Any]], llm_client, llm_model: str, debug: bool = False) -> List[Dict[str, Any]]:
        """
        Async version of _verify_ner_with_llm_client for parallel processing.
        """
        if not CONFIG_AVAILABLE:
            logger.warning("Config not available, skipping LLM verification")
            return []

        # Check if models are initialized
        if not self.model_manager.is_initialized():
            logger.warning(f"Models not initialized, cannot perform LLM verification for {llm_model}")
            return []

        try:
            # Collect all entities and their context first
            entity_type_descriptions = {}
            for entity_type in self.model_manager.entity_enricher.ENTITY_TYPES:
                if entity_type == 'PERSON' or entity_type == 'PER':
                    entity_type_descriptions[entity_type] = 'Person indicates human beings being mentioned'
                elif entity_type == 'ORG':
                    entity_type_descriptions[entity_type] = 'Organization indicates companies, agencies, institutions'
                elif entity_type == 'GPE':
                    entity_type_descriptions[entity_type] = 'Geopolitical Entity indicates countries, cities, states'
                elif entity_type == 'LOC':
                    entity_type_descriptions[entity_type] = 'Location indicates non-GPE locations, mountain ranges, bodies of water'
                elif entity_type == 'MISC' or entity_type == 'NORP':
                    entity_type_descriptions[entity_type] = 'NORP indicates nationalities, religious or political groups'
                elif entity_type == 'FAC':
                    entity_type_descriptions[entity_type] = 'Facility indicates buildings, airports, highways, bridges'
                elif entity_type == 'PRODUCT':
                    entity_type_descriptions[
                        entity_type] = 'Product indicates objects, vehicles, foods, etc. (not services)'
                elif entity_type == 'EVENT':
                    entity_type_descriptions[entity_type] = 'Event indicates named hurricanes, battles, wars, sports events'
                elif entity_type == 'WORK_OF_ART':
                    entity_type_descriptions[entity_type] = 'Work of Art indicates titles of paintings, songs, etc.'
                elif entity_type == 'LAW':
                    entity_type_descriptions[entity_type] = 'Law indicates named documents made into laws'
                elif entity_type == 'LANGUAGE':
                    entity_type_descriptions[entity_type] = 'Language indicates any named language'
                elif entity_type == 'DATE':
                    entity_type_descriptions[entity_type] = 'Date indicates absolute or relative dates or periods'
                elif entity_type == 'TIME':
                    entity_type_descriptions[entity_type] = 'Times indicate times smaller than a day'
                elif entity_type == 'MONEY':
                    entity_type_descriptions[entity_type] = 'Money indicates monetary values, including unit'
                elif entity_type == 'PERCENT':
                    entity_type_descriptions[entity_type] = 'Percent indicates percentage values'
                elif entity_type == 'ORDINAL':
                    entity_type_descriptions[entity_type] = 'Ordinal indicates ordinal numbers (first, second, etc.)'
                elif entity_type == 'CARDINAL':
                    entity_type_descriptions[entity_type] = 'Cardinal indicates cardinal numbers that do not fall under another type'
                elif entity_type == 'QUANTITY':
                    entity_type_descriptions[entity_type] = 'Quantity indicates measurements, such as weight or distance'
                elif entity_type == 'BOOK':
                    entity_type_descriptions[entity_type] = 'Books indicate titles of published written works'
                elif entity_type == 'MODEL':
                    entity_type_descriptions[entity_type] = 'Models indicate AI models, machine learning models, or other named models'
                elif entity_type == 'SOFTWARE':
                    entity_type_descriptions[entity_type] = 'Software indicates named software applications, programs, or tools'
                elif entity_type == 'OTHER':
                    entity_type_descriptions[entity_type] = 'Not a recognized entity type'
                else:
                    entity_type_descriptions[entity_type] = f'{entity_type} entity'

            # Step 1: Collect all unique context information
            entity_contexts = {}  # Map entity text -> context info

            for e in entities:
                entity_type = e.get('type', 'UNKNOWN')
                entity_text = e.get('text', '').strip()

                if not entity_text:
                    continue

                # Collect context information
                if 'context' in e:
                    ctx = e['context']
                    sentence = ctx.get('sentence', '').strip()

                    # Store context for this entity
                    entity_contexts[entity_text] = {
                        'sentence': sentence,
                        'description': entity_type_descriptions.get(entity_type, f'{entity_type} entity')
                    }
                else:
                    # No context available, use type description only
                    entity_contexts[entity_text] = {
                        'sentence': '',
                        'description': entity_type_descriptions.get(entity_type, f'{entity_type} entity')
                    }

            # Step 2: Prepare entities for verification
            entities_for_verification = []
            for e in entities:
                entity_text = e.get('text', '').strip()
                entity_type = e.get('type', 'UNKNOWN')

                if not entity_text:
                    continue

                entity_data = {
                    'text': entity_text,
                    'type': entity_type,
                    'sentence': entity_contexts.get(entity_text, {}).get('sentence', ''),
                    'desc': entity_contexts.get(entity_text, {}).get('description', f'{entity_type} entity')
                }
                entities_for_verification.append(entity_data)

            if not entities_for_verification:
                return []

            # Step 3: Build the prompt
            import json

            # Build system prompt with all entity type definitions
            system_prompt = "You are an expert at Named Entity Recognition (NER). Verify and correct the following spaCy NER entities.\n\n"

            # Add entity type definitions
            system_prompt += "Entity type definitions:\n"
            for entity_type, description in entity_type_descriptions.items():
                system_prompt += f"- {entity_type}: {description}\n"
            system_prompt += "\n"

            # This provides complete context instead of just individual sentences
            if text and text.strip():
                system_prompt += f"\n\nFull text context:\n{text.strip()}\n"

            # Add instructions for structured markdown format (optimized for gemma3:4b)
            system_prompt += "\n\nReturn verified entities in this simple format:\n\n**1. Verified Entities:**\n* **Entity Name** (TYPE) - Brief reason\n* **Another Entity** (TYPE) - Brief reason\n\n**2. New Entities Found:**\n* **New Entity** (TYPE) - Brief reason\n\n**3. Entities to Remove:**\n* **False Entity** - Reason to remove\n\nUse these entity types: PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT, etc.\n\nBe concise and accurate. Only include entities that actually exist in the text."

            # Minimize JSON - use compact format (no indentation) to reduce tokens
            entities_json = json.dumps(entities_for_verification, separators=(',', ':'))

            # Final prompt with context in header
            full_prompt = f"{system_prompt}\n\nEntities to verify:\n{entities_json}"

            if debug:
                logger.debug(f"Sending verification request to {llm_model} with {len(entities_for_verification)} entities")
                logger.debug(f"Prompt length: {len(full_prompt)} characters")

            # Use SessionLogger for session-based logging (voices integration)
            session_logger = SessionLogger()

            # Log the LLM request
            request_id = session_logger.log_request(
                model=llm_model,
                prompt=full_prompt,
                request_type="ner_verification",
                metadata={
                    'entities_count': len(entities_for_verification),
                    'text_length': len(text) if text else 0
                }
            )

            # Make the async LLM call
            response = await llm_client.acomplete(full_prompt)

            if debug:
                logger.debug(f"LLM response length: {len(response) if response else 0}")

            # Log the response using SessionLogger
            session_logger.log_response(
                model=llm_model,
                response=response,
                request_id=request_id,
                request_type="ner_verification"
            )

            if not response or not response.strip():
                logger.warning(f"Empty response from LLM (model: {llm_model})")
                return []

            verified_entities = None
            json_parsing_attempted = False

            # Try to parse as direct JSON first
            try:
                verified_entities = json.loads(response.strip())
                json_parsing_attempted = True
                if debug:
                    logger.debug(f"Parsed {len(verified_entities) if verified_entities else 0} entities from direct JSON response")

                # Log parsed JSON using SessionLogger
                try:
                    safe_model = llm_model.replace(':', '_').replace('/', '_').replace('\\', '_')
                    json_filename = f"parsed_ner_verification_{safe_model}_{request_id}.json"
                    json_path = session_logger.log_json(verified_entities, json_filename, subfolder="parsed")
                    logger.debug(f"Saved parsed JSON to {json_path}")
                except Exception as json_log_error:
                    logger.warning(f"Failed to log parsed JSON: {json_log_error}")

            except json.JSONDecodeError:
                if debug:
                    logger.debug("Direct JSON parsing failed, trying to extract JSON from markdown code blocks...")

                # Try to extract JSON from markdown code blocks (```json ... ```)
                json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    try:
                        verified_entities = json.loads(json_content)
                        json_parsing_attempted = True
                        if debug:
                            logger.debug(f"Parsed {len(verified_entities) if verified_entities else 0} entities from markdown JSON code block")

                        # Log parsed JSON using SessionLogger
                        try:
                            safe_model = llm_model.replace(':', '_').replace('/', '_').replace('\\', '_')
                            json_filename = f"parsed_markdown_ner_verification_{safe_model}_{request_id}.json"
                            json_path = session_logger.log_json(verified_entities, json_filename, subfolder="parsed")
                            logger.debug(f"Saved parsed markdown JSON to {json_path}")
                        except Exception as json_log_error:
                            logger.warning(f"Failed to log parsed markdown JSON: {json_log_error}")

                    except json.JSONDecodeError as e:
                        if debug:
                            logger.debug(f"Failed to parse JSON from markdown code block: {e}")
                else:
                    if debug:
                        logger.debug("No markdown JSON code block found, trying Gemma3 markdown format...")

                # If we get here, all JSON parsing attempts failed, so try Gemma3 markdown format
                if debug:
                    logger.debug("All JSON parsing failed, trying Gemma3 markdown parser...")
                try:
                    verified_entities = self._parse_gemma3_markdown(response, entities, llm_model)

                    if verified_entities:
                        logger.debug(f"Parsed {len(verified_entities)} entities using Gemma3 markdown parser")

                        # Log parsed entities using SessionLogger
                        try:
                            safe_model = llm_model.replace(':', '_').replace('/', '_').replace('\\', '_')
                            json_filename = f"parsed_gemma3_ner_verification_{safe_model}_{request_id}.json"
                            json_path = session_logger.log_json(verified_entities, json_filename, subfolder="parsed")
                            logger.debug(f"Saved parsed Gemma3 entities as JSON to {json_path}")
                        except Exception as json_log_error:
                            logger.warning(f"Failed to log parsed Gemma3 JSON: {json_log_error}")
                    else:
                        logger.warning(f"No entities found in Gemma3 markdown response (model: {llm_model}), using original entities")
                        return []
                except Exception as gemma_error:
                    logger.warning(f"Failed to parse as Gemma3 markdown (model: {llm_model}): {gemma_error}. Response was: {response[:200]}...")
                    return []

            if verified_entities and len(verified_entities) > 0:

                # Filter out REMOVE entities and entities marked as verified=False
                result = []
                verified_count = discovered_count = corrected_count = removed_count = 0

                # Build a mapping from normalized entity text to original entities for position info
                original_entities_map = {e['text'].strip().lower(): e for e in entities}

                for verified in verified_entities:
                    if not isinstance(verified, dict):
                        continue

                    v_type = verified.get('type')
                    v_text = (verified.get('text') or verified.get('entity') or "").strip()
                    v_reasoning = (verified.get('reasoning') or "").strip()
                    v_verified = verified.get('verified', True)
                    v_new_entity = verified.get('new_entity', False)

                    if v_type == 'REMOVE' or v_verified is False:
                        removed_count += 1
                        continue

                    if v_text:
                        verified_text_lower = v_text.lower()
                        original = original_entities_map.get(verified_text_lower)

                        if original:
                            # Existing entity that was verified/corrected
                            result.append({
                                'text': v_text,
                                'type': v_type or original.get('type', 'UNKNOWN'),
                                'start': original.get('start', 0),
                                'end': original.get('end', 0),
                                'confidence': verified.get('confidence', 1.0),
                                'reasoning': (v_reasoning or f"Verified as {v_type or 'entity'} by LLM model {llm_model}").strip(),
                                'source': 'verified'  # Indicates this was verified from original spaCy entities
                            })
                            verified_count += 1
                        elif v_new_entity:
                            # Explicitly marked as new entity by LLM
                            # Try to find position in the full text if possible
                            start_pos = 0
                            end_pos = 0

                            # If we have the full text, try to find the entity position
                            if text and v_text in text:
                                start_pos = text.find(v_text)
                                if start_pos >= 0:
                                    end_pos = start_pos + len(v_text)

                            result.append({
                                'text': v_text,
                                'type': v_type or 'UNKNOWN',
                                'start': start_pos,
                                'end': end_pos,
                                'confidence': verified.get('confidence', 0.8),  # Slightly lower confidence for new entities
                                'reasoning': (v_reasoning or f"Discovered as {v_type or 'entity'} by LLM model {llm_model}").strip(),
                                'source': 'discovered'  # Indicates this was discovered by LLM
                            })
                            discovered_count += 1
                        else:
                            # LLM says this is not new (came from original list) but we can't find the original
                            # This could happen if spaCy missed it or there's a text variation
                            # Treat as verified but with estimated position
                            start_pos = 0
                            end_pos = 0

                            # If we have the full text, try to find the entity position
                            if text and v_text in text:
                                start_pos = text.find(v_text)
                                if start_pos >= 0:
                                    end_pos = start_pos + len(v_text)

                            result.append({
                                'text': v_text,
                                'type': v_type or 'UNKNOWN',
                                'start': start_pos,
                                'end': end_pos,
                                'confidence': verified.get('confidence', 0.9),  # High confidence since LLM verified it
                                'reasoning': (v_reasoning or f"Verified as {v_type or 'entity'} by LLM model {llm_model}").strip(),
                                'source': 'verified'  # Treat as verified since LLM says it came from original list
                            })
                            verified_count += 1

                if debug:
                    logger.debug(f"LLM verification (model: {llm_model}): Verified {verified_count}, Discovered {discovered_count}, Corrected {corrected_count} entities. Removed {removed_count} false positives.")

                return result
            else:
                logger.warning(f"No valid entities found in LLM verification response (model: {llm_model}), using original entities")
                return []
        except Exception as e:
            logger.warning(f"Async verification failed for {llm_model}: {e}")
            return []

    def _parse_llm_response(self, response_text: str, original_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse LLM response to extract verified entities."""
        entities = []

        # Split response into lines
        lines = response_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line.startswith('**1. Verified Entities:**'):
                current_section = 'verified'
                continue
            elif line.startswith('**2. New Entities Found:**'):
                current_section = 'new'
                continue
            elif line.startswith('**3. Entities to Remove:**'):
                current_section = 'remove'
                continue

            # Skip other headers
            if line.startswith('**') and ('Entities' in line or 'Found' in line or 'Remove' in line):
                continue

            # Process bullet points
            if current_section and re.match(r'^[\*\-\•]\s+', line):
                if current_section == 'remove':
                    # Pattern for removal: * Entity Name (TYPE) - Reason
                    match = re.match(r'^[\*\-\•]\s+(?:\*\*)?([^*]+?)(?:\*\*)?\s*(?:\(([A-Z_]+)\))?\s*(?:\-\s*(.*))?$', line)
                    if match:
                        entity_text = match.group(1).strip()
                        # For removal, we don't add to entities (they get filtered out)
                        continue
                else:
                    # Pattern for entities: * Entity Name (TYPE) - Brief reason
                    match = re.match(r'^[\*\-\•]\s+(?:\*\*)?([^*]+?)(?:\*\*)?\s*\(([A-Z_]+)\)\s*(?:\-\s*(.*))?$', line)
                    if match:
                        entity_text = match.group(1).strip()
                        entity_type = match.group(2).strip()
                        reason = match.group(3).strip() if match.group(3) else ""

                        if entity_text and entity_type:
                            # Try to match with original entity for position info
                            original = next((e for e in original_entities if e['text'].strip().lower() == entity_text.lower()), None)

                            entity = {
                                'text': entity_text,
                                'type': entity_type,
                                'confidence': 0.9,
                                'reasoning': f"Verified by LLM: {reason}" if reason else f"Verified by LLM as {entity_type}"
                            }

                            if original:
                                entity.update({
                                    'start': original.get('start', 0),
                                    'end': original.get('end', 0)
                                })

                            entities.append(entity)

        return entities

    def _parse_general_response(self, response_text: str) -> Dict[str, Any]:
        """Parse general LLM responses into structured format."""
        if not response_text or not response_text.strip():
            return {'text': response_text}

        # Try to parse as JSON first
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Try to parse as structured text
        return self._parse_structured_text(response_text)

    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text/markdown into JSON."""
        lines = text.split('\n')
        result = {}
        current_section = None
        current_list = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line.startswith('#'):
                current_section = line.lstrip('#').strip()
                result[current_section] = {}
                current_list = None
                continue
            elif line.startswith('**') and line.endswith('**'):
                current_section = line.strip('*').strip()
                if current_section not in result:
                    result[current_section] = {}
                current_list = None
                continue

            # Check for key-value pairs
            kv_match = re.match(r'^([^:]+):\s*(.+)$', line)
            if kv_match:
                key = kv_match.group(1).strip()
                value = kv_match.group(2).strip()

                key = re.sub(r'[*_`]', '', key).strip()

                if current_section:
                    if current_section not in result:
                        result[current_section] = {}
                    result[current_section][key] = value
                else:
                    result[key] = value
                continue

            # Check for list items
            list_match = re.match(r'^[\*\-\•]\s+(.+)$', line)
            if list_match:
                item = list_match.group(1).strip()
                item = re.sub(r'[*_`]', '', item).strip()

                if current_section:
                    if current_section not in result:
                        result[current_section] = []
                    if not isinstance(result[current_section], list):
                        result[current_section] = [result[current_section]]
                    result[current_section].append(item)
                else:
                    if 'items' not in result:
                        result['items'] = []
                    result['items'].append(item)
                continue

        return result if result else {'text': text}

    def _merge_multi_model_results(self, all_verified_entities: List[Dict[str, Any]],
                                  original_entities: List[Dict[str, Any]],
                                  successful_models: List[str]) -> List[Dict[str, Any]]:
        """
        Merge verification results from multiple LLM models using consensus-based approach.

        Args:
            all_verified_entities: List of all verified entities from all models
            original_entities: Original spaCy entities
            successful_models: List of models that returned results

        Returns:
            Merged entity list with consensus types and reasoning
        """
        from collections import defaultdict, Counter

        logger.info(f"Merging results from {len(successful_models)} models: {successful_models}")

        # Group entities by normalized text
        entity_groups = defaultdict(list)
        original_entity_map = {e['text'].strip().lower(): e for e in original_entities}

        for entity in all_verified_entities:
            text = entity.get('text', '').strip()
            if text:
                normalized_text = text.lower()
                entity_groups[normalized_text].append(entity)

        merged_entities = []

        for normalized_text, entity_list in entity_groups.items():
            # Count votes for each type
            type_votes = Counter(e.get('type') for e in entity_list if e.get('type'))
            reasoning_list = [e.get('reasoning', '') for e in entity_list if e.get('reasoning')]

            # Determine consensus type (majority vote)
            if type_votes:
                consensus_type, vote_count = type_votes.most_common(1)[0]
                confidence = vote_count / len(successful_models)  # Consensus confidence
            else:
                consensus_type = 'UNKNOWN'
                confidence = 0.0

            # Combine reasoning from all models
            combined_reasoning = '; '.join(filter(None, reasoning_list))

            # Use original entity for position info if available
            original = original_entity_map.get(normalized_text)
            if original:
                merged_entity = {
                    'text': original['text'],  # Use original casing
                    'type': consensus_type,
                    'start': original.get('start', 0),
                    'end': original.get('end', 0),
                    'confidence': confidence,
                    'reasoning': f"Consensus from {len(entity_list)}/{len(successful_models)} models: {combined_reasoning}",
                    'source': 'verified',
                    'source_models': [e.get('source_model') for e in entity_list],
                    'consensus_score': confidence
                }
            else:
                # New entity discovered by LLM
                merged_entity = {
                    'text': entity_list[0]['text'],
                    'type': consensus_type,
                    'start': 0,
                    'end': 0,
                    'confidence': confidence * 0.8,  # Slightly lower confidence for new entities
                    'reasoning': f"Discovered by consensus from {len(entity_list)}/{len(successful_models)} models: {combined_reasoning}",
                    'source': 'discovered',
                    'source_models': [e.get('source_model') for e in entity_list],
                    'consensus_score': confidence
                }

            merged_entities.append(merged_entity)

        logger.info(f"Merged {len(entity_groups)} entity groups into {len(merged_entities)} final entities")
        return merged_entities


    def _split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into sentences using spaCy and build ordered index.
        
        Returns:
            List of sentence dictionaries with 'text', 'index', 'start', 'end'
        """
        try:
            import spacy
            # Try to use spaCy if available (same as EntityEnricher)
            if self.model_manager.entity_enricher.use_spacy:
                # Load spaCy model directly (same as PaleFireCore does)
                try:
                    nlp_model = spacy.load("en_core_web_sm")
                    doc = nlp_model(text)
                    sentences = []
                    for i, sent in enumerate(doc.sents):
                        sentences.append({
                            'text': sent.text,
                            'index': i,
                            'start': sent.start_char,
                            'end': sent.end_char
                        })
                    return sentences
                except OSError:
                    logger.warning("spaCy model 'en_core_web_sm' not found for sentence splitting")
        except Exception as e:
            logger.warning(f"Failed to use spaCy for sentence splitting: {e}")
        
        # Fallback to simple sentence splitting
        import re
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = []
        parts = re.split(sentence_pattern, text)
        current_pos = 0
        for i, part in enumerate(parts):
            if part.strip():
                sentences.append({
                    'text': part.strip(),
                    'index': i,
                    'start': current_pos,
                    'end': current_pos + len(part)
                })
                current_pos += len(part)
        return sentences
    
    def _extract_keywords_ner_deep(self, text: str, num_keywords: int, verify_ner: bool, enricher, blocksize: int = 1) -> List[Dict[str, Any]]:
        """
        Deep processing: split text into sentences and process in blocks.
        
        Args:
            text: Full text to process
            num_keywords: Maximum number of keywords to return
            verify_ner: If True, verify results using LLM
            enricher: EntityEnricher instance
            blocksize: Number of sentences per block (default: 1 = sentence-by-sentence)
            
        Returns:
            List of keywords with sentence index information
        """
        # Validate blocksize
        if blocksize < 1:
            logger.warning(f"Invalid blocksize {blocksize}, using default of 1")
            blocksize = 1
        
        # 1. Split text into sentences with ordered index
        sentences = self._split_into_sentences(text)
        logger.debug(f"Deep mode: split text into {len(sentences)} sentences, blocksize={blocksize}")
        
        # Group sentences into blocks
        blocks = []
        for i in range(0, len(sentences), blocksize):
            block_sentences = sentences[i:i+blocksize]
            block_text = ' '.join(s['text'] for s in block_sentences)
            block_start = block_sentences[0]['start'] if block_sentences else 0
            block_end = block_sentences[-1]['end'] if block_sentences else len(text)
            block_indices = [s['index'] for s in block_sentences]
            blocks.append({
                'text': block_text,
                'sentences': block_sentences,
                'indices': block_indices,
                'start': block_start,
                'end': block_end,
                'block_index': i // blocksize
            })
        
        logger.debug(f"Deep mode: grouped into {len(blocks)} blocks of up to {blocksize} sentences each")
        
        # 2. Process each sentence separately for entity extraction (for accuracy)
        all_entities = []
        sentence_entity_map = {}  # Map sentence_index -> list of entities
        block_entity_map = {}  # Map block_index -> list of all entities in block
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            sentence_index = sentence_info['index']
            sentence_start = sentence_info['start']
            
            # Extract entities from this sentence
            sentence_entities = enricher.extract_entities(sentence_text)
            
            # Adjust entity positions to account for sentence offset in full text
            for entity in sentence_entities:
                # Update start/end positions to be relative to full text
                entity['start'] = entity.get('start', 0) + sentence_start
                entity['end'] = entity.get('end', 0) + sentence_start
                entity['sentence_index'] = sentence_index
                entity['sentence_text'] = sentence_text
                all_entities.append(entity)
            
            sentence_entity_map[sentence_index] = sentence_entities
        
        # Group entities by block for LLM verification
        for block in blocks:
            block_entities = []
            for sentence_index in block['indices']:
                block_entities.extend(sentence_entity_map.get(sentence_index, []))
            block_entity_map[block['block_index']] = block_entities
        
        if not all_entities:
            return []
        
        # 3. Verify with LLM block-by-block if requested
        verified_map = {}  # Maps entity_text.lower() -> verified_entity_data
        verification_successful = False
        llm_model = 'unknown'  # Default model name
        
        if verify_ner:
            try:
                logger.debug(f"Deep mode: starting LLM verification for {len(blocks)} blocks")

                # Prepare blocks for verification (filter out empty blocks)
                blocks_to_verify = [
                    block for block in blocks
                    if block_entity_map.get(block['block_index'], [])
                ]

                logger.debug(f"Deep mode: found {len(blocks_to_verify)} blocks with entities to verify")

                verified_blocks = []

                # Process each block with LLM verification using all configured models
                for block in blocks_to_verify:
                    block_text = block['text']
                    block_index = block['block_index']
                    block_entities = block_entity_map.get(block_index, [])

                    logger.debug(f"Processing block {block_index} with {len(block_entities)} entities")

                    # Create verification prompt for this block
                    system_prompt = "You are an expert at Named Entity Recognition (NER). Verify and correct the following spaCy NER entities.\n\n"

                    # Add entity type definitions
                    entity_type_descriptions = self.model_manager.entity_enricher.ENTITY_TYPES
                    system_prompt += "Entity type definitions:\n"
                    for entity_type in entity_type_descriptions:
                        if entity_type == 'PERSON' or entity_type == 'PER':
                            system_prompt += f"- {entity_type}: Person indicates human beings being mentioned\n"
                        elif entity_type == 'ORG':
                            system_prompt += f"- {entity_type}: Organization indicates companies, agencies, institutions\n"
                        elif entity_type == 'GPE':
                            system_prompt += f"- {entity_type}: Geopolitical Entity indicates countries, cities, states\n"
                        elif entity_type == 'LOC':
                            system_prompt += f"- {entity_type}: Location indicates non-GPE locations, mountain ranges, bodies of water\n"
                        else:
                            system_prompt += f"- {entity_type}: {entity_type} entity\n"
                    system_prompt += "\n"

                    # Add block text context
                    if block_text and block_text.strip():
                        system_prompt += f"Text context:\n{block_text.strip()}\n"

                    # Add instructions
                    system_prompt += "\nReturn verified entities in this simple format:\n\n**1. Verified Entities:**\n* **Entity Name** (TYPE) - Brief reason\n* **Another Entity** (TYPE) - Brief reason\n\n**2. New Entities Found:**\n* **New Entity** (TYPE) - Brief reason\n\n**3. Entities to Remove:**\n* **False Entity** - Reason to remove\n\nUse these entity types: PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT, etc.\n\nBe concise and accurate. Only include entities that actually exist in the text."

                    # Prepare entities for verification
                    entities_json = json.dumps(block_entities, separators=(',', ':'))
                    verification_prompt = f"{system_prompt}\n\nEntities to verify:\n{entities_json}"

                    # Use LLMRequestHandler for this block (uses all configured models)
                    block_request_handler = LLMRequestHandler(agent_instance=self)

                    # Send parallel requests to all verification models for this block
                    block_responses = block_request_handler.send_parallel(
                        prompt=verification_prompt,
                        request_type=f"ner_verification_block_{block_index}",
                        temperature=0.1,
                        max_tokens=500,
                        parse_json=True,  # Enable JSON parsing for NER verification
                        original_entities=block_entities  # Pass original entities for parsing
                    )

                    logger.debug(f"Block {block_index}: received {len(block_responses)} responses from models")

                    # Process responses for this block
                    block_verified_entities = []
                    block_successful_models = []

                    for response in block_responses:
                        if response.success:
                            try:
                                parsed_entities = self._parse_llm_response(response.response, block_entities)
                                for entity in parsed_entities:
                                    if isinstance(entity, dict):
                                        entity_copy = entity.copy()
                                        entity_copy['source_model'] = response.model
                                        entity_copy['block_index'] = block_index
                                        block_verified_entities.append(entity_copy)
                                block_successful_models.append(response.model)
                                logger.debug(f"Block {block_index}: parsed {len(parsed_entities)} entities from {response.model}")
                            except Exception as e:
                                logger.warning(f"Block {block_index}: failed to parse response from {response.model}: {e}")
                        else:
                            logger.warning(f"Block {block_index}: request failed for model {response.model}: {response.error}")

                    verified_blocks.append({
                        'block_index': block_index,
                        'verified_entities': block_verified_entities,
                        'successful_models': block_successful_models,
                        'block_text': block_text
                    })

                    if block_verified_entities:
                        verification_successful = True
                        logger.debug(f"Block {block_index}: verification successful with {len(block_verified_entities)} entities from {len(block_successful_models)} models")
                    else:
                        logger.warning(f"Block {block_index}: no successful verifications")

                logger.debug(f"Deep mode: processed {len(verified_blocks)} blocks, verification_successful={verification_successful}")

                # Process results from all blocks
                verified_map = {}  # Maps entity_text.lower() -> verified_entity_data

                if verification_successful:
                    for block_result in verified_blocks:
                        block_index = block_result['block_index']
                        block_verified_entities = block_result['verified_entities']

                        # Map verified entities back to sentences
                        for verified_entity in block_verified_entities:
                            v_text = (verified_entity.get('text') or "").strip().lower()
                            if v_text:
                                # Find which sentence(s) this entity belongs to based on block
                                verified_entity['sentence_indices'] = [block_index]  # Use block index as sentence index for deep mode
                                verified_map[v_text] = verified_entity

                    logger.debug(f"Deep mode: verified {len(verified_map)} entities across {len(verified_blocks)} blocks")
            except Exception as e:
                logger.error(f"Deep mode LLM verification failed: {e}", exc_info=True)
        
        # 4. Combine and process entities (similar to regular flow)
        # Use the same aggregation logic as extract_keywords_ner
        entity_scores = {}
        entity_types = {}
        entity_reasonings = {}
        entity_verified_status = {}
        entity_display_texts = {}
        entity_sentence_indices = {}  # Track which sentences contain each entity
        
        for entity in all_entities:
            entity_text = entity['text'].strip()
            entity_lower = entity_text.lower()
            normalized_lower = re.sub(r'\s+', ' ', entity_lower).strip()
            sentence_index = entity.get('sentence_index', -1)
            
            # Match with verified entities
            matched_with_llm = False
            if verify_ner and verification_successful:
                ve = verified_map.get(entity_lower) or verified_map.get(normalized_lower)
                if ve:
                    matched_with_llm = True
                    llm_type = ve.get('type', 'UNKNOWN')
                    llm_reasoning = ve.get('reasoning', '').strip()
                    entity_type = llm_type
                    reasoning = llm_reasoning if llm_reasoning else f"Verified as {entity_type} by LLM model {llm_model}"
                else:
                    entity_type = entity.get('type', 'UNKNOWN')
                    reasoning = f"Not verified by LLM model {llm_model} - using spaCy classification ({entity_type})"
            else:
                # No verification requested, or verification failed - use original entity
                entity_type = entity.get('type', 'UNKNOWN')
                reasoning = ''  # No reasoning needed if verification wasn't requested
            
            # Score based on entity type importance
            type_weights = {
                'PER': 1.0, 'ORG': 0.9, 'LOC': 0.8, 'GPE': 0.8,
                'PRODUCT': 0.7, 'EVENT': 0.7, 'BOOK': 0.7, 'MODEL': 0.7, 'SOFTWARE': 0.6,
                'FAC': 0.6, 'WORK_OF_ART': 0.6,
                'LAW': 0.5, 'LANGUAGE': 0.5, 'DATE': 0.4, 'TIME': 0.3,
                'MONEY': 0.3, 'PERCENT': 0.2, 'ORDINAL': 0.2, 'CARDINAL': 0.1, 'QUANTITY': 0.1,
                'OTHER': 0.1,  # Not a recognized entity type - low priority
            }
            
            base_score = type_weights.get(entity_type, 0.5)
            length_bonus = min(len(entity_text.split()) * 0.1, 0.3)
            
            if entity_lower not in entity_scores:
                entity_scores[entity_lower] = 0.0
                entity_types[entity_lower] = entity_type
                entity_display_texts[entity_lower] = entity_text
                entity_reasonings[entity_lower] = reasoning
                entity_verified_status[entity_lower] = matched_with_llm
                entity_sentence_indices[entity_lower] = [sentence_index]
            else:
                # Update reasoning if LLM verified
                if matched_with_llm:
                    entity_reasonings[entity_lower] = reasoning
                    entity_types[entity_lower] = entity_type
                    entity_verified_status[entity_lower] = True
                # Track sentence indices
                if sentence_index not in entity_sentence_indices[entity_lower]:
                    entity_sentence_indices[entity_lower].append(sentence_index)
            
            entity_scores[entity_lower] += base_score + length_bonus
        
        # 5. Convert to keyword format
        keywords = []
        for entity_lower, score in entity_scores.items():
            kw_data = {
                'keyword': entity_display_texts[entity_lower],
                'score': round(score, 4),
                'type': entity_types[entity_lower],
                'sentence_indices': sorted(entity_sentence_indices[entity_lower])  # Add sentence indices
            }
            
            # Add status based on verification
            if verify_ner:
                if entity_lower in entity_verified_status and entity_verified_status[entity_lower]:
                    kw_data['status'] = 'verified'
                else:
                    kw_data['status'] = 'Not verified'
            
            # Add reasoning
            if entity_lower in entity_reasonings:
                reasoning_value = entity_reasonings[entity_lower]
                if reasoning_value:
                    kw_data['reasoning'] = reasoning_value
            
            keywords.append(kw_data)
        
        # Sort and limit
        keywords.sort(key=lambda x: x['score'], reverse=True)
        keywords = keywords[:num_keywords]
        
        # Normalize scores
        if keywords:
            max_score = keywords[0]['score']
            if max_score > 0:
                for kw in keywords:
                    kw['score'] = round(min(kw['score'] / max_score, 1.0), 4)
        
        logger.debug(f"Deep mode: returning {len(keywords)} keywords from {len(sentences)} sentences")
        return keywords
    
    def extract_keywords_ner(self, text: str, num_keywords: int = 20, verify_ner: bool = False, deep: bool = False, blocksize: int = 1) -> List[Dict[str, Any]]:
        """
        Extract keywords using spaCy NER (Named Entity Recognition).
        Returns named entities as keywords, sorted by importance.
        
        Args:
            text: Text to extract keywords from
            num_keywords: Maximum number of keywords to return
            verify_ner: If True, verify results using LLM
            deep: If True, process text sentence-by-sentence with ordered index
            blocksize: Number of sentences per block when deep=True (default: 1 = sentence-by-sentence)
        """
        enricher = self.model_manager.entity_enricher
        
        # If deep mode, split into sentences and process separately
        if deep:
            return self._extract_keywords_ner_deep(text, num_keywords, verify_ner, enricher, blocksize)
        
        # 1. Extract original entities using spaCy (to get all occurrences/frequencies)
        original_entities = enricher.extract_entities(text)
        if not original_entities:
            return []
        
        # 2. Verify with LLM if requested
        verified_map = {} # Maps entity_text.lower() -> verified_entity_data
        verification_successful = False
        if verify_ner:
            try:
                llm_cfg = config.get_llm_config()
                # Parse verification models
                main_model = llm_cfg.get('model', 'unknown')
                verification_model_str = llm_cfg.get('verification_model')
                verification_models = self._parse_verification_models(verification_model_str, main_model)
                logger.debug(f"Starting LLM verification (models: {verification_models}) for {len(original_entities)} entities")
                # We send unique entities to LLM to save tokens, but keep frequencies
                # Actually, verify_ner_with_llm expects the full list to handle context
                verified_entities = self.verify_ner_with_llm(text, original_entities, debug=True)
                
                logger.debug(f"LLM verification returned {len(verified_entities) if verified_entities else 0} entities")
                
                # Check if verification actually succeeded
                # If verification failed, verify_ner_with_llm returns original_entities without reasoning
                if verified_entities and len(verified_entities) > 0:
                    # Check if any entity has reasoning (indicates successful LLM verification)
                    # When verification fails, original entities are returned without reasoning field
                    entities_with_reasoning = [ve for ve in verified_entities if ve.get('reasoning')]
                    has_reasoning = len(entities_with_reasoning) > 0
                    
                    logger.debug(f"Found {len(entities_with_reasoning)} entities with reasoning out of {len(verified_entities)} total")
                    
                    if has_reasoning:
                        verification_successful = True
                        logger.info(f"LLM verification (model: {llm_model}) successful: {len(entities_with_reasoning)} entities verified out of {len(verified_entities)} returned")
                        
                        # Warn if LLM returned significantly fewer entities than spaCy found
                        if len(verified_entities) < len(original_entities) * 0.5:
                            logger.warning(f"LLM (model: {llm_model}) returned only {len(verified_entities)} entities vs {len(original_entities)} from spaCy - some entities may be missing")
                        
                        # Create a map of verified results for easy lookup
                        # Use normalized keys to handle text variations
                        for ve in verified_entities:
                            # Handle both 'text' and 'entity' keys (LLM might use either)
                            v_text = (ve.get('text') or ve.get('entity') or "").strip()
                            if not v_text:
                                logger.debug(f"Skipping entity without text: {ve}")
                                continue
                                
                            v_lower = v_text.lower()
                            v_normalized = re.sub(r'\s+', ' ', v_lower).strip()
                            
                            # Ensure the entity dict has 'text' key for consistency
                            if 'text' not in ve and 'entity' in ve:
                                ve['text'] = ve['entity']
                            
                            # Store with both exact and normalized keys for flexible matching
                            if v_lower not in verified_map:
                                verified_map[v_lower] = ve
                            if v_normalized != v_lower and v_normalized not in verified_map:
                                verified_map[v_normalized] = ve
                        
                        logger.debug(f"Built verified_map with {len(verified_map)} entries")
                    else:
                        # No reasoning found - verification likely failed, use original entities
                        logger.warning(f"LLM verification (model: {llm_model}) returned {len(verified_entities)} entities but NONE have reasoning - treating as failed")
                        # Log sample entities to debug
                        if verified_entities:
                            sample = verified_entities[0]
                            logger.debug(f"Sample entity keys: {list(sample.keys())}, has reasoning: {'reasoning' in sample}, entity: {sample}")
                else:
                    # Empty result - verification failed completely, use original entities
                    logger.warning(f"LLM verification (model: {llm_model}) returned empty result - using original spaCy entities")
            except Exception as e:
                logger.error(f"LLM verification (model: {llm_model}) failed in extract_keywords_ner: {e}", exc_info=True)
        
        # 3. Count frequencies and calculate scores using original occurrences
        entity_scores = {}
        entity_types = {}
        entity_reasonings = {}
        entity_verified_status = {}  # Track if entity was verified by LLM
        entity_display_texts = {} # Keep original casing
        
        logger.debug(f"Processing {len(original_entities)} original entities, verified_map has {len(verified_map)} entries")
        
        for entity in original_entities:
            entity_text = entity['text'].strip()
            entity_lower = entity_text.lower()
            
            # Normalize text for better matching (remove extra spaces, normalize punctuation)
            normalized_lower = re.sub(r'\s+', ' ', entity_lower).strip()
            
            # Determine type and reasoning (from verified map or original)
            matched_with_llm = False  # Track if we matched with LLM verification
            if verify_ner and verification_successful:
                # Try exact match first
                ve = verified_map.get(entity_lower)
                if not ve:
                    # Try normalized match (handles spacing differences like "10/28" vs "10 / 28")
                    ve = verified_map.get(normalized_lower)
                
                # If still not found, try matching by original spaCy type
                # This helps when LLM corrected the type but text has slight variations
                if not ve:
                    original_type = entity.get('type', 'UNKNOWN')
                    # Search verified_map for entities with matching type
                    for key, verified_entity in verified_map.items():
                        verified_type = verified_entity.get('type', 'UNKNOWN')
                        verified_text = (verified_entity.get('text') or "").strip().lower()
                        # Check if types match and texts are similar
                        if verified_type == original_type:
                            # Try fuzzy matching - check if one text contains the other
                            if (entity_lower in verified_text or verified_text in entity_lower or
                                normalized_lower in verified_text or verified_text in normalized_lower):
                                ve = verified_entity
                                logger.debug(f"Matched entity by type '{original_type}': '{entity_text}' -> '{verified_entity.get('text', '')}'")
                                break
                
                if ve:
                    matched_with_llm = True
                    # Use LLM's verified type and reasoning
                    llm_type = ve.get('type', 'UNKNOWN')
                    llm_reasoning = ve.get('reasoning', '').strip()
                    original_type = entity.get('type', 'UNKNOWN')
                    
                    # Always use LLM's type (it verified it)
                    entity_type = llm_type
                    
                    # Only consider verified if LLM provided specific reasoning
                    if llm_reasoning:
                        reasoning = llm_reasoning
                    else:
                        # LLM listed this entity but didn't provide verification reasoning
                        # Treat as not verified - use spaCy classification
                        entity_type = entity.get('type', 'UNKNOWN')
                        reasoning = f"Not verified by LLM model {llm_model} - using spaCy classification ({entity_type})"
                    
                    # Log if type changed from spaCy to LLM
                    if original_type != entity_type:
                        logger.debug(f"Entity '{entity_text}': type changed from '{original_type}' (spaCy) to '{entity_type}' (LLM verified)")
                else:
                    # Entity not in verified map - could mean:
                    # 1. LLM removed it (marked as REMOVE)
                    # 2. Text variation prevented matching
                    # For now, include it with original spaCy classification and note it wasn't verified
                    entity_type = entity.get('type', 'UNKNOWN')
                    reasoning = f"Not verified by LLM model {llm_model} - using spaCy classification ({entity_type})"
            else:
                # No verification requested, or verification failed - use original entity
                entity_type = entity.get('type', 'UNKNOWN')
                # Only add reasoning message if verification was attempted but failed
                reasoning = f"Verification unavailable - using spaCy classification" if (verify_ner and not verification_successful) else ''
            
            # Score based on entity type importance
            type_weights = {
                'PER': 1.0, 'ORG': 0.9, 'LOC': 0.8, 'GPE': 0.8,
                'PRODUCT': 0.7, 'EVENT': 0.7, 'BOOK': 0.7, 'MODEL': 0.7, 'SOFTWARE': 0.6,
                'FAC': 0.6, 'WORK_OF_ART': 0.6,
                'LAW': 0.5, 'LANGUAGE': 0.5, 'DATE': 0.4, 'TIME': 0.3,
                'MONEY': 0.3, 'PERCENT': 0.2, 'ORDINAL': 0.2, 'CARDINAL': 0.1, 'QUANTITY': 0.1,
                'OTHER': 0.1,  # Not a recognized entity type - low priority
            }
            
            base_score = type_weights.get(entity_type, 0.5)
            length_bonus = min(len(entity_text.split()) * 0.1, 0.3)
            
            if entity_lower not in entity_scores:
                entity_scores[entity_lower] = 0.0
                entity_types[entity_lower] = entity_type
                entity_display_texts[entity_lower] = entity_text
                # Always store reasoning if it exists (even if empty, but we set a default above)
                entity_reasonings[entity_lower] = reasoning
                # Track verification status
                entity_verified_status[entity_lower] = matched_with_llm
            else:
                # If we already have this entity, prefer LLM reasoning over default reasoning
                existing_reasoning = entity_reasonings.get(entity_lower, '')
                # Prefer reasoning that mentions the model (LLM verified) or is more detailed
                if reasoning:
                    # Check if current reasoning is from LLM verification
                    is_llm_reasoning = matched_with_llm or (verify_ner and verification_successful and llm_model in reasoning)
                    is_existing_llm_reasoning = llm_model in existing_reasoning if existing_reasoning else False
                    
                    # Prefer LLM reasoning, or more detailed reasoning
                    if is_llm_reasoning and not is_existing_llm_reasoning:
                        # Current is LLM, existing is not - prefer current
                        entity_reasonings[entity_lower] = reasoning
                    elif is_llm_reasoning == is_existing_llm_reasoning and len(reasoning) > len(existing_reasoning):
                        # Both same type (both LLM or both not), prefer more detailed
                        entity_reasonings[entity_lower] = reasoning
                
                # Also update type if LLM verified it (prefer LLM type over spaCy)
                if matched_with_llm:
                    # LLM verified type takes precedence
                    entity_types[entity_lower] = entity_type
                    # Update verification status
                    entity_verified_status[entity_lower] = True
            
            # Accumulate scores based on occurrences
            entity_scores[entity_lower] += base_score + length_bonus
        
        # 4. Convert to keyword format
        keywords = []
        for entity_lower, score in entity_scores.items():
            kw_data = {
                'keyword': entity_display_texts[entity_lower],
                'score': round(score, 4),
                'type': entity_types[entity_lower]
            }
            # Add status based on verification
            if verify_ner:
                # If verification was requested, always include status
                if entity_lower in entity_verified_status and entity_verified_status[entity_lower]:
                    kw_data['status'] = 'verified'
                else:
                    kw_data['status'] = 'Not verified'
            # Always include reasoning if it was set (which it should be for verify_ner=True)
            if entity_lower in entity_reasonings:
                reasoning_value = entity_reasonings[entity_lower]
                if reasoning_value:  # Only add if non-empty
                    kw_data['reasoning'] = reasoning_value
            
            keywords.append(kw_data)
        
        # Sort and limit
        keywords.sort(key=lambda x: x['score'], reverse=True)
        keywords = keywords[:num_keywords]
        
        # Normalize scores
        if keywords:
            max_score = keywords[0]['score']
            if max_score > 0:
                for kw in keywords:
                    kw['score'] = round(min(kw['score'] / max_score, 1.0), 4)
        
        return keywords
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities using the loaded model.
        
        Args:
            text: Text to extract entities from
        
        Returns:
            Dictionary with entity information
        """
        enricher = self.model_manager.entity_enricher
        
        episode = {
            'content': text,
            'type': 'text',
            'description': 'Entity extraction'
        }
        
        return enricher.enrich_episode(episode)
    
    def parse_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Parse a file and extract text.
        
        Args:
            file_path: Path to file to parse
            **kwargs: Parser-specific options
        
        Returns:
            ParseResult as dictionary
        """
        if not PARSERS_AVAILABLE:
            return {
                'success': False,
                'error': 'File parsers not available. Install parsing dependencies.'
            }
        
        try:
            parser = get_parser(file_path)
            result = parser.parse(file_path, **kwargs)
            return result.to_dict()
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get daemon status."""
        status = {
            'running': self.running,
            'models_initialized': self.model_manager.is_initialized(),
            'use_spacy': self.use_spacy,
            'spacy_available': SPACY_AVAILABLE,
            'parsers_available': PARSERS_AVAILABLE
        }
        return status


class AIAgentClient:
    """
    Client for communicating with the AI Agent Daemon.
    Can be used to send requests to a running daemon.
    """
    
    def __init__(self, socket_path: str = '/tmp/palefire_ai_agent.sock'):
        """
        Initialize client.
        
        Args:
            socket_path: Path to Unix socket for communication
        """
        self.socket_path = socket_path
    
    def extract_keywords(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Send keyword extraction request to daemon."""
        # For now, direct access (can be extended to use socket/HTTP)
        # In production, this would communicate via socket or HTTP API
        raise NotImplementedError("Socket communication not yet implemented")
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Send entity extraction request to daemon."""
        raise NotImplementedError("Socket communication not yet implemented")


# Global daemon instance (for singleton pattern)
_daemon_instance: Optional[AIAgentDaemon] = None


def get_daemon(pidfile: Optional[str] = None, use_spacy: bool = True) -> AIAgentDaemon:
    """
    Get or create the global daemon instance.
    
    Args:
        pidfile: Path to PID file
        use_spacy: Whether to use spaCy
    
    Returns:
        AIAgentDaemon instance
    """
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = AIAgentDaemon(pidfile=pidfile, use_spacy=use_spacy)
    return _daemon_instance


if __name__ == '__main__':
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Pale Fire AI Agent Daemon')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'],
                       help='Action to perform')
    parser.add_argument('--pidfile', type=str, default='/tmp/palefire_ai_agent.pid',
                       help='Path to PID file')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as background daemon')
    parser.add_argument('--no-spacy', action='store_true',
                       help='Disable spaCy (use pattern-based NER)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    daemon = AIAgentDaemon(pidfile=args.pidfile, use_spacy=not args.no_spacy)
    
    if args.action == 'start':
        daemon.start(daemon=args.daemon)
    elif args.action == 'stop':
        # Read PID and send SIGTERM
        try:
            with open(args.pidfile, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to process {pid}")
        except FileNotFoundError:
            logger.error("PID file not found. Daemon may not be running.")
        except ProcessLookupError:
            logger.error("Process not found. Removing stale PID file.")
            os.remove(args.pidfile)
    elif args.action == 'restart':
        # Stop then start
        try:
            with open(args.pidfile, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)  # Wait for shutdown
        except:
            pass
        daemon.start(daemon=args.daemon)
    elif args.action == 'status':
        status = daemon.get_status()
        print(json.dumps(status, indent=2))

