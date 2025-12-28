"""
Pale Fire - AI Agent Daemon

Manages long-running AI agents that keep Gensim and spaCy models loaded in memory
to avoid start/stop delays. Provides a daemon service for keyword extraction and NER.
"""

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

# Configure logger
logger = logging.getLogger(__name__)

# LLM Response Logging Helper
def log_llm_response(response: str, model: str, request_type: str = "unknown", request_id: Optional[str] = None) -> None:
    """
    Log LLM response to logs folder for auditing and debugging.

    Args:
        response: The LLM response text
        model: The model name used
        request_type: Type of request (e.g., "ner_verification", "keyword_extraction")
        request_id: Optional request ID for correlation
    """
    try:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Generate timestamp and request ID
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]

        # Create log filename with safe characters
        safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
        filename = f"llm_response_{request_type}_{safe_model}_{timestamp}_{request_id}.txt"
        filepath = os.path.join(logs_dir, filename)

        # Write response to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Request Type: {request_type}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Request ID: {request_id}\n")
            f.write(f"Response Length: {len(response) if response else 0}\n")
            f.write("="*80 + "\n")
            f.write("RESPONSE:\n")
            f.write(response if response else "(empty response)")
            f.write("\n" + "="*80 + "\n")

        logger.debug(f"Logged LLM response to: {filepath}")

    except Exception as e:
        logger.warning(f"Failed to log LLM response: {e}")
        # Don't raise exception to avoid breaking the main flow

# Import models
from modules.KeywordBase import KeywordExtractor
from modules.PaleFireCore import EntityEnricher

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
                    remove_match = re.match(r'^[\*\-\•]\s+(?:\*\*)?([^*]+?)(?:\*\*)?\s*(?:\(([A-Z_]+)\))?\s*(?:\-\s*(.*))?$', line)
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
                    entity_match = re.match(r'^[\*\-\•]\s+(?:\*\*)?([^*]+?)(?:\*\*)?\s*\(([A-Z_]+)\)\s*(?:\-\s*(.*))?$', line)
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

    def _verify_ner_with_llm_client(self, text: str, entities: List[Dict[str, Any]], llm_client, llm_model: str, debug: bool = False, async_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Core verification logic using a pre-configured LLM client.

        Args:
            text: Original text
            entities: List of entities from spaCy
            llm_client: Pre-configured LLM client
            llm_model: Model name for logging
            debug: Enable debug output

        Returns:
            List of verified entities
        """
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
                entity_type_descriptions[entity_type] = 'Product indicates objects, vehicles, foods, etc. (not services)'
            elif entity_type == 'EVENT':
                entity_type_descriptions[entity_type] = 'Event indicates named hurricanes, battles, wars, sports events'
            elif entity_type == 'WORK_OF_ART':
                entity_type_descriptions[entity_type] = 'Work of Art indicates titles of books, songs, etc.'
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
            return entities

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

        # Generate request ID for correlation
        request_id = str(uuid.uuid4())[:8]

        # Log the LLM request for auditing
        try:
            logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
            os.makedirs(logs_dir, exist_ok=True)

            timestamp = time.strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(logs_dir, f"llm_request_{timestamp}_{request_id}.txt")

            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"Model: {llm_model}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Entities count: {len(entities_for_verification)}\n")
                f.write(f"Text length: {len(text) if text else 0}\n")
                f.write(f"Request type: NER verification\n")
                f.write(f"Request ID: {request_id}\n\n")
                f.write(f"PROMPT:\n{full_prompt}\n\n")
        except Exception as log_error:
            logger.warning(f"Failed to log LLM request: {log_error}")

        # Make the LLM call
        try:
            if async_mode:
                import asyncio
                # Use asyncio.run() - nest_asyncio should handle existing event loops
                response = asyncio.run(llm_client.acomplete(full_prompt))
            else:
                response = llm_client.complete(full_prompt)
        except Exception as e:
            logger.warning(f"LLM call failed (model: {llm_model}): {e}, using original entities")
            if debug:
                import traceback
                traceback.print_exc()
            return entities

        if debug:
            logger.debug(f"LLM response length: {len(response) if response else 0}")

        # Log the response using the standardized helper
        log_llm_response(response, llm_model, "ner_verification", request_id)

        if not response or not response.strip():
            logger.warning(f"Empty response from LLM (model: {llm_model})")
            return entities

        verified_entities = None
        json_parsing_attempted = False

        # Try to parse as direct JSON first
        try:
            verified_entities = json.loads(response.strip())
            json_parsing_attempted = True
            if debug:
                logger.debug(f"Parsed {len(verified_entities) if verified_entities else 0} entities from direct JSON response")

            # Log parsed JSON if available
            try:
                json_log_file = log_file.replace('.txt', '_parsed.json')
                with open(json_log_file, 'w', encoding='utf-8') as f:
                    json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                logger.debug(f"Saved parsed JSON to {json_log_file}")
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

                    # Log parsed JSON if available
                    try:
                        json_log_file = log_file.replace('.txt', '_markdown_parsed.json')
                        with open(json_log_file, 'w', encoding='utf-8') as f:
                            json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                        logger.debug(f"Saved parsed markdown JSON to {json_log_file}")
                    except Exception as json_log_error:
                        logger.warning(f"Failed to log parsed markdown JSON: {json_log_error}")

                except json.JSONDecodeError as e:
                    if debug:
                        logger.debug(f"Failed to parse JSON from markdown code block: {e}")
            else:
                if debug:
                    logger.debug("No markdown JSON code block found, trying Gemma3 markdown format...")

        # If JSON parsing was not attempted, try Gemma3 markdown format
        if not json_parsing_attempted:
            logger.debug(f"No JSON found in response, trying Gemma3 markdown parser...")
            try:
                verified_entities = self._parse_gemma3_markdown(response, llm_model, debug)

                if verified_entities:
                    logger.debug(f"Parsed {len(verified_entities)} entities using Gemma3 markdown parser")

                    # Log parsed JSON if available
                    try:
                        json_log_file = log_file.replace('.txt', '_gemma3_parsed.json')
                        with open(json_log_file, 'w', encoding='utf-8') as f:
                            json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                        logger.debug(f"Saved parsed Gemma3 entities as JSON to {json_log_file}")
                    except Exception as json_log_error:
                        logger.warning(f"Failed to log parsed Gemma3 JSON: {json_log_error}")
                else:
                    logger.warning(f"No entities found in Gemma3 markdown response (model: {llm_model}), using original entities")
                    return entities
            except Exception as gemma_error:
                logger.warning(f"Failed to parse as Gemma3 markdown (model: {llm_model}): {gemma_error}. Response was: {response[:200]}...")
                return entities

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
            return entities

    async def _verify_ner_with_llm_client_async(self, text: str, entities: List[Dict[str, Any]], llm_client, llm_model: str, debug: bool = False) -> List[Dict[str, Any]]:
        """
        Async version of _verify_ner_with_llm_client for parallel processing.
        """
        if not CONFIG_AVAILABLE:
            logger.warning("Config not available, skipping LLM verification")
            return entities

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
                    entity_type_descriptions[entity_type] = 'Product indicates objects, vehicles, foods, etc. (not services)'
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
                return entities

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

            # Generate request ID for correlation
            request_id = str(uuid.uuid4())[:8]

            # Log the LLM request for auditing
            try:
                logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
                os.makedirs(logs_dir, exist_ok=True)

                timestamp = time.strftime('%Y%m%d_%H%M%S')
                log_file = os.path.join(logs_dir, f"llm_request_{timestamp}_{request_id}.txt")

                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"Model: {llm_model}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Entities count: {len(entities_for_verification)}\n")
                    f.write(f"Text length: {len(text) if text else 0}\n")
                    f.write(f"Request type: NER verification\n")
                    f.write(f"Request ID: {request_id}\n\n")
                    f.write(f"PROMPT:\n{full_prompt}\n\n")
            except Exception as log_error:
                logger.warning(f"Failed to log LLM request: {log_error}")

            try:
                # Make the async LLM call
                response = await llm_client.acomplete(full_prompt)

                if debug:
                    logger.debug(f"LLM response length: {len(response) if response else 0}")

                # Log the response using the standardized helper
                log_llm_response(response, llm_model, "ner_verification", request_id)

                if not response or not response.strip():
                    logger.warning(f"Empty response from LLM (model: {llm_model})")
                    return entities

                verified_entities = None
                json_parsing_attempted = False

                # Try to parse as direct JSON first
                try:
                    verified_entities = json.loads(response.strip())
                    json_parsing_attempted = True
                    if debug:
                        logger.debug(f"Parsed {len(verified_entities) if verified_entities else 0} entities from direct JSON response")

                    # Log parsed JSON if available
                    try:
                        json_log_file = log_file.replace('.txt', '_parsed.json')
                        with open(json_log_file, 'w', encoding='utf-8') as f:
                            json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                        logger.debug(f"Saved parsed JSON to {json_log_file}")
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

                            # Log parsed JSON if available
                            try:
                                json_log_file = log_file.replace('.txt', '_markdown_parsed.json')
                                with open(json_log_file, 'w', encoding='utf-8') as f:
                                    json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                                logger.debug(f"Saved parsed markdown JSON to {json_log_file}")
                            except Exception as json_log_error:
                                logger.warning(f"Failed to log parsed markdown JSON: {json_log_error}")

                        except json.JSONDecodeError as e:
                            if debug:
                                logger.debug(f"Failed to parse JSON from markdown code block: {e}")
                    else:
                        if debug:
                            logger.debug("No markdown JSON code block found, trying Gemma3 markdown format...")

                # If JSON parsing was not attempted, try structured markdown formats
                if not json_parsing_attempted:
                    logger.debug(f"No JSON found in response, trying structured markdown parsers...")

                    # First try the numbered section format (e.g., **1. Entities...**)
                    try:
                        verified_entities = self._parse_structured_markdown(response, llm_model, debug)

                        if verified_entities:
                            logger.debug(f"Parsed {len(verified_entities)} entities using structured markdown parser")

                            # Log parsed JSON if available
                            try:
                                json_log_file = log_file.replace('.txt', '_structured_parsed.json')
                                with open(json_log_file, 'w', encoding='utf-8') as f:
                                    json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                                logger.debug(f"Saved parsed structured markdown to {json_log_file}")
                            except Exception as json_log_error:
                                logger.warning(f"Failed to log parsed structured markdown: {json_log_error}")
                        else:
                            # Try Gemma3 markdown format as fallback
                            logger.debug(f"No entities found with structured parser, trying Gemma3 markdown parser...")
                            try:
                                verified_entities = self._parse_gemma3_markdown(response, llm_model, debug)

                                if verified_entities:
                                    logger.debug(f"Parsed {len(verified_entities)} entities using Gemma3 markdown parser")

                                    # Log parsed JSON if available
                                    try:
                                        json_log_file = log_file.replace('.txt', '_gemma3_parsed.json')
                                        with open(json_log_file, 'w', encoding='utf-8') as f:
                                            json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                                        logger.debug(f"Saved parsed Gemma3 entities as JSON to {json_log_file}")
                                    except Exception as json_log_error:
                                        logger.warning(f"Failed to log parsed Gemma3 JSON: {json_log_error}")
                                else:
                                    logger.warning(f"No entities found in Gemma3 markdown response (model: {llm_model}), using original entities")
                                    return entities
                            except Exception as gemma_error:
                                logger.warning(f"Failed to parse as Gemma3 markdown (model: {llm_model}): {gemma_error}. Response was: {response[:200]}...")
                                return entities
                    except Exception as structured_error:
                        logger.warning(f"Failed to parse as structured markdown (model: {llm_model}): {structured_error}. Response was: {response[:200]}...")
                        # Try Gemma3 as final fallback
                        try:
                            verified_entities = self._parse_gemma3_markdown(response, llm_model, debug)
                            if verified_entities:
                                logger.debug(f"Parsed {len(verified_entities)} entities using Gemma3 markdown parser")
                            else:
                                logger.warning(f"No entities found in any markdown format (model: {llm_model}), using original entities")
                                return entities
                        except Exception as gemma_error:
                            logger.warning(f"All markdown parsing failed (model: {llm_model}): {gemma_error}. Response was: {response[:200]}...")
                            return entities

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
                    return entities

            except Exception as e:
                logger.warning(f"LLM verification failed (model: {llm_model}): {e}, using original entities")
                if debug:
                    import traceback
                    traceback.print_exc()
                return entities

        except Exception as e:
            logger.warning(f"LLM verification setup failed (model: {llm_model}): {e}, using original entities")
            return entities

    def verify_ner_with_llm(self, text: str, entities: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
        """
        Verify and correct spaCy NER results using LLM.

        Args:
            text: Original text
            entities: List of entities from spaCy
            debug: Enable debug output

        Returns:
            List of verified entities (with REMOVE type for false positives)
        """
        if not CONFIG_AVAILABLE:
            logger.warning("Config not available, skipping LLM verification")
            return entities

        try:
            llm_cfg = config.get_llm_config()
            if not llm_cfg.get('api_key'):
                logger.warning("LLM API key not configured, skipping verification")
                return entities

            # Use verification model if specified, otherwise fallback to main model
            main_model = llm_cfg.get('model', 'unknown')
            verification_model = llm_cfg.get('verification_model')
            # Handle None, empty string, or whitespace-only strings
            llm_model = verification_model if (verification_model and verification_model.strip()) else main_model

            # Log which model is being used
            if verification_model and verification_model != main_model:
                logger.info(f"Using verification model: {llm_model} (main model: {main_model})")
            else:
                logger.warning(f"OLLAMA_VERIFICATION_MODEL not set - using main model for verification: {llm_model}. Set OLLAMA_VERIFICATION_MODEL in .env to use a separate verification model.")

            # Get timeout for verification requests
            verification_timeout = llm_cfg.get('verification_timeout', 300)  # Default 5 minutes

            if debug:
                logger.debug(f"Using verification timeout: {verification_timeout}s")

            llm_client = self._create_ollama_client(llm_model, verification_timeout)
            if not llm_client:
                logger.warning("Failed to create LLM client, skipping verification")
                return entities

            return self._verify_ner_with_llm_client(text, entities, llm_client, llm_model, debug)

        except Exception as e:
            logger.warning(f"LLM verification failed in verify_ner_with_llm: {e}, using original entities")
            if debug:
                import traceback
                traceback.print_exc()
            return entities

    def _split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
            entity_type_descriptions = EntityEnricher.CONTEXT_REQUIRED_TYPES.copy()

            # Add descriptions for any entity types not in CONTEXT_REQUIRED_TYPES
            all_entity_types = set(EntityEnricher.ENTITY_TYPES.values())
            for entity_type in all_entity_types:
                if entity_type not in entity_type_descriptions:
                    # Add a generic description for entity types not in CONTEXT_REQUIRED_TYPES
                    if entity_type == 'PER':
                        entity_type_descriptions[entity_type] = 'Person names indicate human beings being mentioned'
                    elif entity_type == 'ORG':
                        entity_type_descriptions[entity_type] = 'Organization names indicate companies, agencies, institutions, etc.'
                    elif entity_type == 'LOC':
                        entity_type_descriptions[entity_type] = 'Location names indicate physical places like mountains, lakes, etc.'
                    elif entity_type == 'DATE':
                        entity_type_descriptions[entity_type] = 'Dates indicate absolute or relative dates or periods'
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
            
            # Step 2: Build context header
            context_header_parts = []

            # Add entity type descriptions for ALL entity types (not just ones found in this text)
            all_entity_types = set(EntityEnricher.ENTITY_TYPES.values())
            all_entity_types.discard('UNKNOWN')  # Remove UNKNOWN type

            if all_entity_types:
                context_header_parts.append("Entity type definitions:")
                for entity_type in sorted(all_entity_types):
                    description = entity_type_descriptions.get(entity_type, f'{entity_type} entity')
                    context_header_parts.append(f"- {entity_type}: {description}")
            
            context_header = '\n'.join(context_header_parts) if context_header_parts else ''
            
            # Step 3: Prepare entities for verification
            entities_for_verification = []
            for e in entities:
                entity_type = e.get('type', 'UNKNOWN')
                entity_text = e.get('text', '').strip()
                
                if not entity_text:
                    continue
                
                # Get context for this entity
                ctx_info = entity_contexts.get(entity_text, {
                    'sentence': '',
                    'description': entity_type_descriptions.get(entity_type, f'{entity_type} entity')
                })
                
                # Minimal entity data - flat structure (no nested context) to save tokens
                entity_data = {
                    'text': entity_text,
                    'type': entity_type,
                    'sent': ctx_info['sentence'],  # Sentence containing the entity
                    'verified': False
                }
                
                entities_for_verification.append(entity_data)
            
            # Step 4: Build the prompt with context in header
            system_prompt = """Verify spaCy NER entities."""
            
            # Add context header if available
            if context_header:
                system_prompt += f"\n\n{context_header}\n"
            
            # Add the full text context (block text) for better verification
            # This provides complete context instead of just individual sentences
            if text and text.strip():
                system_prompt += f"\n\nFull text context:\n{text.strip()}\n"

            # Add instructions for structured markdown format (optimized for gemma3:4b)
            system_prompt += "\n\nReturn verified entities in this simple format:\n\n**1. Verified Entities:**\n* **Entity Name** (TYPE) - Brief reason\n* **Another Entity** (TYPE) - Brief reason\n\n**2. New Entities Found:**\n* **New Entity** (TYPE) - Brief reason\n\n**3. Entities to Remove:**\n* **False Entity** - Reason to remove\n\nUse these entity types: PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT, etc.\n\nBe concise and accurate. Only include entities that actually exist in the text."
            
            # Minimize JSON - use compact format (no indentation) to reduce tokens
            entities_json = json.dumps(entities_for_verification, separators=(',', ':'))
            
            # Final prompt with context in header
            verification_prompt = f"""{system_prompt}
{entities_json}"""
            
            if debug:
                logger.debug(f"Verifying {len(entities)} entities with LLM (model: {llm_model})")
            
            # Generate request ID for correlation
            request_id = str(uuid.uuid4())[:8]

            # Log the request to logs folder
            try:
                logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
                os.makedirs(logs_dir, exist_ok=True)
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                log_file = os.path.join(logs_dir, f"llm_request_{timestamp}_{request_id}.txt")

                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"Model: {llm_model}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Request type: NER verification\n")
                    f.write(f"Request ID: {request_id}\n\n")
                    f.write(f"PROMPT:\n{verification_prompt}\n\n")
            except Exception as e:
                logger.warning(f"Failed to log LLM request: {e}")

            # Call LLM
            response = llm_client.complete(
                messages=[{"role": "user", "content": verification_prompt}],
                temperature=0.1,
                max_tokens=2000
            )

            # Log the response using the standardized helper
            log_llm_response(response, llm_model, "ner_verification", request_id)
            
            # Extract JSON from response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if not json_match:
                # Try finding any array-like structure
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
            
            verified_entities = None
            
            if json_match:
                try:
                    # Clean up common LLM artifacts before parsing
                    json_str = json_match.group()
                    # Remove potential comments or trailing commas
                    json_str = re.sub(r',(\s*[\]\}])', r'\1', json_str)
                    verified_entities = json.loads(json_str)
                    # Log parsed JSON to file
                    try:
                        log_dir = Path(__file__).parent.parent / 'logs'
                        log_dir.mkdir(parents=True, exist_ok=True)
                        json_log_file = log_dir / f"llm_response_ner_{timestamp}.json"
                        with open(json_log_file, 'w', encoding='utf-8') as f:
                            json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                        logger.debug(f"Saved parsed JSON to {json_log_file}")
                    except Exception as json_log_error:
                        logger.warning(f"Failed to log parsed JSON: {json_log_error}")
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON from LLM (model: {llm_model}): {e}. Trying Gemma3 markdown parser...")
                    # Try Gemma3 markdown format as fallback
                    try:
                        verified_entities = self._parse_gemma3_markdown(response, entities, llm_model)
                        if verified_entities:
                            logger.info(f"Successfully parsed Gemma3 markdown format: {len(verified_entities)} entities")
                            # Log parsed Gemma3 entities as JSON
                            try:
                                log_dir = Path(__file__).parent.parent / 'logs'
                                log_dir.mkdir(parents=True, exist_ok=True)
                                json_log_file = log_dir / f"llm_response_ner_{timestamp}.json"
                                with open(json_log_file, 'w', encoding='utf-8') as f:
                                    json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                                logger.debug(f"Saved parsed Gemma3 entities as JSON to {json_log_file}")
                            except Exception as json_log_error:
                                logger.warning(f"Failed to log parsed Gemma3 JSON: {json_log_error}")
                    except Exception as gemma_error:
                        logger.warning(f"Failed to parse as Gemma3 markdown (model: {llm_model}): {gemma_error}. Response was: {response[:200]}...")
                        return entities
            
            # If no JSON found, try Gemma3 markdown format
            if not verified_entities:
                logger.debug(f"No JSON found in response, trying Gemma3 markdown parser...")
                try:
                    verified_entities = self._parse_gemma3_markdown(response, entities, llm_model)
                    if verified_entities and len(verified_entities) > 0:
                        logger.info(f"Successfully parsed Gemma3 markdown format: {len(verified_entities)} entities")
                        # Log parsed Gemma3 entities as JSON
                        try:
                            log_dir = Path(__file__).parent.parent / 'logs'
                            log_dir.mkdir(parents=True, exist_ok=True)
                            json_log_file = log_dir / f"llm_response_ner_{timestamp}.json"
                            with open(json_log_file, 'w', encoding='utf-8') as f:
                                json.dump(verified_entities, f, indent=2, ensure_ascii=False)
                            logger.debug(f"Saved parsed Gemma3 entities as JSON to {json_log_file}")
                        except Exception as json_log_error:
                            logger.warning(f"Failed to log parsed Gemma3 JSON: {json_log_error}")
                    else:
                        logger.warning(f"No entities found in Gemma3 markdown response (model: {llm_model}), using original entities")
                        return entities
                except Exception as gemma_error:
                    logger.warning(f"Failed to parse as Gemma3 markdown (model: {llm_model}): {gemma_error}. Response was: {response[:200]}...")
                    return entities
            
            if verified_entities and len(verified_entities) > 0:
                
                # Filter out REMOVE entities and entities marked as verified=False
                result = []
                for verified in verified_entities:
                    if not isinstance(verified, dict):
                        continue
                        
                    # Handle both 'text' and 'entity' keys for robustness
                    v_type = verified.get('type')
                    v_text = (verified.get('text') or verified.get('entity') or "").strip()
                    v_reasoning = (verified.get('reasoning') or "").strip()
                    v_verified = verified.get('verified', True)  # Default to True if not specified
                    v_new_entity = verified.get('new_entity', False)  # Default to False for backward compatibility

                    # Skip if type is REMOVE or verified is False
                    if v_type == 'REMOVE' or v_verified is False:
                        continue

                    if v_text:
                        # Find original entity to preserve position info
                        # Use case-insensitive matching and strip whitespace
                        verified_text_lower = v_text.lower()
                        original = next((e for e in entities if e['text'].strip().lower() == verified_text_lower), None)

                        if original:
                            # Existing entity that was verified/corrected
                            result.append({
                                'text': v_text,
                                'type': v_type or original.get('type', 'UNKNOWN'),
                                'start': original.get('start', 0),
                                'end': original.get('end', 0),
                                'confidence': verified.get('confidence', 1.0),
                                'reasoning': (v_reasoning or f"Not verified by LLM model {llm_model} - listed in verified entities but no reasoning provided").strip(),
                                'source': 'verified'  # Indicates this was verified from original spaCy entities
                            })
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
                                'reasoning': (v_reasoning or f"Discovered by LLM model {llm_model} but no reasoning provided").strip(),
                                'source': 'discovered'  # Indicates this was discovered by LLM
                            })
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
                                'reasoning': (v_reasoning or f"Not verified by LLM model {llm_model} - listed as existing but no reasoning provided").strip(),
                                'source': 'verified'  # Treat as verified since LLM says it came from original list
                            })
                
                if debug:
                    original_count = len(entities)
                    verified_count = len([r for r in result if r.get('source') == 'verified'])
                    discovered_count = len([r for r in result if r.get('source') == 'discovered'])
                    corrected_count = len([r for r in result if r.get('source') == 'corrected'])
                    removed_count = original_count - verified_count - corrected_count

                    logger.debug(f"LLM verification (model: {llm_model}): {verified_count} verified, {discovered_count} discovered, {corrected_count} corrected, {removed_count} removed")
                
                return result
            else:
                logger.warning(f"No valid entities found in LLM verification response (model: {llm_model}), using original entities")
                return entities

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
                llm_cfg = config.get_llm_config()
                main_model = llm_cfg.get('model', 'unknown')
                verification_model = llm_cfg.get('verification_model')
                llm_model = verification_model if (verification_model and verification_model.strip()) else main_model
                verification_timeout = llm_cfg.get('verification_timeout', 300)
                parallel_requests = llm_cfg.get('parallel_requests', True)

                logger.debug(f"Deep mode: verifying {len(blocks)} blocks with LLM (model: {llm_model}, blocksize={blocksize}, parallel={parallel_requests})")

                # Prepare blocks for verification (filter out empty blocks)
                blocks_to_verify = [
                    block for block in blocks
                    if block_entity_map.get(block['block_index'], [])
                ]

                verified_blocks = []

                if parallel_requests and len(blocks_to_verify) > 1:
                    # Use asyncio for truly parallel async HTTP requests
                    import asyncio

                    logger.debug(f"Processing {len(blocks_to_verify)} blocks in parallel with asyncio")

                    # Create async verification tasks
                    async def verify_blocks_async():
                        # Create client for async requests
                        client = self._create_ollama_client(llm_model, verification_timeout)

                        # Create async tasks for each block
                        tasks = []
                        for block in blocks_to_verify:
                            block_text = block['text']
                            block_index = block['block_index']
                            block_entities = block_entity_map.get(block_index, [])

                            # Create async task
                            task = self._verify_ner_with_llm_client_async(
                                block_text, block_entities, client, llm_model, debug=False
                            )
                            tasks.append((block_index, task))

                        # Execute all tasks concurrently
                        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

                        # Process results
                        verified_results = []
                        for (block_index, _), result in zip(tasks, results):
                            if isinstance(result, Exception):
                                logger.error(f"Error in parallel verification for block {block_index}: {result}")
                                # Continue with other blocks
                            else:
                                verified_results.append((block_index, result))
                                logger.debug(f"Completed verification for block {block_index}")

                        return verified_results

                    # Apply nest_asyncio to handle nested event loops
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                    except ImportError:
                        logger.warning("nest_asyncio not available - may encounter event loop issues")

                    # Run the async verification
                    try:
                        verified_blocks = asyncio.run(verify_blocks_async())
                        logger.debug(f"Successfully completed parallel async verification of {len(verified_blocks)} blocks")
                    except Exception as e:
                        logger.warning(f"Parallel async verification failed: {e}, falling back to sequential processing")
                        # Fall back to sequential processing
                        verified_blocks = []
                        for block in blocks_to_verify:
                            block_text = block['text']
                            block_index = block['block_index']
                            block_entities = block_entity_map.get(block_index, [])
                            try:
                                verified_block_entities = self.verify_ner_with_llm(
                                    block_text, block_entities, debug=False
                                )
                                verified_blocks.append((block_index, verified_block_entities))
                                logger.debug(f"Completed sequential verification for block {block_index}")
                            except Exception as seq_error:
                                logger.error(f"Sequential processing failed for block {block_index}: {seq_error}")
                                verified_blocks.append((block_index, block_entities))  # Use original
                else:
                    # Sequential processing (fallback or single block)
                    if parallel_requests:
                        logger.debug("Using sequential processing (only 1 block to verify)")
                    else:
                        logger.debug("Using sequential processing (parallel disabled)")

                    for block in blocks_to_verify:
                        block_text = block['text']
                        block_index = block['block_index']
                        block_entities = block_entity_map.get(block_index, [])

                        # Verify entities from this block
                        verified_block_entities = self.verify_ner_with_llm(
                            block_text,
                            block_entities,
                            debug=False
                        )
                        verified_blocks.append((block_index, verified_block_entities))

                # Process results from all blocks
                for block_index, verified_block_entities in verified_blocks:
                    # Map verified entities back to sentences
                    for verified_entity in verified_block_entities:
                        v_text = (verified_entity.get('text') or "").strip().lower()
                        if v_text:
                            # Find which sentence(s) this entity belongs to
                            # Store block indices for reference
                            verified_entity['block_index'] = block_index
                            verified_entity['sentence_indices'] = block['indices']
                            verified_map[v_text] = verified_entity
                    
                    if verified_block_entities:
                        verification_successful = True
                
                logger.debug(f"Deep mode: verified {len(verified_map)} entities across {len(blocks)} blocks")
                
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
                # Use verification model if specified, otherwise fallback to main model
                main_model = llm_cfg.get('model', 'unknown')
                verification_model = llm_cfg.get('verification_model')
                llm_model = verification_model if (verification_model and verification_model.strip()) else main_model
                logger.debug(f"Starting LLM verification (model: {llm_model}) for {len(original_entities)} entities")
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

