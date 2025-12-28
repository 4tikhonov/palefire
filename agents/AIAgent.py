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
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

# Configure logger
logger = logging.getLogger(__name__)

# Import models
from modules.KeywordBase import KeywordExtractor
from modules.PaleFireCore import EntityEnricher

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
    
    def extract_keywords(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract keywords using the loaded model.
        
        Args:
            text: Text to extract keywords from
            **kwargs: Additional arguments for KeywordExtractor.extract()
        
        Returns:
            List of keyword dictionaries
        """
        extractor = self.model_manager.keyword_extractor
        
        # Update extractor parameters if provided
        if kwargs:
            # Create a new extractor with updated parameters
            extractor = KeywordExtractor(**kwargs)
        
        return extractor.extract(text)
    
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

