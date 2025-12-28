#!/usr/bin/env python3
"""
Pale Fire AI Agent Service

Systemd/service manager compatible daemon script for running the AI Agent.
Can be used with systemd, launchd, or other service managers.
"""

import sys
import os
import signal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import AIAgentDaemon
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/palefire_ai_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for service."""
    # Configuration
    pidfile = os.environ.get('PALEFIRE_PIDFILE', '/tmp/palefire_ai_agent.pid')
    use_spacy = os.environ.get('PALEFIRE_USE_SPACY', 'true').lower() == 'true'
    # In Docker, we typically want foreground mode (daemon=False)
    # Docker handles the daemonization
    daemon_mode = os.environ.get('PALEFIRE_DAEMON', 'true').lower() == 'true'
    
    logger.info("Starting Pale Fire AI Agent Service...")
    logger.info(f"PID file: {pidfile}")
    logger.info(f"Use spaCy: {use_spacy}")
    logger.info(f"Daemon mode: {daemon_mode}")
    
    # Create and start daemon
    daemon = AIAgentDaemon(pidfile=pidfile, use_spacy=use_spacy)
    
    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        daemon.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        daemon.start(daemon=daemon_mode)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        daemon.stop()


if __name__ == '__main__':
    main()

