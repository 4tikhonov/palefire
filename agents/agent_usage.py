#!/usr/bin/env python3
"""
Example: Using the AI Agent Daemon

This example shows how to use the AI Agent daemon to extract keywords
and entities without model loading delays.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import get_daemon, AIAgentDaemon
import json


def example_direct_usage():
    """Example: Use daemon directly (models stay loaded)."""
    print("=" * 80)
    print("Example 1: Direct Daemon Usage")
    print("=" * 80)
    
    # Get daemon instance (singleton pattern)
    daemon = get_daemon(use_spacy=True)
    
    # Initialize models (only done once)
    daemon.model_manager.initialize(use_spacy=True)
    
    # Now you can use models without reloading delays
    text = "Artificial intelligence and machine learning are transforming technology."
    
    # Extract keywords (fast - models already loaded)
    keywords = daemon.extract_keywords(text, num_keywords=10)
    print(f"\nExtracted {len(keywords)} keywords:")
    for kw in keywords[:5]:
        print(f"  - {kw['keyword']} (score: {kw['score']:.4f})")
    
    # Extract entities (fast - models already loaded)
    entities = daemon.extract_entities(text)
    print(f"\nExtracted {entities['entity_count']} entities:")
    for entity_type, entity_list in list(entities['entities_by_type'].items())[:3]:
        print(f"  - {entity_type}: {', '.join(entity_list[:3])}")


def example_daemon_service():
    """Example: Run as daemon service."""
    print("\n" + "=" * 80)
    print("Example 2: Running as Daemon Service")
    print("=" * 80)
    print("\nTo run as daemon:")
    print("  python3 palefire-cli.py agent start --daemon")
    print("\nTo check status:")
    print("  python3 palefire-cli.py agent status")
    print("\nTo stop:")
    print("  python3 palefire-cli.py agent stop")


def example_multiple_requests():
    """Example: Multiple requests benefit from loaded models."""
    print("\n" + "=" * 80)
    print("Example 3: Multiple Requests (Models Stay Loaded)")
    print("=" * 80)
    
    daemon = get_daemon(use_spacy=True)
    daemon.model_manager.initialize(use_spacy=True)
    
    texts = [
        "Machine learning algorithms process data efficiently.",
        "Natural language processing helps computers understand human language.",
        "Deep learning neural networks can recognize patterns in images."
    ]
    
    print("\nProcessing multiple texts (models loaded once):")
    for i, text in enumerate(texts, 1):
        keywords = daemon.extract_keywords(text, num_keywords=5)
        print(f"\nText {i}: {text[:50]}...")
        print(f"  Keywords: {', '.join([kw['keyword'] for kw in keywords[:3]])}")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    example_direct_usage()
    example_daemon_service()
    example_multiple_requests()
    
    print("\n" + "=" * 80)
    print("✅ Examples completed")
    print("=" * 80)

