#!/usr/bin/env python3
"""
Pale Fire CLI - Intelligent Knowledge Graph Search System

Command-line interface for ingesting episodes and querying the knowledge graph
with intelligent ranking and question-aware search capabilities.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging import INFO
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import configuration
import config

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# Import Pale Fire core modules
from modules import EntityEnricher, QuestionTypeDetector

# Import utility functions
from utils.palefire_utils import (
    search_episodes,
    search_episodes_with_custom_ranking,
    search_episodes_with_question_aware_ranking,
    export_results_to_json,
    clean_database,
)

# Configure logging from config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)

# Global debug flag (set by CLI argument)
DEBUG = False


def debug_print(*args, **kwargs):
    """Print only if DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)


def load_episodes_from_file(filepath: str) -> list:
    """
    Load episodes from a JSON file.
    
    Expected format:
    [
        {
            "content": "text or json object",
            "type": "text" or "json",
            "description": "description"
        },
        ...
    ]
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        # Convert type strings to EpisodeType
        for episode in data:
            type_str = episode.get('type', 'text')
            if type_str == 'text':
                episode['type'] = EpisodeType.text
            elif type_str == 'json':
                episode['type'] = EpisodeType.json
            else:
                logger.warning(f"Unknown episode type: {type_str}, defaulting to text")
                episode['type'] = EpisodeType.text
        
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filepath}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading episodes: {e}")
        sys.exit(1)


async def ingest_episodes(episodes_data: list, graphiti: Graphiti, use_ner: bool = True, debug: bool = False):
    """Ingest episodes into the knowledge graph with optional NER enrichment."""
    try:
        # Initialize the graph database
        await graphiti.build_indices_and_constraints()
        
        # Initialize NER enricher if requested
        enricher = EntityEnricher(use_spacy=True) if use_ner else None
        
        if debug:
            debug_print('\n' + '='*80)
            debug_print(f'📝 EPISODE INGESTION {"WITH NER ENRICHMENT" if use_ner else ""}')
            debug_print('='*80)
        
        for i, episode in enumerate(episodes_data):
            if debug:
                debug_print(f'\n[Episode {i}] Processing...')
            
            if use_ner and enricher:
                # Enrich episode with NER
                enriched_episode = enricher.enrich_episode(episode)
                
                # Display extracted entities
                if debug and enriched_episode['entities_by_type']:
                    debug_print(f'  ✓ Extracted {enriched_episode["entity_count"]} entities:')
                    for entity_type, entity_list in enriched_episode['entities_by_type'].items():
                        debug_print(f'    - {entity_type}: {", ".join(entity_list[:5])}')
        
                # Create enriched content
                content = enricher.create_enriched_content(enriched_episode)
            else:
                content = (episode['content'] if isinstance(episode['content'], str) 
                          else json.dumps(episode['content']))
            
            # Add to Graphiti
            await graphiti.add_episode(
                name=f'Episode {i}',
                episode_body=content,
                source=episode['type'],
                source_description=episode.get('description', 'No description'),
                reference_time=datetime.now(timezone.utc),
            )
            
            if debug:
                debug_print(f'  ✓ Added to graph: Episode {i}')
        
        if debug:
            debug_print('\n' + '='*80)
            debug_print(f'✅ INGESTION COMPLETE - {len(episodes_data)} episodes added')
            debug_print('='*80)
        
    finally:
        await graphiti.close()


async def search_query(query: str, graphiti: Graphiti, method: str = 'question-aware', export_json: str = None, debug: bool = False):
    """Execute a search query using the specified method."""
    results = None
    try:
        if debug:
            debug_print('\n' + '='*80)
            debug_print(f'🔍 SEARCH: "{query}"')
            debug_print(f'Method: {method}')
            debug_print('='*80)
    
        if method == 'standard':
            results = await search_episodes(graphiti, query, debug=debug)
        elif method == 'connection':
            results = await search_episodes_with_custom_ranking(graphiti, query, debug=debug)
        elif method == 'question-aware':
            # Use question-aware search with full 5-factor ranking
            enricher = EntityEnricher(use_spacy=config.NER_USE_SPACY)
            results = await search_episodes_with_question_aware_ranking(graphiti, query, enricher=enricher, debug=debug)
        else:
            logger.error(f"Unknown search method: {method}")
            results = await search_episodes(graphiti, query, debug=debug)

        # Export to JSON if requested
        if export_json and results:
            export_results_to_json(results, export_json, query, method, debug=debug)
        
        return results
    
    finally:
        await graphiti.close()


def create_cli_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='Pale Fire - Intelligent Knowledge Graph Search System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Ingest episodes from a file with NER enrichment
  %(prog)s ingest episodes.json --ner

  # Query with question-aware ranking
  %(prog)s query "Who was the California Attorney General in 2020?"

  # Query with connection-based ranking
  %(prog)s query "Gavin Newsom" --method connection

  # Export results to JSON
  %(prog)s query "California politics" --export results.json

  # Clean the database
  %(prog)s clean --confirm

  # Enable debug output
  %(prog)s query "test query" --debug
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest episodes from a file')
    ingest_parser.add_argument('file', type=str, help='Path to JSON file containing episodes')
    ingest_parser.add_argument('--ner', action='store_true', default=True,
                               help='Use NER enrichment (default: True)')
    ingest_parser.add_argument('--no-ner', dest='ner', action='store_false',
                               help='Disable NER enrichment')
    ingest_parser.add_argument('--debug', action='store_true',
                               help='Enable debug output (verbose printing)')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Search the knowledge graph')
    query_parser.add_argument('query', type=str, help='Search query')
    query_parser.add_argument('--method', type=str, default='question-aware',
                             choices=['standard', 'connection', 'question-aware'],
                             help='Search method (default: question-aware)')
    query_parser.add_argument('--export', type=str, dest='export_json', metavar='FILE',
                             help='Export results to JSON file')
    query_parser.add_argument('--debug', action='store_true',
                             help='Enable debug output (verbose printing)')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean/clear the Neo4j database')
    clean_parser.add_argument('--confirm', action='store_true',
                             help='Skip confirmation prompt')
    clean_parser.add_argument('--nodes-only', action='store_true',
                             help='Delete only nodes (keep indexes and constraints)')
    clean_parser.add_argument('--debug', action='store_true',
                             help='Enable debug output (verbose printing)')
    
    return parser


def create_graphiti_instance():
    """Create and return a configured Graphiti instance."""
    # Get configuration from config module
    llm_cfg = config.get_llm_config()
    emb_cfg = config.get_embedder_config()
    
    # Configure LLM client
    llm_config = LLMConfig(
        api_key=llm_cfg['api_key'],
        model=llm_cfg['model'],
        small_model=llm_cfg['small_model'],
        base_url=llm_cfg['base_url'],
    )

    llm_client = OpenAIGenericClient(config=llm_config)

    # Initialize Graphiti with configured clients
    graphiti = Graphiti(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=emb_cfg['api_key'],
                embedding_model=emb_cfg['embedding_model'],
                embedding_dim=emb_cfg['embedding_dim'],
                base_url=emb_cfg['base_url'],
            )
        ),
        cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
    )
    
    return graphiti


async def main_cli(args):
    """Main CLI entry point."""
    # Set global debug flag (use getattr in case debug is not set)
    global DEBUG
    DEBUG = getattr(args, 'debug', False)
    
    # Validate configuration
    config.validate_config()
    
    if args.command == 'ingest':
        # Load episodes from file
        episodes_data = load_episodes_from_file(args.file)
        if DEBUG:
            debug_print(f'Loaded {len(episodes_data)} episodes from {args.file}')
        
        # Create Graphiti instance
        graphiti = create_graphiti_instance()
        
        # Ingest episodes
        await ingest_episodes(episodes_data, graphiti, use_ner=args.ner, debug=DEBUG)
        
    elif args.command == 'query':
        # Create Graphiti instance
        graphiti = create_graphiti_instance()
        
        # Execute search
        await search_query(args.query, graphiti, method=args.method, export_json=args.export_json, debug=DEBUG)
        
    elif args.command == 'clean':
        # Create Graphiti instance
        graphiti = create_graphiti_instance()
        
        # Clean database
        await clean_database(graphiti, confirm=args.confirm, nodes_only=args.nodes_only, debug=DEBUG)
        
    else:
        if DEBUG:
            debug_print('No command specified. Use --help for usage information.')
        sys.exit(1)


async def main():
    """Legacy main function for backward compatibility."""
    # Get configuration from config module
    llm_cfg = config.get_llm_config()
    emb_cfg = config.get_embedder_config()
    
    # Configure LLM client
    llm_config = LLMConfig(
        api_key=llm_cfg['api_key'],
        model=llm_cfg['model'],
        small_model=llm_cfg['small_model'],
        base_url=llm_cfg['base_url'],
    )

    llm_client = OpenAIGenericClient(config=llm_config)

    # Initialize Graphiti with configured clients
    graphiti = Graphiti(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=emb_cfg['api_key'],
                embedding_model=emb_cfg['embedding_model'],
                embedding_dim=emb_cfg['embedding_dim'],
                base_url=emb_cfg['base_url'],
            )
        ),
        cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
    )

    ADD = False
    if ADD:
        # Note: Episodes should be loaded from an external file using load_episodes_from_file()
        # Example: episodes_data = load_episodes_from_file('episodes.json')
        # Then use ingest_episodes(episodes_data, graphiti, use_ner=True)
        debug_print(True, "⚠️  Legacy ingestion code disabled. Use 'ingest' command instead:")
        debug_print(True, "   python palefire-cli.py ingest episodes.json --ner")
        await graphiti.close()
    else:
        q = "Who was the California Attorney General in 2020?"
        #q = "Who is Gavin Newsom?"
        
        # Compare standard search vs enhanced ranking
        debug_print(True, '\n' + '='*80)
        debug_print(True, 'STANDARD SEARCH (RRF only)')
        debug_print(True, '='*80)
        await search_episodes(graphiti, q, debug=True)
        
        debug_print(True, '\n' + '='*80)
        debug_print(True, 'ENHANCED SEARCH (RRF + Connection-based Ranking)')
        debug_print(True, '='*80)
        # Use configured weight values
        await search_episodes_with_custom_ranking(graphiti, q, debug=True)


if __name__ == '__main__':
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if args.command:
        asyncio.run(main_cli(args))
    else:
        # If no command specified, run legacy main for backward compatibility
        asyncio.run(main())

