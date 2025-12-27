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
from typing import Optional

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
from modules import EntityEnricher, QuestionTypeDetector, KeywordExtractor

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


def extract_keywords_from_text(
    text: str,
    method: str = 'tfidf',
    num_keywords: int = 10,
    min_word_length: int = 3,
    max_word_length: int = 50,
    use_stemming: bool = False,
    tfidf_weight: float = 1.0,
    textrank_weight: float = 0.5,
    word_freq_weight: float = 0.3,
    position_weight: float = 0.2,
    title_weight: float = 2.0,
    first_sentence_weight: float = 1.5,
    enable_ngrams: bool = True,
    min_ngram: int = 2,
    max_ngram: int = 4,
    ngram_weight: float = 1.2,
    documents_file: Optional[str] = None,
    output_file: Optional[str] = None,
    debug: bool = False
):
    """
    Extract keywords from text using Gensim.
    
    Args:
        text: Input text to extract keywords from
        method: Extraction method ('tfidf', 'textrank', 'word_freq', 'combined')
        num_keywords: Number of keywords to extract
        min_word_length: Minimum word length
        max_word_length: Maximum word length
        use_stemming: Whether to use stemming
        tfidf_weight: Weight for TF-IDF scores
        textrank_weight: Weight for TextRank scores
        word_freq_weight: Weight for word frequency scores
        position_weight: Weight for position-based scoring
        title_weight: Weight multiplier for words in titles
        first_sentence_weight: Weight multiplier for words in first sentence
        documents_file: Optional path to JSON file with documents for IDF
        output_file: Optional path to output JSON file
        debug: Enable debug output
    """
    try:
        if debug:
            import sys
            debug_print('\n' + '='*80, file=sys.stderr)
            debug_print('🔑 KEYWORD EXTRACTION', file=sys.stderr)
            debug_print('='*80, file=sys.stderr)
            debug_print(f'Method: {method}', file=sys.stderr)
            debug_print(f'Number of keywords: {num_keywords}', file=sys.stderr)
            debug_print(f'Text length: {len(text)} characters', file=sys.stderr)
        
        # Load documents if provided
        documents = None
        if documents_file:
            try:
                with open(documents_file, 'r', encoding='utf-8') as f:
                    documents_data = json.load(f)
                    if isinstance(documents_data, list):
                        documents = documents_data
                    else:
                        logger.warning(f"Documents file should contain a list, got {type(documents_data)}")
            except Exception as e:
                logger.error(f"Error loading documents file: {e}")
        
        # Create keyword extractor
        extractor = KeywordExtractor(
            method=method,
            num_keywords=num_keywords,
            min_word_length=min_word_length,
            max_word_length=max_word_length,
            use_stemming=use_stemming,
            tfidf_weight=tfidf_weight,
            textrank_weight=textrank_weight,
            word_freq_weight=word_freq_weight,
            position_weight=position_weight,
            title_weight=title_weight,
            first_sentence_weight=first_sentence_weight,
            enable_ngrams=enable_ngrams,
            min_ngram=min_ngram,
            max_ngram=max_ngram,
            ngram_weight=ngram_weight,
        )
        
        # Extract keywords
        keywords = extractor.extract(text, documents)
        
        if debug:
            import sys
            debug_print(f'\nExtracted {len(keywords)} keywords:', file=sys.stderr)
            for i, kw in enumerate(keywords, 1):
                kw_type = kw.get('type', 'unigram')
                debug_print(f'  {i}. {kw["keyword"]} (score: {kw["score"]:.4f}, type: {kw_type})', file=sys.stderr)
        
        # Prepare output
        output = {
            'method': method,
            'num_keywords': len(keywords),
            'keywords': keywords,
            'parameters': {
                'num_keywords': num_keywords,
                'min_word_length': min_word_length,
                'max_word_length': max_word_length,
                'use_stemming': use_stemming,
                'tfidf_weight': tfidf_weight,
                'textrank_weight': textrank_weight,
                'word_freq_weight': word_freq_weight,
                'position_weight': position_weight,
                'title_weight': title_weight,
                'first_sentence_weight': first_sentence_weight,
                'enable_ngrams': enable_ngrams,
                'min_ngram': min_ngram,
                'max_ngram': max_ngram,
                'ngram_weight': ngram_weight,
            }
        }
        
        # Output results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            if debug:
                import sys
                debug_print(f'\n✅ Keywords saved to {output_file}', file=sys.stderr)
        else:
            # Always output JSON to stdout (debug messages go to stderr)
            print(json.dumps(output, indent=2, ensure_ascii=False))
        
        if debug:
            # Debug messages go to stderr so they don't interfere with JSON output
            import sys
            debug_print('\n' + '='*80, file=sys.stderr)
            debug_print('✅ KEYWORD EXTRACTION COMPLETE', file=sys.stderr)
            debug_print('='*80, file=sys.stderr)
        
    except ImportError as e:
        logger.error(f"Gensim not available: {e}")
        logger.error("Install with: pip install gensim")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


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

  # Extract keywords from text
  %(prog)s keywords "Your text here" --method tfidf --num-keywords 10

  # Extract keywords with custom weights
  %(prog)s keywords "Your text here" --method combined --tfidf-weight 1.5 --textrank-weight 0.8
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
    
    # Keywords command
    keywords_parser = subparsers.add_parser('keywords', help='Extract keywords from text using Gensim')
    keywords_parser.add_argument('text', type=str, help='Text to extract keywords from')
    keywords_parser.add_argument('--method', type=str, default='tfidf',
                                choices=['tfidf', 'textrank', 'word_freq', 'combined'],
                                help='Extraction method (default: tfidf)')
    keywords_parser.add_argument('--num-keywords', type=int, default=10, dest='num_keywords',
                                help='Number of keywords to extract (default: 10)')
    keywords_parser.add_argument('--min-word-length', type=int, default=3, dest='min_word_length',
                                help='Minimum word length (default: 3)')
    keywords_parser.add_argument('--max-word-length', type=int, default=50, dest='max_word_length',
                                help='Maximum word length (default: 50)')
    keywords_parser.add_argument('--use-stemming', action='store_true', dest='use_stemming',
                                help='Use stemming for preprocessing')
    keywords_parser.add_argument('--tfidf-weight', type=float, default=1.0, dest='tfidf_weight',
                                help='Weight for TF-IDF scores in combined method (default: 1.0)')
    keywords_parser.add_argument('--textrank-weight', type=float, default=0.5, dest='textrank_weight',
                                help='Weight for TextRank scores in combined method (default: 0.5)')
    keywords_parser.add_argument('--word-freq-weight', type=float, default=0.3, dest='word_freq_weight',
                                help='Weight for word frequency scores in combined method (default: 0.3)')
    keywords_parser.add_argument('--position-weight', type=float, default=0.2, dest='position_weight',
                                help='Weight for position-based scoring (default: 0.2)')
    keywords_parser.add_argument('--title-weight', type=float, default=2.0, dest='title_weight',
                                help='Weight multiplier for words in titles/headers (default: 2.0)')
    keywords_parser.add_argument('--first-sentence-weight', type=float, default=1.5, dest='first_sentence_weight',
                                help='Weight multiplier for words in first sentence (default: 1.5)')
    keywords_parser.add_argument('--enable-ngrams', action='store_true', default=True, dest='enable_ngrams',
                                help='Enable n-gram extraction (default: True)')
    keywords_parser.add_argument('--no-ngrams', dest='enable_ngrams', action='store_false',
                                help='Disable n-gram extraction')
    keywords_parser.add_argument('--min-ngram', type=int, default=2, dest='min_ngram',
                                help='Minimum n-gram size (1 for unigrams, 2-4 for phrases) (default: 2)')
    keywords_parser.add_argument('--max-ngram', type=int, default=4, dest='max_ngram',
                                help='Maximum n-gram size (2, 3, or 4) (default: 4)')
    keywords_parser.add_argument('--ngram-weight', type=float, default=1.2, dest='ngram_weight',
                                help='Weight multiplier for n-grams (default: 1.2)')
    keywords_parser.add_argument('--documents', type=str, dest='documents_file',
                                help='Path to JSON file with list of documents for IDF calculation')
    keywords_parser.add_argument('-o', '--output', '-output', type=str, dest='output_file',
                                help='Path to output JSON file (default: print to stdout)')
    keywords_parser.add_argument('--debug', action='store_true',
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
    
    elif args.command == 'keywords':
        # Extract keywords from text
        extract_keywords_from_text(
            text=args.text,
            method=args.method,
            num_keywords=args.num_keywords,
            min_word_length=args.min_word_length,
            max_word_length=args.max_word_length,
            use_stemming=args.use_stemming,
            tfidf_weight=args.tfidf_weight,
            textrank_weight=args.textrank_weight,
            word_freq_weight=args.word_freq_weight,
            position_weight=args.position_weight,
            title_weight=args.title_weight,
            first_sentence_weight=args.first_sentence_weight,
            enable_ngrams=args.enable_ngrams,
            min_ngram=args.min_ngram,
            max_ngram=args.max_ngram,
            ngram_weight=args.ngram_weight,
            documents_file=args.documents_file,
            output_file=args.output_file,
            debug=DEBUG
        )
        
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

