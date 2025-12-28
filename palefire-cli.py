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
import time
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
from agents import AIAgentDaemon

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


def extract_keywords_from_parsed_text(text: str, num_keywords: int = 20, 
                                     method: str = 'combined', verify_ner: bool = False,
                                     debug: bool = False) -> Optional[list]:
    """
    Extract keywords from parsed text using the AI Agent daemon.
    
    Args:
        text: Text to extract keywords from
        num_keywords: Number of keywords to extract
        method: Extraction method (tfidf, textrank, word_freq, combined, ner)
        verify_ner: If True and method is 'ner', verify results using LLM
        debug: Enable debug output
        
    Returns:
        List of keywords or None if extraction fails
    """
    if not text or not text.strip():
        return None
    
    if debug:
        import sys
        debug_print('Extracting keywords from parsed text...', file=sys.stderr)
        if verify_ner and method == 'ner':
            debug_print('LLM verification enabled for NER results', file=sys.stderr)
    
    # Ensure daemon is running
    ensure_daemon_running(debug=debug)
    
    try:
        from agents import get_daemon
        daemon = get_daemon(use_spacy=True)
        if not daemon.model_manager.is_initialized():
            daemon.model_manager.initialize(use_spacy=True)
        
        keywords = daemon.extract_keywords(
            text,
            num_keywords=num_keywords,
            method=method,
            verify_ner=verify_ner
        )
        return keywords
    except Exception as e:
        logger.warning(f"Failed to extract keywords: {e}")
        if debug:
            import sys
            debug_print(f'Keyword extraction failed: {e}', file=sys.stderr)
        return None


def extract_file_path_from_prompt(prompt: str) -> Optional[str]:
    """
    Extract file path from prompt using pattern matching (fallback when LLM is not available).
    
    Args:
        prompt: Natural language command
        
    Returns:
        Extracted file path or None
    """
    import re
    import os
    
    # Pattern 1: Quoted paths (single or double quotes) - handles paths with spaces
    # Match everything between quotes that ends with a file extension
    quoted_pattern = r'["\']([^"\']*[^"\']+\.(?:pdf|txt|csv|xlsx|xls|ods|xlsm))["\']'
    match = re.search(quoted_pattern, prompt)
    if match:
        path = match.group(1).strip()
        if os.path.exists(path):
            return path
        # Try expanding ~
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path
    
    # Pattern 2: Absolute paths (starting with /) - handle paths with spaces by looking for file extension
    # Match from / to the file extension, allowing spaces in between
    abs_path_pattern = r'(/[^\s,]*[^\s,]+\.(?:pdf|txt|csv|xlsx|xls|ods|xlsm))'
    matches = re.findall(abs_path_pattern, prompt)
    for path in matches:
        # Clean up any trailing punctuation
        path = path.rstrip('.,;')
        if os.path.exists(path):
            return path
        # Try expanding ~ and resolving
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path
    
    # Pattern 3: Absolute paths with spaces (more flexible)
    # Look for /Users/ or /home/ or any absolute path ending with file extension
    abs_path_flexible = r'(/[^\s,]+(?:/[^\s,]+)*\.(?:pdf|txt|csv|xlsx|xls|ods|xlsm))'
    matches = re.findall(abs_path_flexible, prompt)
    for path in matches:
        path = path.rstrip('.,;')
        if os.path.exists(path):
            return path
    
    # Pattern 4: Relative paths with extensions
    rel_path_pattern = r'([^\s,]+\.(?:pdf|txt|csv|xlsx|xls|ods|xlsm))'
    matches = re.findall(rel_path_pattern, prompt)
    for match in matches:
        # Skip if it's a URL or contains protocol
        if '://' in match or match.startswith('http'):
            continue
        match = match.rstrip('.,;')
        if os.path.exists(match):
            return match
        # Try with current directory
        full_path = os.path.abspath(match)
        if os.path.exists(full_path):
            return full_path
    
    return None


def detect_parser_from_prompt(prompt: str, debug: bool = False) -> Optional[dict]:
    """
    Detect parser type and options from natural language prompt using LLM.
    
    Args:
        prompt: Natural language command (e.g., "parse PDF file example.pdf")
        debug: Enable debug output
        
    Returns:
        Dictionary with parser detection results or None if detection fails
    """
    try:
        # First, try to extract file path using pattern matching (fallback)
        extracted_path = extract_file_path_from_prompt(prompt)
        
        # Try to use LLM for parser detection
        llm_cfg = config.get_llm_config()
        
        # Use LLM if API key is configured (works with both OpenAI and Ollama)
        if llm_cfg.get('api_key'):
            # Try to use simple Ollama client
            try:
                from utils.llm_client import SimpleOllamaClient
                
                llm_client = SimpleOllamaClient(
                    model=llm_cfg['model'],
                    base_url=llm_cfg['base_url'],
                    api_key=llm_cfg['api_key']
                )
                
                # Load parser detection prompt
                prompt_file = Path(__file__).parent / 'prompts' / 'system' / 'parser_detection_prompt.md'
                if prompt_file.exists():
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        system_prompt = f.read()
                else:
                    # Fallback prompt
                    system_prompt = """You are an intelligent file parser selector. Analyze user commands and determine which file parser to use.
Available parsers: pdf, txt, csv, spreadsheet (for .xlsx, .xls, .ods).
Return JSON: {"parser": "pdf|txt|csv|spreadsheet", "file_path": "path", "file_type": "pdf|txt|csv|xlsx|xls|ods", "confidence": 0.0-1.0, "options": {}, "reasoning": "explanation"}"""
                
                # Create detection prompt
                detection_prompt = f"{system_prompt}\n\nUser command: {prompt}\n\nDetect parser and return JSON:"
                
                if debug:
                    import sys
                    debug_print(f'Using LLM to detect parser from prompt: {prompt}', file=sys.stderr)
                
                # Log the request to logs folder
                try:
                    import time
                    from pathlib import Path
                    log_dir = Path(__file__).parent / 'logs'
                    log_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = int(time.time())
                    log_file = log_dir / f"llm_request_parser_{timestamp}.txt"
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.write(detection_prompt)
                except Exception as e:
                    logger.warning(f"Failed to log LLM request: {e}")
                
                # Call LLM
                response = llm_client.complete(
                    messages=[{"role": "user", "content": detection_prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                # Extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    # Ensure file_path is set if LLM didn't extract it
                    if not result.get('file_path') and extracted_path:
                        result['file_path'] = extracted_path
                    if debug:
                        import sys
                        debug_print(f'Parser detection result: {result}', file=sys.stderr)
                    return result
                else:
                    if debug:
                        import sys
                        debug_print(f'No JSON found in LLM response: {response}', file=sys.stderr)
                    # Fallback to pattern-based detection
                    if extracted_path:
                        return _create_fallback_detection(prompt, extracted_path, debug)
                    return None
                    
            except Exception as e:
                if debug:
                    import sys
                    debug_print(f'LLM detection failed: {e}, using fallback', file=sys.stderr)
                # Fallback to pattern-based detection
                if extracted_path:
                    return _create_fallback_detection(prompt, extracted_path, debug)
                return None
        else:
            # No LLM available, use pattern-based detection
            if extracted_path:
                return _create_fallback_detection(prompt, extracted_path, debug)
            return None
            
    except Exception as e:
        if debug:
            import sys
            debug_print(f'Parser detection error: {e}', file=sys.stderr)
        # Last resort: try pattern-based detection
        extracted_path = extract_file_path_from_prompt(prompt)
        if extracted_path:
            return _create_fallback_detection(prompt, extracted_path, debug)
        return None


def _create_fallback_detection(prompt: str, file_path: str, debug: bool = False) -> dict:
    """
    Create parser detection result using pattern matching fallback.
    
    Args:
        prompt: Original prompt
        file_path: Extracted file path
        debug: Enable debug output
        
    Returns:
        Detection result dictionary
    """
    import re
    from pathlib import Path
    
    file_ext = Path(file_path).suffix.lower()
    
    # Determine parser type from extension
    parser_map = {
        '.pdf': 'pdf',
        '.txt': 'txt',
        '.text': 'txt',
        '.csv': 'csv',
        '.xlsx': 'spreadsheet',
        '.xls': 'spreadsheet',
        '.xlsm': 'spreadsheet',
        '.ods': 'spreadsheet',
    }
    
    parser = parser_map.get(file_ext, None)
    if not parser:
        return None
    
    # Extract options from prompt
    options = {}
    
    # Extract max_pages for PDF
    if parser == 'pdf':
        pages_match = re.search(r'(?:first|only|max|limit).*?(\d+).*?page', prompt, re.IGNORECASE)
        if pages_match:
            options['max_pages'] = int(pages_match.group(1))
    
    # Extract delimiter for CSV
    if parser == 'csv':
        delimiter_match = re.search(r'(?:with|using|delimiter).*?([;,\t])', prompt, re.IGNORECASE)
        if delimiter_match:
            options['delimiter'] = delimiter_match.group(1)
    
    # Extract sheet names for spreadsheet
    if parser == 'spreadsheet':
        sheet_match = re.search(r"(?:only|sheet|sheets).*?['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
        if sheet_match:
            options['sheet_names'] = [sheet_match.group(1)]
    
    # Extract keyword extraction options (works for all parsers)
    keyword_patterns = [
        r'extract.*?keyword',
        r'get.*?keyword',
        r'find.*?keyword',
        r'keyword.*?extraction',
        r'extract.*?all.*?keyword',
    ]
    extract_keywords = any(re.search(pattern, prompt, re.IGNORECASE) for pattern in keyword_patterns)
    if extract_keywords:
        options['extract_keywords'] = True
        # Try to extract number of keywords
        num_keywords_match = re.search(r'(?:extract|get|find).*?(\d+).*?keyword', prompt, re.IGNORECASE)
        if num_keywords_match:
            options['num_keywords'] = int(num_keywords_match.group(1))
        elif 'all keywords' in prompt.lower():
            # "all keywords" means extract many keywords
            options['num_keywords'] = 50  # Default for "all"
        
        # Check if NER method is requested
        ner_patterns = [
            r'keyword.*?ner',
            r'keyword.*?with.*?ner',
            r'ner.*?keyword',
            r'named.*?entity.*?keyword',
            r'extract.*?keyword.*?with.*?ner',
            r'use.*?ner.*?for.*?keyword',
        ]
        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in ner_patterns):
            options['keywords_method'] = 'ner'
            
            # Check if verification is requested
            verify_patterns = [
                r'verify',
                r'and.*?verify',
                r'verify.*?ner',
                r'ner.*?verify',
                r'verify.*?result',
                r'check.*?result',
                r'validate',
            ]
            if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in verify_patterns):
                options['verify_ner'] = True
    
    result = {
        'parser': parser,
        'file_path': file_path,
        'file_type': file_ext.lstrip('.'),
        'confidence': 0.8 if file_ext in parser_map else 0.5,
        'options': options,
        'reasoning': f'Pattern-based detection: file extension {file_ext} maps to {parser} parser'
    }
    
    if debug:
        import sys
        debug_print(f'Fallback detection result: {result}', file=sys.stderr)
    
    return result


def parse_file_command(file_path: str, output_file: Optional[str] = None, 
                       extract_keywords: bool = False, keywords_method: str = 'combined',
                       num_keywords: int = 20, verify_ner: bool = False,
                       debug: bool = False, prompt: Optional[str] = None, **parser_options):
    """Parse a file using the appropriate parser."""
    try:
        from agents.parsers import get_parser
        
        # If prompt is provided, try to detect parser and options from it
        if prompt:
            detection_result = detect_parser_from_prompt(prompt, debug=debug)
            if detection_result and detection_result.get('confidence', 0) > 0.5:
                # Use detected parser and file path
                detected_parser = detection_result.get('parser')
                detected_file_path = detection_result.get('file_path') or file_path
                detected_options = detection_result.get('options', {})
                
                if debug:
                    import sys
                    debug_print(f'Detected parser: {detected_parser}, file: {detected_file_path}, options: {detected_options}', file=sys.stderr)
                
                # Override file_path if detected
                if detected_file_path and detected_file_path != file_path:
                    file_path = detected_file_path
                
                # Merge detected options with provided options
                parser_options.update(detected_options)
                
                # Map parser type to actual parser
                if detected_parser == 'pdf':
                    from agents.parsers import PDFParser
                    parser = PDFParser()
                elif detected_parser == 'csv':
                    from agents.parsers import CSVParser
                    parser = CSVParser()
                elif detected_parser == 'txt':
                    from agents.parsers import TXTParser
                    parser = TXTParser()
                elif detected_parser == 'spreadsheet':
                    from agents.parsers import SpreadsheetParser
                    parser = SpreadsheetParser()
                else:
                    # Fallback to auto-detection
                    parser = get_parser(file_path)
            else:
                # Fallback to auto-detection
                if debug:
                    import sys
                    debug_print(f'Parser detection failed or low confidence, using auto-detection', file=sys.stderr)
                parser = get_parser(file_path)
        else:
            if debug:
                import sys
                debug_print(f'Parsing file: {file_path}', file=sys.stderr)
            parser = get_parser(file_path)
        
        # Parse file with options
        result = parser.parse(file_path, **parser_options)
        
        if not result.success:
            logger.error(f"Parsing failed: {result.error}")
            print(json.dumps({'error': result.error}, indent=2))
            return
        
        output = result.to_dict()
        
        # Extract keywords if requested
        if extract_keywords:
            if debug:
                import sys
                debug_print('Extracting keywords from parsed text...', file=sys.stderr)
            
            # Ensure daemon is running
            ensure_daemon_running(debug=debug)
            
            try:
                from agents import get_daemon
                daemon = get_daemon(use_spacy=True)
                if not daemon.model_manager.is_initialized():
                    daemon.model_manager.initialize(use_spacy=True)
                
                keywords = daemon.extract_keywords(
                    result.text,
                    num_keywords=num_keywords,
                    method=keywords_method,
                    verify_ner=verify_ner
                )
                output['keywords'] = keywords
                # Add verified field if NER verification was performed
                if verify_ner and keywords_method == 'ner':
                    output['verified'] = True
            except Exception as e:
                logger.warning(f"Failed to extract keywords: {e}")
                if debug:
                    import sys
                    debug_print(f'Keyword extraction failed: {e}', file=sys.stderr)
        
        # Output results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            if debug:
                import sys
                debug_print(f'✅ Results saved to {output_file}', file=sys.stderr)
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
    
    except ValueError as e:
        logger.error(f"Parser error: {e}")
        print(json.dumps({'error': str(e)}, indent=2))
    except Exception as e:
        logger.error(f"Error parsing file: {e}", exc_info=True)
        print(json.dumps({'error': f"Error parsing file: {str(e)}"}, indent=2))


def parse_txt_command(file_path: str, encoding: str = 'utf-8', 
                     output_file: Optional[str] = None, debug: bool = False,
                     prompt: Optional[str] = None,
                     extract_keywords: bool = False, keywords_method: str = 'combined',
                     num_keywords: int = 20, verify_ner: bool = False):
    """Parse a text file."""
    try:
        from agents.parsers import TXTParser
        
        # If prompt is provided, try to extract file path from it
        if prompt:
            extracted_path = extract_file_path_from_prompt(prompt)
            if extracted_path and (not file_path or not os.path.exists(file_path)):
                file_path = extracted_path
                if debug:
                    import sys
                    debug_print(f'Extracted file path from prompt: {file_path}', file=sys.stderr)
        
        # Validate file path
        if not file_path:
            print(json.dumps({'error': 'No file path provided. Please specify a file or use --prompt with a file path.', 'success': False}, indent=2))
            return
        
        if not os.path.exists(file_path):
            print(json.dumps({'error': f'File not found: {file_path}', 'success': False}, indent=2))
            return
        
        parser = TXTParser(encoding=encoding)
        result = parser.parse(file_path, encoding=encoding)
        
        output = result.to_dict()
        
        # Extract keywords if requested
        if extract_keywords:
            keywords = extract_keywords_from_parsed_text(
                result.text,
                num_keywords=num_keywords,
                method=keywords_method,
                verify_ner=verify_ner,
                debug=debug
            )
            if keywords:
                output['keywords'] = keywords
                # Add verified field if NER verification was performed
                if verify_ner and keywords_method == 'ner':
                    output['verified'] = True
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
    
    except Exception as e:
        logger.error(f"Error parsing text file: {e}", exc_info=True)
        print(json.dumps({'error': f"Error parsing file: {str(e)}"}, indent=2))


def parse_csv_command(file_path: str, delimiter: str = ',', include_headers: bool = True,
                     output_file: Optional[str] = None, debug: bool = False,
                     prompt: Optional[str] = None,
                     extract_keywords: bool = False, keywords_method: str = 'combined',
                     num_keywords: int = 20, verify_ner: bool = False):
    """Parse a CSV file."""
    try:
        from agents.parsers import CSVParser
        
        # If prompt is provided, try to detect parser and options from it
        if prompt:
            # First, try to extract file path from prompt
            extracted_path = extract_file_path_from_prompt(prompt)
            if extracted_path and (not file_path or not os.path.exists(file_path)):
                file_path = extracted_path
                if debug:
                    import sys
                    debug_print(f'Extracted file path from prompt: {file_path}', file=sys.stderr)
            
            # Then try LLM detection for options
            detection_result = detect_parser_from_prompt(prompt, debug=debug)
            if detection_result and detection_result.get('confidence', 0) > 0.5:
                detected_file_path = detection_result.get('file_path')
                detected_options = detection_result.get('options', {})
                if detected_file_path and os.path.exists(detected_file_path):
                    file_path = detected_file_path
                if 'delimiter' in detected_options:
                    delimiter = detected_options['delimiter']
                if 'include_headers' in detected_options:
                    include_headers = detected_options['include_headers']
                if debug:
                    import sys
                    debug_print(f'Using detected file path: {file_path}, delimiter: {delimiter}', file=sys.stderr)
        
        # Validate file path
        if not file_path:
            print(json.dumps({'error': 'No file path provided. Please specify a file or use --prompt with a file path.', 'success': False}, indent=2))
            return
        
        if not os.path.exists(file_path):
            print(json.dumps({'error': f'File not found: {file_path}', 'success': False}, indent=2))
            return
        
        parser = CSVParser(delimiter=delimiter)
        result = parser.parse(file_path, delimiter=delimiter, include_headers=include_headers)
        
        output = result.to_dict()
        
        # Extract keywords if requested
        if extract_keywords:
            keywords = extract_keywords_from_parsed_text(
                result.text,
                num_keywords=num_keywords,
                method=keywords_method,
                verify_ner=verify_ner,
                debug=debug
            )
            if keywords:
                output['keywords'] = keywords
                # Add verified field if NER verification was performed
                if verify_ner and keywords_method == 'ner':
                    output['verified'] = True
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
    
    except Exception as e:
        logger.error(f"Error parsing CSV file: {e}", exc_info=True)
        print(json.dumps({'error': f"Error parsing file: {str(e)}"}, indent=2))


def parse_pdf_command(file_path: str, max_pages: Optional[int] = None, 
                     extract_tables: bool = True, output_file: Optional[str] = None, 
                     debug: bool = False, prompt: Optional[str] = None,
                     extract_keywords: bool = False, keywords_method: str = 'combined',
                     num_keywords: int = 20, verify_ner: bool = False):
    """Parse a PDF file."""
    try:
        from agents.parsers import PDFParser
        
        # If prompt is provided, try to detect parser and options from it
        if prompt:
            # First, always try to extract file path from prompt (pattern matching)
            extracted_path = extract_file_path_from_prompt(prompt)
            if extracted_path and (not file_path or not os.path.exists(file_path)):
                file_path = extracted_path
                if debug:
                    import sys
                    debug_print(f'Extracted file path from prompt: {file_path}', file=sys.stderr)
            
            # Then try LLM detection for parser type and options
            detection_result = detect_parser_from_prompt(prompt, debug=debug)
            if detection_result and detection_result.get('confidence', 0) > 0.5:
                detected_file_path = detection_result.get('file_path')
                detected_options = detection_result.get('options', {})
                # Use detected file path if available and current path is invalid
                if detected_file_path and os.path.exists(detected_file_path):
                    file_path = detected_file_path
                if 'max_pages' in detected_options and detected_options['max_pages']:
                    max_pages = detected_options['max_pages']
                if 'extract_tables' in detected_options:
                    extract_tables = detected_options['extract_tables']
                # Extract keyword extraction options from detection
                if 'extract_keywords' in detected_options and detected_options['extract_keywords']:
                    extract_keywords = True
                    if 'num_keywords' in detected_options:
                        num_keywords = detected_options['num_keywords']
                    if 'keywords_method' in detected_options:
                        keywords_method = detected_options['keywords_method']
                    if 'verify_ner' in detected_options and detected_options['verify_ner']:
                        verify_ner = True
                if debug:
                    import sys
                    debug_print(f'Using detected file path: {file_path}, max_pages: {max_pages}, extract_keywords: {extract_keywords}, verify_ner: {verify_ner}', file=sys.stderr)
            else:
                # Fallback: try pattern-based keyword detection
                import re
                keyword_patterns = [
                    r'extract.*?keyword',
                    r'get.*?keyword',
                    r'find.*?keyword',
                    r'keyword.*?extraction',
                    r'extract.*?all.*?keyword',
                ]
                if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in keyword_patterns):
                    extract_keywords = True
                    num_keywords_match = re.search(r'(?:extract|get|find).*?(\d+).*?keyword', prompt, re.IGNORECASE)
                    if num_keywords_match:
                        num_keywords = int(num_keywords_match.group(1))
                    elif 'all keywords' in prompt.lower():
                        num_keywords = 50  # Default for "all"
                    
                    # Check if NER method is requested
                    ner_patterns = [
                        r'keyword.*?ner',
                        r'keyword.*?with.*?ner',
                        r'ner.*?keyword',
                        r'named.*?entity.*?keyword',
                        r'extract.*?keyword.*?with.*?ner',
                        r'use.*?ner.*?for.*?keyword',
                    ]
                    if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in ner_patterns):
                        keywords_method = 'ner'
                        
                        # Check if verification is requested
                        verify_patterns = [
                            r'verify',
                            r'and.*?verify',
                            r'verify.*?ner',
                            r'ner.*?verify',
                            r'verify.*?result',
                            r'check.*?result',
                            r'validate',
                        ]
                        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in verify_patterns):
                            verify_ner = True
                    
                    if debug:
                        import sys
                        debug_print(f'Detected keyword extraction from prompt: extract_keywords={extract_keywords}, num_keywords={num_keywords}, method={keywords_method}, verify_ner={verify_ner}', file=sys.stderr)
        
        # Validate file path
        if not file_path:
            print(json.dumps({'error': 'No file path provided. Please specify a file or use --prompt with a file path.', 'success': False}, indent=2))
            return
        
        if not os.path.exists(file_path):
            print(json.dumps({'error': f'File not found: {file_path}', 'success': False}, indent=2))
            return
        
        parser = PDFParser()
        result = parser.parse(file_path, max_pages=max_pages, extract_tables=extract_tables)
        
        output = result.to_dict()
        
        # Extract keywords if requested
        if extract_keywords:
            keywords = extract_keywords_from_parsed_text(
                result.text,
                num_keywords=num_keywords,
                method=keywords_method,
                verify_ner=verify_ner,
                debug=debug
            )
            if keywords:
                output['keywords'] = keywords
                # Add verified field if NER verification was performed
                if verify_ner and keywords_method == 'ner':
                    output['verified'] = True
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
    
    except Exception as e:
        logger.error(f"Error parsing PDF file: {e}", exc_info=True)
        print(json.dumps({'error': f"Error parsing file: {str(e)}"}, indent=2))


def parse_spreadsheet_command(file_path: str, sheet_names: Optional[list] = None,
                             include_headers: bool = True, output_file: Optional[str] = None,
                             debug: bool = False, prompt: Optional[str] = None,
                             extract_keywords: bool = False, keywords_method: str = 'combined',
                             num_keywords: int = 20, verify_ner: bool = False):
    """Parse a spreadsheet file."""
    try:
        from agents.parsers import SpreadsheetParser
        
        # If prompt is provided, try to detect parser and options from it
        if prompt:
            detection_result = detect_parser_from_prompt(prompt, debug=debug)
            if detection_result and detection_result.get('confidence', 0) > 0.5:
                detected_file_path = detection_result.get('file_path') or file_path
                detected_options = detection_result.get('options', {})
                if detected_file_path:
                    file_path = detected_file_path
                if 'sheet_names' in detected_options and detected_options['sheet_names']:
                    sheet_names = detected_options['sheet_names']
                if debug:
                    import sys
                    debug_print(f'Using detected file path: {file_path}, sheet_names: {sheet_names}', file=sys.stderr)
        
        parser = SpreadsheetParser()
        result = parser.parse(file_path, sheet_names=sheet_names, include_headers=include_headers)
        
        output = result.to_dict()
        
        # Extract keywords if requested
        if extract_keywords:
            keywords = extract_keywords_from_parsed_text(
                result.text,
                num_keywords=num_keywords,
                method=keywords_method,
                verify_ner=verify_ner,
                debug=debug
            )
            if keywords:
                output['keywords'] = keywords
                # Add verified field if NER verification was performed
                if verify_ner and keywords_method == 'ner':
                    output['verified'] = True
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
    
    except Exception as e:
        logger.error(f"Error parsing spreadsheet file: {e}", exc_info=True)
        print(json.dumps({'error': f"Error parsing file: {str(e)}"}, indent=2))


def ensure_daemon_running(pidfile: str = '/tmp/palefire_ai_agent.pid', use_spacy: bool = True, debug: bool = False) -> bool:
    """
    Check if daemon is running, start it if not.
    
    Returns:
        True if daemon is running (or was started), False otherwise
    """
    import signal as sig
    
    # Check if daemon is running
    daemon_running = False
    try:
        if os.path.exists(pidfile):
            with open(pidfile, 'r') as f:
                pid = int(f.read().strip())
            # Check if process exists
            try:
                os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
                daemon_running = True
            except ProcessLookupError:
                # Stale PID file
                try:
                    os.remove(pidfile)
                except:
                    pass
    except (FileNotFoundError, ValueError):
        pass
    
    if daemon_running:
        if debug:
            import sys
            debug_print(f'✅ Daemon is running (PID: {pid})', file=sys.stderr)
        return True
    
    # Daemon not running, start it
    if debug:
        import sys
        debug_print('⚠️  Daemon not running, starting it automatically...', file=sys.stderr)
    
    try:
        daemon = AIAgentDaemon(pidfile=pidfile, use_spacy=use_spacy)
        
        # Start in background
        daemon.start(daemon=True)
        
        # Wait a moment for daemon to initialize
        time.sleep(1.0)
        
        # Verify it started
        if os.path.exists(pidfile):
            with open(pidfile, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)
                if debug:
                    import sys
                    debug_print(f'✅ Daemon started successfully (PID: {pid})', file=sys.stderr)
                return True
            except ProcessLookupError:
                if debug:
                    import sys
                    debug_print('⚠️  Daemon process not found after start', file=sys.stderr)
                return False
        else:
            if debug:
                import sys
                debug_print('⚠️  PID file not created', file=sys.stderr)
            return False
    except Exception as e:
        if debug:
            import sys
            debug_print(f'❌ Failed to start daemon: {e}', file=sys.stderr)
        logger.warning(f"Failed to start daemon automatically: {e}")
        return False


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
    verify_ner: bool = False,
    debug: bool = False
):
    """
    Extract keywords from text using Gensim or spaCy NER.
    
    Args:
        text: Input text to extract keywords from
        method: Extraction method ('tfidf', 'textrank', 'word_freq', 'combined', 'ner')
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
        verify_ner: If True and method is 'ner', verify results using LLM
        debug: Enable debug output
    """
    try:
        if debug:
            import sys
            debug_print('\n' + '='*80, file=sys.stderr)
            debug_print('🔑 KEYWORD EXTRACTION', file=sys.stderr)
            debug_print('='*80, file=sys.stderr)
            debug_print(f'Method: {method}', file=sys.stderr)
            if method == 'ner' and verify_ner:
                debug_print('LLM verification: enabled', file=sys.stderr)
            debug_print(f'Number of keywords: {num_keywords}', file=sys.stderr)
            debug_print(f'Text length: {len(text)} characters', file=sys.stderr)
        
        # If NER method, use AI Agent daemon
        if method == 'ner':
            keywords = extract_keywords_from_parsed_text(
                text,
                num_keywords=num_keywords,
                method='ner',
                verify_ner=verify_ner,
                debug=debug
            )
            if not keywords:
                keywords = []
        else:
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
                kw_reasoning = f" - reasoning: {kw['reasoning']}" if kw.get('reasoning') else ""
                debug_print(f'  {i}. {kw["keyword"]} (score: {kw["score"]:.4f}, type: {kw_type}){kw_reasoning}', file=sys.stderr)
        
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
        
        # Add verified field if NER verification was performed
        if method == 'ner' and verify_ner:
            output['verified'] = True
        
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

  # Parse file using natural language
  %(prog)s parse --prompt "parse PDF file example.pdf"
  %(prog)s parse --prompt "parse CSV file data.csv with semicolon delimiter"
  %(prog)s parse-spreadsheet --prompt "parse Excel file report.xlsx, only Summary sheet"
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
                                choices=['tfidf', 'textrank', 'word_freq', 'combined', 'ner'],
                                help='Extraction method (default: tfidf). Use "ner" for spaCy NER-based extraction.')
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
    
    # Parse command (file parsing)
    parse_parser = subparsers.add_parser('parse', help='Parse files and extract text')
    parse_parser.add_argument('file', type=str, nargs='?', help='Path to file to parse (optional if using --prompt)')
    parse_parser.add_argument('--prompt', '-p', type=str, dest='prompt',
                             help='Natural language command (e.g., "parse PDF file example.pdf")')
    parse_parser.add_argument('--output', '-o', type=str, dest='output_file',
                             help='Output JSON file (default: print to stdout)')
    parse_parser.add_argument('--extract-keywords', action='store_true', dest='extract_keywords',
                             help='Extract keywords from parsed text')
    parse_parser.add_argument('--keywords-method', type=str, default='combined',
                             choices=['tfidf', 'textrank', 'word_freq', 'combined', 'ner'],
                             help='Keyword extraction method (default: combined). Use "ner" for spaCy NER-based extraction.')
    parse_parser.add_argument('--num-keywords', type=int, default=20, dest='num_keywords',
                             help='Number of keywords to extract (default: 20)')
    parse_parser.add_argument('--debug', action='store_true',
                             help='Enable debug output')
    
    # Parser-specific commands
    txt_parser_cmd = subparsers.add_parser('parse-txt', help='Parse text files (.txt)')
    txt_parser_cmd.add_argument('file', type=str, nargs='?', help='Path to .txt file (optional if using --prompt)')
    txt_parser_cmd.add_argument('--prompt', '-p', type=str, dest='prompt',
                               help='Natural language command (e.g., "parse text file readme.txt")')
    txt_parser_cmd.add_argument('--encoding', type=str, default='utf-8',
                               help='Text encoding (default: utf-8)')
    txt_parser_cmd.add_argument('--output', '-o', type=str, dest='output_file',
                               help='Output JSON file')
    txt_parser_cmd.add_argument('--extract-keywords', action='store_true', dest='extract_keywords',
                               help='Extract keywords from parsed text')
    txt_parser_cmd.add_argument('--keywords-method', type=str, default='combined',
                               choices=['tfidf', 'textrank', 'word_freq', 'combined', 'ner'],
                               help='Keyword extraction method (default: combined). Use "ner" for spaCy NER-based extraction.')
    txt_parser_cmd.add_argument('--num-keywords', type=int, default=20, dest='num_keywords',
                               help='Number of keywords to extract (default: 20)')
    txt_parser_cmd.add_argument('--debug', action='store_true', help='Enable debug output')
    
    csv_parser_cmd = subparsers.add_parser('parse-csv', help='Parse CSV files (.csv)')
    csv_parser_cmd.add_argument('file', type=str, nargs='?', help='Path to .csv file (optional if using --prompt)')
    csv_parser_cmd.add_argument('--prompt', '-p', type=str, dest='prompt',
                               help='Natural language command (e.g., "parse CSV file data.csv with semicolon delimiter")')
    csv_parser_cmd.add_argument('--delimiter', type=str, default=',',
                               help='CSV delimiter (default: ,)')
    csv_parser_cmd.add_argument('--include-headers', action='store_true', default=True,
                               dest='include_headers', help='Include header row')
    csv_parser_cmd.add_argument('--no-headers', action='store_false', dest='include_headers',
                               help='Do not include header row')
    csv_parser_cmd.add_argument('--output', '-o', type=str, dest='output_file',
                               help='Output JSON file')
    csv_parser_cmd.add_argument('--extract-keywords', action='store_true', dest='extract_keywords',
                               help='Extract keywords from parsed text')
    csv_parser_cmd.add_argument('--keywords-method', type=str, default='combined',
                               choices=['tfidf', 'textrank', 'word_freq', 'combined', 'ner'],
                               help='Keyword extraction method (default: combined). Use "ner" for spaCy NER-based extraction.')
    csv_parser_cmd.add_argument('--verify-ner', action='store_true', dest='verify_ner',
                               help='Verify NER results using LLM to remove false positives (only works with --keywords-method ner)')
    csv_parser_cmd.add_argument('--num-keywords', type=int, default=20, dest='num_keywords',
                               help='Number of keywords to extract (default: 20)')
    csv_parser_cmd.add_argument('--debug', action='store_true', help='Enable debug output')
    
    pdf_parser_cmd = subparsers.add_parser('parse-pdf', help='Parse PDF files (.pdf)')
    pdf_parser_cmd.add_argument('file', type=str, nargs='?', help='Path to .pdf file (optional if using --prompt)')
    pdf_parser_cmd.add_argument('--prompt', '-p', type=str, dest='prompt',
                               help='Natural language command (e.g., "parse PDF file report.pdf, first 10 pages")')
    pdf_parser_cmd.add_argument('--max-pages', type=int, dest='max_pages',
                               help='Maximum number of pages to parse')
    pdf_parser_cmd.add_argument('--extract-tables', action='store_true', default=True,
                               dest='extract_tables', help='Extract tables from PDF')
    pdf_parser_cmd.add_argument('--no-tables', action='store_false', dest='extract_tables',
                               help='Do not extract tables')
    pdf_parser_cmd.add_argument('--output', '-o', type=str, dest='output_file',
                               help='Output JSON file')
    pdf_parser_cmd.add_argument('--extract-keywords', action='store_true', dest='extract_keywords',
                               help='Extract keywords from parsed text')
    pdf_parser_cmd.add_argument('--keywords-method', type=str, default='combined',
                               choices=['tfidf', 'textrank', 'word_freq', 'combined', 'ner'],
                               help='Keyword extraction method (default: combined). Use "ner" for spaCy NER-based extraction.')
    pdf_parser_cmd.add_argument('--verify-ner', action='store_true', dest='verify_ner',
                               help='Verify NER results using LLM to remove false positives (only works with --keywords-method ner)')
    pdf_parser_cmd.add_argument('--num-keywords', type=int, default=20, dest='num_keywords',
                               help='Number of keywords to extract (default: 20)')
    pdf_parser_cmd.add_argument('--debug', action='store_true', help='Enable debug output')
    
    spreadsheet_parser_cmd = subparsers.add_parser('parse-spreadsheet', 
                                                  help='Parse spreadsheet files (.xlsx, .xls, .ods)')
    spreadsheet_parser_cmd.add_argument('file', type=str, nargs='?', help='Path to spreadsheet file (optional if using --prompt)')
    spreadsheet_parser_cmd.add_argument('--prompt', '-p', type=str, dest='prompt',
                                       help='Natural language command (e.g., "parse Excel file report.xlsx, only Summary sheet")')
    spreadsheet_parser_cmd.add_argument('--sheets', type=str, nargs='+', dest='sheet_names',
                                       help='Sheet names to parse (default: all)')
    spreadsheet_parser_cmd.add_argument('--include-headers', action='store_true', default=True,
                                       dest='include_headers', help='Include header rows')
    spreadsheet_parser_cmd.add_argument('--no-headers', action='store_false', dest='include_headers',
                                       help='Do not include header rows')
    spreadsheet_parser_cmd.add_argument('--output', '-o', type=str, dest='output_file',
                                       help='Output JSON file')
    spreadsheet_parser_cmd.add_argument('--extract-keywords', action='store_true', dest='extract_keywords',
                                       help='Extract keywords from parsed text')
    spreadsheet_parser_cmd.add_argument('--keywords-method', type=str, default='combined',
                                       choices=['tfidf', 'textrank', 'word_freq', 'combined', 'ner'],
                                       help='Keyword extraction method (default: combined). Use "ner" for spaCy NER-based extraction.')
    spreadsheet_parser_cmd.add_argument('--verify-ner', action='store_true', dest='verify_ner',
                                       help='Verify NER results using LLM to remove false positives (only works with --keywords-method ner)')
    spreadsheet_parser_cmd.add_argument('--num-keywords', type=int, default=20, dest='num_keywords',
                                       help='Number of keywords to extract (default: 20)')
    spreadsheet_parser_cmd.add_argument('--debug', action='store_true', help='Enable debug output')
    
    # Agent command
    agent_parser = subparsers.add_parser('agent', help='Manage AI Agent daemon')
    agent_parser.add_argument('action', type=str, choices=['start', 'stop', 'restart', 'status'],
                            help='Action to perform on the daemon')
    agent_parser.add_argument('--pidfile', type=str, default='/tmp/palefire_ai_agent.pid',
                            help='Path to PID file (default: /tmp/palefire_ai_agent.pid)')
    agent_parser.add_argument('--daemon', '--background', '-d', '-b', action='store_true',
                            help='Run as background daemon (alias: --background, -d, -b)')
    agent_parser.add_argument('--no-spacy', action='store_true', dest='no_spacy',
                            help='Disable spaCy (use pattern-based NER)')
    agent_parser.add_argument('--log-level', type=str, default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            help='Log level (default: INFO)')
    
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
        # Ensure daemon is running (start if needed)
        pidfile = '/tmp/palefire_ai_agent.pid'
        use_spacy = True  # Default to using spaCy
        
        if DEBUG:
            import sys
            debug_print('Checking if daemon is running...', file=sys.stderr)
        
        daemon_available = ensure_daemon_running(pidfile=pidfile, use_spacy=use_spacy, debug=DEBUG)
        
        # Try to use daemon if available, otherwise fall back to direct extraction
        if daemon_available:
            try:
                # Use daemon for faster extraction (models already loaded)
                from agents import get_daemon
                daemon = get_daemon(use_spacy=use_spacy)
                
                # Ensure models are initialized
                if not daemon.model_manager.is_initialized():
                    if DEBUG:
                        import sys
                        debug_print('Initializing daemon models...', file=sys.stderr)
                    daemon.model_manager.initialize(use_spacy=use_spacy)
                
                # Extract keywords using daemon
                if DEBUG:
                    import sys
                    debug_print('Using daemon for keyword extraction (models already loaded)...', file=sys.stderr)
                
                # Create a new extractor with custom parameters if needed
                # The daemon's extractor might have default parameters, so we create a new one
                from modules import KeywordExtractor
                extractor = KeywordExtractor(
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
                )
                
                # Load documents if provided
                documents = None
                if args.documents_file:
                    try:
                        with open(args.documents_file, 'r', encoding='utf-8') as f:
                            documents_data = json.load(f)
                            if isinstance(documents_data, list):
                                documents = documents_data
                    except Exception as e:
                        logger.warning(f"Error loading documents file: {e}")
                
                # Extract keywords using the extractor
                keywords = extractor.extract(args.text, documents)
                
                # Format output
                output = {
                    'method': args.method,
                    'num_keywords': len(keywords),
                    'keywords': keywords,
                    'parameters': {
                        'num_keywords': args.num_keywords,
                        'min_word_length': args.min_word_length,
                        'max_word_length': args.max_word_length,
                        'use_stemming': args.use_stemming,
                        'tfidf_weight': args.tfidf_weight,
                        'textrank_weight': args.textrank_weight,
                        'word_freq_weight': args.word_freq_weight,
                        'position_weight': args.position_weight,
                        'title_weight': args.title_weight,
                        'first_sentence_weight': args.first_sentence_weight,
                        'enable_ngrams': args.enable_ngrams,
                        'min_ngram': args.min_ngram,
                        'max_ngram': args.max_ngram,
                        'ngram_weight': args.ngram_weight,
                    },
                    'daemon_used': True
                }
                
                # Output results
                if args.output_file:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        json.dump(output, f, indent=2, ensure_ascii=False)
                    if DEBUG:
                        import sys
                        debug_print(f'\n✅ Keywords saved to {args.output_file}', file=sys.stderr)
                else:
                    print(json.dumps(output, indent=2, ensure_ascii=False))
                
                return
            except Exception as e:
                if DEBUG:
                    import sys
                    debug_print(f'⚠️  Failed to use daemon: {e}, falling back to direct extraction', file=sys.stderr)
                logger.warning(f"Failed to use daemon, falling back to direct extraction: {e}")
        
        # Fall back to direct extraction (original method)
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
    
    elif args.command == 'parse':
        # Parse file using appropriate parser
        parse_file_command(
            file_path=args.file,
            output_file=args.output_file,
            extract_keywords=args.extract_keywords,
            keywords_method=args.keywords_method,
            num_keywords=args.num_keywords,
            verify_ner=getattr(args, 'verify_ner', False),
            debug=DEBUG
        )
    
    elif args.command == 'parse-txt':
        # Parse text file
        if not args.file and not args.prompt:
            parser.error("Either 'file' argument or '--prompt' option is required")
        parse_txt_command(
            file_path=args.file or '',
            encoding=args.encoding,
            output_file=args.output_file,
            debug=DEBUG,
            prompt=args.prompt,
            extract_keywords=getattr(args, 'extract_keywords', False),
            keywords_method=getattr(args, 'keywords_method', 'combined'),
            num_keywords=getattr(args, 'num_keywords', 20),
            verify_ner=getattr(args, 'verify_ner', False)
        )
    
    elif args.command == 'parse-csv':
        # Parse CSV file
        if not args.file and not args.prompt:
            parser.error("Either 'file' argument or '--prompt' option is required")
        parse_csv_command(
            file_path=args.file or '',
            delimiter=args.delimiter,
            include_headers=args.include_headers,
            output_file=args.output_file,
            debug=DEBUG,
            prompt=args.prompt,
            extract_keywords=getattr(args, 'extract_keywords', False),
            keywords_method=getattr(args, 'keywords_method', 'combined'),
            num_keywords=getattr(args, 'num_keywords', 20),
            verify_ner=getattr(args, 'verify_ner', False)
        )
    
    elif args.command == 'parse-pdf':
        # Parse PDF file
        if not args.file and not args.prompt:
            parser.error("Either 'file' argument or '--prompt' option is required")
        parse_pdf_command(
            file_path=args.file or '',
            max_pages=args.max_pages,
            extract_tables=args.extract_tables,
            output_file=args.output_file,
            debug=DEBUG,
            prompt=args.prompt,
            extract_keywords=getattr(args, 'extract_keywords', False),
            keywords_method=getattr(args, 'keywords_method', 'combined'),
            num_keywords=getattr(args, 'num_keywords', 20),
            verify_ner=getattr(args, 'verify_ner', False)
        )
    
    elif args.command == 'parse-spreadsheet':
        # Parse spreadsheet file
        if not args.file and not args.prompt:
            parser.error("Either 'file' argument or '--prompt' option is required")
        parse_spreadsheet_command(
            file_path=args.file or '',
            sheet_names=args.sheet_names.split(',') if args.sheet_names else None,
            include_headers=args.include_headers,
            output_file=args.output_file,
            debug=DEBUG,
            prompt=args.prompt,
            extract_keywords=getattr(args, 'extract_keywords', False),
            keywords_method=getattr(args, 'keywords_method', 'combined'),
            num_keywords=getattr(args, 'num_keywords', 20),
            verify_ner=getattr(args, 'verify_ner', False)
        )
    
    # Note: Agent command is handled synchronously in __main__ block
    # to avoid async event loop issues with process forking
    
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
    
    # Handle agent command synchronously (before async context)
    # Agent daemon operations don't need async and can cause issues with forking
    if args.command == 'agent':
        import json
        import os
        import signal as sig
        
        daemon = AIAgentDaemon(pidfile=args.pidfile, use_spacy=not args.no_spacy)
        
        # Setup logging for agent command
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if args.action == 'start':
            if args.daemon:
                # Background mode: show PID and status
                print("Starting AI Agent daemon in background mode...")
                print(f"PID file: {args.pidfile}")
                print(f"Use spaCy: {not args.no_spacy}")
                print(f"Log level: {args.log_level}")
                
                # Start in background
                daemon.start(daemon=True)
                
                # Wait a moment to ensure PID file is written
                time.sleep(0.5)
                
                # Read and display PID
                try:
                    if os.path.exists(args.pidfile):
                        with open(args.pidfile, 'r') as f:
                            pid = f.read().strip()
                        print(f"✅ AI Agent daemon started successfully (PID: {pid})")
                        print(f"   Check status: python palefire-cli.py agent status")
                        print(f"   Stop daemon: python palefire-cli.py agent stop")
                        print(f"   View logs: tail -f /tmp/palefire_ai_agent.log")
                    else:
                        print("⚠️  Daemon started but PID file not found yet")
                        print("   Check if daemon is running: python palefire-cli.py agent status")
                except Exception as e:
                    logger.warning(f"Could not read PID file: {e}")
                    print("⚠️  Daemon started but could not read PID")
                    print("   Check if daemon is running: python palefire-cli.py agent status")
            else:
                # Foreground mode: run normally with better output
                print("Starting AI Agent daemon in foreground mode...")
                print("Press Ctrl+C to stop")
                print("-" * 60)
                daemon.start(daemon=False)
        elif args.action == 'stop':
            # Read PID and send SIGTERM
            try:
                with open(args.pidfile, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, sig.SIGTERM)
                logger.info(f"Sent SIGTERM to process {pid}")
                print(f"Stopped daemon (PID: {pid})")
            except FileNotFoundError:
                logger.error("PID file not found. Daemon may not be running.")
                print("Error: Daemon not running (PID file not found)")
            except ProcessLookupError:
                logger.error("Process not found. Removing stale PID file.")
                try:
                    os.remove(args.pidfile)
                except:
                    pass
                print("Error: Daemon process not found (stale PID file removed)")
        elif args.action == 'restart':
            # Stop then start
            try:
                with open(args.pidfile, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, sig.SIGTERM)
                logger.info(f"Sent SIGTERM to process {pid}")
                time.sleep(2)  # Wait for shutdown
            except:
                pass
            if args.daemon:
                print("Restarting AI Agent daemon in background mode...")
                daemon.start(daemon=True)
                time.sleep(0.5)
                try:
                    if os.path.exists(args.pidfile):
                        with open(args.pidfile, 'r') as f:
                            pid = f.read().strip()
                        print(f"✅ AI Agent daemon restarted successfully (PID: {pid})")
                except:
                    pass
            else:
                print("Restarting AI Agent daemon in foreground mode...")
                daemon.start(daemon=False)
        elif args.action == 'status':
            # Check if daemon process is actually running
            daemon_running = False
            pid = None
            process_info = None
            
            try:
                if os.path.exists(args.pidfile):
                    with open(args.pidfile, 'r') as f:
                        pid = int(f.read().strip())
                    
                    # Check if process exists
                    try:
                        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
                        daemon_running = True
                        # Try to get process info
                        import psutil
                        try:
                            proc = psutil.Process(pid)
                            process_info = {
                                'pid': pid,
                                'status': proc.status(),
                                'memory_mb': round(proc.memory_info().rss / 1024 / 1024, 2),
                                'cpu_percent': proc.cpu_percent(interval=0.1),
                                'create_time': datetime.fromtimestamp(proc.create_time()).isoformat()
                            }
                        except:
                            process_info = {'pid': pid, 'status': 'running'}
                    except ProcessLookupError:
                        daemon_running = False
                        # Stale PID file
                        try:
                            os.remove(args.pidfile)
                        except:
                            pass
            except (FileNotFoundError, ValueError) as e:
                daemon_running = False
            
            # Get daemon status (models info)
            status = daemon.get_status()
            
            # Update with actual running status
            status['daemon_running'] = daemon_running
            status['pid'] = pid
            if process_info:
                status['process'] = process_info
            
            print(json.dumps(status, indent=2))
        
        # Exit after handling agent command (don't enter async context)
        sys.exit(0)
    
    # Handle other commands in async context
    if args.command:
        asyncio.run(main_cli(args))
    else:
        # If no command specified, run legacy main for backward compatibility
        asyncio.run(main())

