#!/usr/bin/env python3
"""
Ollama Response Parser

Handles parsing of various Ollama response formats and creates JSON logs.
Supports structured markdown, JSON, and other formats.
"""

import json
import re
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class OllamaResponseParser:
    """
    Parser for Ollama LLM responses with automatic JSON logging.
    """

    def __init__(self, logs_dir: str = None):
        """
        Initialize the response parser.

        Args:
            logs_dir: Directory to store JSON logs. Defaults to 'logs' relative to this module.
        """
        if logs_dir is None:
            # Default to logs directory relative to this module
            module_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(os.path.dirname(module_dir), 'logs')

        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

    def parse_and_log_response(self, response: str, model_name: str, request_type: str = "unknown",
                              request_id: Optional[str] = None, original_entities: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Parse an Ollama response and create JSON logs.

        Args:
            response: The raw response from Ollama
            model_name: Name of the model used
            request_type: Type of request (e.g., "ner_verification", "keyword_extraction")
            request_id: Optional request ID for correlation
            original_entities: Original entities for NER verification (optional)

        Returns:
            List of parsed entities/dictionaries
        """
        if not response or not response.strip():
            logger.warning(f"Empty response from {model_name}")
            return []

        logger.debug(f"Parsing {model_name} response for {request_type}")

        # Try different parsing strategies in order
        parsed_data, parse_method = self._parse_response(response, original_entities or [])

        # Log the parsing result as JSON
        self._log_parsed_response(parsed_data, model_name, request_type, request_id, parse_method)

        return parsed_data

    def _parse_response(self, response: str, original_entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse the response using various strategies.

        Returns:
            Tuple of (parsed_data, parse_method_name)
        """
        # Strategy 1: Direct JSON parsing
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, list):
                logger.debug("Parsed as direct JSON array")
                return parsed, "direct_json"
        except json.JSONDecodeError:
            pass

        # Strategy 2: JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1).strip())
                if isinstance(parsed, list):
                    logger.debug("Parsed as markdown JSON code block")
                    return parsed, "markdown_json"
            except json.JSONDecodeError:
                pass

        # Strategy 3: Llama structured markdown format
        parsed = self._parse_llama_structured_markdown(response, original_entities)
        if parsed:
            logger.debug(f"Parsed {len(parsed)} entities using Llama structured markdown")
            return parsed, "llama_structured_markdown"

        # Strategy 4: General structured markdown format
        parsed = self._parse_structured_markdown(response, original_entities)
        if parsed:
            logger.debug(f"Parsed {len(parsed)} entities using structured markdown")
            return parsed, "structured_markdown"

        # Strategy 5: Gemma3 markdown format (fallback)
        parsed = self._parse_gemma3_markdown(response, original_entities)
        if parsed:
            logger.debug(f"Parsed {len(parsed)} entities using Gemma3 markdown")
            return parsed, "gemma3_markdown"

        # No parsing succeeded
        logger.warning(f"Could not parse response from model, returning empty result")
        return [], "unparsed"

    def _parse_structured_markdown(self, response: str, original_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse structured markdown format (optimized for gemma3:4b).

        Format:
        **1. Verified Entities:**
        * Entity Name (TYPE) - Brief reason
        * **Entity Name** (TYPE) - Brief reason

        **2. New Entities Found:**
        * New Entity (TYPE) - Brief reason

        **3. Entities to Remove:**
        * Entity Name (TYPE) - Reason to remove

        Returns:
            List of parsed entities in verification format
        """
        result = []
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

            # Process bullet points
            if current_section and re.match(r'^[\*\-\•]\s+', line):
                # Skip lines that contain "**None**" (empty section indicators)
                if '**None**' in line:
                    continue

                if current_section == 'remove':
                    # Pattern for removal: * Entity Name (TYPE) - Reason OR * **Entity Name** (TYPE) - Reason
                    remove_match = re.match(r'^[\*\-\•]\s+(?:\*\*)?([^*]+?)(?:\*\*)?\s*(?:\(([A-Z_]+)\))?\s*(?:\-\s*(.*))?$', line)
                    if remove_match:
                        entity_text = remove_match.group(1).strip()
                        entity_type = remove_match.group(2).strip() if remove_match.group(2) else ''
                        reason = remove_match.group(3).strip() if remove_match.group(3) else 'Marked for removal'

                        # Look for matching original entity to preserve position info
                        matched_original = None
                        entity_lower = entity_text.lower()
                        for orig in original_entities:
                            if orig.get('text', '').lower() == entity_lower:
                                matched_original = orig
                                break

                        # Create a "remove" entity (always, since LLM says to remove it)
                        entity = {
                            'text': entity_text,
                            'type': 'REMOVE',
                            'reasoning': f"Marked for removal: {reason}",
                            'verified': True,  # REMOVE entities are verified to be removed
                            'new_entity': False  # REMOVE applies to existing entities
                        }

                        # Preserve position info if we found a match
                        if matched_original:
                            entity.update({
                                'start': matched_original.get('start', 0),
                                'end': matched_original.get('end', 0)
                            })

                        result.append(entity)
                else:
                    # Pattern: * Entity Name (TYPE) - Brief reason OR * **Entity Name** (TYPE) - Brief reason
                    entity_match = re.match(r'^[\*\-\•]\s+(?:\*\*)?([^*]+?)(?:\*\*)?\s*\(([A-Z_]+)\)\s*(?:\-\s*(.*))?$', line)
                    if entity_match:
                        entity_text = entity_match.group(1).strip()
                        entity_type = entity_match.group(2).strip()
                        description = entity_match.group(3).strip() if entity_match.group(3) else ''

                        # Try to match with original entity (case-insensitive)
                        entity_lower = entity_text.lower()
                        matched_original = None
                        for orig in original_entities:
                            if orig.get('text', '').lower() == entity_lower:
                                matched_original = orig
                                break

                        # Create entity in standard format expected by verification logic
                        is_new_entity = not matched_original  # True if discovered, False if verified existing

                        entity = {
                            'text': entity_text,
                            'type': entity_type,
                            'reasoning': description if description else 'No reasoning provided by LLM',
                            'verified': True,  # Structured markdown entities are considered verified
                            'new_entity': is_new_entity
                        }

                        # Preserve position info from original if available
                        if matched_original:
                            entity.update({
                                'start': matched_original.get('start', 0),
                                'end': matched_original.get('end', 0)
                            })

                        result.append(entity)

        return result

    def _parse_llama_structured_markdown(self, response: str, original_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse Llama structured markdown format.

        Format:
        **1. Verified Entities:**
        * **Entity Name** (TYPE) - Description
        * **Entity Name** (TYPE) - Description

        **2. New Entities Found:**
        * **Entity Name** (TYPE) - Description
        * **None**

        **3. Entities to Remove:**
        * **Entity Name** (TYPE) - Reason
        * **False Entity** - None

        Returns:
            List of parsed entities in verification format
        """
        result = []
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

            # Skip other header lines and empty section indicators
            if line.startswith('**') and ('Entities' in line or 'Found' in line or 'Remove' in line):
                continue
            if '**None**' in line:
                continue

            # Process bullet points
            if current_section and re.match(r'^[\*\-\•]\s+', line):
                if current_section == 'remove':
                    # Pattern for Llama removal: * **Entity Name** (TYPE) - Reason
                    remove_match = re.match(r'^[\*\-\•]\s+\*\*([^*]+?)\*\*\s*\(([A-Z_]+)\)\s*(?:\-\s*(.*))?$', line)
                    if remove_match:
                        entity_text = remove_match.group(1).strip()
                        entity_type = remove_match.group(2).strip()
                        reason = remove_match.group(3).strip() if remove_match.group(3) else 'Marked for removal'

                        # Look for matching original entity to preserve position info
                        matched_original = None
                        entity_lower = entity_text.lower()
                        for orig in original_entities:
                            if orig.get('text', '').strip().lower() == entity_lower:
                                matched_original = orig
                                break

                        # Create a "remove" entity
                        entity = {
                            'text': entity_text,
                            'type': 'REMOVE',
                            'reasoning': f"Marked for removal: {reason}",
                            'verified': True,
                            'new_entity': False
                        }

                        # Preserve position info if we found a match
                        if matched_original:
                            entity.update({
                                'start': matched_original.get('start', 0),
                                'end': matched_original.get('end', 0)
                            })

                        result.append(entity)
                else:
                    # Pattern for Llama entities: * **Entity Name** (TYPE) - Description
                    entity_match = re.match(r'^[\*\-\•]\s+\*\*([^*]+?)\*\*\s*\(([A-Z_]+)\)\s*(?:\-\s*(.*))?$', line)
                    if entity_match:
                        entity_text = entity_match.group(1).strip()
                        entity_type = entity_match.group(2).strip()
                        description = entity_match.group(3).strip() if entity_match.group(3) else ''

                        # Try to match with original entity
                        matched_original = None
                        entity_lower = entity_text.lower()
                        for orig in original_entities:
                            if orig.get('text', '').strip().lower() == entity_lower:
                                matched_original = orig
                                break

                        # Create entity in verification format
                        is_new_entity = not matched_original

                        entity = {
                            'text': entity_text,
                            'type': entity_type,
                            'reasoning': description if description else 'No reasoning provided by LLM',
                            'verified': True,
                            'new_entity': is_new_entity
                        }

                        # Preserve position info if we found a match
                        if matched_original:
                            entity.update({
                                'start': matched_original.get('start', 0),
                                'end': matched_original.get('end', 0)
                            })

                        result.append(entity)

        return result

    def _parse_gemma3_markdown(self, response: str, original_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse Gemma3 markdown format as fallback.

        Returns:
            List of parsed entities in verification format
        """
        # This is a simplified version - the full implementation would be more complex
        # For now, return empty list to indicate this method isn't fully implemented here
        return []

    def _log_parsed_response(self, parsed_data: List[Dict[str, Any]], model_name: str,
                           request_type: str, request_id: Optional[str], parse_method: str):
        """
        Log the parsed response data as JSON.
        """
        try:
            import time
            import uuid

            # Generate timestamp and request ID
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            if request_id is None:
                request_id = str(uuid.uuid4())[:8]

            # Create safe filename
            safe_model = model_name.replace(':', '_').replace('/', '_').replace('\\', '_')
            filename = f"parsed_response_{request_type}_{safe_model}_{timestamp}_{request_id}_{parse_method}.json"
            filepath = os.path.join(self.logs_dir, filename)

            # Create log data structure
            log_data = {
                "metadata": {
                    "model": model_name,
                    "request_type": request_type,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "request_id": request_id,
                    "parse_method": parse_method,
                    "data_count": len(parsed_data) if isinstance(parsed_data, list) else 0
                },
                "data": parsed_data
            }

            # Write JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Logged parsed {request_type} response to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to log parsed response: {e}")

    def test_parsing(self, test_response: str, expected_entities: int = None) -> bool:
        """
        Test parsing functionality with a sample response.

        Args:
            test_response: Sample response to parse
            expected_entities: Expected number of entities (optional)

        Returns:
            True if parsing succeeded and matches expectations
        """
        try:
            parsed = self.parse_and_log_response(
                test_response,
                "test_model",
                "test_parsing",
                "test123"
            )

            success = len(parsed) > 0 if expected_entities is None else len(parsed) == expected_entities

            if success:
                logger.info(f"Test parsing successful: {len(parsed)} entities parsed")
            else:
                logger.warning(f"Test parsing failed: expected {expected_entities}, got {len(parsed)}")

            return success

        except Exception as e:
            logger.error(f"Test parsing failed with exception: {e}")
            return False
