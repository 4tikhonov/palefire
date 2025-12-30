"""
Response Parser for LLM responses.

Parses various response formats (JSON, markdown, text) into structured JSON.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Parser for LLM responses into structured JSON format.
    
    Supports:
    - Direct JSON
    - JSON in markdown code blocks
    - Structured text/markdown
    - Plain text (converted to structured format)
    """
    
    @staticmethod
    def parse(response: str, response_type: str = "general") -> Dict[str, Any]:
        """
        Parse LLM response into structured JSON.
        
        Args:
            response: Raw response text from LLM
            response_type: Type of response (e.g., "json", "markdown", "text", "general")
            
        Returns:
            Dictionary with parsed data and metadata
        """
        if not response or not response.strip():
            return {
                'parsed': False,
                'format': 'empty',
                'data': None,
                'error': 'Empty response'
            }
        
        result = {
            'parsed': False,
            'format': 'unknown',
            'data': None,
            'original_response': response,
            'response_length': len(response)
        }
        
        # Try direct JSON first
        try:
            data = json.loads(response.strip())
            result.update({
                'parsed': True,
                'format': 'json',
                'data': data
            })
            return result
        except json.JSONDecodeError:
            pass
        
        # Try JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            json_content = json_match.group(1).strip()
            try:
                data = json.loads(json_content)
                result.update({
                    'parsed': True,
                    'format': 'markdown_json',
                    'data': data
                })
                return result
            except json.JSONDecodeError:
                pass
        
        # Try to extract JSON object from text
        json_obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_obj_match:
            try:
                data = json.loads(json_obj_match.group(0))
                result.update({
                    'parsed': True,
                    'format': 'embedded_json',
                    'data': data
                })
                return result
            except json.JSONDecodeError:
                pass
        
        # Try to parse as structured text/markdown
        structured = ResponseParser._parse_structured_text(response)
        if structured:
            result.update({
                'parsed': True,
                'format': 'structured_text',
                'data': structured
            })
            return result
        
        # Fallback: convert plain text to structured format
        result.update({
            'parsed': True,
            'format': 'plain_text',
            'data': {
                'text': response.strip(),
                'type': 'text_response'
            }
        })
        
        return result
    
    @staticmethod
    def _parse_structured_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Parse structured text/markdown into JSON.
        
        Looks for:
        - Key-value pairs
        - Lists
        - Sections with headers
        """
        lines = text.split('\n')
        result = {}
        current_section = None
        current_list = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers (markdown style)
            if line.startswith('#'):
                current_section = line.lstrip('#').strip()
                result[current_section] = {}
                current_list = None
                continue
            elif line.startswith('**') and line.endswith('**'):
                # Bold header
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
                
                # Remove markdown formatting from key
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
                # Remove markdown formatting
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
        
        return result if result else None
    
    @staticmethod
    def parse_to_json_file(
        response: str,
        output_path: str,
        response_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Parse response and save to JSON file.
        
        Args:
            response: Raw response text
            output_path: Path to save JSON file
            response_type: Type of response
            metadata: Optional metadata to include
            
        Returns:
            True if successful, False otherwise
        """
        try:
            parsed = ResponseParser.parse(response, response_type)
            
            # Add metadata if provided
            if metadata:
                parsed['metadata'] = metadata
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved parsed JSON to: {output_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to parse and save JSON: {e}")
            return False

