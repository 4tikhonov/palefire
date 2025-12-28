"""
Plain Text File Parser

Parses .txt files and extracts text content.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class TXTParser(BaseParser):
    """Parser for plain text files."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize TXT parser.
        
        Args:
            encoding: Text encoding (default: utf-8)
        """
        super().__init__()
        self.encoding = encoding
    
    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        Parse a text file.
        
        Args:
            file_path: Path to .txt file
            **kwargs: Additional options:
                - encoding: Override default encoding
                - max_length: Maximum characters to read (optional)
        
        Returns:
            ParseResult with extracted text
        """
        encoding = kwargs.get('encoding', self.encoding)
        max_length = kwargs.get('max_length', None)
        
        if not self.validate_file(file_path):
            return ParseResult(
                text='',
                error=f"File not found or invalid: {file_path}"
            )
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                if max_length:
                    text = f.read(max_length)
                else:
                    text = f.read()
            
            # Extract metadata
            path = Path(file_path)
            metadata = {
                'filename': path.name,
                'file_size': self.get_file_size(file_path),
                'encoding': encoding,
                'line_count': len(text.splitlines()),
            }
            
            # Split into pages (by double newlines or large chunks)
            pages = self._split_into_pages(text)
            
            return ParseResult(
                text=text,
                metadata=metadata,
                pages=pages
            )
        
        except UnicodeDecodeError as e:
            return ParseResult(
                text='',
                error=f"Encoding error: {e}. Try specifying encoding parameter."
            )
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}", exc_info=True)
            return ParseResult(
                text='',
                error=f"Error parsing file: {str(e)}"
            )
    
    def _split_into_pages(self, text: str, max_chunk_size: int = 5000) -> list:
        """
        Split text into pages/chunks.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        # Try to split on double newlines first
        chunks = text.split('\n\n')
        pages = []
        current_page = ''
        
        for chunk in chunks:
            if len(current_page) + len(chunk) <= max_chunk_size:
                current_page += chunk + '\n\n'
            else:
                if current_page:
                    pages.append(current_page.strip())
                current_page = chunk + '\n\n'
        
        if current_page:
            pages.append(current_page.strip())
        
        return pages if pages else [text]
    
    def get_supported_extensions(self) -> list:
        """Get supported file extensions."""
        return ['.txt', '.text']

