"""
CSV File Parser

Parses .csv files and extracts text content, optionally preserving table structure.
"""

from typing import Dict, Any, Optional, List
import logging
import csv
from pathlib import Path

from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class CSVParser(BaseParser):
    """Parser for CSV files."""
    
    def __init__(self, delimiter: str = ',', quotechar: str = '"'):
        """
        Initialize CSV parser.
        
        Args:
            delimiter: CSV delimiter character
            quotechar: Quote character
        """
        super().__init__()
        self.delimiter = delimiter
        self.quotechar = quotechar
    
    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        Parse a CSV file.
        
        Args:
            file_path: Path to .csv file
            **kwargs: Additional options:
                - delimiter: Override default delimiter
                - quotechar: Override default quote character
                - include_headers: Include header row in text (default: True)
                - include_tables: Include table structure in tables field (default: True)
                - max_rows: Maximum number of rows to read (optional)
        
        Returns:
            ParseResult with extracted text and table data
        """
        delimiter = kwargs.get('delimiter', self.delimiter)
        quotechar = kwargs.get('quotechar', self.quotechar)
        include_headers = kwargs.get('include_headers', True)
        include_tables = kwargs.get('include_tables', True)
        max_rows = kwargs.get('max_rows', None)
        
        if not self.validate_file(file_path):
            return ParseResult(
                text='',
                error=f"File not found or invalid: {file_path}"
            )
        
        try:
            tables = []
            rows = []
            headers = None
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Try to detect delimiter
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                try:
                    detected_delimiter = sniffer.sniff(sample).delimiter
                    delimiter = detected_delimiter
                except:
                    pass  # Use provided delimiter
                
                reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
                
                for i, row in enumerate(reader):
                    if max_rows and i >= max_rows:
                        break
                    
                    if i == 0:
                        headers = row
                        if include_headers:
                            rows.append(row)
                    else:
                        rows.append(row)
            
            # Build text representation
            text_parts = []
            if headers and include_headers:
                text_parts.append(' | '.join(str(cell) for cell in headers))
                text_parts.append('-' * 80)
            
            for row in rows[1:] if headers else rows:
                text_parts.append(' | '.join(str(cell) for cell in row))
            
            text = '\n'.join(text_parts)
            
            # Build table structure
            if include_tables and headers:
                tables.append({
                    'headers': headers,
                    'rows': rows[1:] if headers else rows,
                    'row_count': len(rows) - (1 if headers else 0),
                    'column_count': len(headers) if headers else (len(rows[0]) if rows else 0)
                })
            
            # Extract metadata
            path = Path(file_path)
            metadata = {
                'filename': path.name,
                'file_size': self.get_file_size(file_path),
                'delimiter': delimiter,
                'row_count': len(rows) - (1 if headers else 0),
                'column_count': len(headers) if headers else (len(rows[0]) if rows else 0),
                'has_headers': headers is not None,
            }
            
            return ParseResult(
                text=text,
                metadata=metadata,
                tables=tables
            )
        
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {e}", exc_info=True)
            return ParseResult(
                text='',
                error=f"Error parsing CSV file: {str(e)}"
            )
    
    def get_supported_extensions(self) -> list:
        """Get supported file extensions."""
        return ['.csv']

