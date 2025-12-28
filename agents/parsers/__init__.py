"""
Pale Fire - File Parsers

Parsers for extracting text from various file formats.
"""

from .base_parser import BaseParser, ParseResult
from .txt_parser import TXTParser
from .csv_parser import CSVParser
from .pdf_parser import PDFParser
from .spreadsheet_parser import SpreadsheetParser

__all__ = [
    'BaseParser',
    'ParseResult',
    'TXTParser',
    'CSVParser',
    'PDFParser',
    'SpreadsheetParser',
]

# Parser registry
PARSERS = {
    '.txt': TXTParser,
    '.csv': CSVParser,
    '.pdf': PDFParser,
    '.xlsx': SpreadsheetParser,
    '.xls': SpreadsheetParser,
    '.ods': SpreadsheetParser,
}


def get_parser(file_path: str) -> BaseParser:
    """
    Get appropriate parser for file based on extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        Parser instance
        
    Raises:
        ValueError: If no parser available for file type
    """
    from pathlib import Path
    ext = Path(file_path).suffix.lower()
    
    if ext not in PARSERS:
        raise ValueError(f"No parser available for file type: {ext}")
    
    return PARSERS[ext]()

