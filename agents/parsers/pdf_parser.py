"""
PDF File Parser

Parses PDF files and extracts text content using PyPDF2 or pdfplumber.
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class PDFParser(BaseParser):
    """Parser for PDF files."""
    
    def __init__(self, prefer_pdfplumber: bool = True):
        """
        Initialize PDF parser.
        
        Args:
            prefer_pdfplumber: Prefer pdfplumber over PyPDF2 (better table extraction)
        """
        super().__init__()
        self.prefer_pdfplumber = prefer_pdfplumber
        
        if not PYPDF2_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            logger.warning("No PDF parsing library available. Install with: pip install PyPDF2 or pip install pdfplumber")
    
    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        Parse a PDF file.
        
        Args:
            file_path: Path to .pdf file
            **kwargs: Additional options:
                - prefer_pdfplumber: Override default preference
                - max_pages: Maximum number of pages to parse (optional)
                - extract_tables: Extract tables from PDF (default: True, requires pdfplumber)
        
        Returns:
            ParseResult with extracted text and metadata
        """
        prefer_pdfplumber = kwargs.get('prefer_pdfplumber', self.prefer_pdfplumber)
        max_pages = kwargs.get('max_pages', None)
        extract_tables = kwargs.get('extract_tables', True)
        
        if not self.validate_file(file_path):
            return ParseResult(
                text='',
                error=f"File not found or invalid: {file_path}"
            )
        
        # Try pdfplumber first if available and preferred
        if PDFPLUMBER_AVAILABLE and (prefer_pdfplumber or not PYPDF2_AVAILABLE):
            return self._parse_with_pdfplumber(file_path, max_pages, extract_tables)
        
        # Fall back to PyPDF2
        if PYPDF2_AVAILABLE:
            return self._parse_with_pypdf2(file_path, max_pages)
        
        return ParseResult(
            text='',
            error="No PDF parsing library available. Install with: pip install PyPDF2 or pip install pdfplumber"
        )
    
    def _parse_with_pdfplumber(self, file_path: str, max_pages: Optional[int], extract_tables: bool) -> ParseResult:
        """Parse PDF using pdfplumber."""
        try:
            import pdfplumber
            
            pages_text = []
            tables = []
            metadata = {
                'filename': Path(file_path).name,
                'file_size': self.get_file_size(file_path),
            }
            
            with pdfplumber.open(file_path) as pdf:
                metadata['page_count'] = len(pdf.pages)
                metadata['pdf_metadata'] = pdf.metadata or {}
                
                for i, page in enumerate(pdf.pages):
                    if max_pages and i >= max_pages:
                        break
                    
                    # Extract text
                    page_text = page.extract_text() or ''
                    pages_text.append(page_text)
                    
                    # Extract tables if requested
                    if extract_tables:
                        page_tables = page.extract_tables()
                        for table in page_tables:
                            if table:
                                tables.append({
                                    'page': i + 1,
                                    'rows': table,
                                    'row_count': len(table),
                                    'column_count': len(table[0]) if table else 0
                                })
            
            full_text = '\n\n'.join(pages_text)
            
            return ParseResult(
                text=full_text,
                metadata=metadata,
                pages=pages_text,
                tables=tables
            )
        
        except Exception as e:
            logger.error(f"Error parsing PDF with pdfplumber {file_path}: {e}", exc_info=True)
            return ParseResult(
                text='',
                error=f"Error parsing PDF: {str(e)}"
            )
    
    def _parse_with_pypdf2(self, file_path: str, max_pages: Optional[int]) -> ParseResult:
        """Parse PDF using PyPDF2."""
        try:
            import PyPDF2
            
            pages_text = []
            metadata = {
                'filename': Path(file_path).name,
                'file_size': self.get_file_size(file_path),
            }
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                metadata['page_count'] = len(pdf_reader.pages)
                metadata['pdf_metadata'] = pdf_reader.metadata or {}
                
                for i, page in enumerate(pdf_reader.pages):
                    if max_pages and i >= max_pages:
                        break
                    
                    page_text = page.extract_text() or ''
                    pages_text.append(page_text)
            
            full_text = '\n\n'.join(pages_text)
            
            return ParseResult(
                text=full_text,
                metadata=metadata,
                pages=pages_text
            )
        
        except Exception as e:
            logger.error(f"Error parsing PDF with PyPDF2 {file_path}: {e}", exc_info=True)
            return ParseResult(
                text='',
                error=f"Error parsing PDF: {str(e)}"
            )
    
    def get_supported_extensions(self) -> list:
        """Get supported file extensions."""
        return ['.pdf']

