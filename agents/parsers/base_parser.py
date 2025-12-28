"""
Base Parser Class

Abstract base class for all file parsers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ParseResult:
    """Result of parsing a file."""
    
    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        pages: Optional[List[str]] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None
    ):
        """
        Initialize parse result.
        
        Args:
            text: Extracted text content
            metadata: File metadata (title, author, etc.)
            pages: List of page/section texts (for multi-page documents)
            tables: List of extracted tables
            error: Error message if parsing failed
        """
        self.text = text
        self.metadata = metadata or {}
        self.pages = pages or []
        self.tables = tables or []
        self.error = error
        self.success = error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'metadata': self.metadata,
            'pages': self.pages,
            'tables': self.tables,
            'success': self.success,
            'error': self.error,
        }


class BaseParser(ABC):
    """Abstract base class for file parsers."""
    
    def __init__(self):
        """Initialize parser."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        Parse a file and extract text.
        
        Args:
            file_path: Path to file to parse
            **kwargs: Additional parser-specific options
            
        Returns:
            ParseResult object with extracted content
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of extensions (e.g., ['.txt', '.text'])
        """
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate that file exists and is readable.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is valid
        """
        path = Path(file_path)
        return path.exists() and path.is_file() and path.stat().st_size > 0
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        return Path(file_path).stat().st_size

