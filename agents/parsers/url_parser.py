"""
URL/HTML Parser

Parses HTML pages from URLs using BeautifulSoup to extract text content.
"""

from typing import Dict, Any, Optional
import logging
from urllib.parse import urlparse

from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Install with: pip install requests")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")


class URLParser(BaseParser):
    """Parser for HTML pages from URLs."""
    
    def __init__(self, timeout: int = 30, headers: Optional[Dict[str, str]] = None):
        """
        Initialize URL parser.
        
        Args:
            timeout: Request timeout in seconds (default: 30)
            headers: Custom HTTP headers (default: None)
        """
        super().__init__()
        self.timeout = timeout
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def parse(self, url: str, **kwargs) -> ParseResult:
        """
        Parse an HTML page from a URL.
        
        Args:
            url: URL to fetch and parse
            **kwargs: Additional options:
                - timeout: Override default timeout
                - headers: Override default headers
                - remove_scripts: Remove script and style tags (default: True)
                - remove_comments: Remove HTML comments (default: True)
        
        Returns:
            ParseResult with extracted text
        """
        if not REQUESTS_AVAILABLE:
            return ParseResult(
                text='',
                error="requests library not installed. Install with: pip install requests"
            )
        
        if not BS4_AVAILABLE:
            return ParseResult(
                text='',
                error="beautifulsoup4 library not installed. Install with: pip install beautifulsoup4"
            )
        
        # Validate URL
        if not self._is_valid_url(url):
            return ParseResult(
                text='',
                error=f"Invalid URL: {url}"
            )
        
        timeout = kwargs.get('timeout', self.timeout)
        headers = kwargs.get('headers', self.headers)
        remove_scripts = kwargs.get('remove_scripts', True)
        remove_comments = kwargs.get('remove_comments', True)
        
        try:
            # Fetch the URL
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements if requested
            if remove_scripts:
                for script in soup(['script', 'style', 'noscript']):
                    script.decompose()
            
            # Remove comments if requested
            if remove_comments:
                from bs4 import Comment
                comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    comment.extract()
            
            # Extract text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Extract metadata
            metadata = {
                'url': url,
                'title': soup.title.string if soup.title else '',
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(response.content),
                'text_length': len(text),
            }
            
            # Try to extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                metadata['description'] = meta_desc['content']
            
            # Try to extract meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords and meta_keywords.get('content'):
                metadata['keywords'] = meta_keywords['content']
            
            # Extract links (for reference)
            links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text(strip=True)
                if href:
                    # Make absolute URL if relative
                    absolute_url = self._make_absolute_url(url, href)
                    links.append({
                        'text': link_text,
                        'url': absolute_url
                    })
            metadata['links'] = links[:100]  # Limit to first 100 links
            
            # Split into pages (by sections/headings)
            pages = self._split_into_pages(soup, text)
            
            return ParseResult(
                text=text,
                metadata=metadata,
                pages=pages
            )
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}", exc_info=True)
            return ParseResult(
                text='',
                error=f"Error fetching URL: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}", exc_info=True)
            return ParseResult(
                text='',
                error=f"Error parsing URL: {str(e)}"
            )
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _make_absolute_url(self, base_url: str, relative_url: str) -> str:
        """Convert relative URL to absolute URL."""
        from urllib.parse import urljoin
        return urljoin(base_url, relative_url)
    
    def _split_into_pages(self, soup: BeautifulSoup, text: str, max_chunk_size: int = 5000) -> list:
        """
        Split HTML content into pages/sections based on headings.
        
        Args:
            soup: BeautifulSoup object
            text: Extracted text
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        # Try to split by headings (h1, h2, h3)
        headings = soup.find_all(['h1', 'h2', 'h3'])
        
        if not headings or len(text) <= max_chunk_size:
            # If no headings or text is small, return as single page
            return [text]
        
        pages = []
        current_section = []
        current_size = 0
        
        # Process all elements in order
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'div', 'li']):
            element_text = element.get_text(strip=True)
            if not element_text:
                continue
            
            element_size = len(element_text)
            
            # If adding this element would exceed max size, start new page
            if current_size + element_size > max_chunk_size and current_section:
                pages.append('\n\n'.join(current_section))
                current_section = []
                current_size = 0
            
            current_section.append(element_text)
            current_size += element_size
        
        # Add remaining content
        if current_section:
            pages.append('\n\n'.join(current_section))
        
        return pages if pages else [text]
    
    def get_supported_extensions(self) -> list:
        """Get supported file extensions (URLs don't have extensions)."""
        return []  # URLs don't have file extensions
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if the input is a valid URL."""
        return self._is_valid_url(file_path)

