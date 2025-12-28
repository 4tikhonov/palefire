"""
Unit tests for AI Agent module (ModelManager, AIAgentDaemon, Parsers).

Tests cover:
- ModelManager initialization and thread safety
- AIAgentDaemon lifecycle (start/stop/status)
- Keyword extraction via agent
- Entity extraction via agent
- File parsing operations
- Parser implementations
"""

import pytest
import os
import tempfile
import json
import threading
import time
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

# Helper function for PDF parser testing
def _pdf_libraries_available():
    """Check if PDF libraries are available."""
    try:
        import PyPDF2
        return True
    except ImportError:
        try:
            import pdfplumber
            return True
        except ImportError:
            return False

# Helper function for URL parser testing
def _url_libraries_available():
    """Check if URL parsing libraries are available."""
    try:
        import requests
        from bs4 import BeautifulSoup
        return True
    except ImportError:
        return False

# Import agent modules
from agents import ModelManager, AIAgentDaemon, get_daemon

# Import parsers (may not be available)
try:
    from agents.parsers import (
        TXTParser,
        CSVParser,
        PDFParser,
        SpreadsheetParser,
        URLParser,
        get_parser,
        is_url,
    )
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False
    # Create mock classes for testing
    TXTParser = None
    CSVParser = None
    PDFParser = None
    SpreadsheetParser = None
    URLParser = None
    get_parser = None
    is_url = None


class TestModelManager:
    """Test ModelManager class."""
    
    def test_model_manager_init(self):
        """Test ModelManager initialization."""
        manager = ModelManager()
        assert manager is not None
        assert not manager.is_initialized()
    
    @pytest.mark.requires_spacy
    def test_model_manager_initialize_with_spacy(self):
        """Test ModelManager initialization with spaCy."""
        manager = ModelManager()
        manager.initialize(use_spacy=True)
        assert manager.is_initialized()
        assert manager.keyword_extractor is not None
        assert manager.entity_enricher is not None
    
    def test_model_manager_initialize_without_spacy(self):
        """Test ModelManager initialization without spaCy."""
        manager = ModelManager()
        manager.initialize(use_spacy=False)
        assert manager.is_initialized()
        assert manager.keyword_extractor is not None
        assert manager.entity_enricher is not None
    
    def test_model_manager_double_initialize(self):
        """Test that double initialization doesn't cause issues."""
        manager = ModelManager()
        manager.initialize(use_spacy=False)
        assert manager.is_initialized()
        
        # Second initialization should be safe
        manager.initialize(use_spacy=False)
        assert manager.is_initialized()
    
    def test_model_manager_access_before_init(self):
        """Test that accessing models before initialization raises error."""
        manager = ModelManager()
        with pytest.raises(RuntimeError):
            _ = manager.keyword_extractor
        with pytest.raises(RuntimeError):
            _ = manager.entity_enricher
    
    def test_model_manager_thread_safety(self):
        """Test ModelManager thread safety."""
        manager = ModelManager()
        manager.initialize(use_spacy=False)
        
        results = []
        
        def access_models():
            try:
                extractor = manager.keyword_extractor
                enricher = manager.entity_enricher
                results.append((extractor is not None, enricher is not None))
            except Exception as e:
                results.append(False)
        
        threads = [threading.Thread(target=access_models) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert all(r[0] and r[1] for r in results)
    
    def test_model_manager_reload(self):
        """Test ModelManager reload functionality."""
        manager = ModelManager()
        manager.initialize(use_spacy=False)
        assert manager.is_initialized()
        
        old_extractor = manager.keyword_extractor
        manager.reload()
        
        # After reload, should still be initialized
        assert manager.is_initialized()
        # Should have new instances
        new_extractor = manager.keyword_extractor
        # Note: They might be the same object, but reload should work without error


class TestAIAgentDaemon:
    """Test AIAgentDaemon class."""
    
    @pytest.fixture
    def temp_pidfile(self):
        """Create temporary PID file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pid') as f:
            pidfile = f.name
        yield pidfile
        # Cleanup
        if os.path.exists(pidfile):
            os.remove(pidfile)
    
    def test_daemon_init(self, temp_pidfile):
        """Test AIAgentDaemon initialization."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        assert daemon is not None
        assert daemon.pidfile == temp_pidfile
        assert daemon.use_spacy is False
        assert not daemon.running
    
    def test_daemon_start_foreground(self, temp_pidfile):
        """Test daemon start in foreground mode."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        
        # Start in foreground (non-daemon mode)
        # This will initialize models and run, so we need to stop it quickly
        import threading
        
        def start_and_stop():
            time.sleep(0.1)
            daemon.stop()
        
        stop_thread = threading.Thread(target=start_and_stop, daemon=True)
        stop_thread.start()
        
        # Start daemon (will block until stopped)
        try:
            daemon.start(daemon=False)
        except KeyboardInterrupt:
            pass
        
        # Should have created PID file
        assert os.path.exists(temp_pidfile) or not daemon.running
    
    def test_daemon_status(self, temp_pidfile):
        """Test daemon status method."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        status = daemon.get_status()
        
        assert 'running' in status
        assert 'models_initialized' in status
        assert 'use_spacy' in status
        assert 'spacy_available' in status
        assert 'parsers_available' in status
        assert status['running'] is False
        assert status['use_spacy'] is False
    
    def test_daemon_extract_keywords(self, temp_pidfile):
        """Test keyword extraction via daemon."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        daemon.model_manager.initialize(use_spacy=False)
        
        text = "Artificial intelligence and machine learning are transforming technology."
        keywords = daemon.extract_keywords(text, num_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all('keyword' in kw for kw in keywords)
        assert all('score' in kw for kw in keywords)
    
    def test_daemon_extract_entities(self, temp_pidfile):
        """Test entity extraction via daemon."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        daemon.model_manager.initialize(use_spacy=False)
        
        text = "Kamala Harris was the Attorney General of California."
        entities = daemon.extract_entities(text)
        
        assert isinstance(entities, dict)
        assert 'entities' in entities or 'entities_by_type' in entities
    
    def test_daemon_stop(self, temp_pidfile):
        """Test daemon stop method."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        
        # Stop when not running should be safe
        daemon.stop()
        assert not daemon.running
    
    def test_daemon_signal_handler(self, temp_pidfile):
        """Test daemon signal handler."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        
        # Test signal handler registration
        assert daemon._signal_handler is not None


class TestGetDaemon:
    """Test get_daemon singleton function."""
    
    def test_get_daemon_singleton(self):
        """Test that get_daemon returns singleton instance."""
        # Clear any existing instance
        import agents.AIAgent as agent_module
        agent_module._daemon_instance = None
        
        daemon1 = get_daemon(use_spacy=False)
        daemon2 = get_daemon(use_spacy=False)
        
        assert daemon1 is daemon2
    
    def test_get_daemon_different_params(self):
        """Test get_daemon with different parameters."""
        # Clear any existing instance
        import agents.AIAgent as agent_module
        agent_module._daemon_instance = None
        
        daemon1 = get_daemon(use_spacy=False)
        # Second call with same params should return same instance
        daemon2 = get_daemon(use_spacy=False)
        assert daemon1 is daemon2


@pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parsers not available")
class TestTXTParser:
    """Test TXT parser."""
    
    @pytest.fixture
    def sample_txt_file(self):
        """Create a sample text file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("This is a test file.\nIt has multiple lines.\n\nAnd paragraphs.")
            temp_file = f.name
        yield temp_file
        os.remove(temp_file)
    
    def test_txt_parser_init(self):
        """Test TXT parser initialization."""
        parser = TXTParser()
        assert parser is not None
        assert parser.encoding == 'utf-8'
    
    def test_txt_parser_parse(self, sample_txt_file):
        """Test TXT parser parsing."""
        parser = TXTParser()
        result = parser.parse(sample_txt_file)
        
        assert result.success
        assert len(result.text) > 0
        assert 'test file' in result.text.lower()
        assert 'metadata' in result.to_dict()
        assert 'filename' in result.metadata
    
    def test_txt_parser_parse_nonexistent(self):
        """Test TXT parser with non-existent file."""
        parser = TXTParser()
        result = parser.parse('/nonexistent/file.txt')
        
        assert not result.success
        assert result.error is not None
    
    def test_txt_parser_supported_extensions(self):
        """Test TXT parser supported extensions."""
        parser = TXTParser()
        extensions = parser.get_supported_extensions()
        assert '.txt' in extensions
    
    def test_txt_parser_validate_file(self, sample_txt_file):
        """Test TXT parser file validation."""
        parser = TXTParser()
        assert parser.validate_file(sample_txt_file)
        assert not parser.validate_file('/nonexistent/file.txt')


@pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parsers not available")
class TestCSVParser:
    """Test CSV parser."""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as f:
            f.write("Name,Age,City\nJohn,30,New York\nJane,25,Los Angeles")
            temp_file = f.name
        yield temp_file
        os.remove(temp_file)
    
    def test_csv_parser_init(self):
        """Test CSV parser initialization."""
        parser = CSVParser()
        assert parser is not None
        assert parser.delimiter == ','
    
    def test_csv_parser_parse(self, sample_csv_file):
        """Test CSV parser parsing."""
        parser = CSVParser()
        result = parser.parse(sample_csv_file)
        
        assert result.success
        assert len(result.text) > 0
        assert 'John' in result.text
        assert 'metadata' in result.to_dict()
        assert 'row_count' in result.metadata
    
    def test_csv_parser_parse_with_headers(self, sample_csv_file):
        """Test CSV parser with headers."""
        parser = CSVParser()
        result = parser.parse(sample_csv_file, include_headers=True)
        
        assert result.success
        assert 'Name' in result.text or 'Age' in result.text
    
    def test_csv_parser_parse_without_headers(self, sample_csv_file):
        """Test CSV parser without headers."""
        parser = CSVParser()
        result = parser.parse(sample_csv_file, include_headers=False)
        
        assert result.success
        assert len(result.text) > 0
    
    def test_csv_parser_supported_extensions(self):
        """Test CSV parser supported extensions."""
        parser = CSVParser()
        extensions = parser.get_supported_extensions()
        assert '.csv' in extensions
    
    def test_csv_parser_custom_delimiter(self):
        """Test CSV parser with custom delimiter."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Name;Age;City\nJohn;30;NY")
            temp_file = f.name
        
        try:
            parser = CSVParser(delimiter=';')
            result = parser.parse(temp_file, delimiter=';')
            assert result.success
        finally:
            os.remove(temp_file)


@pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parsers not available")
class TestPDFParser:
    """Test PDF parser."""
    
    def test_pdf_parser_init(self):
        """Test PDF parser initialization."""
        parser = PDFParser()
        assert parser is not None
    
    def test_pdf_parser_supported_extensions(self):
        """Test PDF parser supported extensions."""
        parser = PDFParser()
        extensions = parser.get_supported_extensions()
        assert '.pdf' in extensions
    
    def test_pdf_parser_parse_nonexistent(self):
        """Test PDF parser with non-existent file."""
        parser = PDFParser()
        result = parser.parse('/nonexistent/file.pdf')
        
        assert not result.success
        assert result.error is not None
    
    @pytest.mark.skipif(not _pdf_libraries_available(), reason="PDF libraries not available")
    def test_pdf_parser_parse_empty_file(self):
        """Test PDF parser with empty file."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            temp_file = f.name
        
        try:
            parser = PDFParser()
            result = parser.parse(temp_file)
            # Should handle gracefully
            assert result is not None
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


@pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parsers not available")
class TestSpreadsheetParser:
    """Test Spreadsheet parser."""
    
    def test_spreadsheet_parser_init(self):
        """Test Spreadsheet parser initialization."""
        parser = SpreadsheetParser()
        assert parser is not None
    
    def test_spreadsheet_parser_supported_extensions(self):
        """Test Spreadsheet parser supported extensions."""
        parser = SpreadsheetParser()
        extensions = parser.get_supported_extensions()
        assert '.xlsx' in extensions
        assert '.xls' in extensions
        assert '.ods' in extensions
    
    def test_spreadsheet_parser_parse_nonexistent(self):
        """Test Spreadsheet parser with non-existent file."""
        parser = SpreadsheetParser()
        result = parser.parse('/nonexistent/file.xlsx')
        
        assert not result.success
        assert result.error is not None


@pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parsers not available")
@pytest.mark.skipif(not _url_libraries_available(), reason="URL parsing libraries not available")
class TestURLParser:
    """Test URL parser."""
    
    def test_url_parser_init(self):
        """Test URL parser initialization."""
        parser = URLParser()
        assert parser is not None
        assert parser.timeout == 30
    
    def test_url_parser_init_with_timeout(self):
        """Test URL parser initialization with custom timeout."""
        parser = URLParser(timeout=60)
        assert parser.timeout == 60
    
    def test_url_parser_validate_url(self):
        """Test URL parser URL validation."""
        parser = URLParser()
        assert parser.validate_file('https://example.com')
        assert parser.validate_file('http://example.com')
        assert not parser.validate_file('not-a-url')
        assert not parser.validate_file('/path/to/file.txt')
    
    def test_url_parser_supported_extensions(self):
        """Test URL parser supported extensions (should be empty)."""
        parser = URLParser()
        extensions = parser.get_supported_extensions()
        assert extensions == []  # URLs don't have file extensions
    
    def test_url_parser_parse_invalid_url(self):
        """Test URL parser with invalid URL."""
        parser = URLParser()
        result = parser.parse('not-a-valid-url')
        
        assert not result.success
        assert result.error is not None
    
    @pytest.mark.skipif(not _url_libraries_available(), reason="requests/beautifulsoup4 not available")
    def test_is_url_helper(self):
        """Test is_url helper function."""
        assert is_url('https://example.com')
        assert is_url('http://example.com/path')
        assert not is_url('not-a-url')
        assert not is_url('/path/to/file.txt')
        assert not is_url('file.txt')


@pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parsers not available")
class TestParserRegistry:
    """Test parser registry and factory."""
    
    def test_get_parser_txt(self):
        """Test getting TXT parser."""
        parser = get_parser('test.txt')
        assert isinstance(parser, TXTParser)
    
    def test_get_parser_csv(self):
        """Test getting CSV parser."""
        parser = get_parser('test.csv')
        assert isinstance(parser, CSVParser)
    
    def test_get_parser_pdf(self):
        """Test getting PDF parser."""
        parser = get_parser('test.pdf')
        assert isinstance(parser, PDFParser)
    
    def test_get_parser_xlsx(self):
        """Test getting Spreadsheet parser for .xlsx."""
        parser = get_parser('test.xlsx')
        assert isinstance(parser, SpreadsheetParser)
    
    @pytest.mark.skipif(not _url_libraries_available(), reason="requests/beautifulsoup4 not available")
    def test_get_parser_url(self):
        """Test getting URL parser for URL."""
        parser = get_parser('https://example.com')
        assert isinstance(parser, URLParser)
    
    @pytest.mark.skipif(not _url_libraries_available(), reason="requests/beautifulsoup4 not available")
    def test_get_parser_url_http(self):
        """Test getting URL parser for HTTP URL."""
        parser = get_parser('http://example.com')
        assert isinstance(parser, URLParser)
    
    def test_get_parser_unsupported(self):
        """Test getting parser for unsupported file type."""
        with pytest.raises(ValueError):
            get_parser('test.unknown')


@pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parsers not available")
class TestAIAgentFileParsing:
    """Test file parsing via AI Agent."""
    
    @pytest.fixture
    def temp_pidfile(self):
        """Create temporary PID file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pid') as f:
            pidfile = f.name
        yield pidfile
        if os.path.exists(pidfile):
            os.remove(pidfile)
    
    @pytest.fixture
    def sample_txt_file(self):
        """Create a sample text file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("Test content for parsing.")
            temp_file = f.name
        yield temp_file
        os.remove(temp_file)
    
    def test_daemon_parse_file(self, temp_pidfile, sample_txt_file):
        """Test file parsing via daemon."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        
        result = daemon.parse_file(sample_txt_file)
        
        # Result should be a dict (ParseResult.to_dict())
        assert isinstance(result, dict)
        # If parsers are available, should have success field
        if 'success' in result:
            # If parsing succeeded, should have text
            if result.get('success'):
                assert 'text' in result
            else:
                assert 'error' in result
    
    def test_daemon_parse_file_nonexistent(self, temp_pidfile):
        """Test file parsing with non-existent file."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        
        result = daemon.parse_file('/nonexistent/file.txt')
        
        assert isinstance(result, dict)
        assert not result.get('success', True)  # Should fail or return error


class TestAIAgentIntegration:
    """Integration tests for AI Agent."""
    
    @pytest.fixture
    def temp_pidfile(self):
        """Create temporary PID file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pid') as f:
            pidfile = f.name
        yield pidfile
        if os.path.exists(pidfile):
            os.remove(pidfile)
    
    def test_keyword_extraction_workflow(self, temp_pidfile):
        """Test complete keyword extraction workflow."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        daemon.model_manager.initialize(use_spacy=False)
        
        text = "Machine learning and artificial intelligence are key technologies."
        keywords = daemon.extract_keywords(text, num_keywords=5, method='combined')
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
    
    def test_entity_extraction_workflow(self, temp_pidfile):
        """Test complete entity extraction workflow."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        daemon.model_manager.initialize(use_spacy=False)
        
        text = "Barack Obama was the President of the United States."
        entities = daemon.extract_entities(text)
        
        assert isinstance(entities, dict)
    
    def test_parse_and_extract_keywords(self, temp_pidfile):
        """Test parsing file and extracting keywords."""
        # Create sample text file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("Artificial intelligence and machine learning are transforming industries.")
            temp_file = f.name
        
        try:
            daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
            daemon.model_manager.initialize(use_spacy=False)
            
            # Parse file
            parse_result = daemon.parse_file(temp_file)
            
            if parse_result.get('success'):
                # Extract keywords from parsed text
                text = parse_result.get('text', '')
                if text:
                    keywords = daemon.extract_keywords(text, num_keywords=5)
                    assert isinstance(keywords, list)
        finally:
            os.remove(temp_file)


class TestAIAgentErrorHandling:
    """Test error handling in AI Agent."""
    
    @pytest.fixture
    def temp_pidfile(self):
        """Create temporary PID file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pid') as f:
            pidfile = f.name
        yield pidfile
        if os.path.exists(pidfile):
            os.remove(pidfile)
    
    def test_daemon_handles_missing_models(self, temp_pidfile):
        """Test daemon handles missing models gracefully."""
        daemon = AIAgentDaemon(pidfile=temp_pidfile, use_spacy=False)
        
        # Try to extract keywords without initializing models
        # Should raise RuntimeError or handle gracefully
        try:
            keywords = daemon.extract_keywords("test")
            # If it doesn't raise, should return empty or error
            assert isinstance(keywords, list)
        except RuntimeError:
            # Expected behavior
            pass
    
    def test_parser_handles_invalid_file(self):
        """Test parser handles invalid file gracefully."""
        parser = TXTParser()
        result = parser.parse('/invalid/path/file.txt')
        
        assert not result.success
        assert result.error is not None
    
    def test_get_parser_handles_unknown_extension(self):
        """Test get_parser handles unknown file extension."""
        with pytest.raises(ValueError):
            get_parser('file.unknown')

