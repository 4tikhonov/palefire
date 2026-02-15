"""
Pale Fire - AI Agent Module

Daemon service for keeping Gensim and spaCy models loaded in memory.
"""

from .AIAgent import AIAgentDaemon, ModelManager, AIAgentClient, get_daemon
from .response_parser import OllamaResponseParser
from .entity_merger import EntityMerger

# Import parsers (optional)
try:
    from .parsers import (
        BaseParser,
        ParseResult,
        TXTParser,
        CSVParser,
        PDFParser,
        SpreadsheetParser,
        get_parser,
        PARSERS
    )
    __all__ = [
        'AIAgentDaemon',
        'ModelManager',
        'AIAgentClient',
        'get_daemon',
        'OllamaResponseParser',
        'EntityMerger',
        'BaseParser',
        'ParseResult',
        'TXTParser',
        'CSVParser',
        'PDFParser',
        'SpreadsheetParser',
        'get_parser',
        'PARSERS',
    ]
except ImportError:
    __all__ = ['AIAgentDaemon', 'ModelManager', 'AIAgentClient', 'get_daemon', 'OllamaResponseParser', 'EntityMerger']

