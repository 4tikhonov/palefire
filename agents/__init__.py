"""
Pale Fire - AI Agent Module

Daemon service for keeping Gensim and spaCy models loaded in memory.
"""

from .AIAgent import AIAgentDaemon, ModelManager, AIAgentClient, get_daemon

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
    __all__ = ['AIAgentDaemon', 'ModelManager', 'AIAgentClient', 'get_daemon']

