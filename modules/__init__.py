"""
Pale Fire - Core Modules
"""

from .PaleFireCore import EntityEnricher, QuestionTypeDetector
from .KeywordBase import KeywordExtractor

# API models (optional import)
try:
    from .api_models import (
        SearchMethod,
        EpisodeType,
        KeywordExtractionMethod,
        Episode,
        IngestRequest,
        SearchRequest,
        KeywordExtractionRequest,
        EntityInfo,
        ConnectionInfo,
        ScoringInfo,
        SearchResult,
        SearchResponse,
        StatusResponse,
        ConfigResponse,
        KeywordInfo,
        KeywordExtractionResponse,
    )
    __all__ = [
        'EntityEnricher',
        'QuestionTypeDetector',
        'KeywordExtractor',
        'SearchMethod',
        'EpisodeType',
        'KeywordExtractionMethod',
        'Episode',
        'IngestRequest',
        'SearchRequest',
        'KeywordExtractionRequest',
        'EntityInfo',
        'ConnectionInfo',
        'ScoringInfo',
        'SearchResult',
        'SearchResponse',
        'StatusResponse',
        'ConfigResponse',
        'KeywordInfo',
        'KeywordExtractionResponse',
    ]
except ImportError:
    # API models not available (pydantic may not be installed)
    __all__ = ['EntityEnricher', 'QuestionTypeDetector', 'KeywordExtractor']

