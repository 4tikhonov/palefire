"""
Pale Fire API Models

Pydantic models for API request/response validation.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class SearchMethod(str, Enum):
    """Available search methods."""
    standard = "standard"
    connection = "connection"
    question_aware = "question-aware"


class EpisodeType(str, Enum):
    """Episode content types."""
    text = "text"
    json = "json"


class KeywordExtractionMethod(str, Enum):
    """Available keyword extraction methods."""
    tfidf = "tfidf"
    textrank = "textrank"
    word_freq = "word_freq"
    combined = "combined"


# ============================================================================
# Request Models
# ============================================================================

class Episode(BaseModel):
    """Episode for ingestion."""
    content: Any = Field(..., description="Episode content (string or object)")
    type: EpisodeType = Field(..., description="Content type")
    description: Optional[str] = Field(None, description="Episode description")


class IngestRequest(BaseModel):
    """Request to ingest episodes."""
    episodes: List[Episode] = Field(..., description="List of episodes to ingest")
    enable_ner: bool = Field(True, description="Enable NER enrichment")


class SearchRequest(BaseModel):
    """Request to search the knowledge graph."""
    query: str = Field(..., description="Search query", min_length=1)
    method: SearchMethod = Field(
        SearchMethod.question_aware,
        description="Search method to use"
    )
    limit: Optional[int] = Field(None, description="Maximum number of results", ge=1, le=100)


class KeywordExtractionRequest(BaseModel):
    """Request to extract keywords from text."""
    text: str = Field(..., description="Text to extract keywords from", min_length=1)
    method: KeywordExtractionMethod = Field(
        KeywordExtractionMethod.tfidf,
        description="Extraction method to use"
    )
    num_keywords: int = Field(10, description="Number of keywords to extract", ge=1, le=100)
    min_word_length: int = Field(3, description="Minimum word length", ge=1, le=50)
    max_word_length: int = Field(50, description="Maximum word length", ge=1, le=100)
    use_stemming: bool = Field(False, description="Enable stemming for preprocessing")
    tfidf_weight: float = Field(1.0, description="Weight for TF-IDF scores (combined method)", ge=0.0)
    textrank_weight: float = Field(0.5, description="Weight for TextRank scores (combined method)", ge=0.0)
    word_freq_weight: float = Field(0.3, description="Weight for word frequency scores (combined method)", ge=0.0)
    position_weight: float = Field(0.2, description="Weight for position-based scoring", ge=0.0)
    title_weight: float = Field(2.0, description="Weight multiplier for words in titles/headers", ge=0.0)
    first_sentence_weight: float = Field(1.5, description="Weight multiplier for words in first sentence", ge=0.0)
    enable_ngrams: bool = Field(True, description="Enable n-gram extraction (2-4 word phrases)")
    min_ngram: int = Field(2, description="Minimum n-gram size (1 for unigrams, 2-4 for phrases)", ge=1, le=4)
    max_ngram: int = Field(4, description="Maximum n-gram size (2, 3, or 4)", ge=2, le=4)
    ngram_weight: float = Field(1.2, description="Weight multiplier for n-grams", ge=0.0)
    documents: Optional[List[str]] = Field(None, description="Optional document corpus for IDF calculation")


# ============================================================================
# Response Models
# ============================================================================

class EntityInfo(BaseModel):
    """Entity information."""
    name: str
    type: Optional[str] = None
    labels: List[str] = []
    uuid: Optional[str] = None


class ConnectionInfo(BaseModel):
    """Connection information."""
    count: int
    entities: List[EntityInfo]
    relationship_types: List[str]


class ScoringInfo(BaseModel):
    """Scoring breakdown."""
    final_score: float
    original_score: float
    connection_score: Optional[float] = None
    temporal_score: Optional[float] = None
    query_match_score: Optional[float] = None
    entity_type_score: Optional[float] = None


class SearchResult(BaseModel):
    """Single search result."""
    rank: int
    uuid: str
    name: str
    summary: str
    labels: List[str]
    attributes: Dict[str, Any]
    scoring: Optional[ScoringInfo] = None
    connections: Optional[ConnectionInfo] = None
    recognized_entities: Optional[Dict[str, List[str]]] = None


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    method: str
    total_results: int
    results: List[SearchResult]
    timestamp: str


class StatusResponse(BaseModel):
    """Status response."""
    status: str
    message: str
    database_stats: Optional[Dict[str, int]] = None


class ConfigResponse(BaseModel):
    """Configuration response."""
    neo4j_uri: str
    llm_provider: str
    llm_model: str
    embedder_model: str
    search_method: str
    search_limit: int
    ner_enabled: bool


class KeywordInfo(BaseModel):
    """Single keyword information."""
    keyword: str
    score: float
    type: Optional[str] = Field(None, description="Type of keyword: 'unigram', '2-gram', '3-gram', or '4-gram'")


class KeywordExtractionResponse(BaseModel):
    """Keyword extraction response."""
    method: str
    num_keywords: int
    keywords: List[KeywordInfo]
    parameters: Dict[str, Any]
    timestamp: str

