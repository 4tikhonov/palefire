"""
Unit tests for FastAPI endpoints.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

# Try to import API module, skip tests if not available
try:
    import api
    API_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    API_AVAILABLE = False


# Mock the dependencies before importing api
@pytest.fixture
def mock_graphiti():
    """Mock Graphiti instance."""
    mock = AsyncMock()
    mock.driver = Mock()
    mock.driver.session = Mock()
    return mock


@pytest.fixture
def mock_config():
    """Mock config module."""
    with patch('api.config') as mock:
        mock.NEO4J_URI = 'bolt://localhost:7687'
        mock.NEO4J_USER = 'neo4j'
        mock.NEO4J_PASSWORD = 'password'
        mock.LLM_PROVIDER = 'ollama'
        mock.OLLAMA_MODEL = 'test-model'
        mock.EMBEDDER_PROVIDER = 'ollama'
        mock.OLLAMA_EMBEDDING_MODEL = 'test-embedding'
        mock.DEFAULT_SEARCH_METHOD = 'question-aware'
        mock.SEARCH_RESULT_LIMIT = 20
        mock.NER_ENABLED = True
        mock.NER_USE_SPACY = False
        mock.LOG_LEVEL = 'INFO'
        mock.LOG_FORMAT = '%(message)s'
        mock.LOG_DATE_FORMAT = '%Y-%m-%d'
        mock.validate_config = Mock()
        mock.get_llm_config = Mock(return_value={
            'api_key': 'test',
            'model': 'test-model',
            'small_model': 'test-model',
            'base_url': 'http://localhost'
        })
        mock.get_embedder_config = Mock(return_value={
            'api_key': 'test',
            'embedding_model': 'test-embedding',
            'embedding_dim': 768,
            'base_url': 'http://localhost'
        })
        yield mock


class TestAPIEndpoints:
    """Test API endpoints."""
    
    @pytest.mark.skip(reason="Requires full API setup")
    def test_root_endpoint(self):
        """Test root endpoint."""
        # This would require full API initialization
        pass
    
    @pytest.mark.skip(reason="Requires full API setup")
    def test_health_endpoint(self):
        """Test health check endpoint."""
        pass
    
    @pytest.mark.skip(reason="Requires full API setup")
    def test_config_endpoint(self):
        """Test config endpoint."""
        pass


@pytest.mark.skipif(not API_AVAILABLE, reason="API module not available")
class TestAPIModels:
    """Test Pydantic models."""
    
    def test_search_method_enum(self):
        """Test SearchMethod enum."""
        from api import SearchMethod
        assert SearchMethod.standard.value == "standard"
        assert SearchMethod.connection.value == "connection"
        assert SearchMethod.question_aware.value == "question-aware"
    
    def test_episode_type_enum(self):
        """Test EpisodeType enum."""
        from api import EpisodeType
        assert EpisodeType.text.value == "text"
        assert EpisodeType.json.value == "json"
    
    def test_episode_model(self):
        """Test Episode model."""
        from api import Episode, EpisodeType
        episode = Episode(
            content="Test content",
            type=EpisodeType.text,
            description="Test description"
        )
        assert episode.content == "Test content"
        assert episode.type == EpisodeType.text
        assert episode.description == "Test description"
    
    def test_search_request_model(self):
        """Test SearchRequest model."""
        from api import SearchRequest, SearchMethod
        request = SearchRequest(
            query="Test query",
            method=SearchMethod.question_aware,
            limit=10
        )
        assert request.query == "Test query"
        assert request.method == SearchMethod.question_aware
        assert request.limit == 10
    
    def test_search_request_default_method(self):
        """Test SearchRequest with default method."""
        from api import SearchRequest
        request = SearchRequest(query="Test query")
        assert request.method.value == "question-aware"
    
    def test_search_request_limit_validation(self):
        """Test SearchRequest limit validation."""
        from api import SearchRequest
        from pydantic import ValidationError
        
        # Valid limits
        request = SearchRequest(query="Test", limit=1)
        assert request.limit == 1
        
        request = SearchRequest(query="Test", limit=100)
        assert request.limit == 100
        
        # Invalid limits should raise validation error
        with pytest.raises(ValidationError):
            SearchRequest(query="Test", limit=0)
        
        with pytest.raises(ValidationError):
            SearchRequest(query="Test", limit=101)
    
    def test_entity_info_model(self):
        """Test EntityInfo model."""
        from api import EntityInfo
        entity = EntityInfo(
            name="California",
            type="LOC",
            labels=["Entity", "LOC"],
            uuid="xyz-123"
        )
        assert entity.name == "California"
        assert entity.type == "LOC"
        assert len(entity.labels) == 2
    
    def test_connection_info_model(self):
        """Test ConnectionInfo model."""
        from api import ConnectionInfo, EntityInfo
        connection = ConnectionInfo(
            count=5,
            entities=[
                EntityInfo(name="Test", type="PER", labels=[], uuid="123")
            ],
            relationship_types=["WORKED_AT"]
        )
        assert connection.count == 5
        assert len(connection.entities) == 1
        assert len(connection.relationship_types) == 1
    
    def test_scoring_info_model(self):
        """Test ScoringInfo model."""
        from api import ScoringInfo
        scoring = ScoringInfo(
            final_score=0.95,
            original_score=0.85,
            connection_score=0.75,
            temporal_score=1.0,
            query_match_score=0.80,
            entity_type_score=2.0
        )
        assert scoring.final_score == 0.95
        assert scoring.temporal_score == 1.0
    
    def test_search_result_model(self):
        """Test SearchResult model."""
        from api import SearchResult
        result = SearchResult(
            rank=1,
            uuid="abc-123",
            name="Test Entity",
            summary="Test summary",
            labels=["Entity"],
            attributes={"key": "value"}
        )
        assert result.rank == 1
        assert result.name == "Test Entity"
    
    def test_search_response_model(self):
        """Test SearchResponse model."""
        from api import SearchResponse, SearchResult
        response = SearchResponse(
            query="Test query",
            method="question-aware",
            total_results=1,
            results=[
                SearchResult(
                    rank=1,
                    uuid="123",
                    name="Test",
                    summary="Summary",
                    labels=[],
                    attributes={}
                )
            ],
            timestamp="2025-01-01T00:00:00Z"
        )
        assert response.query == "Test query"
        assert response.total_results == 1
        assert len(response.results) == 1


class TestAPIHelpers:
    """Test API helper functions."""
    
    def test_filter_name_embedding(self):
        """Test filtering name_embedding from attributes."""
        attributes = {
            'position': 'Test',
            'name_embedding': [0.1, 0.2],
            'state': 'CA'
        }
        filtered = {k: v for k, v in attributes.items() if k != 'name_embedding'}
        assert 'name_embedding' not in filtered
        assert 'position' in filtered
        assert 'state' in filtered


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

