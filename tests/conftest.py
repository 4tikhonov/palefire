"""
Shared pytest fixtures for all test modules.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test directories for examples
TEST_ROOT = Path(__file__).parent.parent
EXAMPLES_INPUT_DIR = TEST_ROOT / 'examples' / 'input'
EXAMPLES_OUTPUT_DIR = TEST_ROOT / 'examples' / 'output'


@pytest.fixture
def sample_episode_text():
    """Sample text episode for testing."""
    return {
        'content': 'Kamala Harris served as the Attorney General of California from January 3, 2011 to January 3, 2017.',
        'type': 'text',
        'description': 'Career information'
    }


@pytest.fixture
def sample_episode_json():
    """Sample JSON episode for testing."""
    return {
        'content': {
            'name': 'Kamala Harris',
            'position': 'Attorney General',
            'state': 'California',
            'start_date': '2011-01-03',
            'end_date': '2017-01-03'
        },
        'type': 'json',
        'description': 'Career information'
    }


@pytest.fixture
def sample_query_who():
    """Sample WHO question."""
    return "Who was the California Attorney General in 2020?"


@pytest.fixture
def sample_query_where():
    """Sample WHERE question."""
    return "Where did Kamala Harris work?"


@pytest.fixture
def sample_query_when():
    """Sample WHEN question."""
    return "When did Kamala Harris become Attorney General?"


@pytest.fixture
def mock_node():
    """Mock Neo4j node."""
    node = Mock()
    node.uuid = "test-uuid-123"
    node.name = "Kamala Harris"
    node.summary = "Attorney General of California"
    node.labels = ["Entity", "PER"]
    node.attributes = {
        'position': 'Attorney General',
        'state': 'California',
        'start_date': '2011-01-03',
        'end_date': '2017-01-03'
    }
    return node


@pytest.fixture
def mock_enriched_node():
    """Mock enriched node with entities."""
    return {
        'entities': [
            {'text': 'Kamala Harris', 'type': 'PER'},
            {'text': 'California', 'type': 'LOC'},
            {'text': '2011', 'type': 'DATE'},
            {'text': '2017', 'type': 'DATE'}
        ],
        'entities_by_type': {
            'PER': ['Kamala Harris'],
            'LOC': ['California'],
            'DATE': ['2011', '2017']
        },
        'all_entities': ['Kamala Harris', 'California', '2011', '2017']
    }


@pytest.fixture
def mock_graphiti():
    """Mock Graphiti instance."""
    graphiti = AsyncMock()
    graphiti.driver = Mock()
    
    # Mock session
    session = AsyncMock()
    graphiti.driver.session.return_value.__aenter__.return_value = session
    
    # Mock query result
    result = AsyncMock()
    record = Mock()
    record.__getitem__ = Mock(side_effect=lambda key: {
        'connection_count': 5,
        'connected_entities': [
            {'name': 'California', 'labels': ['Entity', 'LOC'], 'uuid': 'loc-123'},
            {'name': 'San Francisco', 'labels': ['Entity', 'LOC'], 'uuid': 'loc-456'}
        ],
        'relationship_types': ['WORKED_IN', 'LOCATED_IN']
    }.get(key))
    
    result.single.return_value = record
    session.run.return_value = result
    
    return graphiti


@pytest.fixture
def mock_entity_enricher():
    """Mock EntityEnricher."""
    enricher = Mock()
    enricher.enrich_episode.return_value = {
        'entities': [
            {'text': 'Kamala Harris', 'type': 'PER'},
            {'text': 'California', 'type': 'LOC'}
        ],
        'entities_by_type': {
            'PER': ['Kamala Harris'],
            'LOC': ['California']
        },
        'all_entities': ['Kamala Harris', 'California']
    }
    return enricher


@pytest.fixture
def mock_question_detector():
    """Mock QuestionTypeDetector."""
    detector = Mock()
    detector.detect_question_type.return_value = {
        'type': 'WHO',
        'confidence': 0.95,
        'description': 'Question about a person',
        'entity_weights': {
            'PER': 2.0,
            'ORG': 1.2,
            'LOC': 1.0
        }
    }
    return detector


@pytest.fixture
def sample_search_results():
    """Sample search results."""
    return [
        {
            'node': Mock(
                uuid='uuid-1',
                name='Result 1',
                summary='Summary 1',
                labels=['Entity', 'PER'],
                attributes={'key': 'value'}
            ),
            'final_score': 0.95,
            'original_score': 0.85,
            'connection_score': 0.75,
            'temporal_score': 1.0,
            'query_match_score': 0.80,
            'entity_type_score': 2.0,
            'connection_count': 10,
            'connections': {
                'count': 10,
                'entities': [
                    {'name': 'Entity 1', 'labels': ['Entity', 'LOC'], 'uuid': 'e1'}
                ],
                'entity_names_display': ['Entity 1 (LOC)'],
                'relationship_types': ['WORKED_IN']
            },
            'temporal_info': {'query_year': 2020},
            'enriched_node': {
                'entities_by_type': {'PER': ['Person 1']},
                'all_entities': ['Person 1']
            }
        }
    ]


@pytest.fixture
def sample_config():
    """Sample configuration."""
    return {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USER': 'neo4j',
        'NEO4J_PASSWORD': 'password',
        'OLLAMA_BASE_URL': 'http://localhost:11434/v1',
        'OLLAMA_MODEL': 'test-model',
        'OLLAMA_EMBEDDING_MODEL': 'test-embedding',
        'SEARCH_RESULT_LIMIT': 20,
        'SEARCH_TOP_K': 5,
        'WEIGHT_CONNECTION': 0.15,
        'WEIGHT_TEMPORAL': 0.20,
        'WEIGHT_QUERY_MATCH': 0.20,
        'WEIGHT_ENTITY_TYPE': 0.15
    }


@pytest.fixture
def examples_input_dir():
    """Path to examples/input directory for test files."""
    EXAMPLES_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLES_INPUT_DIR


@pytest.fixture
def examples_output_dir():
    """Path to examples/output directory for test outputs."""
    EXAMPLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLES_OUTPUT_DIR


# Pytest configuration hooks

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_neo4j: Tests requiring Neo4j")
    config.addinivalue_line("markers", "requires_spacy: Tests requiring spaCy")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests that require Neo4j
        if "neo4j" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_neo4j)
        
        # Mark tests that require spaCy
        if "spacy" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_spacy)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        else:
            # Default to unit test
            item.add_marker(pytest.mark.unit)

