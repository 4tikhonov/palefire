# Testing Guide

## Overview

Pale Fire includes a comprehensive test suite to ensure code quality and reliability.

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── test_config.py                 # Configuration module tests
├── test_palefire_core.py          # EntityEnricher & QuestionTypeDetector tests
├── test_search_functions.py       # Search and helper function tests
└── test_api.py                    # FastAPI endpoint tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `httpx` - FastAPI testing

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term
```

### Run Specific Test Files

```bash
# Run config tests only
pytest tests/test_config.py

# Run PaleFireCore tests only
pytest tests/test_palefire_core.py

# Run search function tests only
pytest tests/test_search_functions.py

# Run API tests only
pytest tests/test_api.py
```

### Run Specific Test Classes or Methods

```bash
# Run specific test class
pytest tests/test_config.py::TestConfig

# Run specific test method
pytest tests/test_config.py::TestConfig::test_neo4j_config_defaults

# Run tests matching a pattern
pytest -k "test_neo4j"
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run tests that don't require Neo4j
pytest -m "not requires_neo4j"
```

## Test Coverage

### Generate Coverage Report

```bash
# HTML report (opens in browser)
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=. --cov-report=term

# Both
pytest --cov=. --cov-report=html --cov-report=term
```

### Coverage Goals

- **Overall**: > 80%
- **Core modules**: > 90%
- **Critical functions**: 100%

## Test Categories

### Unit Tests

Test individual functions and methods in isolation.

**Location**: All test files
**Marker**: `@pytest.mark.unit`
**Run**: `pytest -m unit`

**Examples**:
- Configuration validation
- Entity extraction
- Question type detection
- Score calculations

### Integration Tests

Test interactions between components.

**Location**: `test_api.py`, `test_search_functions.py`
**Marker**: `@pytest.mark.integration`
**Run**: `pytest -m integration`

**Examples**:
- API endpoints
- Database queries
- Full search pipeline

### Tests Requiring External Services

Some tests require external services:

**Neo4j Tests**:
- Marker: `@pytest.mark.requires_neo4j`
- Skip if Neo4j not available
- Run: `pytest -m requires_neo4j`

**spaCy Tests**:
- Marker: `@pytest.mark.requires_spacy`
- Skip if spaCy not installed
- Run: `pytest -m requires_spacy`

## Writing Tests

### Test Structure

```python
import pytest

class TestMyFeature:
    """Test MyFeature functionality."""
    
    def test_basic_functionality(self):
        """Test basic feature works."""
        result = my_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case handling."""
        result = my_function(edge_case_input)
        assert result is not None
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Test Naming Conventions

- **Files**: `test_*.py`
- **Classes**: `Test*`
- **Methods**: `test_*`
- **Descriptive names**: `test_extract_year_from_query`

### Assertions

```python
# Equality
assert result == expected

# Approximate equality (for floats)
assert result == pytest.approx(0.75, rel=1e-2)

# Membership
assert 'key' in dictionary

# Type checking
assert isinstance(result, dict)

# Exceptions
with pytest.raises(ValueError):
    function_that_raises()

# Exceptions with message match
with pytest.raises(ValueError, match="specific error"):
    function_that_raises()
```

### Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        'content': 'Test content',
        'type': 'text'
    }

def test_with_fixture(sample_data):
    """Test using fixture."""
    result = process_data(sample_data)
    assert result is not None
```

### Mocking

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test with mocked dependency."""
    mock_db = Mock()
    mock_db.query.return_value = [{'id': 1}]
    
    result = function_using_db(mock_db)
    assert result == [{'id': 1}]
    mock_db.query.assert_called_once()

@patch('module.external_api')
def test_with_patch(mock_api):
    """Test with patched external call."""
    mock_api.return_value = {'status': 'ok'}
    result = function_calling_api()
    assert result['status'] == 'ok'
```

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

## Test Examples

### Testing Configuration

```python
def test_config_validation():
    """Test configuration validation."""
    import config
    config.validate_config()  # Should not raise
```

### Testing Entity Extraction

```python
def test_entity_extraction():
    """Test entity extraction."""
    enricher = EntityEnricher(use_spacy=False)
    text = "Kamala Harris worked in California."
    entities = enricher._extract_entities_pattern(text)
    assert len(entities) > 0
```

### Testing Question Detection

```python
def test_who_question():
    """Test WHO question detection."""
    detector = QuestionTypeDetector()
    result = detector.detect_question_type("Who is the president?")
    assert result['type'] == 'WHO'
    assert result['entity_weights']['PER'] > 1.0
```

### Testing Search Logic

```python
def test_score_calculation():
    """Test score calculation."""
    semantic_score = 0.8
    connection_score = 0.6
    weight = 0.3
    
    final = (1 - weight) * semantic_score + weight * connection_score
    assert 0.0 <= final <= 1.0
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Best Practices

### 1. Test Independence

Each test should be independent:
```python
# Good: Independent test
def test_feature():
    data = create_test_data()
    result = process(data)
    assert result == expected

# Bad: Depends on previous test
def test_feature_part2():
    # Assumes test_feature_part1 ran first
    result = get_global_state()
    assert result == expected
```

### 2. Clear Test Names

```python
# Good: Descriptive name
def test_extract_year_from_query_with_2020():
    pass

# Bad: Vague name
def test_extract():
    pass
```

### 3. One Assertion Per Test (when possible)

```python
# Good: Single logical assertion
def test_user_name():
    user = create_user("John")
    assert user.name == "John"

# Acceptable: Related assertions
def test_user_creation():
    user = create_user("John", age=30)
    assert user.name == "John"
    assert user.age == 30
```

### 4. Use Fixtures for Setup

```python
# Good: Fixture for common setup
@pytest.fixture
def enricher():
    return EntityEnricher(use_spacy=False)

def test_with_enricher(enricher):
    result = enricher.enrich_episode(data)
    assert result is not None
```

### 5. Test Edge Cases

```python
def test_empty_input():
    result = process([])
    assert result == []

def test_none_input():
    result = process(None)
    assert result is None

def test_large_input():
    result = process(range(10000))
    assert len(result) == 10000
```

## Troubleshooting

### Tests Not Found

```bash
# Check test discovery
pytest --collect-only

# Verify test file naming
ls tests/test_*.py
```

### Import Errors

```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Async Test Errors

```bash
# Install pytest-asyncio
pip install pytest-asyncio

# Mark async tests
@pytest.mark.asyncio
async def test_async():
    pass
```

### Fixture Not Found

```python
# Define fixture in conftest.py
# tests/conftest.py
@pytest.fixture
def shared_fixture():
    return "shared data"
```

## Performance Testing

### Benchmark Tests

```python
import time

def test_performance():
    """Test function performance."""
    start = time.time()
    result = expensive_function()
    duration = time.time() - start
    
    assert duration < 1.0  # Should complete in < 1 second
    assert result is not None
```

### Load Testing

For API load testing, use tools like:
- `locust`
- `ab` (Apache Bench)
- `wrk`

## See Also

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**Testing Guide v1.0** - Quality Through Testing! ✅

