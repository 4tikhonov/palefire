# Pale Fire Test Suite

Comprehensive unit tests for all Pale Fire modules and functions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## Test Files

| File | Description | Tests |
|------|-------------|-------|
| `test_config.py` | Configuration module | 15+ tests |
| `test_palefire_core.py` | EntityEnricher & QuestionTypeDetector | 25+ tests |
| `test_search_functions.py` | Search and helper functions | 20+ tests |
| `test_api.py` | FastAPI endpoints and models | 15+ tests |

## Test Coverage

Current test coverage by module:

- **config.py**: Configuration validation, defaults, helpers
- **modules/PaleFireCore.py**: Entity extraction, question detection
- **Search functions**: Query parsing, scoring, ranking
- **api.py**: Pydantic models, endpoint logic

## Running Tests

### All Tests
```bash
pytest
```

### Specific File
```bash
pytest tests/test_config.py
```

### Specific Test
```bash
pytest tests/test_config.py::TestConfig::test_neo4j_config_defaults
```

### With Coverage
```bash
pytest --cov=. --cov-report=html --cov-report=term
```

### Verbose Output
```bash
pytest -v
```

## Test Markers

Tests are marked for easy filtering:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.requires_neo4j` - Requires Neo4j
- `@pytest.mark.requires_spacy` - Requires spaCy

Run specific markers:
```bash
pytest -m unit
pytest -m "not slow"
pytest -m "not requires_neo4j"
```

## Writing New Tests

1. Create test file: `test_<module>.py`
2. Create test class: `class Test<Feature>:`
3. Write test methods: `def test_<functionality>():`
4. Run tests: `pytest tests/test_<module>.py`

Example:
```python
class TestMyFeature:
    def test_basic_functionality(self):
        """Test basic feature works."""
        result = my_function()
        assert result is not None
```

## Test Structure

```python
import pytest

class TestFeatureName:
    """Test FeatureName functionality."""
    
    def test_normal_case(self):
        """Test normal operation."""
        assert function() == expected
    
    def test_edge_case(self):
        """Test edge case."""
        assert function(edge_input) == edge_output
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function(invalid_input)
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Manual triggers

## See Also

- [Testing Guide](../docs/TESTING.md) - Complete testing documentation
- [pytest Documentation](https://docs.pytest.org/)
- [Coverage Reports](../htmlcov/index.html) - After running with --cov

---

**Test Suite v1.0** - Ensuring Quality! ✅

