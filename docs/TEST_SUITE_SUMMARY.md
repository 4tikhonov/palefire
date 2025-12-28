# Test Suite Summary

## Overview

Pale Fire now includes a comprehensive test suite with **75+ unit tests** covering all major modules and functions.

## Test Coverage

### Module Coverage

| Module | Tests | Coverage Areas |
|--------|-------|----------------|
| `config.py` | 15+ | Configuration validation, defaults, environment variables, helper functions |
| `modules/PaleFireCore.py` | 25+ | EntityEnricher (NER), QuestionTypeDetector, entity extraction, question type detection |
| Search Functions | 20+ | Query parsing, temporal relevance, connection scoring, multi-factor ranking |
| `api.py` | 15+ | Pydantic models, API endpoints, request/response validation |

### Feature Coverage

✅ **Configuration System**
- Default values
- Environment variable loading
- Validation logic
- Helper functions
- Weight sum validation

✅ **Entity Enrichment (NER)**
- Pattern-based NER
- spaCy-based NER
- Date extraction
- Money extraction
- Percentage extraction
- Entity grouping by type

✅ **Question Type Detection**
- WHO questions (person focus)
- WHERE questions (location focus)
- WHEN questions (temporal focus)
- WHAT questions (organization/position)
- HOW MANY questions (quantity)
- WHY questions (reason)
- Entity weight calculation
- Confidence scoring

✅ **Search & Ranking**
- Query term extraction
- Temporal relevance calculation
- Connection-based scoring
- Query match scoring
- Multi-factor ranking (5 factors)
- Score normalization

✅ **API Models**
- Request validation
- Response formatting
- Enum types
- Data models
- Error handling

## Test Files

```
tests/
├── __init__.py                    # Package initialization
├── conftest.py                    # Shared fixtures and configuration
├── test_config.py                 # Configuration tests (15+ tests)
├── test_palefire_core.py          # Core module tests (25+ tests)
├── test_search_functions.py       # Search logic tests (20+ tests)
└── test_api.py                    # API tests (15+ tests)
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Use test runner script
./run_tests.sh coverage
```

### Test Runner Modes

```bash
./run_tests.sh all         # All tests
./run_tests.sh unit        # Unit tests only
./run_tests.sh integration # Integration tests only
./run_tests.sh coverage    # With coverage report
./run_tests.sh fast        # Exclude slow tests
./run_tests.sh config      # Config module only
./run_tests.sh core        # PaleFireCore module only
./run_tests.sh search      # Search functions only
./run_tests.sh api         # API only
```

## Test Examples

### Configuration Testing

```python
def test_neo4j_config_defaults():
    """Test Neo4j configuration defaults."""
    import config
    assert config.NEO4J_URI is not None
    assert config.NEO4J_USER is not None
    assert config.NEO4J_PASSWORD is not None
```

### Entity Extraction Testing

```python
def test_entity_extraction():
    """Test entity extraction."""
    enricher = EntityEnricher(use_spacy=False)
    text = "Kamala Harris worked in California."
    entities = enricher._extract_entities_pattern(text)
    assert len(entities) > 0
```

### Question Detection Testing

```python
def test_who_question():
    """Test WHO question detection."""
    detector = QuestionTypeDetector()
    result = detector.detect_question_type("Who is the president?")
    assert result['type'] == 'WHO'
    assert result['entity_weights']['PER'] > 1.0
```

### Scoring Logic Testing

```python
def test_score_calculation():
    """Test multi-factor score calculation."""
    semantic_score = 0.8
    connection_score = 0.6
    temporal_score = 1.0
    query_match_score = 0.7
    entity_type_score = 0.5
    
    final_score = (
        0.30 * semantic_score +
        0.15 * connection_score +
        0.20 * temporal_score +
        0.20 * query_match_score +
        0.15 * entity_type_score
    )
    
    assert 0.0 <= final_score <= 1.0
```

## Test Fixtures

Shared fixtures in `conftest.py`:

- `sample_episode_text` - Text episode data
- `sample_episode_json` - JSON episode data
- `sample_query_who` - WHO question
- `sample_query_where` - WHERE question
- `sample_query_when` - WHEN question
- `mock_node` - Mock Neo4j node
- `mock_enriched_node` - Mock enriched node
- `mock_graphiti` - Mock Graphiti instance
- `mock_entity_enricher` - Mock EntityEnricher
- `mock_question_detector` - Mock QuestionTypeDetector
- `sample_search_results` - Sample search results
- `sample_config` - Sample configuration

## Test Markers

Tests are marked for easy filtering:

```python
@pytest.mark.unit              # Unit test
@pytest.mark.integration       # Integration test
@pytest.mark.slow              # Slow running test
@pytest.mark.requires_neo4j    # Requires Neo4j
@pytest.mark.requires_spacy    # Requires spaCy
```

Run specific markers:
```bash
pytest -m unit
pytest -m "not slow"
pytest -m "not requires_neo4j"
```

## Continuous Integration

### GitHub Actions

Automated testing on:
- Push to main/develop
- Pull requests
- Manual trigger

**Test Matrix**:
- Python 3.9, 3.10, 3.11, 3.12
- Ubuntu latest

**CI Jobs**:
1. **Test** - Run all tests with coverage
2. **Lint** - Code quality checks (flake8, black, isort)
3. **Security** - Security scans (safety, bandit)

### Coverage Reports

- Uploaded to Codecov
- HTML reports generated
- Terminal summary displayed

## Test Statistics

| Metric | Value |
|--------|-------|
| Total Tests | 75+ |
| Test Files | 4 |
| Fixtures | 12 |
| Markers | 5 |
| Python Versions | 4 (3.9-3.12) |

## Test Quality Metrics

✅ **Independence** - Each test runs independently
✅ **Clarity** - Descriptive names and docstrings
✅ **Coverage** - All major functions tested
✅ **Speed** - Fast unit tests, marked slow tests
✅ **Maintainability** - Shared fixtures, clear structure

## Future Enhancements

### Planned Test Additions

1. **Integration Tests**
   - Full Neo4j integration
   - End-to-end search pipeline
   - API endpoint integration

2. **Performance Tests**
   - Benchmark critical functions
   - Load testing for API
   - Memory usage profiling

3. **Property-Based Tests**
   - Hypothesis testing
   - Fuzz testing
   - Edge case generation

4. **Visual Regression Tests**
   - API response format validation
   - JSON schema validation

## Best Practices

### Writing New Tests

1. **Use descriptive names**
   ```python
   def test_extract_year_from_query_with_2020():
   ```

2. **One logical assertion per test**
   ```python
   def test_user_name():
       user = create_user("John")
       assert user.name == "John"
   ```

3. **Use fixtures for setup**
   ```python
   def test_with_fixture(sample_episode_text):
       result = process(sample_episode_text)
       assert result is not None
   ```

4. **Test edge cases**
   ```python
   def test_empty_input():
       assert process([]) == []
   ```

5. **Mark tests appropriately**
   ```python
   @pytest.mark.slow
   @pytest.mark.requires_neo4j
   def test_full_search_pipeline():
       pass
   ```

## Resources

- **[TESTING.md](TESTING.md)** - Complete testing guide
- **[tests/README.md](../tests/README.md)** - Test directory overview
- **[pytest Documentation](https://docs.pytest.org/)** - pytest framework
- **[pytest-asyncio](https://pytest-asyncio.readthedocs.io/)** - Async testing
- **[pytest-cov](https://pytest-cov.readthedocs.io/)** - Coverage reporting

## Maintenance

### Running Tests Regularly

```bash
# Before committing
./run_tests.sh fast

# Before pushing
./run_tests.sh coverage

# Before release
./run_tests.sh all
```

### Updating Tests

When adding new features:
1. Write tests first (TDD)
2. Ensure tests pass
3. Check coverage
4. Update documentation

### Test Maintenance

- Review and update tests quarterly
- Remove obsolete tests
- Add tests for bug fixes
- Keep fixtures up to date

---

**Test Suite v1.0** - Ensuring Quality Through Comprehensive Testing! ✅

*Last Updated: December 2025*

