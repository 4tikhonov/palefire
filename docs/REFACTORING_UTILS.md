# Refactoring: Moving Functions to Utils

**Date**: December 27, 2025  
**Type**: Code Organization / Refactoring

## Overview

This document describes the refactoring of `palefire-cli.py` to extract utility functions into a dedicated `utils/palefire_utils.py` module. This improves code organization, reusability, and maintainability.

## Changes Summary

### File Structure

```
backend/palefire/
├── palefire-cli.py          # 421 lines (was 1237 lines)
├── utils/
│   ├── __init__.py          # Module exports
│   └── palefire_utils.py    # 823 lines of utility functions
```

### Lines of Code

- **Old CLI**: 1,237 lines
- **New CLI**: 421 lines (66% reduction)
- **Utils Module**: 823 lines
- **Net Change**: Better organized, more modular

## Moved Functions

The following functions were moved from `palefire-cli.py` to `utils/palefire_utils.py`:

### Connection and Graph Utilities

1. **`get_node_connections_with_entities(graphiti, node_uuid)`**
   - Get connections and connected entity names for a node
   - Returns dict with count, entities (with types), and relationship_types

2. **`get_node_connections(graphiti, node_uuid)`**
   - Get the number of connections (edges) for a given node
   - Wrapper around `get_node_connections_with_entities`

### Temporal Utilities

3. **`extract_temporal_info(graphiti, node_uuid)`**
   - Extract temporal information from node attributes and related episodes
   - Returns dict with temporal data

4. **`calculate_temporal_relevance(node, temporal_info, query_year=None)`**
   - Calculate temporal relevance score based on query date and node's temporal attributes
   - Returns score between 0 and 1

### Query Analysis Utilities

5. **`extract_query_terms(query)`**
   - Extract important terms and entities from the query
   - Returns dict with query analysis (year, terms, proper nouns)

6. **`calculate_query_match_score(node, connection_info, query_terms)`**
   - Calculate how well a node matches the specific query terms
   - Checks node name, summary, connected entities, and attributes
   - Returns score between 0 and 1

### Search Functions

7. **`search_central_node(graphiti, query)`**
   - Use the top search result's UUID as the center node for reranking

8. **`search_episodes(graphiti, query)`**
   - Standard search using RRF hybrid search
   - Returns node search results

9. **`search_episodes_with_custom_ranking(graphiti, query, connection_weight=None)`**
   - Enhanced search with connection-based ranking
   - Weighs entities by their connections

10. **`search_episodes_with_question_aware_ranking(...)`**
    - INTELLIGENT search with question-type detection and entity-type weighting
    - Combines 5 factors: semantic, connection, temporal, query match, entity type
    - Most sophisticated search method

### Export and Database Utilities

11. **`export_results_to_json(results, filepath, query, method)`**
    - Export search results to a JSON file
    - Filters out `name_embedding` and includes entity information

12. **`clean_database(graphiti, confirm=False, nodes_only=False)`**
    - Clean/clear the Neo4j database
    - Supports confirmation prompts and partial cleanup

## Functions Remaining in CLI

The following functions remain in `palefire-cli.py` as they are CLI-specific:

1. **`load_episodes_from_file(filepath)`**
   - Load episodes from a JSON file
   - CLI-specific file handling

2. **`ingest_episodes(episodes_data, graphiti, use_ner=True)`**
   - Ingest episodes into the knowledge graph with optional NER enrichment
   - CLI-specific ingestion workflow

3. **`search_query(query, graphiti, method='question-aware', export_json=None)`**
   - Execute a search query using the specified method
   - CLI-specific search dispatcher

4. **`create_cli_parser()`**
   - Create and configure the argument parser
   - CLI-specific

5. **`create_graphiti_instance()`**
   - Create and return a configured Graphiti instance
   - CLI-specific initialization

6. **`main_cli(args)`**
   - Main CLI entry point
   - CLI-specific

7. **`main()`**
   - Legacy main function for backward compatibility
   - CLI-specific

## Benefits

### 1. **Improved Modularity**
   - Utility functions can now be imported by other modules (e.g., `api.py`, tests)
   - Clear separation between CLI logic and core functionality

### 2. **Better Code Organization**
   - Related functions are grouped together in `utils/palefire_utils.py`
   - CLI file is now focused on command-line interface logic

### 3. **Enhanced Reusability**
   - Search functions can be used by both CLI and API
   - Testing becomes easier with isolated utility functions

### 4. **Easier Maintenance**
   - Changes to core functionality are isolated in utils
   - CLI updates don't require touching search logic

### 5. **Clearer Dependencies**
   - Import structure makes dependencies explicit
   - Easier to understand what each module does

## Usage Examples

### Importing in CLI

```python
from utils.palefire_utils import (
    search_episodes,
    search_episodes_with_custom_ranking,
    search_episodes_with_question_aware_ranking,
    export_results_to_json,
    clean_database,
)
```

### Importing in API

```python
from utils.palefire_utils import (
    search_episodes_with_question_aware_ranking,
    export_results_to_json,
)
```

### Importing in Tests

```python
from utils.palefire_utils import (
    extract_query_terms,
    calculate_query_match_score,
    calculate_temporal_relevance,
)
```

## Migration Notes

### For Developers

1. **No API Changes**: The CLI interface remains exactly the same
2. **Import Updates**: If you were importing from `palefire-cli.py`, update imports to use `utils.palefire_utils`
3. **Testing**: All existing tests should continue to work
4. **New Tests**: Consider adding unit tests for individual utility functions

### For Users

- **No changes required**: The CLI works exactly as before
- All commands remain the same:
  ```bash
  python palefire-cli.py ingest episodes.json --ner
  python palefire-cli.py query "Who was the California Attorney General in 2020?"
  python palefire-cli.py clean --confirm
  ```

## File Organization

### `utils/palefire_utils.py`

Contains all core utility functions organized by category:
- Connection and graph utilities
- Temporal utilities
- Query analysis utilities
- Search functions
- Export and database utilities

### `utils/__init__.py`

Exports all utility functions for easy importing:
```python
from utils import (
    search_episodes_with_question_aware_ranking,
    export_results_to_json,
    # ... other functions
)
```

## Testing

All utility functions should be tested independently:

```bash
# Test imports
python -c "from utils.palefire_utils import search_episodes; print('✅ Import successful')"

# Run unit tests (when available)
pytest tests/test_palefire_utils.py
```

## Future Improvements

1. **Add Unit Tests**: Create comprehensive unit tests for each utility function
2. **Type Hints**: Add type hints to all function signatures
3. **Documentation**: Add docstring examples for complex functions
4. **Performance**: Profile and optimize search functions
5. **Caching**: Consider caching frequently used results

## Related Documentation

- [Architecture](./ARCHITECTURE.md) - Overall system architecture
- [CLI Guide](./CLI_GUIDE.md) - Command-line interface usage
- [API Guide](./API_GUIDE.md) - REST API usage
- [Testing](./TESTING.md) - Testing guidelines

## Verification

To verify the refactoring was successful:

```bash
# Check syntax
python3 -m py_compile palefire-cli.py
python3 -m py_compile utils/palefire_utils.py

# Test CLI still works
python palefire-cli.py --help

# Test imports
python -c "from utils.palefire_utils import search_episodes; print('✅')"
```

## Conclusion

This refactoring significantly improves the codebase organization without changing any functionality. The CLI remains fully backward compatible while the new structure enables better code reuse and testing.

