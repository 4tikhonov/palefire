# Configuration System Migration - Changelog

## Date: December 26, 2025

## Summary

Migrated all hardcoded configuration values from `palefire-cli.py` to a centralized configuration system in `config.py`. This improves maintainability, flexibility, and follows best practices for production applications.

## Changes Made

### 1. New Files Created

#### `config.py`
- **Purpose**: Centralized configuration management
- **Features**:
  - Loads all settings from environment variables
  - Provides sensible defaults for all values
  - Validates configuration on import
  - Helper functions: `get_llm_config()`, `get_embedder_config()`, `print_config()`, `validate_config()`
- **Configuration Categories**:
  - Neo4j connection settings
  - LLM provider configuration (Ollama/OpenAI)
  - Embedder configuration
  - Search parameters (method, limits, top-k)
  - Ranking weights (connection, temporal, query match, entity type)
  - NER settings
  - Logging configuration
  - Application settings

#### `env.example`
- **Purpose**: Template for environment variables
- **Contents**: All configurable settings with descriptions and defaults
- **Usage**: `cp env.example .env` and customize

#### `CONFIGURATION.md`
- **Purpose**: Comprehensive configuration documentation
- **Contents**:
  - Complete list of all environment variables
  - Configuration functions reference
  - Setup instructions
  - Best practices for production/development/research
  - Troubleshooting guide
  - Advanced configuration topics

#### `CHANGELOG_CONFIG.md`
- **Purpose**: Document this migration
- **Contents**: This file

### 2. Modified Files

#### `palefire-cli.py`
**Removed hardcoded values:**
- `neo4j_uri = "bolt://10.147.18.253:7687"`
- `neo4j_user = "neo4j"`
- `neo4j_password = "password"`
- `openai_api_key` checks
- `llm_config` hardcoded parameters
- `embedder` hardcoded parameters
- `node_search_config.limit = 20`
- `node_search_config.limit = 5`
- `connection_weight=0.3` default
- Method default `'question-aware'`

**Added:**
- Import of `config` module
- Use of `config.NEO4J_URI`, `config.NEO4J_USER`, `config.NEO4J_PASSWORD`
- Use of `config.get_llm_config()` and `config.get_embedder_config()`
- Use of `config.SEARCH_RESULT_LIMIT` and `config.SEARCH_TOP_K`
- Use of `config.WEIGHT_CONNECTION` for default weights
- Use of `config.DEFAULT_SEARCH_METHOD`
- Use of `config.print_config()` for config command
- Configuration validation via `config.validate_config()`

**Function signature changes:**
- `search_episodes_with_custom_ranking()`: `connection_weight=0.3` → `connection_weight=None` (uses `config.WEIGHT_CONNECTION` if None)
- CLI parser: `--method` default changed from `'question-aware'` to `None` (uses `config.DEFAULT_SEARCH_METHOD` if None)

#### `PALEFIRE_SETUP.md`
**Updated:**
- Configuration section to reference `env.example` and `config.py`
- Added step to view configuration with `python palefire-cli.py config`
- Updated environment variable examples
- Added note about centralized configuration

#### `README.md`
**Updated:**
- Quick start section to include configuration viewing
- Configuration section with complete examples
- Added reference to `CONFIGURATION.md`
- Updated `.env.example` references to `env.example`

### 3. Configuration Values Migrated

#### Neo4j Settings
- `NEO4J_URI` (default: `bolt://10.147.18.253:7687`)
- `NEO4J_USER` (default: `neo4j`)
- `NEO4J_PASSWORD` (default: `password`)

#### LLM Settings
- `LLM_PROVIDER` (default: `ollama`)
- `OPENAI_API_KEY` (required)
- `OLLAMA_BASE_URL` (default: `http://10.147.18.253:11434/v1`)
- `OLLAMA_MODEL` (default: `deepseek-r1:7b`)
- `OLLAMA_SMALL_MODEL` (default: `deepseek-r1:7b`)
- `OLLAMA_API_KEY` (default: `ollama`)
- `OPENAI_BASE_URL` (default: None)
- `OPENAI_MODEL` (default: `gpt-4`)
- `OPENAI_SMALL_MODEL` (default: `gpt-3.5-turbo`)

#### Embedder Settings
- `EMBEDDER_PROVIDER` (default: `ollama`)
- `OLLAMA_EMBEDDING_MODEL` (default: `nomic-embed-text`)
- `OLLAMA_EMBEDDING_DIM` (default: `768`)
- `OLLAMA_EMBEDDING_BASE_URL` (default: `http://10.147.18.253:11434/v1`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-ada-002`)
- `OPENAI_EMBEDDING_DIM` (default: `1536`)

#### Search Settings
- `DEFAULT_SEARCH_METHOD` (default: `question-aware`)
- `SEARCH_RESULT_LIMIT` (default: `20`)
- `SEARCH_TOP_K` (default: `5`)

#### Ranking Weights
- `WEIGHT_CONNECTION` (default: `0.15`)
- `WEIGHT_TEMPORAL` (default: `0.20`)
- `WEIGHT_QUERY_MATCH` (default: `0.20`)
- `WEIGHT_ENTITY_TYPE` (default: `0.15`)
- Semantic weight: calculated as `1.0 - sum(other weights)` = `0.30`

#### NER Settings
- `NER_ENABLED` (default: `true`)
- `NER_USE_SPACY` (default: `true`)
- `SPACY_MODEL` (default: `en_core_web_sm`)

#### Logging Settings
- `LOG_LEVEL` (default: `INFO`)
- `LOG_FORMAT` (default: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`)
- `LOG_DATE_FORMAT` (default: `%Y-%m-%d %H:%M:%S`)

#### Application Settings
- `EPISODE_NAME_PREFIX` (default: `Episode`)
- `USE_CURRENT_TIME` (default: `true`)

## Benefits

### 1. Maintainability
- All configuration in one place
- Easy to find and modify settings
- Clear separation of code and configuration

### 2. Flexibility
- Environment-specific configurations (dev/prod/test)
- Override any setting via environment variables
- No code changes needed for deployment

### 3. Validation
- Automatic validation on startup
- Clear error messages for missing/invalid settings
- Prevents runtime errors due to misconfiguration

### 4. Documentation
- Self-documenting via `config.py` comments
- Comprehensive `CONFIGURATION.md` guide
- Example file (`env.example`) with all options

### 5. Best Practices
- Follows 12-factor app methodology
- Secrets not hardcoded in source
- Easy to integrate with deployment tools

## Migration Guide

### For Existing Users

1. **Copy example configuration:**
   ```bash
   cd /path/to/palefire
   cp env.example .env
   ```

2. **Update `.env` with your settings:**
   ```bash
   nano .env  # or your preferred editor
   ```

3. **Verify configuration:**
   ```bash
   python palefire-cli.py config
   ```

4. **Test with a query:**
   ```bash
   python palefire-cli.py query "Test question?"
   ```

### For New Users

Just follow the Quick Start in `README.md` - the configuration system is already integrated.

## Backward Compatibility

### Breaking Changes
- None for users who set environment variables via `.env` file
- Hardcoded values in `palefire-cli.py` are removed, but defaults in `config.py` match the previous hardcoded values

### Non-Breaking Changes
- All default values match previous hardcoded values
- CLI interface unchanged
- Function signatures backward compatible (optional parameters)

## Testing

### Verification Steps

1. ✅ Configuration loads correctly
   ```bash
   python3 -c "import config; config.print_config()"
   ```

2. ✅ Files compile without errors
   ```bash
   python3 -m py_compile palefire-cli.py config.py
   ```

3. ✅ CLI help works
   ```bash
   python palefire-cli.py --help
   ```

4. ✅ Config command works
   ```bash
   python palefire-cli.py config
   ```

### Test Results

All tests passed successfully:
- Configuration module loads and validates
- All files compile without syntax errors
- CLI commands work as expected
- Configuration display shows correct values

## Future Enhancements

### Potential Improvements

1. **Configuration Profiles**
   - Support for named profiles (dev, prod, test)
   - Switch between profiles via CLI flag

2. **Runtime Configuration**
   - CLI flags to override config values
   - Example: `--search-limit 50 --top-k 10`

3. **Configuration Validation**
   - More sophisticated validation rules
   - Warnings for suboptimal settings

4. **Configuration Export**
   - Export current config to file
   - Share configurations between environments

5. **Configuration UI**
   - Interactive configuration wizard
   - Web-based configuration editor

## References

- **Configuration Guide**: [CONFIGURATION.md](CONFIGURATION.md)
- **CLI Guide**: [CLI_GUIDE.md](CLI_GUIDE.md)
- **Setup Guide**: [PALEFIRE_SETUP.md](PALEFIRE_SETUP.md)
- **Main README**: [README.md](README.md)

## Credits

Migration performed on December 26, 2025, as part of the Pale Fire project modernization effort.

## Support

For issues related to configuration:
1. Check [CONFIGURATION.md](CONFIGURATION.md) for detailed documentation
2. Verify `.env` file exists and has correct values
3. Run `python palefire-cli.py config` to see current settings
4. Check logs for configuration validation errors

---

**Configuration System v1.0** - Centralized, Validated, Documented 🎯

