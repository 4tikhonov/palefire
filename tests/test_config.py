"""
Unit tests for config module.
"""

import os
import pytest
from unittest.mock import patch


class TestConfig:
    """Test configuration module."""
    
    def test_config_imports(self):
        """Test that config module can be imported."""
        import config
        assert config is not None
    
    def test_neo4j_config_defaults(self):
        """Test Neo4j configuration defaults."""
        import config
        assert config.NEO4J_URI is not None
        assert config.NEO4J_USER is not None
        assert config.NEO4J_PASSWORD is not None
    
    def test_llm_config_defaults(self):
        """Test LLM configuration defaults."""
        import config
        assert config.LLM_PROVIDER in ['ollama', 'openai']
        assert config.OLLAMA_MODEL is not None
        assert config.OLLAMA_BASE_URL is not None
    
    def test_embedder_config_defaults(self):
        """Test embedder configuration defaults."""
        import config
        assert config.EMBEDDER_PROVIDER in ['ollama', 'openai']
        assert config.OLLAMA_EMBEDDING_MODEL is not None
        assert config.OLLAMA_EMBEDDING_DIM > 0
    
    def test_search_config_defaults(self):
        """Test search configuration defaults."""
        import config
        assert config.DEFAULT_SEARCH_METHOD in ['standard', 'connection', 'question-aware']
        assert config.SEARCH_RESULT_LIMIT > 0
        assert config.SEARCH_TOP_K > 0
    
    def test_ranking_weights_sum(self):
        """Test that ranking weights sum to <= 1.0."""
        import config
        total = (config.WEIGHT_CONNECTION + config.WEIGHT_TEMPORAL + 
                config.WEIGHT_QUERY_MATCH + config.WEIGHT_ENTITY_TYPE)
        assert total <= 1.0
        assert total >= 0.0
    
    def test_ranking_weights_valid_range(self):
        """Test that individual weights are in valid range."""
        import config
        assert 0.0 <= config.WEIGHT_CONNECTION <= 1.0
        assert 0.0 <= config.WEIGHT_TEMPORAL <= 1.0
        assert 0.0 <= config.WEIGHT_QUERY_MATCH <= 1.0
        assert 0.0 <= config.WEIGHT_ENTITY_TYPE <= 1.0
    
    def test_ner_config(self):
        """Test NER configuration."""
        import config
        assert isinstance(config.NER_ENABLED, bool)
        assert isinstance(config.NER_USE_SPACY, bool)
        assert config.SPACY_MODEL is not None
    
    def test_log_config(self):
        """Test logging configuration."""
        import config
        assert config.LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert config.LOG_FORMAT is not None
    
    def test_get_llm_config(self):
        """Test get_llm_config helper function."""
        import config
        llm_cfg = config.get_llm_config()
        assert 'api_key' in llm_cfg
        assert 'model' in llm_cfg
        assert 'small_model' in llm_cfg
        assert 'base_url' in llm_cfg
    
    def test_get_embedder_config(self):
        """Test get_embedder_config helper function."""
        import config
        emb_cfg = config.get_embedder_config()
        assert 'api_key' in emb_cfg
        assert 'embedding_model' in emb_cfg
        assert 'embedding_dim' in emb_cfg
        assert 'base_url' in emb_cfg
    
    def test_validate_config_with_valid_config(self):
        """Test validate_config with valid configuration."""
        import config
        # Should not raise an exception
        config.validate_config()
    
    @patch.dict(os.environ, {'NEO4J_URI': ''})
    def test_validate_config_missing_neo4j_uri(self):
        """Test validate_config with missing NEO4J_URI."""
        # Reload config with empty URI
        import importlib
        import config as cfg
        importlib.reload(cfg)
        
        with pytest.raises(ValueError, match="NEO4J_URI"):
            cfg.validate_config()
    
    def test_print_config_runs(self, capsys):
        """Test that print_config runs without error."""
        import config
        config.print_config()
        captured = capsys.readouterr()
        assert "PALE FIRE CONFIGURATION" in captured.out
        assert "Neo4j URI" in captured.out


class TestConfigEnvironmentVariables:
    """Test configuration with environment variables."""
    
    @patch.dict(os.environ, {'SEARCH_RESULT_LIMIT': '50'})
    def test_search_limit_from_env(self):
        """Test that SEARCH_RESULT_LIMIT can be set from environment."""
        import importlib
        import config
        importlib.reload(config)
        assert config.SEARCH_RESULT_LIMIT == 50
    
    @patch.dict(os.environ, {'WEIGHT_CONNECTION': '0.25'})
    def test_weight_from_env(self):
        """Test that weights can be set from environment."""
        import importlib
        import config
        importlib.reload(config)
        assert config.WEIGHT_CONNECTION == 0.25
    
    @patch.dict(os.environ, {'NER_ENABLED': 'false'})
    def test_ner_disabled_from_env(self):
        """Test that NER can be disabled from environment."""
        import importlib
        import config
        importlib.reload(config)
        assert config.NER_ENABLED is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

