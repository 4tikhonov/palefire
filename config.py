"""
Pale Fire Configuration

Centralized configuration for all settings.
Override values using environment variables in .env file.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Neo4j Configuration
# ============================================================================

NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://10.147.18.253:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

# ============================================================================
# LLM Configuration
# ============================================================================

# OpenAI API Key (required by Graphiti, can be placeholder for Ollama)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# LLM Provider: 'ollama' or 'openai'
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'ollama')

# Ollama Configuration
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://10.147.18.253:11434/v1')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'deepseek-r1:7b')
OLLAMA_SMALL_MODEL = os.environ.get('OLLAMA_SMALL_MODEL', 'deepseek-r1:7b')
OLLAMA_VERIFICATION_MODEL = os.environ.get('OLLAMA_VERIFICATION_MODEL', None)  # Optional: comma-separated list of models for NER verification (defaults to OLLAMA_MODEL)
OLLAMA_VERIFICATION_TIMEOUT = int(os.environ.get('OLLAMA_VERIFICATION_TIMEOUT', '300'))  # Timeout in seconds for verification requests (default: 300 = 5 minutes)
OLLAMA_PARALLEL_REQUESTS = os.environ.get('OLLAMA_PARALLEL_REQUESTS', 'true').lower() in ('true', '1', 'yes')  # Enable parallel Ollama requests for better performance (default: True)
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY', 'ollama')  # Placeholder

# OpenAI Configuration
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', None)  # Use default
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4')
OPENAI_SMALL_MODEL = os.environ.get('OPENAI_SMALL_MODEL', 'gpt-3.5-turbo')

# ============================================================================
# Embedder Configuration
# ============================================================================

EMBEDDER_PROVIDER = os.environ.get('EMBEDDER_PROVIDER', 'ollama')

# Ollama Embedder
OLLAMA_EMBEDDING_MODEL = os.environ.get('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
OLLAMA_EMBEDDING_DIM = int(os.environ.get('OLLAMA_EMBEDDING_DIM', '768'))
OLLAMA_EMBEDDING_BASE_URL = os.environ.get('OLLAMA_EMBEDDING_BASE_URL', 'http://10.147.18.253:11434/v1')

# OpenAI Embedder
OPENAI_EMBEDDING_MODEL = os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
OPENAI_EMBEDDING_DIM = int(os.environ.get('OPENAI_EMBEDDING_DIM', '1536'))

# ============================================================================
# Search Configuration
# ============================================================================

# Default search method: 'standard', 'connection', 'question-aware'
DEFAULT_SEARCH_METHOD = os.environ.get('DEFAULT_SEARCH_METHOD', 'question-aware')

# Search result limits
SEARCH_RESULT_LIMIT = int(os.environ.get('SEARCH_RESULT_LIMIT', '20'))
SEARCH_TOP_K = int(os.environ.get('SEARCH_TOP_K', '5'))

# Ranking weights (must sum to <= 1.0)
WEIGHT_CONNECTION = float(os.environ.get('WEIGHT_CONNECTION', '0.15'))
WEIGHT_TEMPORAL = float(os.environ.get('WEIGHT_TEMPORAL', '0.20'))
WEIGHT_QUERY_MATCH = float(os.environ.get('WEIGHT_QUERY_MATCH', '0.20'))
WEIGHT_ENTITY_TYPE = float(os.environ.get('WEIGHT_ENTITY_TYPE', '0.15'))
# Semantic weight is calculated as: 1.0 - sum(other weights)

# ============================================================================
# NER Configuration
# ============================================================================

# Enable NER enrichment by default
NER_ENABLED = os.environ.get('NER_ENABLED', 'true').lower() in ('true', '1', 'yes')

# Use spaCy if available, otherwise fall back to pattern-based
NER_USE_SPACY = os.environ.get('NER_USE_SPACY', 'true').lower() in ('true', '1', 'yes')

# spaCy model name
SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.environ.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG_DATE_FORMAT = os.environ.get('LOG_DATE_FORMAT', '%Y-%m-%d %H:%M:%S')

# ============================================================================
# Application Configuration
# ============================================================================

# Episode naming prefix
EPISODE_NAME_PREFIX = os.environ.get('EPISODE_NAME_PREFIX', 'Episode')

# Reference time for episodes (use current time if not specified)
USE_CURRENT_TIME = os.environ.get('USE_CURRENT_TIME', 'true').lower() in ('true', '1', 'yes')

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration and raise errors for missing required values."""
    errors = []
    
    if not NEO4J_URI:
        errors.append("NEO4J_URI must be set")
    
    if not NEO4J_USER:
        errors.append("NEO4J_USER must be set")
    
    if not NEO4J_PASSWORD:
        errors.append("NEO4J_PASSWORD must be set")
    
    if LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY must be set when using OpenAI provider")
    
    # Validate weights sum to <= 1.0
    total_weight = WEIGHT_CONNECTION + WEIGHT_TEMPORAL + WEIGHT_QUERY_MATCH + WEIGHT_ENTITY_TYPE
    if total_weight > 1.0:
        errors.append(f"Sum of ranking weights ({total_weight}) exceeds 1.0")
    
    if errors:
        raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))

# ============================================================================
# Helper Functions
# ============================================================================

def get_llm_config():
    """Get LLM configuration based on provider."""
    if LLM_PROVIDER == 'ollama':
        # Parse verification models: support comma-separated list or single model
        verification_models = []
        if OLLAMA_VERIFICATION_MODEL and OLLAMA_VERIFICATION_MODEL.strip():
            # Split by comma and strip whitespace, filter out empty strings
            verification_models = [m.strip() for m in OLLAMA_VERIFICATION_MODEL.split(',') if m.strip()]
            # If we have valid models, use them; otherwise fallback to main model
            if not verification_models:
                verification_models = [OLLAMA_MODEL]
        else:
            verification_models = [OLLAMA_MODEL]
        
        # Ensure we always have at least one model
        if not verification_models:
            verification_models = [OLLAMA_MODEL]
        
        # Keep single model for backward compatibility
        verification_model = verification_models[0] if verification_models else OLLAMA_MODEL
        
        return {
            'api_key': OLLAMA_API_KEY,
            'model': OLLAMA_MODEL,
            'small_model': OLLAMA_SMALL_MODEL,
            'verification_model': verification_model,  # Single model for backward compatibility
            'verification_models': verification_models,  # List of all verification models
            'verification_timeout': OLLAMA_VERIFICATION_TIMEOUT,
            'parallel_requests': OLLAMA_PARALLEL_REQUESTS,
            'base_url': OLLAMA_BASE_URL,
        }
    else:  # openai
        return {
            'api_key': OPENAI_API_KEY,
            'model': OPENAI_MODEL,
            'small_model': OPENAI_SMALL_MODEL,
            'verification_model': OPENAI_MODEL,  # Use same model for verification
            'parallel_requests': True,  # OpenAI API typically supports parallel requests
            'base_url': OPENAI_BASE_URL,
        }

def get_embedder_config():
    """Get embedder configuration based on provider."""
    if EMBEDDER_PROVIDER == 'ollama':
        return {
            'api_key': OLLAMA_API_KEY,
            'embedding_model': OLLAMA_EMBEDDING_MODEL,
            'embedding_dim': OLLAMA_EMBEDDING_DIM,
            'base_url': OLLAMA_EMBEDDING_BASE_URL,
        }
    else:  # openai
        return {
            'api_key': OPENAI_API_KEY,
            'embedding_model': OPENAI_EMBEDDING_MODEL,
            'embedding_dim': OPENAI_EMBEDDING_DIM,
            'base_url': OPENAI_BASE_URL,
        }

def print_config():
    """Print current configuration (for debugging)."""
    print("="*80)
    print("⚙️  PALE FIRE CONFIGURATION")
    print("="*80)
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Neo4j User: {NEO4J_USER}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    
    llm_cfg = get_llm_config()
    print(f"LLM Model: {llm_cfg['model']}")
    verification_models = llm_cfg.get('verification_models', [llm_cfg.get('verification_model', llm_cfg['model'])])
    if len(verification_models) > 1:
        print(f"LLM Verification Models: {', '.join(verification_models)}")
    else:
        print(f"LLM Verification Model: {verification_models[0] if verification_models else llm_cfg.get('verification_model', llm_cfg['model'])}")
    print(f"LLM Parallel Requests: {llm_cfg.get('parallel_requests', True)}")
    print(f"LLM Base URL: {llm_cfg['base_url']}")
    
    emb_cfg = get_embedder_config()
    print(f"Embedder Provider: {EMBEDDER_PROVIDER}")
    print(f"Embedder Model: {emb_cfg['embedding_model']}")
    print(f"Embedder Dimensions: {emb_cfg['embedding_dim']}")
    
    print(f"\nSearch Configuration:")
    print(f"  Default Method: {DEFAULT_SEARCH_METHOD}")
    print(f"  Result Limit: {SEARCH_RESULT_LIMIT}")
    print(f"  Top K: {SEARCH_TOP_K}")
    
    print(f"\nRanking Weights:")
    print(f"  Connection: {WEIGHT_CONNECTION}")
    print(f"  Temporal: {WEIGHT_TEMPORAL}")
    print(f"  Query Match: {WEIGHT_QUERY_MATCH}")
    print(f"  Entity Type: {WEIGHT_ENTITY_TYPE}")
    semantic_weight = 1.0 - (WEIGHT_CONNECTION + WEIGHT_TEMPORAL + WEIGHT_QUERY_MATCH + WEIGHT_ENTITY_TYPE)
    print(f"  Semantic: {semantic_weight:.2f}")
    
    print(f"\nNER Configuration:")
    print(f"  Enabled: {NER_ENABLED}")
    print(f"  Use spaCy: {NER_USE_SPACY}")
    print(f"  spaCy Model: {SPACY_MODEL}")
    
    print("="*80)

# Validate configuration on import
try:
    validate_config()
except ValueError as e:
    import sys
    print(f"Configuration Error: {e}", file=sys.stderr)
    # Don't exit here, let the application handle it

