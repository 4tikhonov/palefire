"""
Pale Fire FastAPI Wrapper

REST API for Pale Fire knowledge graph search system.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from pathlib import Path

# Import Pale Fire components
import config
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from modules import EntityEnricher, QuestionTypeDetector, KeywordExtractor
from modules.api_models import (
    SearchMethod,
    EpisodeType,
    Episode,
    IngestRequest,
    SearchRequest,
    EntityInfo,
    ConnectionInfo,
    ScoringInfo,
    SearchResult,
    SearchResponse,
    StatusResponse,
    ConfigResponse,
    KeywordExtractionMethod,
    KeywordExtractionRequest,
    KeywordInfo,
    KeywordExtractionResponse,
    FileParseResponse,
    AgentStatusResponse,
    EntityExtractionRequest,
    EntityExtractionResponse,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pale Fire API",
    description="Intelligent Knowledge Graph Search System with 5-Factor Ranking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
graphiti_instance: Optional[Graphiti] = None
enricher_instance: Optional[EntityEnricher] = None
detector_instance: Optional[QuestionTypeDetector] = None
keyword_extractor_instance: Optional[KeywordExtractor] = None

# AI Agent daemon (optional)
try:
    from agents import get_daemon, AIAgentDaemon
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    logger.warning("AI Agent not available")


# ============================================================================
# Startup and Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Graphiti on startup."""
    global graphiti_instance, enricher_instance, detector_instance, keyword_extractor_instance
    
    try:
        logger.info("Initializing Pale Fire API...")
        
        # Validate configuration
        config.validate_config()
        
        # Get configuration
        llm_cfg = config.get_llm_config()
        emb_cfg = config.get_embedder_config()
        
        # Configure LLM client
        llm_config = LLMConfig(
            api_key=llm_cfg['api_key'],
            model=llm_cfg['model'],
            small_model=llm_cfg['small_model'],
            base_url=llm_cfg['base_url'],
        )
        
        llm_client = OpenAIGenericClient(config=llm_config)
        
        # Initialize Graphiti
        graphiti_instance = Graphiti(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            llm_client=llm_client,
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=emb_cfg['api_key'],
                    embedding_model=emb_cfg['embedding_model'],
                    embedding_dim=emb_cfg['embedding_dim'],
                    base_url=emb_cfg['base_url'],
                )
            ),
            cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
        )
        
        # Initialize enricher and detector
        enricher_instance = EntityEnricher(use_spacy=config.NER_USE_SPACY)
        detector_instance = QuestionTypeDetector()
        
        # Initialize keyword extractor
        try:
            keyword_extractor_instance = KeywordExtractor()
            logger.info("✅ Keyword extractor initialized")
        except ImportError as e:
            logger.warning(f"Keyword extractor not available (gensim may not be installed): {e}")
            keyword_extractor_instance = None
        
        logger.info("✅ Pale Fire API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Pale Fire API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close Graphiti on shutdown."""
    global graphiti_instance
    
    if graphiti_instance:
        try:
            await graphiti_instance.close()
            logger.info("✅ Pale Fire API shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - API status."""
    return StatusResponse(
        status="ok",
        message="Pale Fire API is running"
    )


@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        async with graphiti_instance.driver.session() as session:
            result = await session.run("RETURN 1")
            await result.single()
        
        # Get database stats
        async with graphiti_instance.driver.session() as session:
            result = await session.run("""
                MATCH (n)
                RETURN count(n) as node_count,
                       count{(n)-[]->()} as relationship_count
            """)
            record = await result.single()
            stats = {
                "nodes": record['node_count'] if record else 0,
                "relationships": record['relationship_count'] if record else 0
            }
        
        return StatusResponse(
            status="healthy",
            message="All systems operational",
            database_stats=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration."""
    return ConfigResponse(
        neo4j_uri=config.NEO4J_URI,
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else config.OPENAI_MODEL,
        embedder_model=config.OLLAMA_EMBEDDING_MODEL if config.EMBEDDER_PROVIDER == 'ollama' else config.OPENAI_EMBEDDING_MODEL,
        search_method=config.DEFAULT_SEARCH_METHOD,
        search_limit=config.SEARCH_RESULT_LIMIT,
        ner_enabled=config.NER_ENABLED
    )


@app.post("/ingest", response_model=StatusResponse)
async def ingest_episodes(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest episodes into the knowledge graph.
    
    This endpoint accepts a list of episodes and ingests them into the graph.
    NER enrichment can be enabled/disabled per request.
    """
    try:
        logger.info(f"Ingesting {len(request.episodes)} episodes (NER: {request.enable_ner})")
        
        enricher = enricher_instance if request.enable_ner else None
        ingested_count = 0
        
        for i, episode in enumerate(request.episodes, 1):
            # Enrich episode if NER is enabled
            episode_data = {
                'content': episode.content,
                'type': episode.type.value,
                'description': episode.description or 'No description'
            }
            
            if enricher:
                episode_data = enricher.enrich_episode(episode_data)
            
            # Prepare content
            content = (episode_data['content'] if isinstance(episode_data['content'], str)
                      else str(episode_data['content']))
            
            # Add to Graphiti
            await graphiti_instance.add_episode(
                name=f"{config.EPISODE_NAME_PREFIX} {i}",
                episode_body=content,
                source=episode_data['type'],
                source_description=episode_data['description'],
                reference_time=datetime.now(timezone.utc),
            )
            
            ingested_count += 1
        
        return StatusResponse(
            status="success",
            message=f"Successfully ingested {ingested_count} episodes"
        )
        
    except Exception as e:
        logger.error(f"Error ingesting episodes: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search the knowledge graph.
    
    Performs intelligent search using the specified method and returns
    ranked results with detailed scoring information.
    """
    try:
        logger.info(f"Searching: '{request.query}' (method: {request.method.value})")
        
        # Use simplified search for API
        # For full functionality, extract search functions to a separate module
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
        
        # Execute basic search
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = request.limit or config.SEARCH_RESULT_LIMIT
        
        node_search_results = await graphiti_instance._search(
            query=request.query,
            config=node_search_config,
        )
        
        # Convert to dict format
        results = [{'node': node, 'original_score': getattr(node, 'score', 1.0)} 
                  for node in node_search_results.nodes]
        
        # Apply limit if specified
        limit = request.limit or config.SEARCH_TOP_K
        results = results[:limit]
        
        # Convert results to response format
        search_results = []
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                node = result.get('node')
                
                # Filter out name_embedding
                attributes = {}
                if hasattr(node, 'attributes') and node.attributes:
                    attributes = {k: v for k, v in node.attributes.items() if k != 'name_embedding'}
                
                # Build result
                search_result = SearchResult(
                    rank=i,
                    uuid=node.uuid if node else "",
                    name=node.name if node else "",
                    summary=node.summary if node else "",
                    labels=node.labels if hasattr(node, 'labels') else [],
                    attributes=attributes,
                )
                
                # Add scoring if available
                if 'final_score' in result:
                    search_result.scoring = ScoringInfo(
                        final_score=result.get('final_score', 0),
                        original_score=result.get('original_score', 0),
                        connection_score=result.get('connection_score'),
                        temporal_score=result.get('temporal_score'),
                        query_match_score=result.get('query_match_score'),
                        entity_type_score=result.get('entity_type_score'),
                    )
                
                # Add connections if available
                if 'connected_entities' in result:
                    entities = []
                    for entity in result['connected_entities']:
                        if isinstance(entity, dict):
                            entities.append(EntityInfo(
                                name=entity['name'],
                                type=entity.get('type'),
                                labels=entity.get('labels', []),
                                uuid=entity.get('uuid')
                            ))
                        else:
                            entities.append(EntityInfo(name=str(entity)))
                    
                    search_result.connections = ConnectionInfo(
                        count=result.get('connection_count', 0),
                        entities=entities,
                        relationship_types=result.get('relationship_types', [])
                    )
                
                # Add recognized entities if available
                if 'enriched_node' in result:
                    search_result.recognized_entities = result['enriched_node'].get('entities_by_type', {})
                
                search_results.append(search_result)
        
        return SearchResponse(
            query=request.query,
            method=request.method.value,
            total_results=len(search_results),
            results=search_results,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/keywords", response_model=KeywordExtractionResponse)
async def extract_keywords(request: KeywordExtractionRequest):
    """
    Extract keywords from text using Gensim.
    
    Supports multiple extraction methods (TF-IDF, TextRank, word frequency, combined)
    with configurable weights and parameters.
    """
    if keyword_extractor_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Keyword extraction not available. Install gensim: pip install gensim>=4.3.0"
        )
    
    try:
        logger.info(f"Extracting keywords (method: {request.method.value}, num: {request.num_keywords})")
        
        # Create extractor with requested parameters
        extractor = KeywordExtractor(
            method=request.method.value,
            num_keywords=request.num_keywords,
            min_word_length=request.min_word_length,
            max_word_length=request.max_word_length,
            use_stemming=request.use_stemming,
            tfidf_weight=request.tfidf_weight,
            textrank_weight=request.textrank_weight,
            word_freq_weight=request.word_freq_weight,
            position_weight=request.position_weight,
            title_weight=request.title_weight,
            first_sentence_weight=request.first_sentence_weight,
            enable_ngrams=request.enable_ngrams,
            min_ngram=request.min_ngram,
            max_ngram=request.max_ngram,
            ngram_weight=request.ngram_weight,
        )
        
        # Extract keywords
        keywords = extractor.extract(request.text, request.documents)
        
        # Convert to response format
        keyword_list = [
            KeywordInfo(
                keyword=kw['keyword'],
                score=kw['score'],
                type=kw.get('type', 'unigram')
            )
            for kw in keywords
        ]
        
        # Build parameters dict
        parameters = {
            'num_keywords': request.num_keywords,
            'min_word_length': request.min_word_length,
            'max_word_length': request.max_word_length,
            'use_stemming': request.use_stemming,
            'tfidf_weight': request.tfidf_weight,
            'textrank_weight': request.textrank_weight,
            'word_freq_weight': request.word_freq_weight,
            'position_weight': request.position_weight,
            'title_weight': request.title_weight,
            'first_sentence_weight': request.first_sentence_weight,
            'enable_ngrams': request.enable_ngrams,
            'min_ngram': request.min_ngram,
            'max_ngram': request.max_ngram,
            'ngram_weight': request.ngram_weight,
        }
        
        return KeywordExtractionResponse(
            method=request.method.value,
            num_keywords=len(keyword_list),
            keywords=keyword_list,
            parameters=parameters,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except ImportError as e:
        logger.error(f"Gensim not available: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Keyword extraction requires gensim: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {str(e)}")


@app.delete("/clean", response_model=StatusResponse)
async def clean_database():
    """
    Clean/clear the Neo4j database.
    
    ⚠️ WARNING: This permanently deletes all data!
    """
    try:
        logger.warning("Cleaning database...")
        
        # Get stats before cleaning
        async with graphiti_instance.driver.session() as session:
            result = await session.run("""
                MATCH (n)
                RETURN count(n) as node_count,
                       count{(n)-[]->()} as relationship_count
            """)
            record = await result.single()
            node_count = record['node_count'] if record else 0
            rel_count = record['relationship_count'] if record else 0
        
        if node_count == 0:
            return StatusResponse(
                status="success",
                message="Database is already empty"
            )
        
        # Delete all nodes and relationships
        async with graphiti_instance.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        
        logger.info(f"Database cleaned: {node_count} nodes, {rel_count} relationships deleted")
        
        return StatusResponse(
            status="success",
            message=f"Database cleaned successfully. Deleted {node_count} nodes and {rel_count} relationships."
        )
        
    except Exception as e:
        logger.error(f"Error cleaning database: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# ============================================================================
# File Parsing Endpoints
# ============================================================================

@app.post("/parse", response_model=FileParseResponse)
async def parse_file(
    file: UploadFile = File(...),
    file_type: Optional[str] = Form(None, description="Override file type detection (txt, csv, pdf, xlsx, xls, ods)"),
    delimiter: Optional[str] = Form(None, description="CSV delimiter (for CSV files)"),
    include_headers: bool = Form(True, description="Include headers in CSV output"),
    max_pages: Optional[int] = Form(None, description="Maximum pages to parse (for PDF files)", ge=1),
    extract_tables: bool = Form(True, description="Extract tables (for PDF/spreadsheet files)"),
    sheet_names: Optional[str] = Form(None, description="Comma-separated sheet names (for spreadsheet files)"),
):
    """
    Parse a file and extract text content.
    
    Supports multiple file types:
    - TXT: Plain text files
    - CSV: Comma-separated values
    - PDF: PDF documents
    - Spreadsheets: Excel (.xlsx, .xls) and OpenDocument (.ods)
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="File parsing requires AI Agent. Install parsing dependencies: pip install PyPDF2 openpyxl xlrd odfpy"
        )
    
    try:
        # Save uploaded file to temporary location
        file_extension = Path(file.filename).suffix.lower() if file.filename else ''
        
        # Determine file type
        detected_type = file_type or file_extension.lstrip('.')
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Get daemon instance
            daemon = get_daemon(use_spacy=False)
            
            # Prepare parser options
            parser_options = {}
            if delimiter:
                parser_options['delimiter'] = delimiter
            if not include_headers:
                parser_options['include_headers'] = False
            if max_pages:
                parser_options['max_pages'] = max_pages
            if not extract_tables:
                parser_options['extract_tables'] = False
            if sheet_names:
                parser_options['sheet_names'] = [s.strip() for s in sheet_names.split(',')]
            
            # Parse file
            result = daemon.parse_file(tmp_path, file_type=detected_type, **parser_options)
            
            return FileParseResponse(
                success=result.get('success', False),
                text=result.get('text', ''),
                metadata=result.get('metadata', {}),
                pages=result.get('pages', []),
                tables=result.get('tables', []),
                error=result.get('error'),
                file_type=detected_type,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(status_code=500, detail=f"File parsing failed: {str(e)}")


# ============================================================================
# AI Agent Endpoints
# ============================================================================

@app.get("/agent/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """
    Get AI Agent daemon status.
    
    Returns information about the daemon's running state, model initialization,
    and system resources.
    """
    if not AGENT_AVAILABLE:
        return AgentStatusResponse(
            running=False,
            models_initialized=False,
            use_spacy=False,
            spacy_available=False,
            parsers_available=False,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    try:
        daemon = get_daemon(use_spacy=False)
        status = daemon.get_status()
        
        # Try to get process information if daemon is running
        pid = None
        memory_mb = None
        cpu_percent = None
        
        try:
            import psutil
            pidfile = '/tmp/palefire_ai_agent.pid'
            if os.path.exists(pidfile):
                with open(pidfile, 'r') as f:
                    pid = int(f.read().strip())
                
                try:
                    process = psutil.Process(pid)
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent(interval=0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pid = None
        except ImportError:
            pass  # psutil not available
        
        return AgentStatusResponse(
            running=status.get('running', False),
            models_initialized=status.get('models_initialized', False),
            use_spacy=status.get('use_spacy', False),
            spacy_available=status.get('spacy_available', False),
            parsers_available=status.get('parsers_available', False),
            pid=pid,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@app.post("/entities", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """
    Extract named entities from text using spaCy or pattern-based extraction.
    
    Uses the AI Agent daemon if available for faster extraction (models stay loaded).
    Falls back to direct EntityEnricher if daemon is not available.
    """
    try:
        # Try to use daemon if available
        if AGENT_AVAILABLE:
            try:
                daemon = get_daemon(use_spacy=config.NER_USE_SPACY)
                
                # Initialize models if not already initialized
                if not daemon.model_manager.is_initialized():
                    daemon.model_manager.initialize(use_spacy=config.NER_USE_SPACY)
                
                # Extract entities using daemon
                entities_dict = daemon.extract_entities(request.text)
                
                return EntityExtractionResponse(
                    entities=entities_dict.get('entities', []),
                    entities_by_type=entities_dict.get('entities_by_type', {}),
                    all_entities=entities_dict.get('all_entities', []),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as e:
                logger.warning(f"Failed to use daemon for entity extraction: {e}, falling back to direct extraction")
        
        # Fall back to direct extraction
        if enricher_instance is None:
            raise HTTPException(
                status_code=503,
                detail="Entity extraction not available. EntityEnricher not initialized."
            )
        
        # Extract entities directly
        episode = {
            'content': request.text,
            'type': 'text',
            'description': 'Entity extraction'
        }
        
        enriched = enricher_instance.enrich_episode(episode)
        
        return EntityExtractionResponse(
            entities=enriched.get('entities', []),
            entities_by_type=enriched.get('entities_by_type', {}),
            all_entities=enriched.get('all_entities', []),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

