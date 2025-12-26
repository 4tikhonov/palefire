"""
Pale Fire FastAPI Wrapper

REST API for Pale Fire knowledge graph search system.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import Pale Fire components
import config
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from modules import EntityEnricher, QuestionTypeDetector

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

# Global Graphiti instance (initialized on startup)
graphiti_instance: Optional[Graphiti] = None
enricher_instance: Optional[EntityEnricher] = None
detector_instance: Optional[QuestionTypeDetector] = None


# ============================================================================
# Pydantic Models
# ============================================================================

class SearchMethod(str, Enum):
    """Available search methods."""
    standard = "standard"
    connection = "connection"
    question_aware = "question-aware"


class EpisodeType(str, Enum):
    """Episode content types."""
    text = "text"
    json = "json"


class Episode(BaseModel):
    """Episode for ingestion."""
    content: Any = Field(..., description="Episode content (string or object)")
    type: EpisodeType = Field(..., description="Content type")
    description: Optional[str] = Field(None, description="Episode description")


class IngestRequest(BaseModel):
    """Request to ingest episodes."""
    episodes: List[Episode] = Field(..., description="List of episodes to ingest")
    enable_ner: bool = Field(True, description="Enable NER enrichment")


class SearchRequest(BaseModel):
    """Request to search the knowledge graph."""
    query: str = Field(..., description="Search query", min_length=1)
    method: SearchMethod = Field(
        SearchMethod.question_aware,
        description="Search method to use"
    )
    limit: Optional[int] = Field(None, description="Maximum number of results", ge=1, le=100)


class EntityInfo(BaseModel):
    """Entity information."""
    name: str
    type: Optional[str] = None
    labels: List[str] = []
    uuid: Optional[str] = None


class ConnectionInfo(BaseModel):
    """Connection information."""
    count: int
    entities: List[EntityInfo]
    relationship_types: List[str]


class ScoringInfo(BaseModel):
    """Scoring breakdown."""
    final_score: float
    original_score: float
    connection_score: Optional[float] = None
    temporal_score: Optional[float] = None
    query_match_score: Optional[float] = None
    entity_type_score: Optional[float] = None


class SearchResult(BaseModel):
    """Single search result."""
    rank: int
    uuid: str
    name: str
    summary: str
    labels: List[str]
    attributes: Dict[str, Any]
    scoring: Optional[ScoringInfo] = None
    connections: Optional[ConnectionInfo] = None
    recognized_entities: Optional[Dict[str, List[str]]] = None


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    method: str
    total_results: int
    results: List[SearchResult]
    timestamp: str


class StatusResponse(BaseModel):
    """Status response."""
    status: str
    message: str
    database_stats: Optional[Dict[str, int]] = None


class ConfigResponse(BaseModel):
    """Configuration response."""
    neo4j_uri: str
    llm_provider: str
    llm_model: str
    embedder_model: str
    search_method: str
    search_limit: int
    ner_enabled: bool


# ============================================================================
# Startup and Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Graphiti on startup."""
    global graphiti_instance, enricher_instance, detector_instance
    
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

