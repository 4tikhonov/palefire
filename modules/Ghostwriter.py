"""
Ghostwriter Module for Palefire

This module ports the Ghostwriter Agent's functionality to Palefire.
It provides capabilities for:
1. Ingesting content from URLs into a Qdrant vector database.
2. Managing collections of knowledge.
3. Performing RAG-based Q&A using the stored knowledge.
"""

import os
import sys
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant and Sentence Transformers
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    QDRANT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Ghostwriter dependencies: {e}")
    QDRANT_AVAILABLE = False
except Exception as e:
    logger.error(f"Unexpected error importing Ghostwriter dependencies: {e}")
    QDRANT_AVAILABLE = False
    
# Palefire config
import config

# Parsing utilities
from agents.parsers import URLParser

# Graphiti Core
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

class GhostwriterSkill:
    """
    Implements the Ghostwriter Agent's skills within Palefire.
    """
    
    def __init__(self, device: str = None):
        if not QDRANT_AVAILABLE:
            raise ImportError("Ghostwriter requires qdrant-client and sentence-transformers. Please install them.")
            
        # Initialize Qdrant Client
        # Check if QDRANT_HOST is set (e.g. from Docker)
        self.qdrant_host = os.getenv("QDRANT_HOST")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        
        if self.qdrant_host:
            logger.info(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        else:
            # Fallback to local on-disk storage
            logger.info("Using local Qdrant storage at ./qdrant_storage")
            self.client = QdrantClient(path="./qdrant_storage")
        
        # Initialize Embedding Model
        # 'all-MiniLM-L6-v2' is a good balance of speed and performance
        self.model_name = "all-MiniLM-L6-v2"
        # Determine device (prioritize argument, then env var, then auto)
        if not device:
            device = os.getenv("GHOSTWRITER_DEVICE", None)
            
        logger.info(f"Initializing Ghostwriter with device: {device or 'auto'}")
        self.encoder = SentenceTransformer(self.model_name, device=device)
        self.vector_size = 384
        
        # Default collection
        self.default_collection = "palefire_knowledge"
        self._ensure_collection(self.default_collection)
        
        # LLM Client (reuse Palefire's configuration)
        self.llm_client = self._get_llm_client()

    def _get_llm_client(self):
        """Helper to get an LLM client based on Palefire config."""
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        
        llm_cfg = config.get_llm_config()
        logger.info(f"Initializing LLM Client with base_url: {llm_cfg['base_url']}")
        llm_config = LLMConfig(
            api_key=llm_cfg['api_key'],
            model=llm_cfg['model'],
            base_url=llm_cfg['base_url'],
        )
        return OpenAIGenericClient(config=llm_config)

    def _ensure_collection(self, collection_name: str):
        """Ensures that the specified collection exists."""
        try:
            self.client.get_collection(collection_name)
        except Exception:
            logger.info(f"Collection {collection_name} not found, creating...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def ingest_url(self, url: str, collection_name: str = None) -> Dict[str, Any]:
        """
        Ingests content from a URL.
        
        Args:
            url: The URL to ingest.
            collection_name: Optional collection name.
            
        Returns:
            Dictionary with ingestion result.
        """
        target_collection = collection_name or self.default_collection
        self._ensure_collection(target_collection)
        
        print(f"Fetching content from {url}...")
        parser = URLParser()
        result = parser.parse(url)
        
        if not result.success:
            return {"status": "error", "message": f"Failed to parse URL: {result.error}"}
            
        text = result.text
        if not text:
            return {"status": "error", "message": "No text content found at URL."}
            
        # Chunking (simplified)
        chunks = self._chunk_text(text)
        print(f"Split into {len(chunks)} chunks. Embedding...")
        
        points = []
        for i, chunk in enumerate(chunks):
            vector = self.encoder.encode(chunk).tolist()
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "source": url,
                    "content": chunk,
                    "timestamp": datetime.now().isoformat(),
                    "title": result.metadata.get('title', 'Unknown')
                }
            ))
            
        self.client.upsert(
            collection_name=target_collection,
            points=points
        )
        
        return {
            "status": "success",
            "message": f"Successfully ingested {len(chunks)} chunks from {url}",
            "chunks_count": len(chunks)
        }

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Simple text chunking."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    async def ask_question(self, question: str, collection_name: str = None, limit: int = 5) -> Dict[str, Any]:
        """
        Asks a question using RAG.
        
        Args:
            question: The user's question.
            collection_name: Optional collection name.
            limit: Number of context chunks to retrieve.
            
        Returns:
            Answer and sources.
        """
        target_collection = collection_name or self.default_collection
        
        # 1. Retrieve relevant context
        query_vector = self.encoder.encode(question).tolist()
        if hasattr(self.client, 'search'):
            search_result = self.client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                limit=limit
            )
        else:
            # Fallback for newer clients or local mode
            search_result = self.client.query_points(
                collection_name=target_collection,
                query=query_vector,
                limit=limit
            ).points
        
        context_texts = []
        sources = []
        
        for hit in search_result:
            if hit.payload:
                content = hit.payload.get('content')
                source = hit.payload.get('source')
                
                if content:
                    context_texts.append(content)
                if source:
                    sources.append(source)
        
        sources = list(set(sources))
        
        context_str = "\n\n---\n\n".join(context_texts)
        
        if not context_texts:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "sources": []
            }
            
        # 2. Generate Answer using LLM
        # 2. Generate Answer using LLM
        prompt = f"""You are a helpful research assistant. Answer the user's question based ONLY on the following context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context_str}

Question: {question}

Return your answer as a JSON object with a single key "answer".
Example: {{"answer": "The answer is..."}}"""

        try:
            # Use generate_response which expects a list of Message objects
            # and returns a JSON dictionary (as enforced by the client)
            messages = [Message(role="user", content=prompt)]
            response = await self.llm_client.generate_response(messages)
            
            # Extract answer from JSON response
            answer_text = response.get("answer", "No answer found in response.")
            
            return {
                "answer": answer_text,
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources
            }

    def search(self, query: str, collection_name: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Semantic search."""
        target_collection = collection_name or self.default_collection
        query_vector = self.encoder.encode(query).tolist()
        
        if hasattr(self.client, 'search'):
            results = self.client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                limit=limit
            )
        else:
            results = self.client.query_points(
                collection_name=target_collection,
                query=query_vector,
                limit=limit
            ).points
        
        return [
            {
                "score": hit.score,
                "content": hit.payload.get('content', '') if hit.payload else '',
                "source": hit.payload.get('source', 'Unknown') if hit.payload else 'Unknown',
                "title": hit.payload.get('title', 'Unknown') if hit.payload else 'Unknown'
            }
            for hit in results
        ]

    def list_collections(self) -> List[str]:
        """Lists available collections."""
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception:
            return []
