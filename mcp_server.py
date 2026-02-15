from mcp.server.fastmcp import FastMCP
from typing import List, Optional
import sys
import os
import logging
from modules.Ghostwriter import GhostwriterSkill

# Configure logging
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("palefire.mcp")

# Initialize FastMCP server
mcp = FastMCP("Palefire")

# Check for --cpu flag
if "--cpu" in sys.argv:
    os.environ["GHOSTWRITER_DEVICE"] = "cpu"
    sys.argv.remove("--cpu")


# Initialize Ghostwriter skills
# We need to do this lazily or handle potential errors if DB isn't ready immediately.
try:
    skills = GhostwriterSkill()
except Exception as e:
    sys.stderr.write(f"Error initializing Ghostwriter Skills: {e}\n")
    # Don't exit here, allows server to start even if skills fail initially (though tools will fail)
    skills = None

@mcp.tool()
def ingest_url(url: str, collection_name: str = None) -> str:
    """
    Ingest content from a URL into the knowledge base.
    
    Args:
        url: The URL to ingest.
        collection_name: Optional name of the collection to store content in. 
                         If not provided, uses the default collection.
    """
    if not skills:
        return "Error: Ghostwriter skills not initialized."
        
    try:
        # GhostwriterSkill.ingest_url returns a dict with status and message
        # e.g. {'status': 'success', 'message': '...', 'chunks_count': N}
        result = skills.ingest_url(url, collection_name=collection_name)
        if result["status"] == "success":
            chunks = result.get('chunks_count', 'N/A')
            return f"Successfully ingested {url}. Chunks created: {chunks}"
        else:
            return f"Error ingesting URL: {result['message']}"
    except Exception as e:
        return f"Exception during ingestion: {str(e)}"

@mcp.tool()
async def ask_question(question: str, collection_name: str = None, limit: int = 5) -> str:
    """
    Ask a question about the content stored in the knowledge base (RAG).
    
    Args:
        question: The question to ask.
        collection_name: Optional name of the collection to query.
        limit: Maximum number of context chunks to use (default: 5).
    """
    if not skills:
        return "Error: Ghostwriter skills not initialized."
        
    try:
        # GhostwriterSkill.ask_question returns a dict with answer and sources
        result = await skills.ask_question(question, collection_name=collection_name, limit=limit)
        
        answer = result.get('answer', 'No answer generated.')
        sources = result.get('sources', [])
        
        response = str(answer)
        if sources:
            response += "\n\nSources:"
            for source in sources:
                response += f"\n- {source}"
        return response
    except Exception as e:
        return f"Exception during asking: {str(e)}"

@mcp.tool()
def search_content(query: str, collection_name: str = None, limit: int = 5) -> str:
    """
    Search for relevant content in the knowledge base without generating an answer.
    
    Args:
        query: The search query.
        collection_name: Optional name of the collection to search.
        limit: Maximum number of results to return (default: 5).
    """
    if not skills:
        return "Error: Ghostwriter skills not initialized."

    try:
        # GhostwriterSkill.search returns a list of dicts with score, content, source
        results = skills.search(query, collection_name=collection_name, limit=limit)
        
        if not results:
            return "No results found."
            
        output = f"Search Results for '{query}':\n"
        for res in results:
            score = res.get('score', 0)
            source = res.get('source', 'Unknown')
            # Truncate content for cleaner output
            content = res.get('content', '')[:200].replace('\n', ' ') + "..."
            output += f"- [{score:.1%}] {source}: {content}\n"
            
        return output
    except Exception as e:
        return f"Exception during search: {str(e)}"

@mcp.tool()
def list_collections() -> str:
    """
    List all available collections in the vector database.
    """
    if not skills:
        return "Error: Ghostwriter skills not initialized."

    try:
        colls = skills.list_collections()
        if not colls:
            return "No collections found."
            
        output = "Available Collections:\n"
        for c in colls:
            output += f"- {c}\n"
        return output
    except Exception as e:
        return f"Exception listing collections: {str(e)}"

if __name__ == "__main__":
    # If running directly, start the MCP server
    mcp.run()
