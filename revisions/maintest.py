import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://10.147.18.253:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

# OpenAI API key (required by Graphiti)
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Episodes list containing both text and JSON episodes
episodes = [
    {
        'content': 'Kamala Harris is the Attorney General of California. She was previously '
        'the district attorney for San Francisco.',
        'type': EpisodeType.text,
        'description': 'podcast transcript',
    },
    {
        'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
        'type': EpisodeType.text,
        'description': 'podcast transcript',
    },
    {
        'content': {
            'name': 'Gavin Newsom',
            'position': 'Governor',
            'state': 'California',
            'previous_role': 'Lieutenant Governor',
            'previous_location': 'San Francisco',
        },
        'type': EpisodeType.json,
        'description': 'podcast metadata',
    },
    {
        'content': {
            'name': 'Gavin Newsom',
            'position': 'Governor',
            'term_start': 'January 7, 2019',
            'term_end': 'Present',
        },
        'type': EpisodeType.json,
        'description': 'podcast metadata',
    },
]


neo4j_uri = "bolt://10.147.18.253:7687"
neo4j_user = "neo4j"
neo4j_password = "password"
if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

if not openai_api_key:
    raise ValueError('OPENAI_API_KEY must be set')

async def get_node_connections(graphiti, node_uuid):
    """Get the number of connections (edges) for a given node."""
    query = """
    MATCH (n {uuid: $uuid})-[r]-()
    RETURN count(r) as connection_count
    """
    result = await graphiti.driver.execute_query(query, uuid=node_uuid)
    if result and len(result) > 0:
        return result[0].get('connection_count', 0)
    return 0

async def search_central_node(graphiti, query):
    # Use the top search result's UUID as the center node for reranking
    results = await graphiti.search(query)
    if results and len(results) > 0:
        return results[0].source_node_uuid
    return None

async def search_episodes_with_custom_ranking(graphiti, query, connection_weight=0.3):
    """
    Enhanced search with custom ranking that weighs entities by their connections.
    
    Args:
        graphiti: Graphiti instance
        query: Search query string
        connection_weight: Weight factor for connection count (0.0 to 1.0)
                          Higher values give more importance to well-connected nodes
    """
    print('\nPerforming enhanced search with connection-based ranking:')
    print(f'Connection weight factor: {connection_weight}')
    
    # Use a predefined search configuration recipe and modify its limit
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = 20  # Get more results for reranking
    
    # Execute the node search
    node_search_results = await graphiti._search(
        query=query,
        config=node_search_config,
    )
    
    # Enhance results with connection counts and rerank
    enhanced_results = []
    max_connections = 0
    
    print('\nFetching connection counts for each node...')
    for node in node_search_results.nodes:
        # Get connection count for this node
        connection_count = await get_node_connections(graphiti, node.uuid)
        max_connections = max(max_connections, connection_count)
        
        enhanced_results.append({
            'node': node,
            'connection_count': connection_count,
            'original_score': getattr(node, 'score', 1.0)  # Original search score
        })
    
    # Normalize and calculate final scores
    for result in enhanced_results:
        # Normalize connection count (0 to 1)
        normalized_connections = (
            result['connection_count'] / max_connections 
            if max_connections > 0 else 0
        )
        
        # Calculate weighted score
        # final_score = (1 - weight) * original_score + weight * connection_score
        result['connection_score'] = normalized_connections
        result['final_score'] = (
            (1 - connection_weight) * result['original_score'] + 
            connection_weight * normalized_connections
        )
    
    # Sort by final score (descending)
    enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Print top results
    print('\n=== Enhanced Search Results (Top 5) ===')
    for i, result in enumerate(enhanced_results[:5], 1):
        node = result['node']
        print(f'\n[Rank {i}]')
        print(f'Node UUID: {node.uuid}')
        print(f'Node Name: {node.name}')
        node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
        print(f'Content Summary: {node_summary}')
        print(f"Node Labels: {', '.join(node.labels)}")
        print(f'Connections: {result["connection_count"]}')
        print(f'Original Score: {result["original_score"]:.4f}')
        print(f'Connection Score: {result["connection_score"]:.4f}')
        print(f'Final Score: {result["final_score"]:.4f}')
        if hasattr(node, 'attributes') and node.attributes:
            print('Attributes:')
            for key, value in node.attributes.items():
                print(f'  {key}: {value}')
        print('---')
    
    return enhanced_results

async def search_episodes(graphiti, query):
    # Example: Perform a node search using _search method with standard recipes
    print(
        '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
    )

    # Use a predefined search configuration recipe and modify its limit
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = 5  # Limit to 5 results

    # Execute the node search
    node_search_results = await graphiti._search(
        query=query,
        config=node_search_config,
    )

    # Print node search results
    print('\nNode Search Results:')
    for node in node_search_results.nodes:
        print(f'Node UUID: {node.uuid}')
        print(f'Node Name: {node.name}')
        node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
        print(f'Content Summary: {node_summary}')
        print(f"Node Labels: {', '.join(node.labels)}")
        print(f'Created At: {node.created_at}')
        if hasattr(node, 'attributes') and node.attributes:
            print('Attributes:')
            for key, value in node.attributes.items():
                print(f'  {key}: {value}')
        print('---')
    return node_search_results

async def main():
    # Initialize Graphiti with Neo4j connection
    #graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    # Configure Ollama LLM client
    llm_config = LLMConfig(
        api_key="ollama",  # Ollama doesn't require a real API key, but some placeholder is needed
        model="deepseek-r1:7b",
        small_model="deepseek-r1:7b",
        base_url="http://10.147.18.253:11434/v1",  # Ollama's OpenAI-compatible endpoint
    )

    llm_client = OpenAIGenericClient(config=llm_config)

    # Initialize Graphiti with Ollama clients
    graphiti = Graphiti(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        llm_client=llm_client,
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="ollama",  # Placeholder API key
                embedding_model="nomic-embed-text",
                embedding_dim=768,
                base_url="http://10.147.18.253:11434/v1",
            )
        ),
        cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
    )

    ADD = False
    if ADD:
        try:
            # Initialize the graph database with graphiti's indices. This only needs to be done once.
            await graphiti.build_indices_and_constraints()
            
            # Add episodes to the graph
            for i, episode in enumerate(episodes):
                await graphiti.add_episode(
                    name=f'Freakonomics Radio {i}',
                    episode_body=episode['content']
                    if isinstance(episode['content'], str)
                    else json.dumps(episode['content']),
                    source=episode['type'],
                    source_description=episode['description'],
                    reference_time=datetime.now(timezone.utc),
                )
                print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

            
        finally:
            # Close the connection
            await graphiti.close()
            print('\nConnection closed')
    else:
        q = "Who was the California Attorney General in 2020?"
        #q = "Who is Gavin Newsom?"
        
        # Compare standard search vs enhanced ranking
        print('\n' + '='*80)
        print('STANDARD SEARCH (RRF only)')
        print('='*80)
        await search_episodes(graphiti, q)
        
        print('\n' + '='*80)
        print('ENHANCED SEARCH (RRF + Connection-based Ranking)')
        print('='*80)
        # Try different weight values: 0.3 = 30% weight on connections
        await search_episodes_with_custom_ranking(graphiti, q, connection_weight=0.3)
        
#        center_node_uuid = await search_central_node(graphiti, q)
#        print(f'Center node UUID: {center_node_uuid}')
if __name__ == '__main__':
    asyncio.run(main())
