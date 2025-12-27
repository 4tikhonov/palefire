"""
Pale Fire Utility Functions

Core utility functions for knowledge graph search, ranking, and data management.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from modules import EntityEnricher, QuestionTypeDetector

logger = logging.getLogger(__name__)


def debug_print(debug: bool, *args, **kwargs):
    """Print only if debug is True."""
    if debug:
        print(*args, **kwargs)


async def get_node_connections_with_entities(graphiti, node_uuid):
    """
    Get the number of connections and connected entity names for a given node.
    Returns dict with count, entities (with types), and relationship_types.
    """
    query = """
    MATCH (n {uuid: $uuid})-[r]-(connected)
    RETURN 
        count(r) as connection_count,
        collect(DISTINCT {
            name: connected.name, 
            labels: labels(connected),
            uuid: connected.uuid
        }) as connected_entities,
        collect(DISTINCT type(r)) as relationship_types
    """
    try:
        async with graphiti.driver.session() as session:
            result = await session.run(query, uuid=node_uuid)
            record = await result.single()
            if record:
                # Process entities to extract entity types from labels
                entities = []
                entity_names = []  # For backward compatibility
                
                for entity_data in record['connected_entities']:
                    name = entity_data['name']
                    labels = entity_data.get('labels', [])
                    
                    # Extract entity type from labels (PER, LOC, ORG, etc.)
                    entity_type = None
                    entity_types = ['PER', 'LOC', 'ORG', 'DATE', 'TIME', 'MONEY', 'PERCENT', 
                                   'GPE', 'NORP', 'FAC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 
                                   'LAW', 'LANGUAGE', 'QUANTITY', 'ORDINAL', 'CARDINAL']
                    
                    for label in labels:
                        if label in entity_types:
                            entity_type = label
                            break
                    
                    entities.append({
                        'name': name,
                        'type': entity_type,
                        'labels': labels,
                        'uuid': entity_data.get('uuid')
                    })
                    entity_names.append(name)
                
                return {
                    'count': record['connection_count'],
                    'entities': entities,
                    'entity_names': entity_names,  # For backward compatibility
                    'relationship_types': record['relationship_types']
                }
        return {'count': 0, 'entities': [], 'entity_names': [], 'relationship_types': []}
    except Exception as e:
        logger.warning(f"Error getting connections for node {node_uuid}: {e}")
        return {'count': 0, 'entities': [], 'entity_names': [], 'relationship_types': []}


async def get_node_connections(graphiti, node_uuid):
    """Get the number of connections (edges) for a given node."""
    result = await get_node_connections_with_entities(graphiti, node_uuid)
    return result['count']


async def extract_temporal_info(graphiti, node_uuid):
    """
    Extract temporal information from node attributes and related episodes.
    Returns dict with temporal data.
    """
    query = """
    MATCH (n {uuid: $uuid})
    OPTIONAL MATCH (n)-[:PART_OF]-(episode)
    RETURN 
        n.created_at as created_at,
        n.valid_at as valid_at,
        n.invalid_at as invalid_at,
        collect(DISTINCT episode.valid_at) as episode_dates,
        properties(n) as node_properties
    """
    try:
        async with graphiti.driver.session() as session:
            result = await session.run(query, uuid=node_uuid)
            record = await result.single()
            if record:
                return {
                    'created_at': record['created_at'],
                    'valid_at': record['valid_at'],
                    'invalid_at': record['invalid_at'],
                    'episode_dates': record['episode_dates'],
                    'properties': record['node_properties']
                }
        return None
    except Exception as e:
        logger.warning(f"Error getting temporal info for node {node_uuid}: {e}")
        return None


def calculate_temporal_relevance(node, temporal_info, query_year=None):
    """
    Calculate temporal relevance score based on query date and node's temporal attributes.
    Returns score between 0 and 1.
    """
    if not query_year or not temporal_info:
        return 1.0  # No temporal filtering
    
    try:
        # Check node properties for date-related fields
        properties = temporal_info.get('properties', {})
        
        # Look for common date fields in properties
        date_fields = ['term_start', 'term_end', 'start_date', 'end_date', 'year', 'date']
        
        for field in date_fields:
            if field in properties:
                value = str(properties[field])
                # Extract year from various date formats
                if str(query_year) in value:
                    return 1.0  # Perfect match
                
                # Check if it's a range (e.g., "2011-2017")
                if '-' in value and len(value) < 20:
                    try:
                        parts = value.split('-')
                        if len(parts) >= 2:
                            start_year = int(''.join(filter(str.isdigit, parts[0]))[:4])
                            end_year = int(''.join(filter(str.isdigit, parts[1]))[:4])
                            if start_year <= query_year <= end_year:
                                return 1.0  # Within range
                    except:
                        pass
        
        # Check valid_at and invalid_at timestamps
        valid_at = temporal_info.get('valid_at')
        invalid_at = temporal_info.get('invalid_at')
        
        if valid_at:
            try:
                valid_date = datetime.fromisoformat(str(valid_at).replace('Z', '+00:00'))
                valid_year = valid_date.year
                
                if invalid_at:
                    invalid_date = datetime.fromisoformat(str(invalid_at).replace('Z', '+00:00'))
                    invalid_year = invalid_date.year
                    
                    if valid_year <= query_year <= invalid_year:
                        return 1.0  # Within validity period
                    else:
                        return 0.3  # Outside validity period
                else:
                    # No end date, check if started before query year
                    if valid_year <= query_year:
                        return 0.8  # Started before, might still be valid
            except:
                pass
        
        # Default: slight penalty for no temporal match
        return 0.5
        
    except Exception as e:
        logger.warning(f"Error calculating temporal relevance: {e}")
        return 1.0


def extract_query_terms(query):
    """
    Extract important terms and entities from the query.
    Returns dict with query analysis.
    """
    # Remove common stop words
    stop_words = {
        'who', 'what', 'when', 'where', 'why', 'how', 'is', 'was', 'were', 'are',
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'as', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can'
    }
    
    # Extract year
    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    query_year = int(year_match.group()) if year_match else None
    
    # Tokenize and clean
    tokens = re.findall(r'\b[a-zA-Z]+\b', query.lower())
    important_terms = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # Extract potential proper nouns (capitalized words in original query)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    
    return {
        'query_year': query_year,
        'important_terms': important_terms,
        'proper_nouns': proper_nouns,
        'original_query': query.lower()
    }


def calculate_query_match_score(node, connection_info, query_terms):
    """
    Calculate how well a node matches the specific query terms.
    Checks node name, summary, connected entities, and attributes.
    Returns score between 0 and 1.
    """
    score = 0.0
    max_score = 0.0
    
    query_lower = query_terms['original_query']
    important_terms = query_terms['important_terms']
    proper_nouns = query_terms['proper_nouns']
    
    # 1. Check if node name matches query terms (weight: 3.0)
    node_name_lower = node.name.lower()
    max_score += 3.0
    
    # Exact match in query
    if node_name_lower in query_lower:
        score += 3.0
    # Partial match of important terms
    else:
        term_matches = sum(1 for term in important_terms if term in node_name_lower)
        score += (term_matches / max(len(important_terms), 1)) * 3.0
    
    # 2. Check if node name matches proper nouns (weight: 2.0)
    max_score += 2.0
    for proper_noun in proper_nouns:
        if proper_noun.lower() in node_name_lower or node_name_lower in proper_noun.lower():
            score += 2.0
            break
    
    # 3. Check node summary for query terms (weight: 1.5)
    max_score += 1.5
    if hasattr(node, 'summary') and node.summary:
        summary_lower = node.summary.lower()
        term_matches = sum(1 for term in important_terms if term in summary_lower)
        score += (term_matches / max(len(important_terms), 1)) * 1.5
    
    # 4. Check connected entities for query terms (weight: 1.0)
    max_score += 1.0
    if connection_info.get('entities'):
        # Handle both old format (list of strings) and new format (list of dicts)
        connected_names = []
        for entity in connection_info['entities']:
            if isinstance(entity, dict):
                connected_names.append(entity['name'])
            else:
                connected_names.append(str(entity))
        connected_text = ' '.join(connected_names).lower()
        term_matches = sum(1 for term in important_terms if term in connected_text)
        score += (term_matches / max(len(important_terms), 1)) * 1.0
    
    # 5. Check node attributes for query terms (weight: 1.0)
    max_score += 1.0
    if hasattr(node, 'attributes') and node.attributes:
        attr_text = ' '.join(str(v).lower() for v in node.attributes.values())
        term_matches = sum(1 for term in important_terms if term in attr_text)
        score += (term_matches / max(len(important_terms), 1)) * 1.0
    
    # 6. Check node labels for query terms (weight: 0.5)
    max_score += 0.5
    if hasattr(node, 'labels') and node.labels:
        labels_text = ' '.join(node.labels).lower()
        term_matches = sum(1 for term in important_terms if term in labels_text)
        score += (term_matches / max(len(important_terms), 1)) * 0.5
    
    # Normalize to 0-1 range
    normalized_score = score / max_score if max_score > 0 else 0.0
    
    return normalized_score


async def search_central_node(graphiti, query):
    """Use the top search result's UUID as the center node for reranking."""
    results = await graphiti.search(query)
    if results and len(results) > 0:
        return results[0].source_node_uuid
    return None


async def search_episodes_with_custom_ranking(graphiti, query, connection_weight=None, debug: bool = False):
    """
    Enhanced search with custom ranking that weighs entities by their connections.
    
    Args:
        graphiti: Graphiti instance
        query: Search query string
        connection_weight: Weight factor for connection count (0.0 to 1.0)
                          Higher values give more importance to well-connected nodes
        debug: If True, enable verbose debug output
    """
    if connection_weight is None:
        connection_weight = config.WEIGHT_CONNECTION
    
    debug_print(debug, '\nPerforming enhanced search with connection-based ranking:')
    debug_print(debug, f'Connection weight factor: {connection_weight}')
    
    # Use a predefined search configuration recipe and modify its limit
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = config.SEARCH_RESULT_LIMIT  # Get more results for reranking
    
    # Execute the node search
    node_search_results = await graphiti._search(
        query=query,
        config=node_search_config,
    )
    
    # Enhance results with connection counts and rerank
    enhanced_results = []
    max_connections = 0
    
    debug_print(debug, '\nFetching connection counts for each node...')
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
    debug_print(debug, '\n=== Enhanced Search Results (Top 5) ===')
    for i, result in enumerate(enhanced_results[:5], 1):
        node = result['node']
        debug_print(debug, f'\n[Rank {i}]')
        debug_print(debug, f'Node UUID: {node.uuid}')
        debug_print(debug, f'Node Name: {node.name}')
        node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
        debug_print(debug, f'Content Summary: {node_summary}')
        debug_print(debug, f"Node Labels: {', '.join(node.labels)}")
        debug_print(debug, f'Connections: {result["connection_count"]}')
        debug_print(debug, f'Original Score: {result["original_score"]:.4f}')
        debug_print(debug, f'Connection Score: {result["connection_score"]:.4f}')
        debug_print(debug, f'Final Score: {result["final_score"]:.4f}')
        if hasattr(node, 'attributes') and node.attributes:
            debug_print(debug, 'Attributes:')
            for key, value in node.attributes.items():
                debug_print(debug, f'  {key}: {value}')
        debug_print(debug, '---')
    
    return enhanced_results


async def search_episodes(graphiti, query, debug: bool = False):
    """Standard search using RRF hybrid search."""
    debug_print(debug,
        '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
    )
    logger.info(f"Standard search with query: '{query}'")
    debug_print(debug, f'🔎 Query: "{query}"')

    # Use a predefined search configuration recipe and modify its limit
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = config.SEARCH_TOP_K  # Limit to configured top K results

    # Execute the node search
    node_search_results = await graphiti._search(
        query=query,
        config=node_search_config,
    )
    debug_print(debug, f'   Found {len(node_search_results.nodes)} results')

    # Print node search results
    debug_print(debug, '\nNode Search Results:')
    for node in node_search_results.nodes:
        debug_print(debug, f'Node UUID: {node.uuid}')
        debug_print(debug, f'Node Name: {node.name}')
        node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
        debug_print(debug, f'Content Summary: {node_summary}')
        debug_print(debug, f"Node Labels: {', '.join(node.labels)}")
        debug_print(debug, f'Created At: {node.created_at}')
        if hasattr(node, 'attributes') and node.attributes:
            debug_print(debug, 'Attributes:')
            for key, value in node.attributes.items():
                debug_print(debug, f'  {key}: {value}')
        debug_print(debug, '---')
    return node_search_results


async def search_episodes_with_question_aware_ranking(
    graphiti,
    query,
    enricher=None,
    connection_weight=None,
    temporal_weight=None,
    query_match_weight=None,
    entity_type_weight=None,
    query_year=None,
    debug: bool = False
):
    """
    INTELLIGENT search with question-type detection and entity-type weighting.
    
    Combines 5 factors:
    1. Semantic relevance (RRF score from hybrid search)
    2. Connection count (how well-connected the entity is)
    3. Temporal relevance (does it match the time period)
    4. Query term matching (how well it matches specific query terms)
    5. Entity type matching (entity types relevant to question type)
    """
    # Use config defaults if not specified
    if connection_weight is None:
        connection_weight = config.WEIGHT_CONNECTION
    if temporal_weight is None:
        temporal_weight = config.WEIGHT_TEMPORAL
    if query_match_weight is None:
        query_match_weight = config.WEIGHT_QUERY_MATCH
    if entity_type_weight is None:
        entity_type_weight = config.WEIGHT_ENTITY_TYPE
    
    # Initialize question type detector
    q_detector = QuestionTypeDetector()
    
    # Detect question type
    question_info = q_detector.detect_question_type(query)
    
    # Auto-detect year in query if not provided
    if query_year is None:
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            query_year = int(year_match.group())
    
    # Extract query terms for matching
    query_terms = extract_query_terms(query)
    
    # Calculate semantic weight (remaining weight after other factors)
    semantic_weight = 1.0 - connection_weight - temporal_weight - query_match_weight - entity_type_weight
    
    debug_print(debug, '\n🧠 Performing QUESTION-AWARE search with intelligent ranking:')
    debug_print(debug, f'  Question Type: {question_info["type"]} - {question_info["description"]}')
    debug_print(debug, f'  Confidence: {question_info["confidence"]:.2f}')
    if question_info['entity_weights']:
        debug_print(debug, f'  Entity Type Preferences: {", ".join(f"{k}={v:.1f}x" for k, v in list(question_info["entity_weights"].items())[:5])}')
    debug_print(debug, f'\n  Weight Distribution:')
    debug_print(debug, f'    Semantic: {semantic_weight:.2f} (RRF hybrid search)')
    debug_print(debug, f'    Connection: {connection_weight:.2f} (graph connectivity)')
    debug_print(debug, f'    Temporal: {temporal_weight:.2f} (time period match)')
    debug_print(debug, f'    Query Match: {query_match_weight:.2f} (term matching)')
    debug_print(debug, f'    Entity Type: {entity_type_weight:.2f} (question-type alignment)')
    if query_year:
        debug_print(debug, f'  Query year: {query_year}')
    
    # Use a predefined search configuration recipe and modify its limit
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = config.SEARCH_RESULT_LIMIT  # Get more results for reranking
    
    # Execute the node search
    logger.info(f"Executing search with query: '{query}'")
    debug_print(debug, f'\n🔎 Searching for: "{query}"')
    node_search_results = await graphiti._search(
        query=query,
        config=node_search_config,
    )
    debug_print(debug, f'   Found {len(node_search_results.nodes)} initial results from hybrid search')
    
    # Enhance results with all factors and rerank
    enhanced_results = []
    max_connections = 0
    
    # Initialize enricher if not provided
    if enricher is None:
        enricher = EntityEnricher(use_spacy=config.NER_USE_SPACY)
    
    debug_print(debug, '\nAnalyzing nodes with question-aware scoring...')
    for node in node_search_results.nodes:
        # Get connection details
        connection_info = await get_node_connections_with_entities(graphiti, node.uuid)
        max_connections = max(max_connections, connection_info['count'])
        
        # Get temporal information
        temporal_info = await extract_temporal_info(graphiti, node.uuid)
        temporal_score = calculate_temporal_relevance(node, temporal_info, query_year)
        
        # Calculate query term matching score
        query_match_score = calculate_query_match_score(node, connection_info, query_terms)
        
        # Extract entities from node content for entity type matching
        node_text = f"{node.name} {node.summary}"
        if hasattr(node, 'attributes') and node.attributes:
            node_text += " " + " ".join(str(v) for v in node.attributes.values())
        
        # Create a pseudo-episode for entity extraction
        pseudo_episode = {'content': node_text, 'type': 'text', 'description': 'node'}
        enriched_node = enricher.enrich_episode(pseudo_episode)
        
        # Calculate entity type matching score
        entity_type_score = q_detector.apply_entity_type_weights(
            node, enriched_node, question_info['entity_weights']
        )
        
        enhanced_results.append({
            'node': node,
            'connection_count': connection_info['count'],
            'connected_entities': connection_info['entities'],
            'relationship_types': connection_info['relationship_types'],
            'temporal_info': temporal_info,
            'temporal_score': temporal_score,
            'query_match_score': query_match_score,
            'entity_type_score': entity_type_score,
            'enriched_node': enriched_node,
            'original_score': getattr(node, 'score', 1.0)
        })
    
    # Normalize and calculate final scores
    for result in enhanced_results:
        # Normalize connection count (0 to 1)
        normalized_connections = (
            result['connection_count'] / max_connections 
            if max_connections > 0 else 0
        )
        
        # Keep entity type score as-is (0.5-2.0) to preserve the 2x boost effect
        # This allows nodes with matching entity types to get up to 2x boost
        entity_type_multiplier = result['entity_type_score']
        
        # Calculate weighted final score with all 5 factors
        # Entity type acts as a multiplier on the base score
        result['connection_score'] = normalized_connections
        result['entity_type_multiplier'] = entity_type_multiplier
        
        base_score = (
            semantic_weight * result['original_score'] + 
            connection_weight * normalized_connections +
            temporal_weight * result['temporal_score'] +
            query_match_weight * result['query_match_score']
        )
        
        # Apply entity type as a multiplier to amplify matching results
        result['final_score'] = base_score * (1.0 + entity_type_weight * (entity_type_multiplier - 1.0))
    
    # Sort by final score (descending)
    enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Print top results
    debug_print(debug, '\n' + '='*80)
    debug_print(debug, f'🏆 QUESTION-AWARE RANKING RESULTS (Top {config.SEARCH_TOP_K})')
    debug_print(debug, '='*80)
    
    for i, result in enumerate(enhanced_results[:config.SEARCH_TOP_K], 1):
        node = result['node']
        debug_print(debug, f'\n[Rank {i}] {node.name}')
        debug_print(debug, f'UUID: {node.uuid}')
        node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
        debug_print(debug, f'Summary: {node_summary}')
        
        # Display extracted entity types
        entities_by_type = result['enriched_node'].get('entities_by_type', {})
        if entities_by_type:
            debug_print(debug, f'\n🏷️  Entity Types in Node:')
            for ent_type, ents in list(entities_by_type.items())[:5]:
                weight = question_info['entity_weights'].get(ent_type, 1.0)
                indicator = '⭐' if weight > 1.2 else ''
                debug_print(debug, f'    {ent_type}: {", ".join(ents[:3])} {indicator}')
        
        # Display connection information
        debug_print(debug, f'\n📊 Connection Analysis:')
        debug_print(debug, f'  Total: {result["connection_count"]} connections')
        if result['connected_entities']:
            # Format entities with their types
            entity_strs = []
            for entity in result['connected_entities'][:6]:
                if isinstance(entity, dict):
                    name = entity['name']
                    entity_type = entity.get('type')
                    if entity_type:
                        entity_strs.append(f"{name} ({entity_type})")
                    else:
                        entity_strs.append(name)
                else:
                    entity_strs.append(str(entity))
            
            entities_str = ', '.join(entity_strs)
            if len(result['connected_entities']) > 6:
                entities_str += f' ... (+{len(result["connected_entities"]) - 6} more)'
            debug_print(debug, f'  Connected to: {entities_str}')
        
        # Display temporal information
        if result['temporal_info'] and result['temporal_info'].get('properties'):
            debug_print(debug, f'\n🕐 Temporal Data:')
            props = result['temporal_info']['properties']
            date_fields = ['term_start', 'term_end', 'start_date', 'end_date', 'year']
            for field in date_fields:
                if field in props:
                    debug_print(debug, f'  {field}: {props[field]}')
        
        # Display comprehensive scoring
        debug_print(debug, f'\n📈 SCORING BREAKDOWN:')
        debug_print(debug, f'  ├─ Semantic (RRF):     {result["original_score"]:.4f} × {semantic_weight:.2f} = {result["original_score"] * semantic_weight:.4f}')
        debug_print(debug, f'  ├─ Connections:        {result["connection_score"]:.4f} × {connection_weight:.2f} = {result["connection_score"] * connection_weight:.4f}')
        debug_print(debug, f'  ├─ Temporal Match:     {result["temporal_score"]:.4f} × {temporal_weight:.2f} = {result["temporal_score"] * temporal_weight:.4f}')
        debug_print(debug, f'  ├─ Query Term Match:   {result["query_match_score"]:.4f} × {query_match_weight:.2f} = {result["query_match_score"] * query_match_weight:.4f}')
        entity_boost = result["entity_type_multiplier"]
        boost_indicator = f' 🎯 {entity_boost:.1f}x BOOST!' if entity_boost >= 1.5 else ''
        debug_print(debug, f'  ├─ Entity Type Match:  {entity_boost:.4f} multiplier{boost_indicator}')
        debug_print(debug, f'  └─ FINAL SCORE:        {result["final_score"]:.4f}')
        
        debug_print(debug, '─' * 80)
    
    return enhanced_results


def export_results_to_json(results, filepath: str, query: str, method: str, debug: bool = False):
    """
    Export search results to a JSON file.
    
    Args:
        results: Search results (list of dicts or nodes)
        filepath: Path to output JSON file
        query: Original search query
        method: Search method used
        debug: If True, enable verbose debug output
    """
    try:
        # Prepare export data
        export_data = {
            'query': query,
            'method': method,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_results': len(results) if results else 0,
            'results': []
        }
        
        # Convert results to serializable format
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                # Enhanced results from question-aware or custom ranking
                node = result.get('node')
                
                # Filter out name_embedding from attributes
                attributes = {}
                if hasattr(node, 'attributes') and node.attributes:
                    attributes = {k: v for k, v in node.attributes.items() if k != 'name_embedding'}
                
                result_data = {
                    'rank': i,
                    'uuid': node.uuid if node else None,
                    'name': node.name if node else None,
                    'summary': node.summary if node else None,
                    'labels': node.labels if hasattr(node, 'labels') else [],
                    'attributes': attributes,
                    'scoring': {
                        'final_score': result.get('final_score', 0),
                        'original_score': result.get('original_score', 0),
                        'connection_score': result.get('connection_score', 0),
                        'temporal_score': result.get('temporal_score', 0),
                        'query_match_score': result.get('query_match_score', 0),
                        'entity_type_multiplier': result.get('entity_type_multiplier', 1.0),
                    },
                    'connections': {
                        'count': result.get('connection_count', 0),
                        'entities': result.get('connected_entities', []),
                        'relationship_types': result.get('relationship_types', [])
                    },
                    'temporal_info': result.get('temporal_info', {}),
                    'recognized_entities': result.get('enriched_node', {}).get('entities_by_type', {}),
                    'all_entities': result.get('enriched_node', {}).get('entities', [])
                }
            else:
                # Simple node results from standard search
                
                # Filter out name_embedding from attributes
                attributes = {}
                if hasattr(result, 'attributes') and result.attributes:
                    attributes = {k: v for k, v in result.attributes.items() if k != 'name_embedding'}
                
                result_data = {
                    'rank': i,
                    'uuid': result.uuid if hasattr(result, 'uuid') else None,
                    'name': result.name if hasattr(result, 'name') else str(result),
                    'summary': result.summary if hasattr(result, 'summary') else None,
                    'labels': result.labels if hasattr(result, 'labels') else [],
                    'attributes': attributes,
                    'score': getattr(result, 'score', None)
                }
            
            export_data['results'].append(result_data)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        debug_print(debug, f'\n💾 Results exported to: {filepath}')
        debug_print(debug, f'   Total results: {export_data["total_results"]}')
        
    except Exception as e:
        logger.error(f'Error exporting results to JSON: {e}')
        debug_print(debug, f'❌ Failed to export results: {e}')


async def clean_database(graphiti, confirm=False, nodes_only=False, debug: bool = False):
    """
    Clean/clear the Neo4j database.
    
    Args:
        graphiti: Graphiti instance
        confirm: If True, skip confirmation prompt
        nodes_only: If True, delete only nodes (keep indexes and constraints)
        debug: If True, enable verbose debug output
    """
    try:
        # Get database statistics before cleaning
        stats_query = """
        MATCH (n)
        RETURN 
            count(n) as node_count,
            count{(n)-[]->()} as relationship_count
        """
        
        async with graphiti.driver.session() as session:
            result = await session.run(stats_query)
            record = await result.single()
            node_count = record['node_count'] if record else 0
            rel_count = record['relationship_count'] if record else 0
        
        debug_print(debug, '\n' + '='*80)
        debug_print(debug, '🗑️  DATABASE CLEANUP')
        debug_print(debug, '='*80)
        debug_print(debug, f'Current database contents:')
        debug_print(debug, f'  Nodes: {node_count}')
        debug_print(debug, f'  Relationships: {rel_count}')
        debug_print(debug, '')
        
        if node_count == 0:
            debug_print(debug, '✅ Database is already empty!')
            return
        
        # Confirmation prompt
        if not confirm:
            debug_print(debug, '⚠️  WARNING: This will permanently delete all data!')
            if nodes_only:
                debug_print(debug, '   Mode: Nodes only (indexes and constraints will be preserved)')
            else:
                debug_print(debug, '   Mode: Complete cleanup (all nodes, relationships, and data)')
            debug_print(debug, '')
            response = input('Are you sure you want to continue? (yes/no): ')
            if response.lower() not in ['yes', 'y']:
                debug_print(debug, '❌ Cleanup cancelled.')
                return
        
        debug_print(debug, '\n🔄 Cleaning database...')
        
        if nodes_only:
            # Delete only nodes and relationships
            delete_query = """
            MATCH (n)
            DETACH DELETE n
            """
        else:
            # Complete cleanup including all data
            delete_query = """
            MATCH (n)
            DETACH DELETE n
            """
        
        async with graphiti.driver.session() as session:
            await session.run(delete_query)
        
        # Verify cleanup
        async with graphiti.driver.session() as session:
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            remaining = record['count'] if record else 0
        
        debug_print(debug, '\n' + '='*80)
        if remaining == 0:
            debug_print(debug, '✅ DATABASE CLEANED SUCCESSFULLY')
            debug_print(debug, '='*80)
            debug_print(debug, f'Deleted:')
            debug_print(debug, f'  Nodes: {node_count}')
            debug_print(debug, f'  Relationships: {rel_count}')
            debug_print(debug, '')
            debug_print(debug, 'The database is now empty and ready for new data.')
        else:
            debug_print(debug, '⚠️  CLEANUP INCOMPLETE')
            debug_print(debug, '='*80)
            debug_print(debug, f'Remaining nodes: {remaining}')
            debug_print(debug, 'Some nodes may not have been deleted. Please check manually.')
            debug_print(debug, '='*80)
            
    except Exception as e:
        logger.error(f'Error cleaning database: {e}')
        debug_print(debug, f'\n❌ Error cleaning database: {e}')
        raise
    finally:
        await graphiti.close()

