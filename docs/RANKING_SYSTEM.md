# Multi-Factor Ranking System for Graphiti

## Overview

This advanced ranking system combines **4 independent factors** to rank search results:

1. **Semantic Relevance** (RRF): Hybrid search combining vector similarity + keyword matching
2. **Connectivity**: How well-connected the entity is in the knowledge graph
3. **Temporal Relevance**: Whether the entity was active during the query time period
4. **Query Term Matching**: How well the entity matches specific terms in the query

This multi-factor approach is especially powerful for complex queries like:
- "Who was the California Attorney General in 2020?" (temporal + specific role)
- "What position did Gavin Newsom hold in San Francisco?" (location + role matching)

## How It Works

### 1. **Standard Search (Baseline)**
The original `search_episodes()` function uses:
- **Hybrid Search**: Combines semantic (vector) and keyword search
- **RRF (Reciprocal Rank Fusion)**: Merges results from both methods
- Returns top 5 results based on RRF scores only

### 2. **Enhanced Search with Connection Ranking**
The `search_episodes_with_custom_ranking()` function adds:
- **Connection Count**: Counts how many relationships each node has
- **Weighted Scoring**: Combines original RRF score with connection importance
- **Configurable Weight**: Adjustable parameter to control connection influence

### 3. **Temporal-Aware Search**
The `search_episodes_with_temporal_ranking()` function adds:
- **Automatic Year Detection**: Extracts years from queries (e.g., "2020" from the query)
- **Temporal Matching**: Checks if entities were valid/active during the query time period
- **Date Range Support**: Handles term_start/term_end, valid_at/invalid_at timestamps
- **Triple Weighting**: Balances semantic relevance, connections, AND temporal match

### 4. **Multi-Factor Search**
The `search_episodes_with_multi_factor_ranking()` function provides:
- **Query Term Extraction**: Identifies important terms and proper nouns from the query
- **Comprehensive Matching**: Checks node name, summary, attributes, connected entities, and labels
- **Weighted Term Scoring**: Different weights for different match types (name > summary > attributes)
- **Quadruple Weighting**: Balances ALL four factors for optimal results

### 5. **Question-Aware Search (RECOMMENDED)**
The new `search_episodes_with_question_aware_ranking()` function adds:
- **Question Type Detection**: Automatically detects WHO/WHERE/WHEN/WHAT/WHY/HOW questions
- **Entity Type Intelligence**: Adjusts weights based on question type (WHO → boost PER entities)
- **Intent Understanding**: Understands what kind of answer the user wants
- **5-Factor Ranking**: Adds entity type matching as the 5th ranking factor

## Scoring Formulas

### Connection-Based Ranking
```
final_score = (1 - connection_weight) × original_score + connection_weight × normalized_connections
```

### Temporal-Aware Ranking
```
final_score = semantic_weight × original_score + 
              connection_weight × normalized_connections +
              temporal_weight × temporal_relevance

where: semantic_weight = 1.0 - connection_weight - temporal_weight
```

### Multi-Factor Ranking
```
final_score = semantic_weight × original_score + 
              connection_weight × normalized_connections +
              temporal_weight × temporal_relevance +
              query_match_weight × query_term_match

where: semantic_weight = 1.0 - connection_weight - temporal_weight - query_match_weight
```

### Question-Aware Ranking (RECOMMENDED)
```
final_score = semantic_weight × original_score + 
              connection_weight × normalized_connections +
              temporal_weight × temporal_relevance +
              query_match_weight × query_term_match +
              entity_type_weight × entity_type_match

where: semantic_weight = 1.0 - connection_weight - temporal_weight - query_match_weight - entity_type_weight
```

**Entity Type Match:**
- Automatically detects question type (WHO/WHERE/WHEN/etc.)
- Applies appropriate entity type weights
- Example: WHO questions boost PER entities 2.0x, penalize LOC entities 0.3x

Where:
- `original_score`: RRF score from hybrid search (0 to 1)
- `normalized_connections`: Node's connections / max_connections (0 to 1)
- `temporal_relevance`: How well the node matches the query time period (0 to 1)
  - **1.0**: Perfect temporal match (entity was active during query year)
  - **0.8**: Likely match (entity started before query year, no end date)
  - **0.5**: No temporal information or weak match
  - **0.3**: Outside validity period
- `query_term_match`: How well the node matches specific query terms (0 to 1)
  - Checks: node name (weight 3.0), proper nouns (2.0), summary (1.5), connected entities (1.0), attributes (1.0), labels (0.5)
  - Normalized to 0-1 range

### Weight Examples:

**Connection-Only:**
- **0.3**: 70% RRF, 30% connections (recommended default)

**Temporal-Aware:**
- **connection_weight=0.2, temporal_weight=0.3**: 50% semantic, 20% connections, 30% temporal
- **connection_weight=0.1, temporal_weight=0.5**: 40% semantic, 10% connections, 50% temporal (strong temporal focus)

**Multi-Factor (RECOMMENDED for complex queries):**
- **connection=0.15, temporal=0.25, query_match=0.25**: 35% semantic, 15% connections, 25% temporal, 25% query match (balanced)
- **connection=0.1, temporal=0.3, query_match=0.3**: 30% semantic, 10% connections, 30% temporal, 30% query match (focus on matching)
- **connection=0.2, temporal=0.2, query_match=0.3**: 30% semantic, 20% connections, 20% temporal, 30% query match (connectivity + matching)

## Key Functions

### `get_node_connections_with_entities(graphiti, node_uuid)`
Queries Neo4j to get comprehensive connection information for a node:
```cypher
MATCH (n {uuid: $uuid})-[r]-(connected)
RETURN 
    count(r) as connection_count,
    collect(DISTINCT connected.name) as connected_entities,
    collect(DISTINCT type(r)) as relationship_types
```

Returns a dictionary with:
- `count`: Number of connections
- `entities`: List of connected entity names
- `relationship_types`: List of relationship types

### `get_node_connections(graphiti, node_uuid)`
Simplified version that returns just the connection count.

### `extract_temporal_info(graphiti, node_uuid)`
Extracts temporal information from node properties and related episodes:
```cypher
MATCH (n {uuid: $uuid})
OPTIONAL MATCH (n)-[:PART_OF]-(episode)
RETURN 
    n.created_at, n.valid_at, n.invalid_at,
    collect(DISTINCT episode.valid_at) as episode_dates,
    properties(n) as node_properties
```

Returns temporal data including:
- `created_at`, `valid_at`, `invalid_at`: Timestamps
- `episode_dates`: Related episode timestamps
- `properties`: All node properties (may contain date fields like `term_start`, `term_end`)

### `calculate_temporal_relevance(node, temporal_info, query_year)`
Calculates how well a node matches the query time period:
- Checks property fields: `term_start`, `term_end`, `start_date`, `end_date`, `year`, `date`
- Parses date ranges (e.g., "2011-2017")
- Checks validity timestamps (`valid_at`, `invalid_at`)
- Returns score 0.0 to 1.0 based on temporal match quality

### `extract_query_terms(query)`
Extracts important information from the query:
- Removes stop words (who, what, when, the, a, etc.)
- Identifies important terms (length > 2, not stop words)
- Extracts proper nouns (capitalized words)
- Auto-detects year mentions
- Returns dict with all extracted information

### `calculate_query_match_score(node, connection_info, query_terms)`
Calculates how well a node matches specific query terms:
- **Node name match** (weight 3.0): Exact or partial match in node name
- **Proper noun match** (weight 2.0): Matches capitalized terms from query
- **Summary match** (weight 1.5): Terms found in node summary
- **Connected entities match** (weight 1.0): Terms in connected entity names
- **Attributes match** (weight 1.0): Terms in node attributes
- **Labels match** (weight 0.5): Terms in node labels
- Returns normalized score 0.0 to 1.0

### `search_episodes_with_custom_ranking(graphiti, query, connection_weight=0.3)`
Main enhanced search function that:
1. Performs initial hybrid search (gets top 20 candidates)
2. Fetches connection counts for each node
3. Normalizes connection scores
4. Calculates weighted final scores
5. Re-ranks and returns top 5 results

## Usage

```python
# Standard search (RRF only)
await search_episodes(graphiti, "Who was the California Attorney General in 2020?")

# Enhanced search with connection ranking
await search_episodes_with_custom_ranking(
    graphiti, 
    "Who was the California Attorney General in 2020?",
    connection_weight=0.3  # 30% weight on connections
)

# Temporal-aware search (for date queries)
await search_episodes_with_temporal_ranking(
    graphiti,
    "Who was the California Attorney General in 2020?",
    connection_weight=0.2,  # 20% weight on connections
    temporal_weight=0.3,    # 30% weight on temporal match
    query_year=2020         # Optional: auto-detected if not provided
)

# Multi-factor search (RECOMMENDED for complex queries)
await search_episodes_with_multi_factor_ranking(
    graphiti,
    "Who was the California Attorney General in 2020?",
    connection_weight=0.15,      # 15% weight on connections
    temporal_weight=0.25,        # 25% weight on temporal match
    query_match_weight=0.25,     # 25% weight on query term matching
    query_year=2020              # Optional: auto-detected if not provided
)
# Remaining 35% goes to semantic relevance (RRF)
```

## Output Format

Each result shows:

### Basic Information:
- **Node UUID**: Unique identifier
- **Node Name**: Entity name
- **Content Summary**: Brief description
- **Node Labels**: Entity types

### 📊 Connection Analysis:
- **Total Connections**: Number of relationships
- **Connected To**: Names of connected entities (up to 10 shown)
- **Relationship Types**: Types of relationships (e.g., "RELATED_TO", "WORKS_FOR")

### 🕐 Temporal Information:
- **term_start / term_end**: Start and end dates from node properties
- **Other date fields**: year, date, start_date, end_date, etc.

### 📈 Scoring Breakdown (Multi-Factor):
Shows detailed breakdown of each factor:
```
├─ Semantic (RRF):     0.8500 × 0.35 = 0.2975
├─ Connections:        0.7500 × 0.15 = 0.1125
├─ Temporal Match:     1.0000 × 0.25 = 0.2500
├─ Query Term Match:   0.9200 × 0.25 = 0.2300
└─ FINAL SCORE:        0.8900
```

Each line shows:
- Factor score (0-1) × weight = contribution to final score

### 🏷️ Attributes:
- Additional metadata associated with the node

## Benefits

1. **Promotes Central Entities**: Entities with many relationships (like "California", "Governor") rank higher
2. **Context Awareness**: Well-connected nodes are often more important in the knowledge graph
3. **Flexible Tuning**: Adjust `connection_weight` based on your use case
4. **Transparent Scoring**: Shows all score components for debugging
5. **Explainable Results**: Displays connected entities and relationship types, making it clear WHY a node ranks highly
6. **Relationship Insights**: See what types of relationships exist (e.g., "WORKS_FOR", "LOCATED_IN")

## When to Use Each Approach

### Use Standard Search when:
- No specific time period mentioned
- Simple entity lookup
- Speed is critical (fewest database queries)
- Broad exploratory queries

### Use Connection-Based Ranking when:
- You want to find central/important entities
- Exploring entity relationships
- No temporal context needed
- Finding "hub" nodes in the graph

### Use Temporal-Aware Ranking when:
- Query mentions a specific year or date
- Historical queries (e.g., "Who was X in 2020?")
- Time-sensitive information needed
- Entities have term limits or validity periods

### Use Multi-Factor Ranking when:
- Complex queries with multiple constraints
- Queries with specific terms AND dates
- Need to balance multiple relevance signals
- Want accurate results without question-type detection

### Use Question-Aware Ranking when: ⭐ RECOMMENDED
- **Any WHO/WHERE/WHEN/WHAT/WHY/HOW question**
- Want the system to understand user intent automatically
- Need entity-type aware results (WHO → people, WHERE → places)
- Maximum accuracy for natural language questions
- Production use with diverse query types

## Tuning Recommendations

### For Connection-Based Ranking:
- **Factual Queries**: Lower weight (0.2-0.3) - prioritize semantic match
- **Entity Discovery**: Higher weight (0.5-0.7) - find central entities
- **Relationship Exploration**: High weight (0.7-1.0) - find hub nodes

### For Temporal-Aware Ranking:
- **Recent queries** (last 5 years): `temporal_weight=0.3-0.4`
- **Historical queries** (10+ years ago): `temporal_weight=0.4-0.5`
- **Precise date queries** ("in 2020"): `temporal_weight=0.5`
- **Vague temporal queries** ("recently"): `temporal_weight=0.2`

## Example Comparison

Query: "Who was the California Attorney General in 2020?"

**Standard Search (RRF only)** might return:
1. Kamala Harris (high semantic match, but was AG 2011-2017, NOT in 2020)
2. California (keyword match)
3. Attorney General (keyword match)

**Enhanced Search (Connection-based)** might return:
1. California (moderate semantic + very high connections)
2. Kamala Harris (high semantic + moderate connections, but wrong time period)
3. Governor role (moderate semantic + high connections)

**Temporal-Aware Search (Connection + Temporal)** returns:
1. **Xavier Becerra** (moderate semantic + moderate connections + PERFECT temporal match: AG 2017-2021)
2. Kamala Harris (high semantic + moderate connections + LOW temporal: AG 2011-2017)
3. California (moderate semantic + very high connections + neutral temporal)

The temporal-aware version correctly identifies that Xavier Becerra was the AG in 2020, even if Kamala Harris has a stronger semantic match, because it factors in the temporal relevance!

## Example Output

```
[Rank 1]
Node UUID: abc-123-def
Node Name: Kamala Harris

📊 Connection Analysis:
  Total Connections: 15
  Connected To: California, Attorney General, San Francisco, District Attorney, Governor, Senate, ...
  Relationship Types: WORKS_FOR, LOCATED_IN, HOLDS_POSITION, RELATED_TO

📈 Scoring Breakdown:
  Original Score (RRF): 0.8500
  Connection Score: 0.7500
  Final Weighted Score: 0.8200

🏷️ Attributes:
  position: Attorney General
  state: California
  term_start: 2011-01-03
  term_end: 2017-01-03
---
```

This makes it immediately clear:
- **Why** the entity ranks highly (high RRF score + many connections)
- **What** it's connected to (California, Attorney General role, etc.)
- **How** it's connected (relationship types like WORKS_FOR, LOCATED_IN)

