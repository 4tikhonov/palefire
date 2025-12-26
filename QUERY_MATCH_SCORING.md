# Query Term Matching System

## Overview

The query term matching component analyzes how well each node matches specific terms and entities mentioned in the query. This goes beyond semantic similarity to check for explicit term matches across multiple node properties.

## How It Works

### 1. Query Analysis (`extract_query_terms`)

First, the system analyzes the query to extract:

```python
query = "Who was the California Attorney General in 2020?"

# Extracted information:
{
    'query_year': 2020,
    'important_terms': ['california', 'attorney', 'general'],
    'proper_nouns': ['California', 'Attorney', 'General'],
    'original_query': 'who was the california attorney general in 2020?'
}
```

**Process:**
1. Extract year using regex `\b(19|20)\d{2}\b`
2. Remove stop words (who, what, when, the, a, is, was, etc.)
3. Keep important terms (length > 2, not stop words)
4. Extract proper nouns (capitalized words)

### 2. Term Matching (`calculate_query_match_score`)

Then, for each node, the system checks for matches across 6 different properties:

| Property | Weight | What It Checks |
|----------|--------|----------------|
| **Node Name** | 3.0 | Exact or partial match of query terms in the node's name |
| **Proper Nouns** | 2.0 | Capitalized terms from query found in node name |
| **Summary** | 1.5 | Query terms found in node's content summary |
| **Connected Entities** | 1.0 | Query terms in names of connected nodes |
| **Attributes** | 1.0 | Query terms in node's attribute values |
| **Labels** | 0.5 | Query terms in node's type labels |

**Total Max Score:** 9.0 (normalized to 0-1 range)

### 3. Scoring Examples

#### Example 1: Perfect Match
```
Query: "Who was the California Attorney General in 2020?"
Node: "Kamala Harris"

Scoring:
- Node name: "Kamala Harris" not in query → 0/3.0
- Proper nouns: No match → 0/2.0
- Summary: "Attorney General of California..." → 3/3 terms → 1.5/1.5 ✓
- Connected: ["California", "Attorney General", "San Francisco"] → 2/3 terms → 0.67/1.0
- Attributes: {position: "Attorney General", state: "California"} → 2/3 terms → 0.67/1.0
- Labels: ["Person", "Official"] → 0/3 terms → 0/0.5

Raw Score: 2.84 / 9.0 = 0.316
```

#### Example 2: Entity Name Match
```
Query: "What position did Gavin Newsom hold?"
Node: "Gavin Newsom"

Scoring:
- Node name: "Gavin Newsom" in query → 3.0/3.0 ✓✓✓
- Proper nouns: "Gavin Newsom" matches → 2.0/2.0 ✓✓
- Summary: "Governor of California, position..." → 1/2 terms → 0.75/1.5
- Connected: ["Governor", "California", "Lieutenant"] → 1/2 terms → 0.5/1.0
- Attributes: {position: "Governor"} → 1/2 terms → 0.5/1.0
- Labels: ["Person"] → 0/2 terms → 0/0.5

Raw Score: 6.75 / 9.0 = 0.750
```

## Integration with Multi-Factor Ranking

The query match score is combined with other factors:

```
final_score = 0.35 × semantic_score +      # RRF hybrid search
              0.15 × connection_score +     # Graph connectivity
              0.25 × temporal_score +       # Time period match
              0.25 × query_match_score      # Term matching (THIS)
```

## Why This Matters

### Problem Without Query Matching:
Query: "Who was the California Attorney General in 2020?"

**Semantic search alone might return:**
1. "California" (high semantic match, but not a person)
2. "Attorney General" (concept, not a specific person)
3. "Kamala Harris" (person, but wrong time period)

### Solution With Query Matching:
**Multi-factor search returns:**
1. "Xavier Becerra" (moderate semantic + perfect temporal + high query match)
   - Summary mentions "Attorney General" and "California"
   - Attributes: term_start=2017, term_end=2021 (includes 2020)
   - Connected to: California, Attorney General, etc.

## Tuning Guidelines

### High Query Match Weight (0.3-0.4)
Use when:
- Query has specific proper nouns (names, places)
- Looking for exact entity matches
- Query has technical terms or specific roles

### Medium Query Match Weight (0.2-0.3) - RECOMMENDED
Use when:
- Balanced queries with some specific terms
- Want to combine term matching with semantic understanding
- General purpose queries

### Low Query Match Weight (0.1-0.2)
Use when:
- Very broad queries
- Conceptual questions
- Prioritizing semantic similarity over exact matches

## Examples by Query Type

### Specific Entity Query
```
Query: "Where did Kamala Harris work?"
Recommended: query_match_weight=0.3

Why: "Kamala Harris" is a proper noun that should match exactly
```

### Role-Based Query
```
Query: "Who was the California Attorney General in 2020?"
Recommended: query_match_weight=0.25

Why: Multiple specific terms (California, Attorney General) plus temporal
```

### Conceptual Query
```
Query: "What government positions exist in California?"
Recommended: query_match_weight=0.15

Why: Broad conceptual query, semantic understanding more important
```

## Benefits

1. **Precision**: Finds entities that explicitly match query terms
2. **Proper Noun Handling**: Correctly identifies named entities
3. **Multi-Property Search**: Checks name, summary, attributes, connections
4. **Weighted Importance**: Node name matches count more than label matches
5. **Complementary**: Works alongside semantic search, not replacing it

## Limitations

- Requires exact or partial string matches (case-insensitive)
- May miss synonyms (e.g., "AG" vs "Attorney General")
- Stop words are removed (may miss some context)
- English-language focused

## Future Enhancements

Potential improvements:
- Synonym expansion (AG → Attorney General)
- Fuzzy matching for misspellings
- Multi-language support
- Entity type awareness (Person vs Place vs Organization)
- Phrase matching (multi-word terms)

