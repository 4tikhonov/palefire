# Question Type Detection and Entity Type Weighting

## Overview

The Question Type Detection system automatically analyzes queries to determine what kind of answer the user is looking for, then adjusts entity type weights accordingly. This dramatically improves search accuracy by understanding user intent.

## How It Works

### 1. Question Type Detection

The system recognizes 8 different question types:

| Question Type | Patterns | Expected Answer | Example |
|---------------|----------|-----------------|---------|
| **WHO** | who, whom, whose, which person | Person (PER) | "Who was the California Attorney General?" |
| **WHERE** | where, which place/location | Location (LOC) | "Where did Kamala Harris work?" |
| **WHEN** | when, what time/date/year | Date/Time | "When was Gavin Newsom governor?" |
| **WHAT_ORG** | what organization/company | Organization (ORG) | "What organization did she lead?" |
| **WHAT_POSITION** | what position/role/title | Position/Role | "What position did Harris hold?" |
| **HOW_MANY** | how many/much, what number | Numbers/Quantities | "How many years was she AG?" |
| **WHY** | why, what reason/cause | Reasons/Events | "Why did she leave office?" |
| **WHAT_EVENT** | what event/happened | Events | "What happened in 2020?" |

### 2. Entity Type Weighting

Based on the detected question type, the system applies different weights to entity types:

#### WHO Questions (Person-focused)
```python
entity_weights = {
    'PER': 2.0,    # ⭐⭐ Strong preference for persons
    'ORG': 0.5,    # Organizations less relevant
    'LOC': 0.3,    # Locations even less relevant
    'DATE': 0.8,   # Dates somewhat relevant (context)
}
```

**Example:**
```
Query: "Who was the California Attorney General in 2020?"

Node A: "Xavier Becerra" (PER entity)
- Entity type score: 2.0 (high boost) ✅

Node B: "California" (LOC entity)
- Entity type score: 0.3 (penalty) ❌

Result: Person entities rank higher
```

#### WHERE Questions (Location-focused)
```python
entity_weights = {
    'LOC': 2.0,    # ⭐⭐ Strong preference for locations
    'GPE': 2.0,    # Geopolitical entities
    'FAC': 1.5,    # Facilities
    'PER': 0.5,    # Persons less relevant
    'ORG': 0.7,    # Organizations somewhat relevant
}
```

**Example:**
```
Query: "Where did Kamala Harris work as district attorney?"

Node A: "San Francisco" (LOC entity)
- Entity type score: 2.0 (high boost) ✅

Node B: "Kamala Harris" (PER entity)
- Entity type score: 0.5 (penalty) ❌

Result: Location entities rank higher
```

#### WHEN Questions (Time-focused)
```python
entity_weights = {
    'DATE': 2.0,   # ⭐⭐ Strong preference for dates
    'TIME': 2.0,   # Times
    'EVENT': 1.5,  # Events (often have dates)
    'PER': 0.8,    # Persons somewhat relevant
}
```

### 3. Scoring Integration

The entity type score is integrated as the 5th factor in the ranking system:

```
final_score = 0.30 × semantic_score +        # RRF hybrid search
              0.15 × connection_score +       # Graph connectivity
              0.20 × temporal_score +         # Time period match
              0.20 × query_match_score +      # Term matching
              0.15 × entity_type_score        # Question-type alignment ⭐
```

## Complete Question Type Patterns

### WHO - Person Queries
**Patterns:**
- `who`, `whom`, `whose`
- `which person`, `what person`
- `name the person`

**Entity Weights:**
- PER: 2.0x (strong boost)
- ORG: 0.5x (penalty)
- LOC: 0.3x (penalty)
- DATE: 0.8x (slight penalty)

**Use Case:** Finding specific people, identifying individuals

### WHERE - Location Queries
**Patterns:**
- `where`
- `which place/location`
- `in which city/state/country`
- `what location`

**Entity Weights:**
- LOC: 2.0x (strong boost)
- GPE: 2.0x (geopolitical entities)
- FAC: 1.5x (facilities)
- PER: 0.5x (penalty)
- ORG: 0.7x (slight penalty)

**Use Case:** Finding places, locations, geographic information

### WHEN - Time Queries
**Patterns:**
- `when`
- `what time/date/year`
- `in which year`
- `how long`
- `what period`

**Entity Weights:**
- DATE: 2.0x (strong boost)
- TIME: 2.0x (strong boost)
- EVENT: 1.5x (boost)
- PER/ORG/LOC: 0.8x (slight penalty)

**Use Case:** Finding dates, time periods, durations

### WHAT_ORG - Organization Queries
**Patterns:**
- `what organization/company/agency`
- `which organization/institution`
- `name the organization`

**Entity Weights:**
- ORG: 2.0x (strong boost)
- PER: 0.7x (slight penalty)
- LOC: 0.8x (slight penalty)

**Use Case:** Finding organizations, companies, agencies

### WHAT_POSITION - Role/Title Queries
**Patterns:**
- `what position/role/title/job`
- `which position/office`
- `attorney general`, `governor`, `senator`, etc.

**Entity Weights:**
- PER: 1.5x (boost)
- ORG: 1.5x (boost)
- LOC: 1.2x (slight boost)
- DATE: 1.5x (boost)

**Use Case:** Finding roles, positions, job titles

### HOW_MANY - Quantity Queries
**Patterns:**
- `how many/much`
- `what number/percentage/amount`

**Entity Weights:**
- CARDINAL: 2.0x (numbers)
- MONEY: 2.0x (money amounts)
- PERCENT: 2.0x (percentages)
- QUANTITY: 2.0x (quantities)

**Use Case:** Finding numbers, counts, amounts

### WHY - Reason Queries
**Patterns:**
- `why`
- `what reason/cause`
- `why did`

**Entity Weights:**
- EVENT: 1.5x (boost)
- PER: 1.2x (slight boost)
- ORG: 1.2x (slight boost)
- DATE: 1.2x (slight boost)

**Use Case:** Finding reasons, causes, explanations

### WHAT_EVENT - Event Queries
**Patterns:**
- `what event/incident/occurrence`
- `what happened`

**Entity Weights:**
- EVENT: 2.0x (strong boost)
- DATE: 1.5x (boost)
- LOC: 1.3x (boost)
- PER: 1.2x (slight boost)

**Use Case:** Finding events, incidents, happenings

## Example Comparisons

### Example 1: WHO Question

**Query:** "Who was the California Attorney General in 2020?"

**Without Question-Type Detection:**
```
Results:
1. California (LOC) - high semantic match
2. Attorney General (ORG) - role match
3. Xavier Becerra (PER) - person match
```

**With Question-Type Detection:**
```
Question Type: WHO (Person-focused)
Entity Type Weights: PER=2.0x, LOC=0.3x, ORG=0.5x

Results:
1. Xavier Becerra (PER) - person match + 2.0x boost ✅
2. Kamala Harris (PER) - person match + 2.0x boost
3. California (LOC) - high semantic but 0.3x penalty
```

**Improvement:** Correctly prioritizes person entities!

### Example 2: WHERE Question

**Query:** "Where did Kamala Harris work as district attorney?"

**Without Question-Type Detection:**
```
Results:
1. Kamala Harris (PER) - name match
2. District Attorney (ORG) - role match
3. San Francisco (LOC) - location match
```

**With Question-Type Detection:**
```
Question Type: WHERE (Location-focused)
Entity Type Weights: LOC=2.0x, PER=0.5x, ORG=0.7x

Results:
1. San Francisco (LOC) - location + 2.0x boost ✅
2. California (LOC) - location + 2.0x boost
3. District Attorney (ORG) - role but 0.7x penalty
```

**Improvement:** Correctly prioritizes location entities!

### Example 3: WHEN Question

**Query:** "When did Gavin Newsom become governor?"

**Without Question-Type Detection:**
```
Results:
1. Gavin Newsom (PER) - name match
2. Governor (ORG) - role match
3. January 7, 2019 (DATE) - date match
```

**With Question-Type Detection:**
```
Question Type: WHEN (Time-focused)
Entity Type Weights: DATE=2.0x, PER=0.8x, ORG=0.8x

Results:
1. January 7, 2019 (DATE) - date + 2.0x boost ✅
2. 2019 (DATE) - year + 2.0x boost
3. Gavin Newsom (PER) - name but 0.8x penalty
```

**Improvement:** Correctly prioritizes date entities!

## Usage

### Basic Usage

```python
from maintest import QuestionTypeDetector

# Initialize detector
detector = QuestionTypeDetector()

# Detect question type
query = "Who was the California Attorney General in 2020?"
question_info = detector.detect_question_type(query)

print(f"Type: {question_info['type']}")
print(f"Description: {question_info['description']}")
print(f"Confidence: {question_info['confidence']}")
print(f"Entity Weights: {question_info['entity_weights']}")
```

**Output:**
```
Type: WHO
Description: Person-focused query
Confidence: 1.0
Entity Weights: {'PER': 2.0, 'ORG': 0.5, 'LOC': 0.3, 'DATE': 0.8}
```

### Integrated Search

```python
# Use question-aware search
await search_episodes_with_question_aware_ranking(
    graphiti,
    "Who was the California Attorney General in 2020?",
    enricher=enricher,
    connection_weight=0.15,
    temporal_weight=0.20,
    query_match_weight=0.20,
    entity_type_weight=0.15  # 15% weight for entity type matching
)
```

## Benefits

### 1. **Intent Understanding**
- Automatically detects what kind of answer user wants
- No manual configuration needed
- Works across different query phrasings

### 2. **Improved Accuracy**
- WHO questions return people, not places
- WHERE questions return locations, not people
- WHEN questions return dates, not entities

### 3. **Confidence Scoring**
- Provides confidence score for question type detection
- Multiple pattern matches increase confidence
- Fallback to general search if uncertain

### 4. **Flexible Weighting**
- Entity type weight can be adjusted (0.0 to 1.0)
- Higher weight = stronger entity type preference
- Lower weight = more balanced with other factors

## Configuration

### Adjusting Entity Type Weight

```python
# Strong entity type preference (20%)
entity_type_weight=0.20

# Moderate entity type preference (15%) - RECOMMENDED
entity_type_weight=0.15

# Light entity type preference (10%)
entity_type_weight=0.10

# No entity type preference (0%)
entity_type_weight=0.0
```

### Custom Question Patterns

You can extend the `QUESTION_PATTERNS` dictionary to add custom patterns:

```python
QUESTION_PATTERNS['CUSTOM_TYPE'] = {
    'patterns': [r'\bcustom pattern\b'],
    'entity_weights': {'PER': 1.5, 'LOC': 1.2},
    'description': 'Custom query type'
}
```

## Performance Impact

- **Detection Speed:** ~1-5ms per query (regex matching)
- **Entity Extraction:** ~50-500ms per node (depends on spaCy)
- **Overall Impact:** +10-20% search time for significantly better accuracy

## Limitations

1. **English-Only:** Patterns are English-specific
2. **Pattern-Based:** May miss complex or unusual phrasings
3. **Entity Extraction Dependency:** Requires NER for best results
4. **Single Question Type:** Detects primary question type only

## Future Enhancements

1. **Multi-Language Support:** Patterns for other languages
2. **ML-Based Detection:** Use ML model instead of regex patterns
3. **Multi-Question Handling:** Handle compound questions
4. **Context Awareness:** Consider conversation history
5. **Custom Entity Types:** Support domain-specific entity types
6. **Learning System:** Learn from user feedback to improve weights

## Best Practices

1. **Use with NER:** Entity type weighting works best with NER enrichment
2. **Tune Weights:** Adjust `entity_type_weight` based on your use case
3. **Monitor Confidence:** Low confidence may indicate unclear question
4. **Combine with Temporal:** Use temporal ranking for date-specific queries
5. **Test Patterns:** Test with various query phrasings

