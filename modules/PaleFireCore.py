"""
Pale Fire Core - Entity Enrichment and Question Type Detection

This module contains the core classes for:
1. EntityEnricher - NER-based entity extraction and enrichment
2. QuestionTypeDetector - Question type detection and entity type weighting
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)

# Try to import spaCy for NER, fallback to pattern-based if not available
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    logger.warning("spaCy not installed. Using pattern-based NER fallback. Install with: pip install spacy")
    SPACY_AVAILABLE = False
    nlp = None


# ============================================================================
# NER (Named Entity Recognition) Enrichment System
# ============================================================================

class EntityEnricher:
    """
    Enriches text with Named Entity Recognition (NER) tags.
    Supports both spaCy-based and pattern-based extraction.
    """
    
    # Entity type mappings
    ENTITY_TYPES = {
        'PERSON': 'PER',
        'PER': 'PER',
        'ORG': 'ORG',
        'ORGANIZATION': 'ORG',
        'GPE': 'LOC',  # Geopolitical entity
        'LOC': 'LOC',
        'LOCATION': 'LOC',
        'DATE': 'DATE',
        'TIME': 'TIME',
        'MONEY': 'MONEY',
        'PERCENT': 'PERCENT',
        'FACILITY': 'FAC',
        'PRODUCT': 'PRODUCT',
        'EVENT': 'EVENT',
        'LAW': 'LAW',
        'LANGUAGE': 'LANGUAGE',
        'NORP': 'NORP',  # Nationalities, religious/political groups
        'WORK_OF_ART': 'WORK_OF_ART',
        'ORDINAL': 'ORDINAL',
        'CARDINAL': 'CARDINAL',
        'QUANTITY': 'QUANTITY',
        'OTHER': 'OTHER'  # Not a recognized entity type
    }
    
    # Entity types that need rich context for LLM verification
    # These are the important named entities that often have false positives
    CONTEXT_REQUIRED_TYPES = {
        'PER': 'Person names indicate human beings being mentioned',
        'ORG': 'Organization name indicate companies, agencies, institutions, none-profits, etc.',
        'LOC': 'Locations indicate physical places like mountains, lakes, etc.',
        'GPE': 'Geopolitical entities indicate countries, cities, states, etc.',
        'PRODUCT': 'Products indicate objects, vehicles, foods, etc. (not services)',
        'EVENT': 'Events indicate named hurricanes, battles, wars, sports events, etc.',
        'FAC': 'Facilities indicate buildings, airports, highways, bridges, etc.',
        'WORK_OF_ART': 'Works of art indicate titles of books, songs, etc.',
        'LAW': 'Laws indicate named documents made into laws.',
        'NORP': 'NORP indicates nationalities or religious or political groups.',
        'DATE': 'Dates indicate absolute or relative dates or periods.',
        'TIME': 'Times indicate times smaller than a day.',
        'MONEY': 'Money indicates monetary values, including unit.',
        'PERCENT': 'Percent indicates percentage values.',
        'LANGUAGE': 'Language indicates any named language.',
        'ORDINAL': 'Ordinal indicates ordinal numbers (first, second, etc.).',
        'CARDINAL': 'Cardinal indicates cardinal numbers that do not fall under another type.',
        'QUANTITY': 'Quantity indicates measurements, such as weight or distance.',
        'OTHER': 'Not a recognized entity type',
    }
    
    def __init__(self, use_spacy=True):
        self.use_spacy = use_spacy and SPACY_AVAILABLE and nlp is not None
        if self.use_spacy:
            logger.info("✓ Using spaCy for NER")
        else:
            logger.info("⚠ Using pattern-based NER (install spaCy for better results)")
    
    def extract_entities_spacy(self, text: str, context_window: int = 100) -> List[Dict[str, Any]]:
        """
        Extract entities using spaCy NER with rich context for important entity types.
        
        Args:
            text: Text to extract entities from
            context_window: Number of characters before/after entity to include as context
        
        Returns:
            List of entity dictionaries. Important entity types (PER, ORG, LOC, etc.) include 'context' field.
        """
        if not self.use_spacy:
            return []
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_type = self.ENTITY_TYPES.get(ent.label_, ent.label_)
            
            # Base entity data
            entity_data = {
                'text': ent.text,
                'type': entity_type,
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_  # Original spaCy label
            }
            
            # Only add rich context for important entity types that need verification
            if entity_type in self.CONTEXT_REQUIRED_TYPES:
                # Extract rich context around the entity
                context_start = max(0, ent.start_char - context_window)
                context_end = min(len(text), ent.end_char + context_window)
                context_before = text[context_start:ent.start_char].strip()
                context_after = text[ent.end_char:context_end].strip()
                
                # Get sentence context (spaCy provides sentence boundaries)
                sentence_start = ent.sent.start_char if hasattr(ent, 'sent') and ent.sent else context_start
                sentence_end = ent.sent.end_char if hasattr(ent, 'sent') and ent.sent else context_end
                sentence_context = text[sentence_start:sentence_end].strip()
                
                # Get surrounding sentences for broader context
                if hasattr(ent, 'sent') and ent.sent:
                    # Get previous sentence
                    prev_sent = None
                    try:
                        # Iterate through all sentences to find the one before ent.sent
                        current_sent = ent.sent
                        for sent in doc.sents:
                            if sent.end == current_sent.start:
                                prev_sent = sent
                                break
                    except:
                        pass
                    
                    # Get next sentence
                    next_sent = None
                    try:
                        # Find the next sentence after this one
                        current_sent = ent.sent
                        for sent in doc.sents:
                            if sent.start > current_sent.end:
                                next_sent = sent
                                break
                    except:
                        pass
                    
                    # Build extended context
                    extended_context_parts = []
                    if prev_sent:
                        extended_context_parts.append(prev_sent.text.strip())
                    extended_context_parts.append(sentence_context)
                    if next_sent:
                        extended_context_parts.append(next_sent.text.strip())
                    extended_context = ' '.join(extended_context_parts)
                else:
                    extended_context = sentence_context
                
                # Add context only for important entity types
                entity_data['context'] = {
                    'description': self.CONTEXT_REQUIRED_TYPES.get(entity_type, ''),
                    'before': context_before,
                    'after': context_after,
                    'sentence': sentence_context,
                    'extended': extended_context,  # Full sentence + surrounding sentences
                    'window_size': context_window
                }
            
            entities.append(entity_data)
        
        return entities
    
    def extract_entities_pattern(self, text: str) -> List[Dict[str, Any]]:
        """Fallback pattern-based entity extraction."""
        entities = []
        
        # Pattern for capitalized names (potential persons/locations)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        for match in re.finditer(name_pattern, text):
            # Simple heuristic: 2 words = likely person, 1 word = likely location
            words = match.group().split()
            entity_type = 'PER' if len(words) >= 2 else 'LOC'
            entities.append({
                'text': match.group(),
                'type': entity_type,
                'start': match.start(),
                'end': match.end(),
                'label': entity_type
            })
        
        # Pattern for dates
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        for match in re.finditer(date_pattern, text):
            entities.append({
                'text': match.group(),
                'type': 'DATE',
                'start': match.start(),
                'end': match.end(),
                'label': 'DATE'
            })
        
        # Pattern for years
        year_pattern = r'\b(19|20)\d{2}\b'
        for match in re.finditer(year_pattern, text):
            entities.append({
                'text': match.group(),
                'type': 'DATE',
                'start': match.start(),
                'end': match.end(),
                'label': 'DATE'
            })
        
        return entities
    
    def extract_entities(self, text: str, context_window: int = 100) -> List[Dict[str, Any]]:
        """
        Extract entities using available method.
        
        Args:
            text: Text to extract entities from
            context_window: Number of characters before/after entity to include as context (for spaCy only)
        
        Returns:
            List of entity dictionaries with context information
        """
        if self.use_spacy:
            return self.extract_entities_spacy(text, context_window=context_window)
        else:
            return self.extract_entities_pattern(text)
    
    def enrich_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich an episode with NER entities.
        
        Args:
            episode: Episode dict with 'content', 'type', 'description'
        
        Returns:
            Enriched episode with 'entities' field added
        """
        content = episode['content']
        
        # Handle both text and JSON content
        if isinstance(content, str):
            text = content
        else:
            # For JSON content, extract text from values
            text = ' '.join(str(v) for v in content.values() if v)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity['text'])
        
        # Remove duplicates while preserving order
        for entity_type in entities_by_type:
            seen = set()
            unique = []
            for item in entities_by_type[entity_type]:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            entities_by_type[entity_type] = unique
        
        # Create enriched episode
        enriched = episode.copy()
        enriched['entities'] = entities
        enriched['entities_by_type'] = entities_by_type
        enriched['entity_count'] = len(entities)
        
        return enriched
    
    def create_enriched_content(self, episode: Dict[str, Any]) -> str:
        """
        Create enriched content string with entity annotations.
        
        Format: "Original text [ENTITIES: PER: Kamala Harris; LOC: California]"
        """
        content = episode['content']
        if isinstance(content, str):
            text = content
        else:
            text = json.dumps(content)
        
        entities_by_type = episode.get('entities_by_type', {})
        
        if not entities_by_type:
            return text
        
        # Create entity annotation
        entity_parts = []
        for entity_type, entity_list in sorted(entities_by_type.items()):
            if entity_list:
                entities_str = ', '.join(entity_list[:5])  # Limit to 5 per type
                if len(entity_list) > 5:
                    entities_str += f' (+{len(entity_list) - 5} more)'
                entity_parts.append(f'{entity_type}: {entities_str}')
        
        entity_annotation = '; '.join(entity_parts)
        
        return f"{text}\n\n[ENTITIES: {entity_annotation}]"


# ============================================================================
# Question Type Detection and Entity Type Weighting
# ============================================================================

class QuestionTypeDetector:
    """
    Detects the type of question and maps it to relevant entity types.
    Adjusts entity weights based on question type for more accurate results.
    """
    
    # Question patterns and their corresponding entity type preferences
    QUESTION_PATTERNS = {
        'WHO': {
            'patterns': [
                r'\bwho\b',
                r'\bwhom\b',
                r'\bwhose\b',
                r'\bwhich person\b',
                r'\bwhat person\b',
                r'\bname the person\b',
            ],
            'entity_weights': {
                'PER': 2.0,      # Strong preference for persons
                'ORG': 0.5,      # Organizations less relevant
                'LOC': 0.3,      # Locations even less relevant
                'DATE': 0.8,     # Dates somewhat relevant (for context)
            },
            'description': 'Person-focused query'
        },
        'WHERE': {
            'patterns': [
                r'\bwhere\b',
                r'\bwhich place\b',
                r'\bwhich location\b',
                r'\bwhat location\b',
                r'\bin which (city|state|country|place)\b',
                r'\bwhat (city|state|country|place)\b',
            ],
            'entity_weights': {
                'LOC': 2.0,      # Strong preference for locations
                'GPE': 2.0,      # Geopolitical entities
                'FAC': 1.5,      # Facilities
                'PER': 0.5,      # Persons less relevant
                'ORG': 0.7,      # Organizations somewhat relevant
                'DATE': 0.5,
            },
            'description': 'Location-focused query'
        },
        'WHEN': {
            'patterns': [
                r'\bwhen\b',
                r'\bwhat time\b',
                r'\bwhat date\b',
                r'\bwhat year\b',
                r'\bin which year\b',
                r'\bhow long\b',
                r'\bwhat period\b',
            ],
            'entity_weights': {
                'DATE': 2.0,     # Strong preference for dates
                'TIME': 2.0,     # Times
                'EVENT': 1.5,    # Events (often have dates)
                'PER': 0.8,      # Persons somewhat relevant
                'ORG': 0.8,
                'LOC': 0.8,
            },
            'description': 'Time-focused query'
        },
        'WHAT_ORG': {
            'patterns': [
                r'\bwhat (organization|company|agency|institution|department)\b',
                r'\bwhich (organization|company|agency|institution|department)\b',
                r'\bname the (organization|company|agency|institution)\b',
            ],
            'entity_weights': {
                'ORG': 2.0,      # Strong preference for organizations
                'PER': 0.7,      # Persons somewhat relevant
                'LOC': 0.8,      # Locations somewhat relevant
                'DATE': 0.8,
            },
            'description': 'Organization-focused query'
        },
        'WHAT_POSITION': {
            'patterns': [
                r'\bwhat (position|role|title|job|office)\b',
                r'\bwhich (position|role|title|job|office)\b',
                r'\bwhat was .* (position|role|title)\b',
                r'\b(attorney general|governor|senator|president|mayor|director)\b',
            ],
            'entity_weights': {
                'PER': 1.5,      # Persons relevant (who holds position)
                'ORG': 1.5,      # Organizations relevant (position in org)
                'LOC': 1.2,      # Locations relevant (position in location)
                'DATE': 1.5,     # Dates important (when held position)
            },
            'description': 'Position/role-focused query'
        },
        'HOW_MANY': {
            'patterns': [
                r'\bhow many\b',
                r'\bhow much\b',
                r'\bwhat number\b',
                r'\bwhat percentage\b',
                r'\bwhat amount\b',
            ],
            'entity_weights': {
                'CARDINAL': 2.0,  # Numbers
                'MONEY': 2.0,     # Money amounts
                'PERCENT': 2.0,   # Percentages
                'QUANTITY': 2.0,  # Quantities
                'DATE': 1.0,
                'PER': 0.8,
                'ORG': 0.8,
                'LOC': 0.8,
            },
            'description': 'Quantity-focused query'
        },
        'WHY': {
            'patterns': [
                r'\bwhy\b',
                r'\bwhat reason\b',
                r'\bwhat cause\b',
                r'\bwhy did\b',
            ],
            'entity_weights': {
                'EVENT': 1.5,    # Events often have reasons
                'PER': 1.2,      # Persons make decisions
                'ORG': 1.2,      # Organizations make decisions
                'DATE': 1.2,     # Context of when
                'LOC': 1.0,
            },
            'description': 'Reason-focused query'
        },
        'WHAT_EVENT': {
            'patterns': [
                r'\bwhat (event|incident|occurrence|happening)\b',
                r'\bwhich (event|incident|occurrence)\b',
                r'\bwhat happened\b',
            ],
            'entity_weights': {
                'EVENT': 2.0,    # Strong preference for events
                'DATE': 1.5,     # When it happened
                'LOC': 1.3,      # Where it happened
                'PER': 1.2,      # Who was involved
                'ORG': 1.2,
            },
            'description': 'Event-focused query'
        },
    }
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for q_type, config in self.QUESTION_PATTERNS.items():
            self.compiled_patterns[q_type] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in config['patterns']
            ]
    
    def detect_question_type(self, query: str) -> Dict[str, Any]:
        """
        Detect the type of question and return entity type weights.
        
        Returns:
            dict with 'type', 'description', 'entity_weights', 'confidence'
        """
        query_lower = query.lower()
        
        # Check each question type
        matches = []
        for q_type, patterns in self.compiled_patterns.items():
            match_count = sum(1 for pattern in patterns if pattern.search(query_lower))
            if match_count > 0:
                matches.append({
                    'type': q_type,
                    'match_count': match_count,
                    'config': self.QUESTION_PATTERNS[q_type]
                })
        
        if not matches:
            # Default: balanced weights
            return {
                'type': 'GENERAL',
                'description': 'General query',
                'entity_weights': {},  # No special weighting
                'confidence': 0.0
            }
        
        # Sort by match count and take the best match
        matches.sort(key=lambda x: x['match_count'], reverse=True)
        best_match = matches[0]
        
        # Calculate confidence based on match count
        confidence = min(best_match['match_count'] / 2.0, 1.0)
        
        return {
            'type': best_match['type'],
            'description': best_match['config']['description'],
            'entity_weights': best_match['config']['entity_weights'],
            'confidence': confidence
        }
    
    def apply_entity_type_weights(
        self, 
        node, 
        enriched_episode: Optional[Dict[str, Any]], 
        entity_weights: Dict[str, float]
    ) -> float:
        """
        Calculate weighted score based on entity types present in the node.
        
        Args:
            node: The search result node
            enriched_episode: Enriched episode data with entity information
            entity_weights: Weights for each entity type
        
        Returns:
            float: Weighted score (0.0 to 2.0+)
        """
        if not entity_weights or not enriched_episode:
            return 1.0  # Neutral score
        
        entities_by_type = enriched_episode.get('entities_by_type', {})
        
        if not entities_by_type:
            return 1.0  # No entities, neutral score
        
        # Calculate weighted score based on entity types present
        total_weight = 0.0
        total_entities = 0
        
        for entity_type, entities in entities_by_type.items():
            if entities:  # Has entities of this type
                weight = entity_weights.get(entity_type, 1.0)
                entity_count = len(entities)
                total_weight += weight * entity_count
                total_entities += entity_count
        
        if total_entities == 0:
            return 1.0
        
        # Average weighted score
        avg_weight = total_weight / total_entities
        
        # Normalize to reasonable range (0.5 to 2.0)
        # This prevents extreme values while still providing meaningful boost
        normalized_score = max(0.5, min(2.0, avg_weight))
        
        return normalized_score

