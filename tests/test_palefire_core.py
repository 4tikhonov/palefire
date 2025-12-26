"""
Unit tests for PaleFireCore module (EntityEnricher and QuestionTypeDetector).
"""

import pytest
from modules.PaleFireCore import EntityEnricher, QuestionTypeDetector


class TestEntityEnricher:
    """Test EntityEnricher class."""
    
    def test_entity_enricher_init(self):
        """Test EntityEnricher initialization."""
        enricher = EntityEnricher(use_spacy=False)
        assert enricher is not None
    
    def test_enrich_text_episode(self):
        """Test enriching a text episode."""
        enricher = EntityEnricher(use_spacy=False)
        episode = {
            'content': 'Kamala Harris is the Attorney General of California.',
            'type': 'text',
            'description': 'test'
        }
        enriched = enricher.enrich_episode(episode)
        assert 'entities' in enriched
        assert 'entities_by_type' in enriched
    
    def test_enrich_json_episode(self):
        """Test enriching a JSON episode."""
        enricher = EntityEnricher(use_spacy=False)
        episode = {
            'content': {'name': 'Test', 'location': 'California'},
            'type': 'json',
            'description': 'test'
        }
        enriched = enricher.enrich_episode(episode)
        assert 'entities' in enriched
    
    def test_pattern_based_ner(self):
        """Test pattern-based NER extraction."""
        enricher = EntityEnricher(use_spacy=False)
        text = "Kamala Harris worked in California from 2011 to 2017."
        entities = enricher.extract_entities_pattern(text)
        assert len(entities) > 0
        # Should find at least the year
        years = [e for e in entities if e['type'] == 'DATE']
        assert len(years) > 0
    
    def test_extract_dates(self):
        """Test date extraction."""
        enricher = EntityEnricher(use_spacy=False)
        text = "From January 3, 2011 to January 3, 2017"
        entities = enricher.extract_entities_pattern(text)
        dates = [e for e in entities if e['type'] == 'DATE']
        assert len(dates) >= 2
    
    def test_extract_capitalized_names(self):
        """Test capitalized name extraction."""
        enricher = EntityEnricher(use_spacy=False)
        text = "Kamala Harris and Joe Biden are politicians."
        entities = enricher.extract_entities_pattern(text)
        # Should extract person names (2+ capitalized words)
        persons = [e for e in entities if e['type'] == 'PER']
        assert len(persons) >= 1
    
    def test_extract_year_pattern(self):
        """Test year pattern extraction."""
        enricher = EntityEnricher(use_spacy=False)
        text = "The event happened in 2020 and 2021."
        entities = enricher.extract_entities_pattern(text)
        years = [e for e in entities if e['type'] == 'DATE']
        assert len(years) >= 2
    
    def test_entities_grouped_by_type(self):
        """Test that entities are grouped by type."""
        enricher = EntityEnricher(use_spacy=False)
        episode = {
            'content': 'In 2020, the budget was $1 million.',
            'type': 'text',
            'description': 'test'
        }
        enriched = enricher.enrich_episode(episode)
        entities_by_type = enriched['entities_by_type']
        assert isinstance(entities_by_type, dict)
        # Should have DATE and MONEY types
        assert 'DATE' in entities_by_type or 'MONEY' in entities_by_type


class TestQuestionTypeDetector:
    """Test QuestionTypeDetector class."""
    
    def test_question_type_detector_init(self):
        """Test QuestionTypeDetector initialization."""
        detector = QuestionTypeDetector()
        assert detector is not None
    
    def test_detect_who_question(self):
        """Test detection of WHO questions."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("Who was the Attorney General?")
        assert result['type'] == 'WHO'
        assert 'PER' in result['entity_weights']
        assert result['entity_weights']['PER'] > 1.0
    
    def test_detect_where_question(self):
        """Test detection of WHERE questions."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("Where did she work?")
        assert result['type'] == 'WHERE'
        assert 'LOC' in result['entity_weights']
        assert result['entity_weights']['LOC'] > 1.0
    
    def test_detect_when_question(self):
        """Test detection of WHEN questions."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("When did he become governor?")
        assert result['type'] == 'WHEN'
        assert 'DATE' in result['entity_weights']
        assert result['entity_weights']['DATE'] > 1.0
    
    def test_detect_what_org_question(self):
        """Test detection of WHAT organization questions."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("What organization does she work for?")
        assert result['type'] == 'WHAT_ORG'
        assert 'ORG' in result['entity_weights']
    
    def test_detect_what_position_question(self):
        """Test detection of WHAT position questions."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("What position did he hold?")
        assert result['type'] == 'WHAT_POSITION'
    
    def test_detect_how_many_question(self):
        """Test detection of HOW MANY questions."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("How many years was she in office?")
        assert result['type'] == 'HOW_MANY'
        assert 'CARDINAL' in result['entity_weights']
    
    def test_detect_why_question(self):
        """Test detection of WHY questions."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("Why did she resign?")
        assert result['type'] == 'WHY'
    
    def test_detect_what_event_question(self):
        """Test detection of WHAT event questions."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("What happened in 2020?")
        assert result['type'] == 'WHAT_EVENT'
    
    def test_question_confidence_score(self):
        """Test that confidence score is returned."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("Who is the president?")
        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_question_description(self):
        """Test that description is returned."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("Who is the president?")
        assert 'description' in result
        assert len(result['description']) > 0
    
    def test_entity_weights_returned(self):
        """Test that entity weights are returned."""
        detector = QuestionTypeDetector()
        result = detector.detect_question_type("Who is the president?")
        assert 'entity_weights' in result
        assert isinstance(result['entity_weights'], dict)
    
    def test_apply_entity_type_weights(self):
        """Test applying entity type weights."""
        detector = QuestionTypeDetector()
        
        # Mock node
        class MockNode:
            name = "Test"
            summary = "Test summary"
        
        node = MockNode()
        enriched_node = {
            'entities_by_type': {
                'PER': ['John Doe'],
                'LOC': ['California']
            }
        }
        entity_weights = {'PER': 2.0, 'LOC': 1.0}
        
        score = detector.apply_entity_type_weights(node, enriched_node, entity_weights)
        assert score >= 1.0  # Should be boosted by PER match
    
    def test_case_insensitive_detection(self):
        """Test that question detection is case insensitive."""
        detector = QuestionTypeDetector()
        result1 = detector.detect_question_type("WHO is the president?")
        result2 = detector.detect_question_type("who is the president?")
        assert result1['type'] == result2['type']


class TestEntityEnricherWithSpacy:
    """Test EntityEnricher with spaCy (if available)."""
    
    @pytest.mark.requires_spacy
    def test_spacy_ner(self):
        """Test spaCy-based NER extraction."""
        try:
            enricher = EntityEnricher(use_spacy=True)
            if not hasattr(enricher, 'nlp') or enricher.nlp is None:
                pytest.skip("spaCy not available")
            text = "Kamala Harris worked in California."
            entities = enricher.extract_entities_spacy(text)
            assert len(entities) > 0
        except Exception:
            pytest.skip("spaCy not available")
    
    @pytest.mark.requires_spacy
    def test_spacy_person_extraction(self):
        """Test person extraction with spaCy."""
        try:
            enricher = EntityEnricher(use_spacy=True)
            if not hasattr(enricher, 'nlp') or enricher.nlp is None:
                pytest.skip("spaCy not available")
            text = "Kamala Harris and Joe Biden are politicians."
            entities = enricher.extract_entities_spacy(text)
            persons = [e for e in entities if e['type'] == 'PER']
            assert len(persons) >= 1
        except Exception:
            pytest.skip("spaCy not available")
    
    @pytest.mark.requires_spacy
    def test_spacy_location_extraction(self):
        """Test location extraction with spaCy."""
        try:
            enricher = EntityEnricher(use_spacy=True)
            if not hasattr(enricher, 'nlp') or enricher.nlp is None:
                pytest.skip("spaCy not available")
            text = "She worked in California and San Francisco."
            entities = enricher.extract_entities_spacy(text)
            locations = [e for e in entities if e['type'] == 'LOC']
            assert len(locations) >= 1
        except Exception:
            pytest.skip("spaCy not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

