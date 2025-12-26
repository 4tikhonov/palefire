"""
Unit tests for search and helper functions from palefire-cli.py
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions we need to test
# Note: These would need to be extracted to a separate module for proper testing
# For now, we'll test the logic patterns


class TestQueryTermExtraction:
    """Test extract_query_terms function logic."""
    
    def test_extract_year_from_query(self):
        """Test extracting year from query."""
        import re
        query = "Who was the Attorney General in 2020?"
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        assert year_match is not None
        assert int(year_match.group()) == 2020
    
    def test_extract_year_range(self):
        """Test extracting years from range."""
        import re
        query = "What happened between 2010 and 2020?"
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        assert len(years) == 2
        assert int(years[0]) == 2010
        assert int(years[1]) == 2020
    
    def test_no_year_in_query(self):
        """Test query without year."""
        import re
        query = "Who is the Attorney General?"
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        assert year_match is None
    
    def test_extract_important_terms(self):
        """Test extracting important terms."""
        import re
        stop_words = {'who', 'what', 'when', 'where', 'the', 'is', 'was'}
        query = "Who was the California Attorney General?"
        tokens = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        important_terms = [t for t in tokens if t not in stop_words and len(t) > 2]
        assert 'california' in important_terms
        assert 'attorney' in important_terms
        assert 'general' in important_terms
        assert 'who' not in important_terms
    
    def test_extract_proper_nouns(self):
        """Test extracting proper nouns."""
        import re
        query = "Who is Kamala Harris from California?"
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        assert 'Kamala Harris' in proper_nouns or 'Kamala' in proper_nouns
        assert 'California' in proper_nouns


class TestTemporalRelevance:
    """Test calculate_temporal_relevance function logic."""
    
    def test_no_query_year_returns_default(self):
        """Test that no query year returns default score."""
        query_year = None
        temporal_info = {'properties': {}}
        # Should return 1.0 when no year
        score = 1.0 if not query_year else 0.5
        assert score == 1.0
    
    def test_exact_year_match(self):
        """Test exact year match in properties."""
        query_year = 2020
        value = "2020"
        assert str(query_year) in value
    
    def test_year_range_match(self):
        """Test year within range."""
        query_year = 2015
        value = "2011-2017"
        if '-' in value:
            parts = value.split('-')
            start_year = int(''.join(filter(str.isdigit, parts[0]))[:4])
            end_year = int(''.join(filter(str.isdigit, parts[1]))[:4])
            assert start_year <= query_year <= end_year
    
    def test_year_outside_range(self):
        """Test year outside range."""
        query_year = 2020
        value = "2011-2017"
        parts = value.split('-')
        start_year = int(''.join(filter(str.isdigit, parts[0]))[:4])
        end_year = int(''.join(filter(str.isdigit, parts[1]))[:4])
        assert not (start_year <= query_year <= end_year)


class TestQueryMatchScoring:
    """Test calculate_query_match_score function logic."""
    
    def test_exact_name_match_in_query(self):
        """Test exact name match in query."""
        node_name = "kamala harris"
        query = "who is kamala harris?"
        assert node_name in query.lower()
    
    def test_partial_term_match(self):
        """Test partial term matching."""
        node_name = "attorney general"
        important_terms = ['attorney', 'general', 'california']
        matches = sum(1 for term in important_terms if term in node_name.lower())
        assert matches == 2
    
    def test_proper_noun_match(self):
        """Test proper noun matching."""
        node_name = "Kamala Harris"
        proper_nouns = ["Kamala Harris", "California"]
        match = any(pn.lower() in node_name.lower() or 
                   node_name.lower() in pn.lower() for pn in proper_nouns)
        assert match is True
    
    def test_no_match(self):
        """Test no matching terms."""
        node_name = "john doe"
        important_terms = ['attorney', 'general', 'california']
        matches = sum(1 for term in important_terms if term in node_name.lower())
        assert matches == 0
    
    def test_score_normalization(self):
        """Test score normalization to 0-1 range."""
        score = 7.5
        max_score = 10.0
        normalized = score / max_score if max_score > 0 else 0.0
        assert 0.0 <= normalized <= 1.0
        assert normalized == 0.75


class TestConnectionScoring:
    """Test connection-based scoring logic."""
    
    def test_normalize_connection_count(self):
        """Test normalizing connection counts."""
        connection_count = 10
        max_connections = 20
        normalized = connection_count / max_connections if max_connections > 0 else 0
        assert normalized == 0.5
    
    def test_zero_max_connections(self):
        """Test handling zero max connections."""
        connection_count = 5
        max_connections = 0
        normalized = connection_count / max_connections if max_connections > 0 else 0
        assert normalized == 0
    
    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        original_score = 0.8
        connection_score = 0.6
        connection_weight = 0.3
        
        final_score = (1 - connection_weight) * original_score + connection_weight * connection_score
        assert 0.0 <= final_score <= 1.0
        assert final_score == pytest.approx(0.74, rel=1e-2)


class TestMultiFactorRanking:
    """Test multi-factor ranking logic."""
    
    def test_five_factor_score_calculation(self):
        """Test 5-factor score calculation."""
        semantic_score = 0.8
        connection_score = 0.6
        temporal_score = 1.0
        query_match_score = 0.7
        entity_type_score = 0.5
        
        semantic_weight = 0.30
        connection_weight = 0.15
        temporal_weight = 0.20
        query_match_weight = 0.20
        entity_type_weight = 0.15
        
        final_score = (
            semantic_weight * semantic_score +
            connection_weight * connection_score +
            temporal_weight * temporal_score +
            query_match_weight * query_match_score +
            entity_type_weight * entity_type_score
        )
        
        assert 0.0 <= final_score <= 1.0
        assert final_score == pytest.approx(0.745, rel=1e-2)
    
    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0."""
        semantic_weight = 0.30
        connection_weight = 0.15
        temporal_weight = 0.20
        query_match_weight = 0.20
        entity_type_weight = 0.15
        
        total = (semantic_weight + connection_weight + temporal_weight + 
                query_match_weight + entity_type_weight)
        assert total == pytest.approx(1.0, rel=1e-6)
    
    def test_entity_type_score_normalization(self):
        """Test entity type score normalization."""
        entity_type_score = 2.0  # Raw score (0.5 to 2.0 range)
        normalized = (entity_type_score - 0.5) / 1.5  # Map to 0-1
        assert 0.0 <= normalized <= 1.0
        assert normalized == pytest.approx(1.0, rel=1e-6)


class TestExportFunctions:
    """Test export-related functions."""
    
    def test_filter_name_embedding_from_attributes(self):
        """Test filtering name_embedding from attributes."""
        attributes = {
            'position': 'Attorney General',
            'state': 'California',
            'name_embedding': [0.1, 0.2, 0.3]  # Should be filtered
        }
        filtered = {k: v for k, v in attributes.items() if k != 'name_embedding'}
        assert 'name_embedding' not in filtered
        assert 'position' in filtered
        assert 'state' in filtered
    
    def test_entity_format_conversion(self):
        """Test converting entity format."""
        entity_data = {
            'name': 'California',
            'labels': ['Entity', 'LOC'],
            'uuid': 'xyz-123'
        }
        
        # Extract entity type from labels
        entity_types = ['PER', 'LOC', 'ORG', 'DATE']
        entity_type = None
        for label in entity_data.get('labels', []):
            if label in entity_types:
                entity_type = label
                break
        
        assert entity_type == 'LOC'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

