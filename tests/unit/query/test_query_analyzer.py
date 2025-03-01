"""
Tests for the QueryAnalyzer component.
"""

import pytest
from unittest.mock import MagicMock

from memoryweave.query.analyzer import SimpleQueryAnalyzer
from memoryweave.interfaces.retrieval import QueryType


class TestSimpleQueryAnalyzer:
    """Test suite for the SimpleQueryAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a query analyzer for testing."""
        return SimpleQueryAnalyzer()

    def test_analyze_personal_query(self, analyzer):
        """Test analyzing personal queries."""
        # Personal queries
        personal_queries = [
            "What is my favorite color?",
            "Tell me about my family",
            "Do you remember where I put my keys?",
            "I like chocolate ice cream",
            "My mother's name is Sarah",
        ]

        for query in personal_queries:
            query_type = analyzer.analyze(query)
            assert query_type == QueryType.PERSONAL, f"Query '{query}' should be PERSONAL"

    def test_analyze_factual_query(self, analyzer):
        """Test analyzing factual queries."""
        # Factual queries
        factual_queries = [
            "What is the capital of France?",
            "Who is the president of the United States?",
            "Define quantum physics",
            "Explain how photosynthesis works",
            "Tell me about the history of Rome",
        ]

        for query in factual_queries:
            query_type = analyzer.analyze(query)
            assert query_type == QueryType.FACTUAL, f"Query '{query}' should be FACTUAL"

    def test_analyze_temporal_query(self, analyzer):
        """Test analyzing temporal queries."""
        # Temporal queries
        temporal_queries = [
            "When did we last talk about this?",
            "What time does the movie start?",
            "Tell me about the recent developments",
            "What happened during the Renaissance period?",
            "How long ago was the dinosaur extinction?",
        ]

        for query in temporal_queries:
            query_type = analyzer.analyze(query)
            assert query_type == QueryType.TEMPORAL, f"Query '{query}' should be TEMPORAL"

    def test_analyze_unknown_query(self, analyzer):
        """Test analyzing ambiguous queries."""
        # Ambiguous queries
        unknown_queries = ["Hello", "Yes", "Continue", "Go on", "This is a test"]

        for query in unknown_queries:
            query_type = analyzer.analyze(query)
            assert query_type == QueryType.UNKNOWN, f"Query '{query}' should be UNKNOWN"

    def test_extract_keywords(self, analyzer):
        """Test keyword extraction."""
        query = "What is the capital of France and its population?"

        keywords = analyzer.extract_keywords(query)

        # Check that important keywords were extracted
        assert "capital" in keywords
        assert "france" in keywords
        assert "population" in keywords

        # Check that stopwords were removed
        assert "what" not in keywords
        assert "is" not in keywords
        assert "the" not in keywords
        assert "of" not in keywords
        assert "and" not in keywords
        assert "its" not in keywords

    def test_extract_entities(self, analyzer):
        """Test entity extraction."""
        query = "Did John Smith visit Paris last summer with Microsoft?"

        entities = analyzer.extract_entities(query)

        # Check that entities were extracted
        assert "John Smith" in entities
        assert "Paris" in entities
        assert "Microsoft" in entities

    def test_configure(self, analyzer):
        """Test configuration of the analyzer."""
        # Initial defaults
        assert analyzer._config["min_keyword_length"] == 3
        assert analyzer._config["max_keywords"] == 10

        # Configure analyzer
        analyzer.configure(
            {
                "min_keyword_length": 4,
                "max_keywords": 5,
                "stopwords": {"test", "example"},
                "personal_patterns": ["\\bours\\b", "\\bwe\\b"],
            }
        )

        # Check updated configuration
        assert analyzer._config["min_keyword_length"] == 4
        assert analyzer._config["max_keywords"] == 5
        assert "test" in analyzer._stopwords
        assert "example" in analyzer._stopwords

        # Test with newly configured patterns
        query = "We should consider our options"
        query_type = analyzer.analyze(query)
        assert query_type == QueryType.PERSONAL
