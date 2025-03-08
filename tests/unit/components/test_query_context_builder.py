# tests/unit/components/test_query_context_builder.py
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pytest

from memoryweave.components.query_context_builder import QueryContextBuilder
from memoryweave.nlp.extraction import NLPExtractor


class TestQueryContextBuilder:
    """Test the QueryContextBuilder component."""

    @pytest.fixture
    def builder(self):
        """Create a query context builder for testing."""
        builder = QueryContextBuilder()
        builder.initialize({})
        return builder

    def test_extract_temporal_markers_explicit(self, builder):
        """Test extracting explicit temporal markers from queries."""
        # Test with date reference
        query = "What happened on January 15?"
        result = builder._extract_temporal_information(query)

        assert result["has_temporal_reference"] is True
        assert result["time_type"] == "absolute"
        assert any("january 15" in expr.lower() for expr in result["time_expressions"])

        # Test with relative time
        query = "What did I do 3 days ago?"
        result = builder._extract_temporal_information(query)

        assert result["has_temporal_reference"] is True
        assert result["time_type"] == "relative"
        assert "3 days ago" in result["time_expressions"]
        assert result["relative_time"] is not None

        # Calculate expected timestamp (approximately)
        expected_time = (datetime.now() - timedelta(days=3)).timestamp()
        # Allow 10-second tolerance for test timing differences
        assert abs(result["relative_time"] - expected_time) < 10

    def test_extract_temporal_markers_implicit(self, builder):
        """Test extracting implicit temporal markers from queries."""
        # Test with implicit recent reference
        query = "Tell me about my recent activities"
        result = builder._extract_temporal_information(query)

        assert result["has_temporal_reference"] is True
        assert "recent" in result["time_keywords"]
        assert "implied_timeframe" in result
        assert result["implied_timeframe"]["focus"] == "recent"

        # Test with past tense verbs
        query = "What happened during the meeting?"
        result = builder._extract_temporal_information(query)

        assert result["has_temporal_reference"] is True
        assert "implied_timeframe" in result
        assert result["implied_timeframe"]["focus"] == "past"

    def test_parse_relative_time(self, builder):
        """Test parsing relative time expressions."""
        # Test "yesterday"
        relative_time = builder._parse_relative_time("yesterday")
        expected_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=1)
        # Allow small difference due to test execution timing
        assert abs(relative_time - expected_time.timestamp()) < 5

        # Test "2 weeks ago"
        relative_time = builder._parse_relative_time("2 weeks ago")
        expected_time = datetime.now() - timedelta(weeks=2)
        # Allow larger difference due to approximation
        assert abs(relative_time - expected_time.timestamp()) < 3600  # Within 1 hour

        # Test "last month"
        relative_time = builder._parse_relative_time("last month")
        expected_time = datetime.now() - timedelta(days=30)
        # Rough approximation, allow larger difference
        assert abs(relative_time - expected_time.timestamp()) < 86400  # Within 1 day

    def test_build_conversation_context(self, builder):
        """Test building context from conversation history."""
        # Mock conversation history
        conversation_history = [
            {
                "user": "Tell me about machine learning",
                "system": "Machine learning is a field of AI...",
            },
            {
                "user": "What are neural networks?",
                "system": "Neural networks are computing systems...",
            },
        ]

        # Mock NLP extractor behavior for deterministic testing
        with patch.object(NLPExtractor, "extract_entities") as mock_extract_entities:
            with patch.object(NLPExtractor, "extract_keywords") as mock_extract_keywords:
                # Set up mock returns
                mock_extract_entities.side_effect = [
                    ["machine learning", "AI"],
                    ["neural networks", "computing systems"],
                ]
                mock_extract_keywords.side_effect = [
                    ["machine", "learning", "field", "AI"],
                    ["neural", "networks", "computing", "systems"],
                ]

                # Call the method
                context = {}
                context["conversation_history"] = conversation_history
                result = builder._build_conversation_context("What are transformers?", context)

                # Check result
                assert "recent_entities" in result
                assert "recent_topics" in result
                assert "turns" in result
                assert result["turns"] == 2

                # Check entities and topics were extracted
                assert "machine learning" in result["recent_entities"]
                assert "neural networks" in result["recent_entities"]
                assert "machine" in result["recent_topics"]
                assert "networks" in result["recent_topics"]

    def test_enrich_embedding(self, builder):
        """Test enriching query embedding with context."""
        # Create a dummy embedding and context
        query_embedding = np.array([0.1, 0.2, 0.3])
        context = {
            "entities": ["Python", "machine learning"],
            "conversation_context": {
                "recent_topics": ["programming", "AI"],
                "recent_entities": ["coding", "algorithms"],
            },
        }

        # Create a mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.4, 0.5, 0.6])
        context["embedding_model"] = mock_model

        # Enrich the embedding
        enriched = builder._enrich_embedding(query_embedding, context)

        # Verify the model was called with appropriate text
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert "Entities: Python, machine learning" in call_args
        assert "Topics: programming, AI" in call_args

        # Verify embedding was altered
        assert not np.array_equal(query_embedding, enriched)

        # Verify it's normalized
        assert abs(np.linalg.norm(enriched) - 1.0) < 1e-6

    def test_process_query(self, builder):
        """Test the full process_query method."""
        # Create a mock NLP extractor
        with patch.object(NLPExtractor, "extract_entities") as mock_extract_entities:
            with patch.object(builder, "_extract_temporal_information") as mock_extract_temporal:
                # Set up mock returns
                mock_extract_entities.return_value = ["Python", "programming"]
                mock_extract_temporal.return_value = {
                    "has_temporal_reference": True,
                    "time_type": "relative",
                    "relative_time": datetime.now().timestamp() - 86400,
                    "time_expressions": ["yesterday"],
                    "time_keywords": ["yesterday"],
                }

                # Call process_query
                context = {"query_embedding": np.array([0.1, 0.2, 0.3])}
                result = builder.process_query("What did I learn about Python yesterday?", context)

                # Verify result contains expected keys
                assert "entities" in result
                assert "temporal_markers" in result
                assert "original_query_embedding" in result
                assert "query_embedding" in result

                # Verify entity extraction was called
                mock_extract_entities.assert_called_once()

                # Verify temporal extraction was called
                mock_extract_temporal.assert_called_once()
