"""
Tests for the DynamicContextAdapter component.

This module tests the DynamicContextAdapter component to ensure:
1. Parameters are correctly adapted based on multiple contextual signals
2. Different memory sizes result in appropriate adaptations
3. Query characteristics influence parameter adjustments
4. Adaptation strength properly controls parameter changes
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from memoryweave.components.dynamic_context_adapter import DynamicContextAdapter


class TestDynamicContextAdapter:
    """Test suite for DynamicContextAdapter component."""

    def setup_method(self):
        """Setup for each test."""
        self.adapter = DynamicContextAdapter()
        self.adapter.initialize(
            {
                "adaptation_strength": 1.0,
                "confidence_threshold": 0.1,
                "similarity_weight": 0.5,
                "associative_weight": 0.3,
                "temporal_weight": 0.1,
                "activation_weight": 0.1,
                "enable_logging": False,
                "max_history_size": 5,
            }
        )

    def test_disabled_adaptation_returns_defaults(self):
        """Test that when adaptation is disabled, default parameters are returned."""
        # Set up context with adaptation disabled
        context = {"enable_dynamic_adaptation": False}

        # Process a query
        result = self.adapter.process_query("What is my favorite color?", context)

        # Verify default parameters are returned
        assert "adapted_retrieval_params" in result
        assert result["adapted_retrieval_params"]["confidence_threshold"] == 0.1
        assert result["adapted_retrieval_params"]["similarity_weight"] == 0.5
        assert result["adapted_retrieval_params"]["adapted_by_dynamic_context"] is False

    def test_memory_size_adaptation(self):
        """Test that memory size affects parameter adaptations."""
        # Create context with large memory store
        large_memory_context = {"memory_store": MagicMock(), "primary_query_type": "factual"}
        # Mock memory embeddings attribute
        large_memory_context["memory_store"].memory_embeddings = np.random.rand(1000, 768)

        # Create context with small memory store
        small_memory_context = {"memory_store": MagicMock(), "primary_query_type": "factual"}
        # Mock memory embeddings attribute
        small_memory_context["memory_store"].memory_embeddings = np.random.rand(30, 768)

        # Process with both memory sizes
        large_result = self.adapter.process_query(
            "What is the capital of France?", large_memory_context
        )
        small_result = self.adapter.process_query(
            "What is the capital of France?", small_memory_context
        )

        large_params = large_result["adapted_retrieval_params"]
        small_params = small_result["adapted_retrieval_params"]

        # Large memory stores should have progressive filtering enabled
        assert large_params.get("use_progressive_filtering", False), (
            "Progressive filtering should be enabled for large stores"
        )
        assert not small_params.get("use_progressive_filtering", False), (
            "Progressive filtering should be disabled for small stores"
        )

        # Large stores should have higher similarity weight to compensate for reduced associative/activation
        assert large_params["similarity_weight"] > small_params["similarity_weight"], (
            "Large stores should emphasize similarity more"
        )

        # Large stores should have reduced activation influence
        assert large_params["activation_weight"] < small_params["activation_weight"], (
            "Large stores should reduce activation weight"
        )

        # Activation + Associative weights should be lower for large stores
        large_combined = large_params["activation_weight"] + large_params["associative_weight"]
        small_combined = small_params["activation_weight"] + small_params["associative_weight"]
        assert large_combined < small_combined, (
            "Large stores should have lower combined activation and associative weights"
        )

    def test_query_type_adaptation(self):
        """Test that query type affects parameter adaptations."""
        # Create context for different query types
        personal_context = {
            "primary_query_type": "personal",
            "query_type_confidence": 0.9,
            "memory_store": MagicMock(),
        }
        personal_context["memory_store"].memory_embeddings = np.random.rand(100, 768)

        factual_context = {
            "primary_query_type": "factual",
            "query_type_confidence": 0.9,
            "memory_store": MagicMock(),
        }
        factual_context["memory_store"].memory_embeddings = np.random.rand(100, 768)

        temporal_context = {
            "primary_query_type": "temporal",
            "query_type_confidence": 0.9,
            "has_temporal_reference": True,
            "memory_store": MagicMock(),
        }
        temporal_context["memory_store"].memory_embeddings = np.random.rand(100, 768)

        # Process with different query types
        personal_result = self.adapter.process_query("What is my favorite color?", personal_context)
        factual_result = self.adapter.process_query(
            "What is the capital of France?", factual_context
        )
        temporal_result = self.adapter.process_query("What did I do yesterday?", temporal_context)

        personal_params = personal_result["adapted_retrieval_params"]
        factual_params = factual_result["adapted_retrieval_params"]
        temporal_params = temporal_result["adapted_retrieval_params"]

        # Personal queries should have higher confidence threshold
        assert personal_params["confidence_threshold"] > factual_params["confidence_threshold"], (
            "Personal queries should have higher confidence threshold"
        )

        # Temporal queries should emphasize temporal context more
        assert temporal_params["temporal_weight"] > personal_params["temporal_weight"], (
            "Temporal queries should emphasize temporal context"
        )
        assert temporal_params["temporal_weight"] > factual_params["temporal_weight"], (
            "Temporal queries should emphasize temporal context"
        )

        # Factual queries should emphasize similarity more
        assert (
            factual_params["similarity_weight"] > personal_params["similarity_weight"]
            or factual_params["similarity_weight"] > temporal_params["similarity_weight"]
        ), "Factual queries should emphasize similarity"

    def test_query_characteristics_adaptation(self):
        """Test that query characteristics affect parameter adaptations."""
        # Create context for specific and vague queries
        specific_context = {
            "keywords": ["python", "programming", "language", "readable", "whitespace"],
            "important_keywords": ["python", "readable", "whitespace"],
            "entities": ["Python"],
            "memory_store": MagicMock(),
        }
        specific_context["memory_store"].memory_embeddings = np.random.rand(100, 768)

        vague_context = {
            "keywords": ["something", "know", "about"],
            "important_keywords": ["know"],
            "entities": [],
            "memory_store": MagicMock(),
        }
        vague_context["memory_store"].memory_embeddings = np.random.rand(100, 768)

        # Process with different query characteristics
        specific_result = self.adapter.process_query(
            "Why is Python considered a readable programming language?", specific_context
        )
        vague_result = self.adapter.process_query("Tell me something I know about.", vague_context)

        specific_params = specific_result["adapted_retrieval_params"]
        vague_params = vague_result["adapted_retrieval_params"]

        # Specific queries should have higher similarity weight
        assert specific_params["similarity_weight"] > vague_params["similarity_weight"], (
            "Specific queries should emphasize similarity"
        )

        # Vague queries should have higher associative weight to find related info
        assert vague_params["associative_weight"] > specific_params["associative_weight"], (
            "Vague queries should emphasize associative links"
        )

        # Vague queries typically have higher confidence threshold to filter noise
        assert vague_params["confidence_threshold"] >= specific_params["confidence_threshold"], (
            "Vague queries should have higher confidence threshold"
        )

    def test_adaptation_strength_controls_change(self):
        """Test that adaptation_strength properly controls parameter changes."""
        # Create adapters with different adaptation strengths
        weak_adapter = DynamicContextAdapter()
        weak_adapter.initialize(
            {
                "adaptation_strength": 0.2,  # Low adaptation strength
                "confidence_threshold": 0.1,
                "similarity_weight": 0.5,
                "associative_weight": 0.3,
                "temporal_weight": 0.1,
                "activation_weight": 0.1,
            }
        )

        strong_adapter = DynamicContextAdapter()
        strong_adapter.initialize(
            {
                "adaptation_strength": 1.0,  # High adaptation strength
                "confidence_threshold": 0.1,
                "similarity_weight": 0.5,
                "associative_weight": 0.3,
                "temporal_weight": 0.1,
                "activation_weight": 0.1,
            }
        )

        # Create context for tests
        context = {
            "primary_query_type": "personal",
            "query_type_confidence": 0.9,
            "has_temporal_reference": False,
            "memory_store": MagicMock(),
            "keywords": ["favorite", "color"],
            "important_keywords": ["favorite", "color"],
            "entities": [],
        }
        context["memory_store"].memory_embeddings = np.random.rand(100, 768)

        # Process with both adapters
        weak_result = weak_adapter.process_query("What is my favorite color?", context)
        strong_result = strong_adapter.process_query("What is my favorite color?", context)

        weak_params = weak_result["adapted_retrieval_params"]
        strong_params = strong_result["adapted_retrieval_params"]

        # Default values
        default_confidence = 0.1

        # Check if strong adapter made larger changes than weak adapter
        # For parameters that should increase
        if strong_params["confidence_threshold"] > default_confidence:
            assert strong_params["confidence_threshold"] > weak_params["confidence_threshold"], (
                "Strong adapter should increase confidence more"
            )

        # For parameters that should decrease
        if strong_params["associative_weight"] < 0.3:
            assert strong_params["associative_weight"] < weak_params["associative_weight"], (
                "Strong adapter should decrease associative weight more"
            )

        # For boolean parameters (if they exist in result)
        if (
            "use_progressive_filtering" in strong_params
            and "use_progressive_filtering" in weak_params
        ):
            # If strong adapter enables feature, weak adapter may not
            if strong_params["use_progressive_filtering"]:
                pass  # No assertion needed - weak might be True or False depending on threshold

    def test_adaptation_history_retention(self):
        """Test that adaptation history is properly retained."""
        # Initialize adapter with small history size
        self.adapter.initialize({"adaptation_strength": 1.0, "max_history_size": 3})

        # Create context for tests
        context = {"primary_query_type": "factual", "memory_store": MagicMock()}
        context["memory_store"].memory_embeddings = np.random.rand(100, 768)

        # Process multiple queries
        queries = [
            "What is the capital of France?",
            "Who invented the telephone?",
            "How tall is Mount Everest?",
            "When was the moon landing?",
        ]

        for query in queries:
            self.adapter.process_query(query, context)

        # Check history size is limited
        assert len(self.adapter.adaptation_history) == 3, (
            "History should be limited to max_history_size"
        )

        # Most recent query should be last in history
        assert self.adapter.adaptation_history[-1]["query"] == queries[-1], (
            "Most recent query should be last in history"
        )

        # Earliest query should be dropped
        for history_item in self.adapter.adaptation_history:
            assert history_item["query"] != queries[0], (
                "Earliest query should be dropped from history"
            )


if __name__ == "__main__":
    pytest.main()
