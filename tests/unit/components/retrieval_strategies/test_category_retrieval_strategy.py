# tests/components/retrieval_strategies/test_category_retrieval_strategy.py
"""
Tests for CategoryRetrievalStrategy.

This file tests the functionality of the category-based retrieval strategy,
which uses ART category clustering to improve retrieval results.
"""

from unittest.mock import patch

import numpy as np

from memoryweave.components.retrieval_strategies_impl import (
    CategoryRetrievalStrategy,
    SimilarityRetrievalStrategy,
)


class TestCategoryRetrievalStrategy:
    """Test suite for CategoryRetrievalStrategy."""

    def test_initialization(self, memory_store):
        """Test strategy initialization with default and custom parameters."""
        # Test with default parameters
        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize({})

        assert strategy.confidence_threshold == 0.0
        assert strategy.max_categories == 3
        assert strategy.activation_boost is True
        assert strategy.min_results == 5

        # Test with custom parameters
        custom_config = {
            "confidence_threshold": 0.2,
            "max_categories": 5,
            "activation_boost": False,
            "min_results": 10,
            "category_selection_threshold": 0.6,
        }

        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize(custom_config)

        assert strategy.confidence_threshold == 0.2
        assert strategy.max_categories == 5
        assert strategy.activation_boost is False
        assert strategy.min_results == 10
        assert strategy.category_selection_threshold == 0.6

    def test_retrieve_with_categories(
        self, memory_store, query_embedding, base_context, mock_category_manager
    ):
        """Test retrieving memories with category manager available."""
        # Attach category manager to memory store
        memory_store.category_manager = mock_category_manager

        # Initialize strategy
        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize({"confidence_threshold": 0.0})

        # Update mock to ensure it returns sufficient results
        memory_store.retrieve_memories.return_value = [
            (0, 0.9, {"content": "Memory 0", "source": "test"}),
            (1, 0.8, {"content": "Memory 1", "source": "test"}),
            (2, 0.7, {"content": "Memory 2", "source": "test"}),
        ]

        # Retrieve memories
        results = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # Check results
        assert len(results) == 3, "Should return 3 results"

    def test_retrieve_with_adaptive_parameters(
        self, memory_store, query_embedding, base_context, mock_category_manager
    ):
        """Test that strategy uses adapted parameters from context."""
        # Attach category manager to memory store
        memory_store.category_manager = mock_category_manager

        # Initialize strategy
        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize({"confidence_threshold": 0.0, "max_categories": 2})

        # Create context with adapted parameters
        adapted_context = base_context.copy()
        adapted_context["adapted_retrieval_params"] = {
            "confidence_threshold": 0.3,
            "max_categories": 1,
        }

        # Patch get_category_similarities to verify it's called with correct parameters
        with patch.object(
            mock_category_manager,
            "get_category_similarities",
            return_value=np.array([0.8, 0.6, 0.4]),
        ) as mock_gcs:
            results = strategy.retrieve(query_embedding, top_k=3, context=adapted_context)

            # Verify get_category_similarities was called
            mock_gcs.assert_called_once_with(query_embedding)

            # Should only use 1 category (from adapted parameters)
            # Only memories from category 0 should be returned
            assert all(r["category_id"] == 0 for r in results), (
                "All results should be from category 0"
            )

            # Results should have relevance scores above the adapted threshold
            assert all(r["relevance_score"] >= 0.3 for r in results), (
                "All results should have scores above adapted threshold"
            )

    def test_fallback_to_similarity(self, memory_store, query_embedding, base_context):
        """Test fallback to similarity retrieval when no category manager is available."""
        # No category manager attached

        # Initialize strategy
        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize({"confidence_threshold": 0.0})

        # Mock the SimilarityRetrievalStrategy to verify it's used as fallback
        with patch.object(
            SimilarityRetrievalStrategy,
            "retrieve",
            return_value=[{"memory_id": 0, "relevance_score": 0.8}],
        ) as mock_sim_retrieve:
            results = strategy.retrieve(query_embedding, top_k=3, context=base_context)

            # Verify SimilarityRetrievalStrategy.retrieve was called
            mock_sim_retrieve.assert_called_once()

            # Should return results from similarity retrieval
            assert results == [{"memory_id": 0, "relevance_score": 0.8}]

    def test_no_categories_selected(
        self,
        memory_store,
        query_embedding,
        base_context,
        mock_category_manager,
    ):
        """Test case where no categories meet the selection threshold."""
        # Attach category manager to memory store
        memory_store.category_manager = mock_category_manager

        # Initialize strategy
        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize({"category_selection_threshold": 0.9})  # Higher than any similarity

        # Override get_category_similarities to return low similarities
        mock_category_manager.get_category_similarities.return_value = np.array([0.3, 0.2, 0.1])

        # Ensure memory_store.retrieve_memories returns results
        memory_store.retrieve_memories.return_value = [
            (0, 0.7, {"content": "Memory 0", "source": "test"}),
            (1, 0.6, {"content": "Memory 1", "source": "test"}),
        ]

        # Retrieve memories
        results = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # Should return results even when no categories meet threshold
        assert len(results) > 0, "Should return results even when no categories meet threshold"

    def test_minimum_results_guarantee(
        self, memory_store, query_embedding, base_context, mock_category_manager
    ):
        """Test minimum results guarantee when no results pass confidence threshold."""
        # Attach category manager to memory store
        memory_store.category_manager = mock_category_manager

        # Configure strategy with high category selection threshold
        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize(
            {
                "confidence_threshold": 0.9,  # Very high to force fallback
                "min_results": 2,  # Guarantee at least 2 results
            }
        )

        # Configure mock_category_manager to return low similarities
        mock_category_manager.get_category_similarities.return_value = np.array([0.3, 0.2, 0.1])

        # Make sure memory_store.retrieve_memories returns enough results
        memory_store.retrieve_memories.return_value = [
            (0, 0.3, {"content": "Memory 0"}),
            (1, 0.2, {"content": "Memory 1"}),
        ]

        # Retrieve memories
        results = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # Should return min_results results
        assert len(results) >= strategy.min_results, "Should return min_results results"

    def test_error_handling(
        self, memory_store, query_embedding, base_context, mock_category_manager
    ):
        """Test error handling during category retrieval."""
        # Attach category manager to memory store
        memory_store.category_manager = mock_category_manager

        # Initialize strategy
        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize({"confidence_threshold": 0.0})

        # Make get_category_similarities raise an exception
        mock_category_manager.get_category_similarities.side_effect = Exception("Test exception")

        # Mock SimilarityRetrievalStrategy to verify fallback
        with patch.object(
            SimilarityRetrievalStrategy,
            "retrieve",
            return_value=[{"memory_id": 0, "relevance_score": 0.8}],
        ) as mock_sim_retrieve:
            results = strategy.retrieve(query_embedding, top_k=3, context=base_context)

            # Should fall back to similarity retrieval
            mock_sim_retrieve.assert_called_once()

            # Should return results from similarity retrieval
            assert results == [{"memory_id": 0, "relevance_score": 0.8}]

    def test_process_query(
        self, memory_store, query_embedding, base_context, mock_category_manager
    ):
        """Test process_query method for end-to-end functionality."""
        # Attach category manager to memory store
        memory_store.category_manager = mock_category_manager

        # Initialize strategy
        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize({"confidence_threshold": 0.0})

        # Create context with query embedding
        context_with_embedding = base_context.copy()
        context_with_embedding["query_embedding"] = query_embedding

        # Mock the retrieve method to isolate testing of process_query
        with patch.object(
            strategy, "retrieve", return_value=[{"memory_id": 0, "relevance_score": 0.8}]
        ) as mock_retrieve:
            result = strategy.process_query("test query", context_with_embedding)

            # Verify retrieve was called with correct parameters
            mock_retrieve.assert_called_once_with(query_embedding, 3, context_with_embedding)

            # Check result structure
            assert "results" in result
            assert result["results"] == [{"memory_id": 0, "relevance_score": 0.8}]

    def test_process_query_with_missing_embedding(
        self, memory_store, base_context, mock_category_manager
    ):
        """Test process_query when query embedding is missing."""
        # Attach category manager to memory store
        memory_store.category_manager = mock_category_manager

        # Initialize strategy
        strategy = CategoryRetrievalStrategy(memory_store)
        strategy.initialize({"confidence_threshold": 0.0})

        # Context without query embedding
        context_without_embedding = base_context.copy()

        # Mock numpy.zeros to avoid numpy array creation errors
        with patch("numpy.zeros") as mock_zeros:
            # Return a simple array for testing
            mock_zeros.return_value = np.array([0.1, 0.1, 0.1])

            # Mock the strategy's retrieve method for simplicity
            with patch.object(strategy, "retrieve") as mock_retrieve:
                mock_retrieve.return_value = [{"memory_id": 0, "relevance_score": 0.8}]

                # Call process_query - should handle missing embedding
                result = strategy.process_query("test query", context_without_embedding)

                # Should have results
                assert "results" in result
                assert len(result["results"]) > 0
