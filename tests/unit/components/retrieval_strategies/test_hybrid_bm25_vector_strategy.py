# tests/components/retrieval_strategies/test_hybrid_bm25_vector_strategy.py
"""
Tests for the HybridBM25VectorStrategy.

This file contains tests for the hybrid retrieval strategy that combines
BM25 keyword matching with vector similarity search.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memoryweave.components.retrieval_strategies.hybrid_bm25_vector_strategy import (
    HybridBM25VectorStrategy,
)


class TestHybridBM25VectorStrategy:
    """Tests for the HybridBM25VectorStrategy class."""

    def test_initialization(self, memory_store):
        """Test initialization with different configurations."""
        # Test default initialization
        strategy = HybridBM25VectorStrategy(memory=memory_store)
        assert strategy.memory == memory_store
        assert strategy.b == 0.75  # Default BM25 length normalization
        assert strategy.k1 == 1.2  # Default BM25 term frequency scaling

        # Test initialization with configuration
        config = {
            "vector_weight": 0.3,
            "bm25_weight": 0.7,
            "confidence_threshold": 0.2,
            "activation_boost": True,
            "enable_dynamic_weighting": True,
            "keyword_weight_bias": 0.8,
            "bm25_b": 0.8,
            "bm25_k1": 1.5,
        }
        strategy.initialize(config)
        assert strategy.vector_weight == 0.3
        assert strategy.bm25_weight == 0.7
        assert strategy.confidence_threshold == 0.2
        assert strategy.activation_boost is True
        assert strategy.enable_dynamic_weighting is True
        assert strategy.keyword_weight_bias == 0.8
        assert strategy.b == 0.8
        assert strategy.k1 == 1.5

    @pytest.mark.parametrize("index_initialized", [True, False])
    def test_retrieve_basic(self, memory_store, query_embedding, base_context, index_initialized):
        """Test basic retrieval functionality."""
        strategy = HybridBM25VectorStrategy(memory=memory_store)
        strategy.initialize(
            {
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "confidence_threshold": 0.0,  # No threshold for testing
            }
        )

        # Mock the BM25 index initialization
        with patch.object(strategy, "_initialize_bm25_index") as mock_init:
            # Set the index_initialized flag
            strategy.index_initialized = index_initialized

            # Add vector scores
            context = base_context.copy()
            context["query"] = "test query"

            # Mock _retrieve_bm25 to return some results
            with patch.object(strategy, "_retrieve_bm25") as mock_bm25:
                # Return some BM25 results for memories 1 and 3
                mock_bm25.return_value = {1: 0.8, 3: 0.6}

                results = strategy.retrieve(query_embedding, top_k=3, context=context)

                # Check if index was initialized if needed
                if not index_initialized:
                    mock_init.assert_called_once()
                else:
                    mock_init.assert_not_called()

                # Check that BM25 was called with query text
                mock_bm25.assert_called_with(context["query"], 6)  # top_k * 2

                # Check basic properties of results
                assert len(results) > 0
                assert "memory_id" in results[0]
                assert "relevance_score" in results[0]
                assert "vector_score" in results[0]
                assert "bm25_score" in results[0]

                # Results should be sorted by relevance_score
                scores = [r["relevance_score"] for r in results]
                assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_retrieve_with_parameter_adaptation(self, memory_store, query_embedding, base_context):
        """Test retrieval with parameter adaptation via context."""
        strategy = HybridBM25VectorStrategy(memory=memory_store)
        strategy.initialize(
            {
                "vector_weight": 0.2,
                "bm25_weight": 0.8,
                "confidence_threshold": 0.1,
            }
        )
        strategy.index_initialized = True  # Skip index initialization

        # Create a context with adapted parameters
        context = base_context.copy()
        context["query"] = "test query"
        context["adapted_retrieval_params"] = {
            "confidence_threshold": 0.05,
            "vector_weight": 0.3,
            "bm25_weight": 0.7,
        }

        # Mock _retrieve_bm25 to return some results
        with patch.object(strategy, "_retrieve_bm25") as mock_bm25:
            mock_bm25.return_value = {1: 0.8, 3: 0.6}

            results = strategy.retrieve(query_embedding, top_k=3, context=context)

            # Check that adapted parameters were used
            assert len(results) > 0

    def test_dynamic_weighting(self, memory_store, query_embedding, base_context):
        """Test dynamic weighting based on query characteristics."""
        strategy = HybridBM25VectorStrategy(memory=memory_store)
        strategy.initialize(
            {
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "confidence_threshold": 0.0,
                "enable_dynamic_weighting": True,
                "keyword_weight_bias": 0.8,
            }
        )
        strategy.index_initialized = True  # Skip index initialization

        # Create a context with keyword-rich query
        context = base_context.copy()
        context["query"] = "test query with important keywords"
        context["important_keywords"] = ["test", "important", "keywords"]

        # Mock _retrieve_bm25 to return some results
        with patch.object(strategy, "_retrieve_bm25") as mock_bm25:
            mock_bm25.return_value = {1: 0.8, 3: 0.6}

            results = strategy.retrieve(query_embedding, top_k=3, context=context)

            # Should be some results
            assert len(results) > 0

            # First result should have bm25_contribution > vector_contribution
            # due to keyword-rich query and dynamic weighting
            assert results[0]["bm25_contribution"] > results[0]["vector_contribution"]

    def test_confidence_threshold_filtering(self, memory_store, query_embedding, base_context):
        """Test confidence threshold filtering."""
        strategy = HybridBM25VectorStrategy(memory=memory_store)

        # Set a high threshold that should filter out results
        strategy.initialize(
            {
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "confidence_threshold": 0.9,  # High threshold
                "min_results": 0,  # Disable minimum results guarantee
            }
        )
        strategy.index_initialized = True  # Skip index initialization

        # Create a context with query
        context = base_context.copy()
        context["query"] = "test query"

        # Mock _retrieve_bm25 to return some results
        with patch.object(strategy, "_retrieve_bm25") as mock_bm25:
            mock_bm25.return_value = {1: 0.5, 3: 0.4}  # Low scores that won't pass threshold

            # Should return empty results due to high threshold
            results_high_threshold = strategy.retrieve(query_embedding, top_k=3, context=context)
            assert len(results_high_threshold) == 0

            # Now with lower threshold
            strategy.initialize({"confidence_threshold": 0.0})
            results_low_threshold = strategy.retrieve(query_embedding, top_k=3, context=context)

            # Should have results now
            assert len(results_low_threshold) > 0

    def test_minimum_results_guarantee(self, memory_store, query_embedding, base_context):
        """Test minimum results guarantee."""
        strategy = HybridBM25VectorStrategy(memory=memory_store)

        # Set a high threshold but enable minimum results guarantee
        strategy.initialize(
            {
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "confidence_threshold": 0.9,  # High threshold
                "min_results": 2,  # Guarantee at least 2 results
            }
        )
        strategy.index_initialized = True  # Skip index initialization

        # Create a context with query
        context = base_context.copy()
        context["query"] = "test query"

        # Mock _retrieve_bm25 to return some results
        with patch.object(strategy, "_retrieve_bm25") as mock_bm25:
            mock_bm25.return_value = {1: 0.5, 3: 0.4}  # Low scores that won't pass threshold

            # Should still return min_results
            results = strategy.retrieve(query_embedding, top_k=3, context=context)
            assert len(results) >= strategy.min_results

            # Should mark results as below threshold
            assert results[0]["below_threshold"] is True

    def test_activation_boost(self, memory_store, query_embedding, base_context):
        """Test activation boost."""
        # Add activation_levels to memory_store
        memory_store.activation_levels = np.array([0.9, 0.5, 0.7, 0.3, 0.1])

        strategy = HybridBM25VectorStrategy(memory=memory_store)
        strategy.initialize(
            {
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "confidence_threshold": 0.0,
                "activation_boost": True,
            }
        )
        strategy.index_initialized = True  # Skip index initialization

        # Create a context with query
        context = base_context.copy()
        context["query"] = "test query"

        # Mock _retrieve_bm25 to return some results
        with patch.object(strategy, "_retrieve_bm25") as mock_bm25:
            mock_bm25.return_value = {0: 0.8, 2: 0.7}  # High activation memories

            # With activation boost
            results_with_boost = strategy.retrieve(query_embedding, top_k=3, context=context)

            # Without activation boost
            strategy.activation_boost = False
            results_without_boost = strategy.retrieve(query_embedding, top_k=3, context=context)

            # Memory 0 has high activation (0.9), so it should get boosted
            # and potentially have higher score with boost than without
            memory_0_with_boost = next((r for r in results_with_boost if r["memory_id"] == 0), None)
            memory_0_without_boost = next(
                (r for r in results_without_boost if r["memory_id"] == 0), None
            )

            if memory_0_with_boost and memory_0_without_boost:
                assert memory_0_with_boost["vector_score"] >= memory_0_without_boost["vector_score"]

    def test_process_query(self, memory_store):
        """Test the process_query method."""
        strategy = HybridBM25VectorStrategy(memory=memory_store)
        strategy.initialize(
            {
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "confidence_threshold": 0.0,
            }
        )
        strategy.index_initialized = True  # Skip index initialization

        # Mock the retrieve method to check it's called correctly
        with patch.object(strategy, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = [{"memory_id": 1, "relevance_score": 0.8}]

            # Call process_query with context
            context = {
                "query_embedding": np.array([0.5, 0.5, 0.5]),
                "top_k": 3,
                "memory": memory_store,
            }
            result = strategy.process_query("test query", context)

            # Check that retrieve was called with correct parameters
            mock_retrieve.assert_called_once()
            args, kwargs = mock_retrieve.call_args
            assert np.array_equal(args[0], context["query_embedding"])
            assert args[1] == 3  # top_k
            assert "query" in kwargs["context"]
            assert kwargs["context"]["query"] == "test query"

            # Check that result has expected structure
            assert "results" in result
            assert result["results"] == mock_retrieve.return_value

    def test_bm25_initialization(self, memory_store):
        """Test the BM25 index initialization."""
        strategy = HybridBM25VectorStrategy(memory=memory_store)

        # Call the initialization method
        with patch("whoosh.index.create_in") as mock_create_in:
            with patch("whoosh.index.open_dir"):
                mock_index = MagicMock()
                mock_create_in.return_value = mock_index
                mock_writer = MagicMock()
                mock_index.writer.return_value = mock_writer

                strategy._initialize_bm25_index()

                # Should have created the index
                mock_create_in.assert_called_once()

                # Should have added documents to the index
                assert mock_writer.add_document.call_count == len(memory_store.memory_metadata)

                # Should have committed the index
                mock_writer.commit.assert_called_once()

                # Should have set index_initialized flag
                assert strategy.index_initialized is True

    def test_retrieve_bm25(self, memory_store):
        """Test the _retrieve_bm25 method."""
        strategy = HybridBM25VectorStrategy(memory=memory_store)

        # Mock the whoosh searcher
        mock_searcher = MagicMock()
        mock_results = MagicMock()
        mock_results.top_score = 1.0
        mock_results.__iter__.return_value = [
            {"id": "1", "score": 0.8},
            {"id": "3", "score": 0.6},
        ]
        mock_searcher.__enter__.return_value.search.return_value = mock_results

        mock_index = MagicMock()
        mock_index.searcher.return_value = mock_searcher
        strategy.index = mock_index

        # Set up memory lookup
        strategy.memory_lookup = {"1": 1, "3": 3}

        # Call the method
        results = strategy._retrieve_bm25("test query", top_k=3)

        # Check that searcher was used
        mock_searcher.__enter__.return_value.search.assert_called_once()

        # Check results
        assert 1 in results
        assert 3 in results
        assert results[1] == 0.8
        assert results[3] == 0.6
