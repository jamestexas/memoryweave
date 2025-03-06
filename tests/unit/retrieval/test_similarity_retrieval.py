"""
Tests for the SimilarityRetrievalStrategy.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from memoryweave.interfaces.memory import IMemoryStore, IVectorStore
from memoryweave.retrieval.similarity import SimilarityRetrievalStrategy


class TestSimilarityRetrievalStrategy:
    """Test suite for the SimilarityRetrievalStrategy class."""

    @pytest.fixture
    def mock_memory_store(self):
        """Create a mock memory store."""
        mock_store = MagicMock(spec=IMemoryStore)

        # Configure the mock to return test data
        def mock_get(memory_id):
            from memoryweave.interfaces.memory import Memory

            return Memory(
                id=memory_id,
                embedding=np.array([0.1, 0.2, 0.3]),
                content={"text": f"Memory {memory_id}"},
                metadata={"source": "test", "importance": 0.8},
            )

        mock_store.get.side_effect = mock_get
        return mock_store

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = MagicMock(spec=IVectorStore)

        # Configure the mock to return test search results
        def mock_search(query_vector, k, threshold=None):
            return [("memory1", 0.95), ("memory2", 0.85), ("memory3", 0.75)][:k]

        mock_store.search.side_effect = mock_search
        return mock_store

    def test_init(self, mock_memory_store, mock_vector_store):
        """Test initialization of strategy."""
        strategy = SimilarityRetrievalStrategy(mock_memory_store, mock_vector_store)

        assert strategy._memory_store == mock_memory_store
        assert strategy._vector_store == mock_vector_store
        assert strategy._default_params["similarity_threshold"] == 0.6
        assert strategy._default_params["max_results"] == 10

    def test_retrieve_basic(self, mock_memory_store, mock_vector_store):
        """Test basic retrieval functionality."""
        strategy = SimilarityRetrievalStrategy(mock_memory_store, mock_vector_store)

        # Create query embedding
        query_embedding = np.array([0.1, 0.2, 0.3])

        # Retrieve memories
        results = strategy.retrieve(query_embedding)

        # Verify the correct methods were called
        mock_vector_store.search.assert_called_once()

        # Check call arguments more carefully
        assert mock_vector_store.search.call_args, "Vector search not appropriately called"
        # Check that search was called with the right arguments
        mock_vector_store.search.assert_called_with(
            query_vector=query_embedding, k=10, threshold=0.6
        )

        # Verify the correct memories were retrieved
        assert len(results) == 3
        assert results[0]["memory_id"] == "memory1"
        assert results[0]["relevance_score"] == 0.95
        assert results[0]["content"] == "Memory memory1"

        # Check all memories were retrieved via memory_store.get
        assert mock_memory_store.get.call_count == 3

    def test_retrieve_with_parameters(self, mock_memory_store, mock_vector_store):
        """Test retrieval with custom parameters."""
        strategy = SimilarityRetrievalStrategy(mock_memory_store, mock_vector_store)

        # Create query embedding
        query_embedding = np.array([0.1, 0.2, 0.3])

        # Custom parameters
        parameters = {"similarity_threshold": 0.8, "max_results": 2}

        # Retrieve memories
        results = strategy.retrieve(query_embedding, parameters)

        # Verify the correct parameters were used
        mock_vector_store.search.assert_called_once()

        # Check that search was called with the right arguments
        mock_vector_store.search.assert_called_with(
            query_vector=query_embedding, k=2, threshold=0.8
        )

        # Verify the correct number of results
        assert len(results) == 2

    def test_configure(self, mock_memory_store, mock_vector_store):
        """Test configuring the strategy."""
        strategy = SimilarityRetrievalStrategy(mock_memory_store, mock_vector_store)

        # Initial defaults
        assert strategy._default_params["similarity_threshold"] == 0.6
        assert strategy._default_params["max_results"] == 10

        # Configure with new defaults
        strategy.configure({"similarity_threshold": 0.7, "max_results": 5})

        # Verify updated defaults
        assert strategy._default_params["similarity_threshold"] == 0.7
        assert strategy._default_params["max_results"] == 5

        # Configure with partial update
        strategy.configure({"similarity_threshold": 0.8})

        # Verify partially updated defaults
        assert strategy._default_params["similarity_threshold"] == 0.8
        assert strategy._default_params["max_results"] == 5
