"""
Tests for the SimilarityRetrievalStrategy.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from memoryweave.components.retrieval_strategies_impl import SimilarityRetrievalStrategy
from memoryweave.interfaces.memory import IMemoryStore


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

        # Mock retrieve_memories method
        def mock_retrieve_memories(
            query_embedding, top_k=10, confidence_threshold=0.0, activation_boost=True
        ):
            # Return tuple of (memory_id, score, metadata)
            return [
                ("memory1", 0.95, {"content": "Memory memory1", "source": "test"}),
                ("memory2", 0.85, {"content": "Memory memory2", "source": "test"}),
                ("memory3", 0.75, {"content": "Memory memory3", "source": "test"}),
            ][:top_k]

        mock_store.get.side_effect = mock_get
        mock_store.retrieve_memories.side_effect = mock_retrieve_memories
        return mock_store

    def test_init(self, mock_memory_store):
        """Test initialization of strategy."""
        strategy = SimilarityRetrievalStrategy(mock_memory_store)

        assert strategy.memory == mock_memory_store
        # Check default parameters are set
        assert hasattr(strategy, "confidence_threshold")
        assert hasattr(strategy, "activation_boost")
        assert hasattr(strategy, "min_results")

    def test_initialize(self, mock_memory_store):
        """Test initializing the strategy with configuration."""
        strategy = SimilarityRetrievalStrategy(mock_memory_store)

        # Initialize with custom parameters
        strategy.initialize(
            config={
                "confidence_threshold": 0.7,
                "activation_boost": False,
                "min_results": 5,
            }
        )

        # Verify the parameters were set
        assert strategy.confidence_threshold == 0.7
        assert strategy.activation_boost is False
        assert strategy.min_results == 5

    def test_retrieve_basic(self, mock_memory_store):
        """Test basic retrieval functionality."""
        strategy = SimilarityRetrievalStrategy(mock_memory_store)

        # Create query embedding
        query_embedding = np.array([0.1, 0.2, 0.3])

        # Set up context
        context = {"memory": mock_memory_store}

        # Retrieve memories
        results = strategy.retrieve(query_embedding, 3, context)

        # Verify the correct methods were called
        mock_memory_store.retrieve_memories.assert_called_once()

        # Check that retrieve_memories was called with the right arguments
        mock_memory_store.retrieve_memories.assert_called_with(
            query_embedding=query_embedding,
            top_k=3,
            confidence_threshold=0.0,
            activation_boost=True,
        )

        # Verify the correct memories were retrieved
        assert len(results) == 3
        assert results[0]["memory_id"] == "memory1"
        assert results[0]["relevance_score"] == 0.95
        assert results[0]["content"] == "Memory memory1"

    def test_retrieve_with_parameters(self, mock_memory_store):
        """Test retrieval with custom parameters."""
        strategy = SimilarityRetrievalStrategy(mock_memory_store)
        strategy.initialize({"confidence_threshold": 0.3})

        # Create query embedding
        query_embedding = np.array([0.1, 0.2, 0.3])

        # Create context with adapted parameters
        context = {
            "memory": mock_memory_store,
            "adapted_retrieval_params": {"confidence_threshold": 0.8},
        }

        # Retrieve memories
        results = strategy.retrieve(query_embedding, 2, context)

        # Verify the correct parameters were used
        mock_memory_store.retrieve_memories.assert_called_once()

        # Check that retrieve_memories was called with the right arguments
        mock_memory_store.retrieve_memories.assert_called_with(
            query_embedding=query_embedding,
            top_k=2,
            confidence_threshold=0.8,
            activation_boost=True,
        )

        # Verify the correct number of results
        assert len(results) == 2

    def test_process_query(self, mock_memory_store):
        """Test the process_query method."""
        strategy = SimilarityRetrievalStrategy(mock_memory_store)

        # Set up a query and context
        query = "test query"
        query_embedding = np.array([0.1, 0.2, 0.3])
        context = {"query_embedding": query_embedding, "memory": mock_memory_store, "top_k": 3}

        # Process the query
        result = strategy.process_query(query, context)

        # Verify we got results back
        assert "results" in result
        assert len(result["results"]) == 3

        # Check the first result
        assert result["results"][0]["memory_id"] == "memory1"
        assert result["results"][0]["relevance_score"] == 0.95
