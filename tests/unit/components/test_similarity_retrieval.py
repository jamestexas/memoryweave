# ruff: noqa: S101
"""
Tests for the SimilarityRetrievalStrategy.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from memoryweave.components.retrieval_strategies_impl import SimilarityRetrievalStrategy
from memoryweave.interfaces.memory import IMemoryStore


@pytest.fixture
def mock_memory_store():
    """Fixture to create a mock memory store."""
    mock_store = MagicMock(spec=IMemoryStore)

    def mock_get(memory_id):
        from memoryweave.interfaces.memory import Memory

        return Memory(
            id=memory_id,
            embedding=np.array([0.1, 0.2, 0.3]),
            content={"text": f"Memory {memory_id}"},
            metadata={"source": "test", "importance": 0.8},
        )

    def mock_retrieve_memories(
        query_embedding,
        top_k=10,
        confidence_threshold=0.0,
        activation_boost=True,
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


def test_init(mock_memory_store):
    """Test initialization of the strategy."""
    strategy = SimilarityRetrievalStrategy(memory=mock_memory_store)
    assert strategy.memory is mock_memory_store, "Memory store should be assigned correctly"
    for attr in ("confidence_threshold", "activation_boost", "min_results"):
        assert hasattr(strategy, attr), f"Strategy missing attribute: {attr}"


def test_initialize(mock_memory_store):
    """Test initializing the strategy with configuration."""
    strategy = SimilarityRetrievalStrategy(memory=mock_memory_store)
    config = {
        "confidence_threshold": 0.7,
        "activation_boost": False,
        "min_results": 5,
    }
    strategy.initialize(config=config)
    assert strategy.confidence_threshold == 0.7, "Confidence threshold not set correctly"
    assert strategy.activation_boost is False, "Activation boost not set correctly"
    assert strategy.min_results == 5, "Min results not set correctly"


def test_retrieve_basic(mock_memory_store):
    """Test basic retrieval functionality."""
    strategy = SimilarityRetrievalStrategy(memory=mock_memory_store)
    query_embedding = np.array([0.1, 0.2, 0.3])
    context = {"memory": mock_memory_store}
    results = strategy.retrieve(query_embedding, 3, context)

    mock_memory_store.retrieve_memories.assert_called_once_with(
        query_embedding=query_embedding,
        top_k=3,
        confidence_threshold=0.0,
        activation_boost=True,
    )

    assert len(results) == 3, "Expected 3 results"
    first_result = results[0]
    assert first_result["memory_id"] == "memory1", "First result memory_id mismatch"
    assert first_result["relevance_score"] == 0.95, "First result relevance_score mismatch"
    assert first_result["content"] == "Memory memory1", "First result content mismatch"


def test_retrieve_with_parameters(mock_memory_store):
    """Test retrieval with custom parameters."""
    strategy = SimilarityRetrievalStrategy(memory=mock_memory_store)
    strategy.initialize({"confidence_threshold": 0.3})
    query_embedding = np.array([0.1, 0.2, 0.3])
    context = {
        "memory": mock_memory_store,
        "adapted_retrieval_params": {"confidence_threshold": 0.8},
    }
    results = strategy.retrieve(query_embedding, 2, context)

    mock_memory_store.retrieve_memories.assert_called_once_with(
        query_embedding=query_embedding,
        top_k=2,
        confidence_threshold=0.8,
        activation_boost=True,
    )

    assert len(results) == 2, "Expected 2 results"


def test_process_query(mock_memory_store):
    """Test the process_query method."""
    strategy = SimilarityRetrievalStrategy(memory=mock_memory_store)
    query = "test query"
    query_embedding = np.array([0.1, 0.2, 0.3])
    context = {"query_embedding": query_embedding, "memory": mock_memory_store, "top_k": 3}
    result = strategy.process_query(query, context)

    assert "results" in result, "Result should contain 'results' key"
    results = result["results"]
    assert len(results) == 3, "Expected 3 results in processed query"
    first_result = results[0]
    assert first_result["memory_id"] == "memory1", (
        "First result memory_id mismatch in processed query"
    )
    assert first_result["relevance_score"] == 0.95, (
        "First result relevance_score mismatch in processed query"
    )
