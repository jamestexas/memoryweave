# tests/components/retrieval_strategies/conftest.py
"""Test fixtures for retrieval strategy tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def memory_store():
    # Create a mock memory store with test data
    mock_store = MagicMock()

    # Create test embeddings and metadata
    mock_store.memory_embeddings = np.array(
        [
            [0.1, 0.2, 0.3],  # Memory 0
            [0.4, 0.5, 0.6],  # Memory 1
            [0.7, 0.8, 0.9],  # Memory 2
            [0.9, 0.8, 0.7],  # Memory 3
            [0.6, 0.5, 0.4],  # Memory 4
        ]
    )

    # Set normalized embeddings (unit vectors)
    norms = np.linalg.norm(mock_store.memory_embeddings, axis=1, keepdims=True)
    mock_store.memory_embeddings = mock_store.memory_embeddings / norms

    # Create test metadata
    mock_store.memory_metadata = [
        {"content": "Memory content 0", "created_at": 100},
        {"content": "Memory content 1", "created_at": 200},
        {"content": "Memory content 2", "created_at": 300},
        {"content": "Memory content 3", "created_at": 400},
        {"content": "Memory content 4", "created_at": 500},
    ]

    # Mock get method
    mock_store.get = lambda idx: type(
        "Memory",
        (),
        {
            "id": idx,
            "content": mock_store.memory_metadata[idx]["content"],
            "metadata": mock_store.memory_metadata[idx],
        },
    )

    # Mock get_all method
    mock_store.get_all = lambda: [mock_store.get(i) for i in range(len(mock_store.memory_metadata))]

    # Mock search_by_vector method
    def mock_search_by_vector(query_vector, limit=10, threshold=0.0):
        similarities = np.dot(mock_store.memory_embeddings, query_vector)

        results = []
        for idx in np.argsort(-similarities)[:limit]:
            if similarities[idx] >= threshold:
                results.append(
                    {
                        "memory_id": idx,
                        "content": mock_store.memory_metadata[idx]["content"],
                        "score": float(similarities[idx]),
                        **mock_store.memory_metadata[idx],
                    }
                )
        return results

    mock_store.search_by_vector = mock_search_by_vector

    return mock_store


@pytest.fixture
def query_embedding():
    # Create a test query embedding
    embedding = np.array([0.5, 0.5, 0.5])
    return embedding / np.linalg.norm(embedding)  # Normalize


@pytest.fixture
def base_context():
    return {
        "query": "Test query",
        "top_k": 3,
        "current_time": 1000,
    }


@pytest.fixture
def mock_category_manager():
    """Create a mock category manager for testing."""
    mock_cm = MagicMock()

    # Mock category similarities
    mock_cm.get_category_similarities.return_value = np.array([0.8, 0.6, 0.4])

    # Mock categories dictionary
    mock_cm.categories = {0: "Category 0", 1: "Category 1", 2: "Category 2"}

    # Mock get_memories_for_category
    def get_memories_for_category(cat_idx):
        # Map categories to memories
        category_to_memories = {
            0: [0, 2],  # Category 0 contains memories 0 and 2
            1: [1, 3],  # Category 1 contains memories 1 and 3
            2: [4],  # Category 2 contains memory 4
        }
        return category_to_memories.get(cat_idx, [])

    mock_cm.get_memories_for_category = get_memories_for_category

    # Mock get_category_for_memory
    def get_category_for_memory(memory_id):
        # Map memories to categories
        memory_to_category = {
            0: 0,
            2: 0,  # Memories 0 and 2 belong to category 0
            1: 1,
            3: 1,  # Memories 1 and 3 belong to category 1
            4: 2,  # Memory 4 belongs to category 2
        }
        return memory_to_category.get(memory_id, -1)

    mock_cm.get_category_for_memory = get_category_for_memory

    return mock_cm


@pytest.fixture
def associative_linker():
    """Create a mock associative linker."""
    mock_linker = MagicMock()
    mock_linker.traverse_associative_network.return_value = {
        1: 0.9,  # Memory 1 with high strength
        3: 0.7,  # Memory 3 with medium strength
    }
    return mock_linker


@pytest.fixture
def temporal_context():
    """Create a mock temporal context."""
    mock_context = MagicMock()
    mock_context.extract_time_references.return_value = {
        "has_temporal_reference": True,
        "time_type": "relative",
        "relative_time": 300,
        "time_keywords": ["yesterday"],
    }
    return mock_context


@pytest.fixture
def activation_manager():
    """Create a mock activation manager."""
    mock_manager = MagicMock()
    mock_manager.get_activated_memories.return_value = {
        0: 0.9,  # Memory 0 with high activation
        2: 0.7,  # Memory 2 with medium activation
    }
    return mock_manager


@pytest.fixture
def memory_store_with_arrays():
    """Create a memory store with array data for vector operations."""
    mock_store = MagicMock()
    mock_store.memory_embeddings = np.array(
        [
            [0.1, 0.2, 0.3],  # Memory 0
            [0.4, 0.5, 0.6],  # Memory 1
            [0.7, 0.8, 0.9],  # Memory 2
            [0.9, 0.8, 0.7],  # Memory 3
            [0.6, 0.5, 0.4],  # Memory 4
        ]
    )

    # Normalize for vector similarity
    norms = np.linalg.norm(mock_store.memory_embeddings, axis=1, keepdims=True)
    mock_store.memory_embeddings = mock_store.memory_embeddings / norms

    mock_store.memory_metadata = [
        {"content": "Memory content 0", "created_at": 100},
        {"content": "Memory content 1", "created_at": 200},
        {"content": "Memory content 2", "created_at": 300},
        {"content": "Memory content 3", "created_at": 400},
        {"content": "Memory content 4", "created_at": 500},
    ]

    # Add activation levels
    mock_store.activation_levels = np.array([0.9, 0.5, 0.7, 0.3, 0.1])

    # Add temporal markers
    mock_store.temporal_markers = np.array([100, 200, 300, 400, 500])

    return mock_store
