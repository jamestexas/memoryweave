#!/usr/bin/env python3
# memoryweave/benchmarks/simple_memory_test.py
"""
Simple memory test to validate the refactored memory storage system.

This script tests the core functionality of memory storage, retrieval,
and vector search with the refactored implementation.

Usage:
    uv run python memoryweave/benchmarks/simple_memory_test.py

"""

import logging
import time

import numpy as np

from memoryweave.storage.refactored import MemoryAdapter, StandardMemoryStore

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def generate_test_memories(count: int, dimension: int = 768) -> list[tuple[np.ndarray, str, dict]]:
    """Generate test memories with embeddings, content, and metadata."""
    memories = []

    for i in range(count):
        # Create unique embedding
        embedding = np.random.randn(dimension)
        embedding = embedding / np.linalg.norm(embedding)

        # Create content and metadata
        content = f"This is test memory {i} with some specific content about topic {i % 5}"
        metadata = {
            "type": "test",
            "created_at": time.time(),
            "importance": i / count,
            "topic": f"topic_{i % 5}",
        }

        memories.append((embedding, content, metadata))

    return memories


def test_basic_operations():
    """Test basic memory operations (add, get, remove)."""
    logger.info("Testing basic memory operations...")

    # Create memory store and adapter
    store = StandardMemoryStore()
    adapter = MemoryAdapter(store)

    # Add a memory
    embedding = np.random.randn(768)
    embedding = embedding / np.linalg.norm(embedding)
    content = "This is a test memory"
    metadata = {"type": "test", "importance": 0.8}

    memory_id = adapter.add(embedding, content, metadata)
    logger.info(f"Added memory with ID: {memory_id}")

    # Retrieve the memory
    memory = adapter.get(memory_id)
    logger.info(f"Retrieved memory: {memory.id}")

    # Verify content and metadata
    assert memory.id == memory_id, "Memory ID mismatch"
    if isinstance(memory.content, dict):
        assert memory.content.get("text") == content, "Content mismatch"
    else:
        assert str(memory.content) == content, "Content mismatch"
    assert memory.metadata.get("type") == "test", "Metadata mismatch"

    # Remove the memory
    adapter.remove(memory_id)
    logger.info(f"Removed memory with ID: {memory_id}")

    # Verify removal
    try:
        adapter.get(memory_id)
        raise AssertionError("Memory still exists after removal")
    except KeyError:
        logger.info("Memory removal verified")

    logger.info("Basic operations test passed!")


def test_vector_search():
    """Test vector similarity search."""
    logger.info("Testing vector similarity search...")

    # Create memory store and adapter
    store = StandardMemoryStore()
    adapter = MemoryAdapter(store)

    # Generate test memories
    test_count = 100
    dimension = 768
    test_memories = generate_test_memories(test_count, dimension)

    # Add memories
    memory_ids = []
    for embedding, content, metadata in test_memories:
        memory_id = adapter.add(embedding, content, metadata)
        memory_ids.append(memory_id)

    logger.info(f"Added {test_count} test memories")

    # Generate a query vector similar to a specific memory
    target_idx = 42  # Choose a specific memory
    target_embedding = test_memories[target_idx][0]

    # Add some noise to create query vector
    noise = np.random.randn(dimension) * 0.1
    query_vector = target_embedding + noise
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Perform search
    results = adapter.search_by_vector(query_vector, limit=5)
    logger.info(f"Search returned {len(results)} results")

    # Verify search results
    assert len(results) > 0, "Search returned no results"

    # Check if target memory is in results
    found_target = False
    for result in results:
        if result["id"] == memory_ids[target_idx]:
            found_target = True
            logger.info(f"Found target memory with score: {result['score']}")
            break

    assert found_target, "Target memory not found in search results"

    logger.info("Vector search test passed!")


def test_id_resolution():
    """Test ID resolution between string and integer IDs."""
    logger.info("Testing ID resolution...")

    # Create memory store and adapter
    store = StandardMemoryStore()
    adapter = MemoryAdapter(store)

    # Add memories
    memory_count = 10
    memory_ids = []

    for i in range(memory_count):
        embedding = np.random.randn(768)
        embedding = embedding / np.linalg.norm(embedding)
        content = f"Memory {i}"
        metadata = {"index": i}

        memory_id = adapter.add(embedding, content, metadata)
        memory_ids.append(memory_id)

    logger.info(f"Added {memory_count} memories")

    # Test retrieval with different ID formats
    for i, original_id in enumerate(memory_ids):
        # Test with original string ID
        memory1 = adapter.get(original_id)

        # Test with index-based ID (integer)
        memory2 = adapter.get(i)

        # Test with string version of index
        memory3 = adapter.get(str(i))

        # Verify they all point to the same memory
        assert memory1.id == memory2.id == memory3.id, f"ID resolution failed for memory {i}"
        logger.info(f"Successfully resolved ID {i} to {memory1.id}")

    logger.info("ID resolution test passed!")


def run_all_tests():
    """Run all memory tests."""
    logger.info("Starting memory system tests...")

    test_basic_operations()
    test_vector_search()
    test_id_resolution()

    logger.info("All tests passed!")


if __name__ == "__main__":
    run_all_tests()
