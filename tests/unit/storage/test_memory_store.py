"""
Tests for the MemoryStore component.
"""

import numpy as np
import pytest

from memoryweave.storage.memory_store import MemoryStore


class TestMemoryStore:
    """Test suite for the MemoryStore class."""

    def test_init(self):
        """Test initialization of memory store."""
        store = MemoryStore()
        assert len(store.get_ids()) == 0
        assert store.get_embeddings().shape == (0, 0)

    def test_add_memory(self):
        """Test adding a memory to the store."""
        store = MemoryStore()
        embedding = np.array([0.1, 0.2, 0.3])
        content = "Test memory content"
        metadata = {"source": "test", "importance": 0.8}

        # Add memory
        memory_id = store.add(embedding, content, metadata)

        # Verify memory was added
        assert memory_id in store.get_ids()
        assert len(store.get_ids()) == 1
        assert store.get_embeddings().shape == (1, 3)
        
        # Retrieve and verify memory
        memory = store.get(memory_id)
        assert memory["id"] == memory_id
        assert np.array_equal(memory["embedding"], embedding)
        assert memory["content"]["text"] == content
        assert memory["metadata"]["source"] == "test"
        assert memory["metadata"]["importance"] == 0.8

    def test_get_memory(self):
        """Test retrieving a memory by ID."""
        store = MemoryStore()
        embedding = np.array([0.1, 0.2, 0.3])
        content = "Test memory content"
        
        # Add memory
        memory_id = store.add(embedding, content)
        
        # Get memory
        memory = store.get(memory_id)
        
        # Verify memory
        assert memory["id"] == memory_id
        assert np.array_equal(memory["embedding"], embedding)
        assert memory["content"]["text"] == content
        
        # Test non-existent memory
        with pytest.raises(KeyError):
            store.get("non_existent_id")

    def test_get_all_memories(self):
        """Test retrieving all memories."""
        store = MemoryStore()
        
        # Add memories
        ids = []
        for i in range(5):
            embedding = np.array([0.1 * i, 0.2 * i, 0.3 * i])
            content = f"Memory {i}"
            memory_id = store.add(embedding, content)
            ids.append(memory_id)
        
        # Get all memories
        memories = store.get_all()
        
        # Verify memories
        assert len(memories) == 5
        memory_ids = [memory["id"] for memory in memories]
        for memory_id in ids:
            assert memory_id in memory_ids

    def test_get_embeddings(self):
        """Test retrieving all embeddings as a matrix."""
        store = MemoryStore()
        
        # Add memories
        for i in range(3):
            embedding = np.array([0.1 * i, 0.2 * i, 0.3 * i])
            content = f"Memory {i}"
            store.add(embedding, content)
        
        # Get embeddings
        embeddings = store.get_embeddings()
        
        # Verify embeddings
        assert embeddings.shape == (3, 3)
        # Check first memory embedding
        assert np.array_equal(embeddings[0], np.array([0.0, 0.0, 0.0]))
        # Check second memory embedding
        assert np.array_equal(embeddings[1], np.array([0.1, 0.2, 0.3]))

    def test_update_activation(self):
        """Test updating activation level of a memory."""
        store = MemoryStore()
        embedding = np.array([0.1, 0.2, 0.3])
        content = "Test memory"
        
        # Add memory
        memory_id = store.add(embedding, content)
        
        # Update activation
        store.update_activation(memory_id, 0.5)
        
        # Verify activation was updated
        memory_meta = store._metadata[memory_id]
        assert memory_meta.activation == 0.5
        
        # Update again
        store.update_activation(memory_id, 0.3)
        assert memory_meta.activation == 0.8
        
        # Test non-existent memory
        with pytest.raises(KeyError):
            store.update_activation("non_existent_id", 0.5)

    def test_update_metadata(self):
        """Test updating metadata of a memory."""
        store = MemoryStore()
        embedding = np.array([0.1, 0.2, 0.3])
        content = "Test memory"
        
        # Add memory
        memory_id = store.add(embedding, content, {"initial": "value"})
        
        # Update metadata
        store.update_metadata(memory_id, {"new_key": "new_value"})
        
        # Verify metadata was updated
        memory = store.get(memory_id)
        assert memory["metadata"]["initial"] == "value"
        assert memory["metadata"]["new_key"] == "new_value"
        
        # Test non-existent memory
        with pytest.raises(KeyError):
            store.update_metadata("non_existent_id", {"key": "value"})

    def test_remove_memory(self):
        """Test removing a memory from the store."""
        store = MemoryStore()
        embedding = np.array([0.1, 0.2, 0.3])
        content = "Test memory"
        
        # Add memory
        memory_id = store.add(embedding, content)
        
        # Verify memory exists
        assert memory_id in store.get_ids()
        
        # Remove memory
        store.remove(memory_id)
        
        # Verify memory was removed
        assert memory_id not in store.get_ids()
        
        # Test removing non-existent memory
        with pytest.raises(KeyError):
            store.remove("non_existent_id")

    def test_clear(self):
        """Test clearing all memories from the store."""
        store = MemoryStore()
        
        # Add memories
        for i in range(5):
            embedding = np.array([0.1 * i, 0.2 * i, 0.3 * i])
            content = f"Memory {i}"
            store.add(embedding, content)
        
        # Verify memories were added
        assert len(store.get_ids()) == 5
        
        # Clear store
        store.clear()
        
        # Verify store is empty
        assert len(store.get_ids()) == 0
        assert store.get_embeddings().shape == (0, 0)

    def test_consolidate(self):
        """Test consolidating memories to stay within capacity."""
        store = MemoryStore()
        
        # Add memories
        ids = []
        for i in range(10):
            embedding = np.array([0.1 * i, 0.2 * i, 0.3 * i])
            content = f"Memory {i}"
            memory_id = store.add(embedding, content)
            # Set different activation levels
            store.update_activation(memory_id, 0.1 * i)
            ids.append(memory_id)
        
        # Consolidate to 5 memories
        removed_ids = store.consolidate(5)
        
        # Verify consolidation
        assert len(store.get_ids()) == 5
        assert len(removed_ids) == 5
        
        # Lower activation memories should be removed first
        for memory_id in removed_ids:
            # These should be the lower activation ones (0.0-0.4)
            assert memory_id in ids[:5]
        
        # Higher activation memories should remain
        remaining_ids = store.get_ids()
        for memory_id in ids[5:]:
            assert memory_id in remaining_ids