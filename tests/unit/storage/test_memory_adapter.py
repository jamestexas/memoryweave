import unittest
from unittest.mock import MagicMock

import numpy as np

from memoryweave.storage.adapter import MemoryAdapter
from memoryweave.storage.memory_store import StandardMemoryStore
from memoryweave.storage.vector_search.base import IVectorSearchProvider


class TestMemoryAdapter(unittest.TestCase):
    def _check_memory_content(self, memory):
        # Check if content is a dict with 'text' field
        if isinstance(memory.content, dict) and "text" in memory.content:
            self.assertEqual(memory.content["text"], self.test_content)
        else:
            self.assertEqual(memory.content, self.test_content)

    def setUp(self):
        self.store = StandardMemoryStore()
        self.vector_search = MagicMock(spec=IVectorSearchProvider)
        self.adapter = MemoryAdapter(self.store, self.vector_search)
        self.test_embedding = np.array([0.1, 0.2, 0.3])
        self.test_content = "Test memory content"
        self.test_metadata = {"type": "test", "importance": 0.7}

    def test_initialization(self):
        # Test that adapter is initialized with store and vector search
        self.assertEqual(self.adapter.memory_store, self.store)
        self.assertEqual(self.adapter._vector_search, self.vector_search)

        # Test initialization without vector search
        adapter_no_vs = MemoryAdapter(self.store)
        self.assertEqual(adapter_no_vs.memory_store, self.store)
        self.assertIsNone(adapter_no_vs._vector_search)

    def test_set_vector_search(self):
        # Create a new vector search provider mock
        new_vector_search = MagicMock(spec=IVectorSearchProvider)

        # Set the new vector search provider
        self.adapter.set_vector_search(new_vector_search)

        # Verify it was set
        self.assertEqual(self.adapter._vector_search, new_vector_search)

        # Verify cache was invalidated
        self.assertTrue(self.adapter._invalidated)

    def test_add_and_get(self):
        # Test adding a memory through the adapter
        memory_id = self.adapter.add(self.test_embedding, self.test_content, self.test_metadata)

        # Verify the store's add method was called
        self.assertTrue(isinstance(memory_id, str))

        # Test getting the memory through the adapter
        memory = self.adapter.get(memory_id)

        # Verify the memory properties
        self.assertEqual(memory.id, memory_id)
        self.assertTrue(np.array_equal(memory.embedding, self.test_embedding))
        self._check_memory_content(memory=memory)

        self.assertEqual(memory.metadata, self.test_metadata)

    def test_get_all(self):
        # Add some memories
        self.adapter.add(self.test_embedding, "Content 1", {"index": 1})
        self.adapter.add(self.test_embedding, "Content 2", {"index": 2})

        # Get all memories
        memories = self.adapter.get_all()

        # Should have 2 memories
        self.assertEqual(len(memories), 2)

        # Verify contents
        memory_contents = [m.content for m in memories]
        self.assertIn("Content 1", memory_contents)
        self.assertIn("Content 2", memory_contents)

    def test_memory_embeddings_property(self):
        # Add some memories
        self.adapter.add(np.array([0.1, 0.2, 0.3]), "Content 1")
        self.adapter.add(np.array([0.4, 0.5, 0.6]), "Content 2")

        # Get embeddings matrix
        embeddings = self.adapter.memory_embeddings

        # Should be a 2x3 matrix
        self.assertEqual(embeddings.shape, (2, 3))

        # Verify cache is built
        self.assertFalse(self.adapter._invalidated)
        self.assertIsNotNone(self.adapter._embeddings_matrix)

    def test_memory_metadata_property(self):
        # Add some memories
        self.adapter.add(self.test_embedding, "Content 1", {"index": 1})
        self.adapter.add(self.test_embedding, "Content 2", {"index": 2})

        # Get metadata
        metadata = self.adapter.memory_metadata

        # Should have 2 entries
        self.assertEqual(len(metadata), 2)

        # Verify additional fields are added
        self.assertTrue(all("memory_id" in m for m in metadata))
        self.assertTrue(all("original_id" in m for m in metadata))
        self.assertTrue(all("content" in m for m in metadata))

    def test_memory_ids_property(self):
        # Add some memories
        id1 = self.adapter.add(self.test_embedding, "Content 1")
        id2 = self.adapter.add(self.test_embedding, "Content 2")

        # Get IDs
        ids = self.adapter.memory_ids

        # Should have 2 IDs
        self.assertEqual(len(ids), 2)

        # Verify IDs
        self.assertIn(id1, ids)
        self.assertIn(id2, ids)

    def test_invalidate_cache(self):
        # Add some memories
        self.adapter.add(self.test_embedding, "Content 1")

        # Access properties to build cache
        _ = self.adapter.memory_embeddings
        _ = self.adapter.memory_metadata
        _ = self.adapter.memory_ids

        # Cache should be built
        self.assertFalse(self.adapter._invalidated)
        self.assertIsNotNone(self.adapter._embeddings_matrix)
        self.assertIsNotNone(self.adapter._metadata_dict)
        self.assertIsNotNone(self.adapter._ids_list)

        # Invalidate cache
        self.adapter.invalidate_cache()

        # Cache should be invalidated
        self.assertTrue(self.adapter._invalidated)
        self.assertIsNone(self.adapter._embeddings_matrix)
        self.assertIsNone(self.adapter._metadata_dict)
        self.assertIsNone(self.adapter._ids_list)

    def test_resolve_id(self):
        # Add memories to populate ID mappings
        id1 = self.adapter.add(self.test_embedding, "Content 1")

        # Force build cache to populate ID mappings
        _ = self.adapter.memory_embeddings

        # Test resolving by internal index
        internal_id = self.adapter._id_to_index_map[id1]
        resolved_id = self.adapter._resolve_id(str(internal_id))
        self.assertEqual(resolved_id, id1)

        # Test resolving by original ID
        resolved_id = self.adapter._resolve_id(id1)
        self.assertEqual(resolved_id, id1)

        # Test resolving an unknown ID
        unknown_id = "unknown_id"
        resolved_id = self.adapter._resolve_id(unknown_id)
        self.assertEqual(resolved_id, unknown_id)

    def test_update_metadata(self):
        # Add a memory
        memory_id = self.adapter.add(self.test_embedding, self.test_content, self.test_metadata)

        # Update metadata
        new_metadata = {"type": "updated", "importance": 0.9}
        self.adapter.update_metadata(memory_id, new_metadata)

        # Get the memory and verify metadata
        memory = self.adapter.get(memory_id)
        self.assertEqual(memory.metadata["type"], "updated")
        self.assertEqual(memory.metadata["importance"], 0.9)

        # Cache should be invalidated
        self.assertTrue(self.adapter._invalidated)

    def test_remove(self):
        # Add a memory
        memory_id = self.adapter.add(self.test_embedding, self.test_content)

        # Remove it
        self.adapter.remove(memory_id)

        # Try to get it - should raise KeyError
        with self.assertRaises(KeyError):
            self.adapter.get(memory_id)

        # Cache should be invalidated
        self.assertTrue(self.adapter._invalidated)

    def test_clear(self):
        # Add some memories
        self.adapter.add(self.test_embedding, "Content 1")
        self.adapter.add(self.test_embedding, "Content 2")

        # Clear
        self.adapter.clear()

        # Should have no memories
        memories = self.adapter.get_all()
        self.assertEqual(len(memories), 0)

        # Cache should be invalidated
        self.assertTrue(self.adapter._invalidated)

    def test_search_by_vector_with_provider(self):
        # Setup mock vector search results
        self.vector_search.search.return_value = [(0, 0.9), (1, 0.8)]

        # Add some memories
        id1 = self.adapter.add(self.test_embedding, "Content 1", {"tag": "one"})
        id2 = self.adapter.add(self.test_embedding, "Content 2", {"tag": "two"})

        # Force cache building to populate ID mappings
        _ = self.adapter.memory_embeddings

        # Setup ID mappings for test
        self.adapter._index_to_id_map = {0: id1, 1: id2}

        # Search
        query_vector = np.array([0.1, 0.2, 0.3])
        results = self.adapter.search_by_vector(query_vector, limit=2)

        # Vector search should be called
        self.vector_search.search.assert_called_once_with(query_vector, 2, None)

        # Should have 2 results
        self.assertEqual(len(results), 2)

        # Verify result structure
        self.assertEqual(results[0]["id"], id1)
        self.assertEqual(results[0]["memory_id"], id1)
        self.assertEqual(results[0]["score"], 0.9)
        self.assertEqual(results[0]["relevance_score"], 0.9)
        self.assertEqual(results[0]["metadata"]["tag"], "one")

        self.assertEqual(results[1]["id"], id2)
        self.assertEqual(results[1]["memory_id"], id2)
        self.assertEqual(results[1]["score"], 0.8)
        self.assertEqual(results[1]["relevance_score"], 0.8)
        self.assertEqual(results[1]["metadata"]["tag"], "two")

    def test_search_by_vector_direct(self):
        # Don't use vector search provider for this test
        self.adapter._vector_search = None

        # Add some memories with different embeddings
        id1 = self.adapter.add(np.array([0.9, 0.1, 0.1]), "Content 1")
        _id2 = self.adapter.add(np.array([0.1, 0.9, 0.1]), "Content 2")
        _id3 = self.adapter.add(np.array([0.1, 0.1, 0.9]), "Content 3")

        # Search with a vector similar to the first memory
        query_vector = np.array([0.95, 0.05, 0.05])
        results = self.adapter.search_by_vector(query_vector, limit=2)

        # Should have 2 results
        self.assertEqual(len(results), 2)

        # First result should be the first memory
        self.assertEqual(results[0]["id"], id1)

        # Search with threshold
        results = self.adapter.search_by_vector(query_vector, limit=2, threshold=0.9)

        # Should have 1 result (only the first memory is above threshold)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], id1)

    def test_build_cache_error_handling(self):
        # Setup mock store to raise exception
        mock_store = MagicMock(spec=StandardMemoryStore)
        mock_store.get_all.side_effect = Exception("Test exception")

        # Create adapter with mock store
        adapter = MemoryAdapter(mock_store)

        # Try to access property that builds cache
        embeddings = adapter.memory_embeddings

        # Should handle exception and return empty array
        self.assertEqual(embeddings.shape, (0, 768))

        # Other properties should also be initialized
        self.assertEqual(adapter._metadata_dict, [])
        self.assertEqual(adapter._ids_list, [])
        self.assertEqual(adapter._index_to_id_map, {})
        self.assertEqual(adapter._id_to_index_map, {})


if __name__ == "__main__":
    unittest.main()
