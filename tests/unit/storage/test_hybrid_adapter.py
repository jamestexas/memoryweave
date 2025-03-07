import unittest

import numpy as np

from memoryweave.storage.hybrid_adapter import HybridMemoryAdapter
from memoryweave.storage.hybrid_store import HybridMemoryStore


class TestHybridMemoryAdapter(unittest.TestCase):
    def setUp(self):
        self.store = HybridMemoryStore()
        self.adapter = HybridMemoryAdapter(self.store)
        self.test_embedding = np.array([0.1, 0.2, 0.3])
        self.test_content = "Test memory content"
        self.test_metadata = {"type": "test", "importance": 0.7}

        # Chunk test data
        self.test_chunks = [
            {"text": "Chunk 1", "metadata": {"chunk_index": 0, "position": "start"}},
            {"text": "Chunk 2", "metadata": {"chunk_index": 1, "position": "middle"}},
            {"text": "Chunk 3", "metadata": {"chunk_index": 2, "position": "end"}},
        ]
        self.test_chunk_embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.7, 0.8, 0.9]),
        ]

    def test_initialization(self):
        # Test that adapter is initialized with correct store
        self.assertEqual(self.adapter.memory_store, self.store)
        self.assertEqual(self.adapter.hybrid_store, self.store)

        # Cache properties should be None initially
        self.assertIsNone(self.adapter._chunk_embeddings_cache)
        self.assertIsNone(self.adapter._chunk_metadata_cache)
        self.assertIsNone(self.adapter._chunk_ids_cache)

    def _check_memory_content(self, memory):
        # Check if content is a dict with 'text' field
        if isinstance(memory.content, dict) and "text" in memory.content:
            self.assertEqual(memory.content["text"], self.test_content)
        else:
            self.assertEqual(memory.content, self.test_content)

    def test_add_and_get(self):
        # Test adding a memory
        memory_id = self.adapter.add(self.test_embedding, self.test_content, self.test_metadata)

        # Verify the memory was added
        memory = self.adapter.get(memory_id)
        # Check if content is a dict with 'text' field
        self._check_memory_content(memory=memory)
        self.assertTrue(np.array_equal(memory.embedding, self.test_embedding))
        self.assertEqual(memory.metadata, self.test_metadata)

    def test_add_hybrid(self):
        # Test adding a hybrid memory
        memory_id = self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Verify memory was added
        memory = self.adapter.get(memory_id)
        self._check_memory_content(memory=memory)
        self.assertEqual(memory.metadata, self.test_metadata)

        # Verify it's a hybrid memory
        self.assertTrue(self.adapter.is_hybrid(memory_id))

        # Verify chunks were added
        chunks = self.adapter.get_chunks(memory_id)
        self.assertEqual(len(chunks), 3)

        # Verify cache is invalidated
        self.assertTrue(self.adapter._invalidated)

    def test_is_hybrid(self):
        # Add a regular memory
        regular_id = self.adapter.add(self.test_embedding, "Regular content")

        # Add a hybrid memory
        hybrid_id = self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Check is_hybrid
        self.assertFalse(self.adapter.is_hybrid(regular_id))
        self.assertTrue(self.adapter.is_hybrid(hybrid_id))

    def test_chunk_embeddings_property(self):
        # Add a hybrid memory to have some data
        _memory_id = self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Get chunk embeddings via property
        embeddings = self.adapter.chunk_embeddings

        # Should be a 3x3 matrix
        self.assertEqual(embeddings.shape, (3, 3))

        # Verify cache is built
        self.assertFalse(self.adapter._invalidated)
        self.assertIsNotNone(self.adapter._chunk_embeddings_cache)

    def test_chunk_metadata_property(self):
        # Add a hybrid memory to have some data
        memory_id = self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Get chunk metadata via property
        metadata = self.adapter.chunk_metadata

        # Should have 3 items
        self.assertEqual(len(metadata), 3)

        # Verify metadata content
        for i, meta in enumerate(metadata):
            # Should include metadata from chunks
            self.assertEqual(meta["position"], self.test_chunks[i]["metadata"]["position"])

            # Should include metadata from memory
            self.assertEqual(meta["type"], self.test_metadata["type"])
            self.assertEqual(meta["importance"], self.test_metadata["importance"])

            # Should include additional fields
            self.assertEqual(meta["memory_id"], memory_id)
            self.assertEqual(meta["chunk_index"], i)
            self.assertTrue(meta["is_hybrid"])

    def test_chunk_ids_property(self):
        # Add a hybrid memory to have some data
        memory_id = self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Get chunk IDs via property
        ids = self.adapter.chunk_ids

        # Should have 3 items
        self.assertEqual(len(ids), 3)

        # Verify IDs are (memory_id, chunk_index) tuples
        for i, chunk_id in enumerate(ids):
            self.assertTrue(isinstance(chunk_id, tuple))
            self.assertEqual(chunk_id[0], memory_id)
            self.assertEqual(chunk_id[1], i)

    def test_invalidate_cache(self):
        # Add a hybrid memory to have some data
        _memory_id = self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Access properties to build cache
        _ = self.adapter.chunk_embeddings
        _ = self.adapter.chunk_metadata
        _ = self.adapter.chunk_ids

        # Cache should be built
        self.assertFalse(self.adapter._invalidated)
        self.assertIsNotNone(self.adapter._chunk_embeddings_cache)
        self.assertIsNotNone(self.adapter._chunk_metadata_cache)
        self.assertIsNotNone(self.adapter._chunk_ids_cache)

        # Invalidate cache
        self.adapter.invalidate_cache()

        # Cache should be invalidated
        self.assertTrue(self.adapter._invalidated)
        self.assertIsNone(self.adapter._chunk_embeddings_cache)
        self.assertIsNone(self.adapter._chunk_metadata_cache)
        self.assertIsNone(self.adapter._chunk_ids_cache)

    def test_search_chunks(self):
        # Add a hybrid memory to have some data
        memory_id = self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Search with a vector close to the third chunk
        query_vector = np.array([0.7, 0.8, 0.9])
        results = self.adapter.search_chunks(query_vector, limit=2)

        # Should have 2 results
        self.assertEqual(len(results), 2)

        # First result should be the third chunk
        self.assertEqual(results[0]["memory_id"], memory_id)
        self.assertEqual(results[0]["chunk_index"], 2)

        # Search with threshold
        results = self.adapter.search_chunks(query_vector, threshold=0.95)

        # Should have 1 result (only the third chunk is above threshold)
        self.assertEqual(len(results), 1)

    def test_search_hybrid(self):
        # Add a hybrid memory
        _hybrid_id = self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Add a regular memory
        regular_id = self.adapter.add(
            np.array([0.9, 0.1, 0.1]), "Regular content", {"type": "regular"}
        )

        # Search with a vector close to the regular memory
        query_vector = np.array([0.9, 0.1, 0.1])
        results = self.adapter.search_hybrid(query_vector, limit=2)

        # Should have 2 results
        self.assertEqual(len(results), 2)

        # First result should be the regular memory
        self.assertEqual(results[0]["memory_id"], regular_id)
        self.assertFalse(results[0].get("is_hybrid", False))

        # Search with threshold and keywords
        results = self.adapter.search_hybrid(
            query_vector,
            limit=2,
            threshold=0.1,
            keywords=["content"],  # This keyword is in both memories
        )

        # Should have 2 results
        self.assertEqual(len(results), 2)

        # Both should have keyword_matches
        self.assertTrue(all("keyword_matches" in r for r in results))

    def test_get_chunk_count(self):
        # Add two hybrid memories
        self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks[:2],
            chunk_embeddings=self.test_chunk_embeddings[:2],
            original_content="Another content",
            metadata={"type": "another"},
        )

        # Get chunk count
        count = self.adapter.get_chunk_count()

        # Should be 5 (3 + 2)
        self.assertEqual(count, 5)

    def test_get_average_chunks_per_memory(self):
        # Add two hybrid memories
        self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks[:2],
            chunk_embeddings=self.test_chunk_embeddings[:2],
            original_content="Another content",
            metadata={"type": "another"},
        )

        # Get average chunks per memory
        avg = self.adapter.get_average_chunks_per_memory()

        # Should be 2.5 (5 chunks / 2 memories)
        self.assertEqual(avg, 2.5)

    def test_empty_chunk_properties(self):
        # Test accessing chunk properties when no chunks exist

        # Should return an empty array with correct shape
        embeddings = self.adapter.chunk_embeddings
        self.assertEqual(embeddings.shape[0], 0)  # No rows
        self.assertTrue(embeddings.shape[1] > 0)  # But has columns

        # Should return empty lists
        self.assertEqual(self.adapter.chunk_metadata, [])
        self.assertEqual(self.adapter.chunk_ids, [])

    def test_build_chunk_cache(self):
        # Add a hybrid memory to have some data
        memory_id = self.adapter.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Add a regular memory
        _regular_id = self.adapter.add(self.test_embedding, "Regular content")

        # Build chunk cache
        self.adapter._build_chunk_cache()

        # Cache should be built
        self.assertFalse(self.adapter._invalidated)
        self.assertIsNotNone(self.adapter._chunk_embeddings_cache)
        self.assertIsNotNone(self.adapter._chunk_metadata_cache)
        self.assertIsNotNone(self.adapter._chunk_ids_cache)

        # Should only include chunks from hybrid memories
        self.assertEqual(self.adapter._chunk_embeddings_cache.shape, (3, 3))
        self.assertEqual(len(self.adapter._chunk_metadata_cache), 3)
        self.assertEqual(len(self.adapter._chunk_ids_cache), 3)

        # All chunk IDs should reference the hybrid memory
        for chunk_id in self.adapter._chunk_ids_cache:
            self.assertEqual(chunk_id[0], memory_id)


if __name__ == "__main__":
    unittest.main()
