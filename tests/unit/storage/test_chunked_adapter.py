import unittest

import numpy as np

from memoryweave.storage.chunked_adapter import ChunkedMemoryAdapter
from memoryweave.storage.chunked_store import ChunkedMemoryStore


class TestChunkedMemoryAdapter(unittest.TestCase):
    def _check_memory_content(self, memory):
        # Check if content is a dict with 'text' field
        if isinstance(memory.content, dict) and "text" in memory.content:
            self.assertEqual(memory.content["text"], self.test_content)
        else:
            self.assertEqual(memory.content, self.test_content)

    def setUp(self):
        self.store = ChunkedMemoryStore()
        self.adapter = ChunkedMemoryAdapter(self.store)
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
        self.assertEqual(self.adapter.chunked_store, self.store)

        # Cache properties should be None initially
        self.assertIsNone(self.adapter._chunk_embeddings_cache)
        self.assertIsNone(self.adapter._chunk_metadata_cache)
        self.assertIsNone(self.adapter._chunk_ids_cache)

    def test_add_and_get(self):
        # Test adding a memory
        memory_id = self.adapter.add(self.test_embedding, self.test_content, self.test_metadata)

        # Verify the memory was added
        memory = self.adapter.get(memory_id)
        self._check_memory_content(memory)
        self.assertTrue(np.array_equal(memory.embedding, self.test_embedding))
        self.assertEqual(memory.metadata, self.test_metadata)

    def test_add_chunked(self):
        # Test adding a chunked memory
        memory_id = self.adapter.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Verify memory was added
        memory = self.adapter.get(memory_id)
        self._check_memory_content(memory)
        self.assertEqual(memory.metadata, self.test_metadata)

        # Verify chunks were added
        chunks = self.adapter.get_chunks(memory_id)
        self.assertEqual(len(chunks), 3)

        # Verify cache is invalidated
        self.assertTrue(self.adapter._invalidated)

    def test_chunk_embeddings_property(self):
        # Add a chunked memory to have some data
        _memory_id = self.adapter.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Get chunk embeddings via property
        embeddings = self.adapter.chunk_embeddings

        # Should be a 3x3 matrix
        self.assertEqual(embeddings.shape, (3, 3))

        # Verify cache is built
        self.assertFalse(self.adapter._invalidated)
        self.assertIsNotNone(self.adapter._chunk_embeddings_cache)

    def test_chunk_metadata_property(self):
        # Add a chunked memory to have some data
        memory_id = self.adapter.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
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

    def test_chunk_ids_property(self):
        # Add a chunked memory to have some data
        memory_id = self.adapter.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
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
        # Add a chunked memory to have some data
        _memory_id = self.adapter.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
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
        # Add a chunked memory to have some data
        memory_id = self.adapter.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
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

    def test_get_chunk_count(self):
        # Add two chunked memories
        self.adapter.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        self.adapter.add_chunked(
            self.test_chunks[:2],
            self.test_chunk_embeddings[:2],
            "Another content",
            {"type": "another"},
        )

        # Get chunk count
        count = self.adapter.get_chunk_count()

        # Should be 5 (3 + 2)
        self.assertEqual(count, 5)

    def test_get_average_chunks_per_memory(self):
        # Add two chunked memories
        self.adapter.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        self.adapter.add_chunked(
            self.test_chunks[:2],
            self.test_chunk_embeddings[:2],
            "Another content",
            {"type": "another"},
        )

        # Get average chunks per memory
        avg = self.adapter.get_average_chunks_per_memory()

        # Should be 2.5 (5 chunks / 2 memories)
        self.assertEqual(avg, 2.5)

    def test_resolve_id(self):
        # Add a memory to get an ID
        memory_id = self.adapter.add(self.test_embedding, self.test_content)

        # Force build cache to populate ID mappings
        _ = self.adapter.memory_embeddings

        # Get the internal index
        internal_idx = self.adapter._id_to_index_map[memory_id]

        # Resolve internal index
        resolved_id = self.adapter._resolve_id(str(internal_idx))

        # Should resolve to the original memory ID
        self.assertEqual(resolved_id, memory_id)

        # Resolve original ID
        resolved_id = self.adapter._resolve_id(memory_id)

        # Should remain the same
        self.assertEqual(resolved_id, memory_id)

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
        # Add a chunked memory to have some data
        memory_id = self.adapter.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
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

        # Should only include chunks from chunked memories
        self.assertEqual(self.adapter._chunk_embeddings_cache.shape, (3, 3))
        self.assertEqual(len(self.adapter._chunk_metadata_cache), 3)
        self.assertEqual(len(self.adapter._chunk_ids_cache), 3)

        # All chunk IDs should reference the chunked memory
        for chunk_id in self.adapter._chunk_ids_cache:
            self.assertEqual(chunk_id[0], memory_id)


if __name__ == "__main__":
    unittest.main()
