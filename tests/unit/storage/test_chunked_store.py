import unittest

import numpy as np

from memoryweave.storage.chunked_store import ChunkedMemoryStore, ChunkInfo


class TestChunkedMemoryStore(unittest.TestCase):
    def _check_memory_content(self, memory):
        # Check if content is a dict with 'text' field
        if isinstance(memory.content, dict) and "text" in memory.content:
            self.assertEqual(memory.content["text"], self.test_content)
        else:
            self.assertEqual(memory.content, self.test_content)

    def setUp(self):
        self.store = ChunkedMemoryStore()
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

    def test_add_and_get(self):
        # Test regular add without chunking
        memory_id = self.store.add(self.test_embedding, self.test_content, self.test_metadata)

        # Verify add worked
        self.assertIsNotNone(memory_id)

        # Get the memory
        memory = self.store.get(memory_id)

        # Verify memory properties
        self.assertEqual(memory.id, memory_id)
        self.assertTrue(np.array_equal(memory.embedding, self.test_embedding))
        self._check_memory_content(memory)
        self.assertEqual(memory.metadata, self.test_metadata)

        # No chunks should be associated
        chunks = self.store.get_chunks(memory_id)
        self.assertEqual(len(chunks), 0)

    def test_add_chunked(self):
        # Test adding a chunked memory
        memory_id = self.store.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Verify add worked
        self.assertIsNotNone(memory_id)

        # Get the memory
        memory = self.store.get(memory_id)

        # Verify memory properties
        self.assertEqual(memory.id, memory_id)
        self._check_memory_content(memory)
        self.assertEqual(memory.metadata, self.test_metadata)

        # Combined embedding should be mean of chunk embeddings
        expected_embedding = np.mean(self.test_chunk_embeddings, axis=0)
        self.assertTrue(np.allclose(memory.embedding, expected_embedding))

        # Get chunks
        chunks = self.store.get_chunks(memory_id)

        # Should have 3 chunks
        self.assertEqual(len(chunks), 3)

        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)
            self.assertEqual(chunk.text, self.test_chunks[i]["text"])
            self.assertEqual(
                chunk.metadata["position"], self.test_chunks[i]["metadata"]["position"]
            )
            self.assertTrue(np.array_equal(chunk.embedding, self.test_chunk_embeddings[i]))

    def test_add_chunked_validation(self):
        # Test validation of chunks and embeddings

        # Different number of chunks and embeddings
        with self.assertRaises(ValueError):
            self.store.add_chunked(
                self.test_chunks[:2],  # Only 2 chunks
                self.test_chunk_embeddings,  # 3 embeddings
                self.test_content,
                self.test_metadata,
            )

        # Empty chunks
        with self.assertRaises(ValueError):
            self.store.add_chunked(
                [],  # No chunks
                [],  # No embeddings
                self.test_content,
                self.test_metadata,
            )

    def test_get_chunk_embeddings(self):
        # Add a chunked memory
        memory_id = self.store.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Get chunk embeddings
        embeddings = self.store.get_chunk_embeddings(memory_id)

        # Should have 3 embeddings
        self.assertEqual(len(embeddings), 3)

        # Verify embeddings
        for i, embedding in enumerate(embeddings):
            self.assertTrue(np.array_equal(embedding, self.test_chunk_embeddings[i]))

    def test_search_chunks(self):
        # Add a chunked memory
        memory_id = self.store.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Search with a vector close to the third chunk
        query_vector = np.array([0.7, 0.8, 0.9])
        results = self.store.search_chunks(query_vector, limit=2)

        # Should have 2 results
        self.assertEqual(len(results), 2)

        # First result should be the third chunk
        self.assertEqual(results[0]["memory_id"], memory_id)
        self.assertEqual(results[0]["chunk_index"], 2)
        self.assertAlmostEqual(results[0]["chunk_similarity"], 1.0, places=6)
        self.assertEqual(results[0]["content"], "Chunk 3")

        # Search with threshold
        results = self.store.search_chunks(query_vector, limit=3, threshold=0.95)

        # Should have 1 result (only the third chunk is above threshold)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_index"], 2)

    def test_get_nonexistent_chunks(self):
        # Add a regular memory
        memory_id = self.store.add(self.test_embedding, self.test_content)

        # Get chunks for memory with no chunks
        chunks = self.store.get_chunks(memory_id)

        # Should be empty list
        self.assertEqual(len(chunks), 0)

        # Get chunks for nonexistent memory
        chunks = self.store.get_chunks("nonexistent_id")

        # Should be empty list
        self.assertEqual(len(chunks), 0)

    def test_remove(self):
        # Add a chunked memory
        memory_id = self.store.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Verify chunks exist
        self.assertEqual(len(self.store.get_chunks(memory_id)), 3)

        # Remove the memory
        self.store.remove(memory_id)

        # Verify memory is removed
        with self.assertRaises(KeyError):
            self.store.get(memory_id)

        # Verify chunks are removed
        self.assertEqual(len(self.store.get_chunks(memory_id)), 0)

    def test_clear(self):
        # Add a chunked memory
        self.store.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Add a regular memory
        self.store.add(self.test_embedding, "Regular content")

        # Verify we have memories
        self.assertEqual(len(self.store.get_all()), 2)

        # Clear the store
        self.store.clear()

        # Verify everything is cleared
        self.assertEqual(len(self.store.get_all()), 0)
        self.assertEqual(self.store.get_chunk_count(), 0)

    def test_get_chunk_count(self):
        # Add a chunked memory
        self.store.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Add another chunked memory with 2 chunks
        self.store.add_chunked(
            self.test_chunks[:2],
            self.test_chunk_embeddings[:2],
            "Another content",
            {"type": "another"},
        )

        # Should have 5 chunks total (3 + 2)
        self.assertEqual(self.store.get_chunk_count(), 5)

    def test_get_average_chunks_per_memory(self):
        # Add a chunked memory with 3 chunks
        self.store.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Add another chunked memory with 2 chunks
        self.store.add_chunked(
            self.test_chunks[:2],
            self.test_chunk_embeddings[:2],
            "Another content",
            {"type": "another"},
        )

        # Average should be 2.5 (5 chunks / 2 memories)
        self.assertEqual(self.store.get_average_chunks_per_memory(), 2.5)

        # Add a regular memory (no chunks)
        self.store.add(self.test_embedding, "Regular content")

        # Average should still be 2.5 (still 5 chunks / 2 chunked memories)
        self.assertEqual(self.store.get_average_chunks_per_memory(), 2.5)

    def test_consolidate(self):
        # Add a chunked memory with low importance
        mem_id1 = self.store.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, "Low importance", {"importance": 0.1}
        )

        # Add a chunked memory with high importance
        mem_id2 = self.store.add_chunked(
            self.test_chunks[:2],
            self.test_chunk_embeddings[:2],
            "High importance",
            {"importance": 0.9},
        )

        # Get all memories
        memories = self.store.get_all()

        # Set activation based on importance
        for memory in memories:
            importance = memory.metadata.get("importance", 0)
            self.store._base_store._metadata[memory.id].activation = importance

        # Consolidate to keep only 1 memory
        removed_ids = self.store.consolidate(1)

        # Should have removed 1 memory (the one with lower activation/importance)
        self.assertEqual(len(removed_ids), 1)
        self.assertEqual(removed_ids[0], mem_id1)  # Should have removed the low importance memory

        # Should have 1 memory left
        memories = self.store.get_all()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].id, mem_id2)

        # Should have 2 chunks left (from the high importance memory)
        self.assertEqual(self.store.get_chunk_count(), 2)

    def test_update_metadata(self):
        # Add a chunked memory
        memory_id = self.store.add_chunked(
            self.test_chunks, self.test_chunk_embeddings, self.test_content, self.test_metadata
        )

        # Update metadata
        new_metadata = {"type": "updated", "importance": 0.9, "new_field": "new_value"}
        self.store.update_metadata(memory_id, new_metadata)

        # Get the memory
        memory = self.store.get(memory_id)

        # Verify metadata is updated
        self.assertEqual(memory.metadata["type"], "updated")
        self.assertEqual(memory.metadata["importance"], 0.9)
        self.assertEqual(memory.metadata["new_field"], "new_value")

    def test_chunk_info_class(self):
        # Test ChunkInfo class initialization
        chunk = ChunkInfo(
            chunk_index=1,
            embedding=np.array([0.1, 0.2, 0.3]),
            text="Test chunk",
            metadata={"position": "middle"},
        )

        # Verify properties
        self.assertEqual(chunk.chunk_index, 1)
        self.assertTrue(np.array_equal(chunk.embedding, np.array([0.1, 0.2, 0.3])))
        self.assertEqual(chunk.text, "Test chunk")
        self.assertEqual(chunk.metadata["position"], "middle")

        # Test with no metadata
        chunk = ChunkInfo(chunk_index=2, embedding=np.array([0.4, 0.5, 0.6]), text="Another test")

        # Metadata should be initialized to empty dict
        self.assertEqual(chunk.metadata, {})

    def test_empty_search_chunks(self):
        # Search with no memories
        query_vector = np.array([0.1, 0.2, 0.3])
        results = self.store.search_chunks(query_vector)

        # Should have no results
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
