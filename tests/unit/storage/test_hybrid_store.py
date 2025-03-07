import unittest

import numpy as np

from memoryweave.storage.hybrid_store import ChunkInfo, HybridMemoryInfo, HybridMemoryStore


class TestHybridMemoryStore(unittest.TestCase):
    def setUp(self):
        self.store = HybridMemoryStore()
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
        # Test regular add without hybridization
        memory_id = self.store.add(self.test_embedding, self.test_content, self.test_metadata)

        # Verify add worked
        self.assertIsNotNone(memory_id)

        # Get the memory
        memory = self.store.get(memory_id)

        # Verify memory properties
        self.assertEqual(memory.id, memory_id)
        self.assertTrue(np.array_equal(memory.embedding, self.test_embedding))
        self.assertEqual(memory.content["text"], self.test_content)
        self.assertEqual(memory.metadata, self.test_metadata)

        # Memory should not be hybrid
        self.assertFalse(self.store.is_hybrid(memory_id))

    def test_add_hybrid(self):
        # Test adding a hybrid memory
        memory_id = self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Verify add worked
        self.assertIsNotNone(memory_id)

        # Get the memory
        memory = self.store.get(memory_id)

        # Verify memory properties
        self.assertEqual(memory.id, memory_id)
        self.assertTrue(np.array_equal(memory.embedding, self.test_embedding))
        self.assertEqual(memory.content["text"], self.test_content)
        self.assertEqual(memory.metadata, self.test_metadata)

        # Memory should be hybrid
        self.assertTrue(self.store.is_hybrid(memory_id))

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

    def test_add_hybrid_validation(self):
        # Test validation of chunks and embeddings

        # Different number of chunks and embeddings
        with self.assertRaises(ValueError):
            self.store.add_hybrid(
                full_embedding=self.test_embedding,
                chunks=self.test_chunks[:2],  # Only 2 chunks
                chunk_embeddings=self.test_chunk_embeddings,  # 3 embeddings
                original_content=self.test_content,
                metadata=self.test_metadata,
            )

    def test_is_hybrid(self):
        # Add a regular memory
        regular_id = self.store.add(self.test_embedding, "Regular content")

        # Add a hybrid memory
        hybrid_id = self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Check is_hybrid
        self.assertFalse(self.store.is_hybrid(regular_id))
        self.assertTrue(self.store.is_hybrid(hybrid_id))

        # Check for nonexistent memory
        self.assertFalse(self.store.is_hybrid("nonexistent_id"))

    def test_get_chunks(self):
        # Add a hybrid memory
        memory_id = self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Get chunks
        chunks = self.store.get_chunks(memory_id)

        # Should have 3 chunks
        self.assertEqual(len(chunks), 3)

        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)
            self.assertEqual(chunk.text, self.test_chunks[i]["text"])

        # Get chunks for nonexistent memory
        chunks = self.store.get_chunks("nonexistent_id")
        self.assertEqual(len(chunks), 0)

        # Get chunks for non-hybrid memory
        regular_id = self.store.add(self.test_embedding, "Regular content")
        chunks = self.store.get_chunks(regular_id)
        self.assertEqual(len(chunks), 0)

    def test_get_chunk_embeddings(self):
        # Add a hybrid memory
        memory_id = self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Get chunk embeddings
        embeddings = self.store.get_chunk_embeddings(memory_id)

        # Should have 3 embeddings
        self.assertEqual(len(embeddings), 3)

        # Verify embeddings
        for i, embedding in enumerate(embeddings):
            self.assertTrue(np.array_equal(embedding, self.test_chunk_embeddings[i]))

    def test_search_chunks(self):
        # Create chunks
        chunks = [
            {"text": "Chunk 1", "metadata": {"index": 0}},
            {"text": "Chunk 2", "metadata": {"index": 1}},
            {"text": "Chunk 3", "metadata": {"index": 2}},
        ]

        # Create distinct embeddings
        chunk_embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        # Add hybrid memory
        _memory_id = self.store.add_hybrid(
            np.array([0.5, 0.5, 0.5]), chunks, chunk_embeddings, "Original content"
        )

        # Use query that matches first chunk
        query_embedding = np.array([1.0, 0.0, 0.0])

        # Search without threshold
        results = self.store.search_chunks(query_embedding, limit=3, threshold=None)

        # Should find at least one result
        self.assertGreaterEqual(len(results), 1)

        # First result should have high similarity
        if results:
            self.assertGreater(results[0]["chunk_similarity"], 0.9)

    def test_search_hybrid(self):
        # Create test data
        full_embedding = np.array([1.0, 0.0, 0.0])
        chunks = [
            {"text": "Chunk 1", "metadata": {"index": 0}},
            {"text": "Chunk 2", "metadata": {"index": 1}},
        ]
        chunk_embeddings = [
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        # Add hybrid memory
        memory_id = self.store.add_hybrid(
            full_embedding, chunks, chunk_embeddings, "Hybrid content"
        )

        # Add regular memory for comparison
        _regular_id = self.store.add(np.array([0.5, 0.5, 0.5]), "Regular content")

        # Search with full embedding query
        query_embedding = np.array([1.0, 0.0, 0.0])

        # Search hybrid without threshold
        results = self.store.search_hybrid(query_embedding, limit=10, threshold=None)

        # Should find at least one result
        self.assertGreaterEqual(len(results), 1)

        # Our memory should be in the results
        memory_ids = [r["memory_id"] for r in results]
        self.assertIn(memory_id, memory_ids)

        # First result should have high similarity
        if results:
            self.assertGreater(results[0]["relevance_score"], 0.8)

    def test_search_hybrid_with_keywords(self):
        # Add a hybrid memory with specific keyword
        _hybrid_id = self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content="This content contains a special keyword: quantum",
            metadata=self.test_metadata,
        )

        # Add a regular memory with the same keyword
        _regular_id = self.store.add(
            np.array([0.9, 0.1, 0.1]), "Regular content with quantum physics", {"type": "regular"}
        )

        # Add another memory without the keyword
        _other_id = self.store.add(
            np.array([0.1, 0.9, 0.1]), "Content without special terms", {"type": "other"}
        )

        # Search with a neutral vector but with keyword matching
        query_vector = np.array([0.33, 0.33, 0.33])  # Neutral vector
        results = self.store.search_hybrid(query_vector, limit=3, keywords=["quantum"])

        # All 3 memories should be returned
        self.assertEqual(len(results), 3)

        # The two memories with "quantum" should be ranked higher
        quantum_matches = [
            result for result in results if "quantum" in str(result.get("content", "")).lower()
        ]
        non_quantum_matches = [
            result for result in results if "quantum" not in str(result.get("content", "")).lower()
        ]

        # Both matches should have higher scores than non-match
        self.assertTrue(
            all(
                qm["relevance_score"] > nm["relevance_score"]
                for qm in quantum_matches
                for nm in non_quantum_matches
            )
        )

        # Should see keyword_matches in results
        for result in quantum_matches:
            self.assertTrue("keyword_matches" in result)
            self.assertTrue(result["keyword_matches"] > 0)

    def test_remove(self):
        # Add a hybrid memory
        memory_id = self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Verify it was added
        self.assertTrue(self.store.is_hybrid(memory_id))

        # Remove the memory
        self.store.remove(memory_id)

        # Verify memory is removed
        with self.assertRaises(KeyError):
            self.store.get(memory_id)

        # Verify it's no longer hybrid
        self.assertFalse(self.store.is_hybrid(memory_id))

        # Verify chunks are removed
        chunks = self.store.get_chunks(memory_id)
        self.assertEqual(len(chunks), 0)

    def test_clear(self):
        # Add a hybrid memory
        self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Add a regular memory
        self.store.add(self.test_embedding, "Regular content")

        # Verify we have memories and chunks
        self.assertEqual(len(self.store.get_all()), 2)
        self.assertTrue(self.store.get_chunk_count() > 0)

        # Clear the store
        self.store.clear()

        # Verify everything is cleared
        self.assertEqual(len(self.store.get_all()), 0)
        self.assertEqual(self.store.get_chunk_count(), 0)
        self.assertEqual(len(self.store._hybrid_info), 0)

    def test_get_chunk_count(self):
        # Add a hybrid memory with 3 chunks
        self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Add another hybrid memory with 2 chunks
        self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks[:2],
            chunk_embeddings=self.test_chunk_embeddings[:2],
            original_content="Another content",
            metadata={"type": "another"},
        )

        # Should have 5 chunks total (3 + 2)
        self.assertEqual(self.store.get_chunk_count(), 5)

    def test_get_average_chunks_per_memory(self):
        # Add a hybrid memory with 3 chunks
        self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
        )

        # Add another hybrid memory with 2 chunks
        self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks[:2],
            chunk_embeddings=self.test_chunk_embeddings[:2],
            original_content="Another content",
            metadata={"type": "another"},
        )

        # Average should be 2.5 (5 chunks / 2 memories)
        self.assertEqual(self.store.get_average_chunks_per_memory(), 2.5)

        # Add a regular memory (no chunks)
        self.store.add(self.test_embedding, "Regular content")

        # Average should still be 2.5 (still 5 chunks / 2 hybrid memories)
        self.assertEqual(self.store.get_average_chunks_per_memory(), 2.5)

    def test_consolidate(self):
        # Add a hybrid memory with low importance
        hybrid_id1 = self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content="Low importance",
            metadata={"importance": 0.1},
        )

        # Add a hybrid memory with high importance
        hybrid_id2 = self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks[:2],
            chunk_embeddings=self.test_chunk_embeddings[:2],
            original_content="High importance",
            metadata={"importance": 0.9},
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
        self.assertEqual(
            removed_ids[0], hybrid_id1
        )  # Should have removed the low importance memory

        # Should have 1 memory left
        memories = self.store.get_all()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].id, hybrid_id2)

        # Should have 2 chunks left (from the high importance memory)
        self.assertEqual(self.store.get_chunk_count(), 2)

        # The remaining memory should still be hybrid
        self.assertTrue(self.store.is_hybrid(hybrid_id2))

    def test_update_metadata(self):
        # Add a hybrid memory
        memory_id = self.store.add_hybrid(
            full_embedding=self.test_embedding,
            chunks=self.test_chunks,
            chunk_embeddings=self.test_chunk_embeddings,
            original_content=self.test_content,
            metadata=self.test_metadata,
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

    def test_hybrid_memory_info_class(self):
        # Test HybridMemoryInfo class initialization
        chunk = ChunkInfo(chunk_index=0, embedding=np.array([0.1, 0.2, 0.3]), text="Test chunk")

        hybrid_info = HybridMemoryInfo(full_embedding=np.array([0.4, 0.5, 0.6]), chunks=[chunk])

        # Verify properties
        self.assertTrue(np.array_equal(hybrid_info.full_embedding, np.array([0.4, 0.5, 0.6])))
        self.assertEqual(len(hybrid_info.chunks), 1)
        self.assertTrue(hybrid_info.is_hybrid)

    def test_empty_search_chunks(self):
        # Search with no memories
        query_vector = np.array([0.1, 0.2, 0.3])
        results = self.store.search_chunks(query_vector)

        # Should have no results
        self.assertEqual(len(results), 0)

    def test_empty_search_hybrid(self):
        # Search with no memories
        query_vector = np.array([0.1, 0.2, 0.3])
        results = self.store.search_hybrid(query_vector)

        # Should have no results
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
