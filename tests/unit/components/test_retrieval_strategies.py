"""
Unit tests for the retrieval strategy components.
"""

import unittest

import numpy as np

from memoryweave.components.retrieval_strategies import (
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
)
from tests.utils.mock_models import MockMemory


class RetrievalStrategiesTest(unittest.TestCase):
    """
    Unit tests for the retrieval strategy components.
    """

    def setUp(self):
        """Set up test environment before each test."""
        # Create memory
        self.memory = MockMemory(embedding_dim=768)

        # Populate memory with test data
        self._populate_test_memory()

        # Create retrieval strategies
        self.similarity_strategy = SimilarityRetrievalStrategy(self.memory)
        self.temporal_strategy = TemporalRetrievalStrategy(self.memory)
        self.hybrid_strategy = HybridRetrievalStrategy(self.memory)

        # Initialize strategies
        self.similarity_strategy.initialize({"confidence_threshold": 0.0})
        self.temporal_strategy.initialize({})
        self.hybrid_strategy.initialize(
            {"relevance_weight": 0.7, "recency_weight": 0.3, "confidence_threshold": 0.0}
        )

    def _populate_test_memory(self):
        """Populate memory with test data."""
        # Create test embeddings
        np.random.seed(42)
        embeddings = []

        # Create 10 random embeddings
        for i in range(10):
            embedding = np.random.randn(768)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

            # Add to memory with metadata
            self.memory.add_memory(embedding, f"Test memory {i}", {"type": "test", "index": i})

    def test_similarity_retrieval(self):
        """Test similarity retrieval strategy."""
        # Create a query embedding similar to the first memory
        query_embedding = self.memory.memory_embeddings[0] * 0.9 + np.random.randn(768) * 0.1
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Retrieve memories
        results = self.similarity_strategy.retrieve(query_embedding, 3, {"memory": self.memory})

        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["memory_id"], 0)  # Most similar should be first

    def test_temporal_retrieval(self):
        """Test temporal retrieval strategy."""
        # Create a dummy query embedding
        query_embedding = np.random.randn(768)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Retrieve memories
        results = self.temporal_strategy.retrieve(query_embedding, 3, {"memory": self.memory})

        # Verify results
        self.assertEqual(len(results), 3)

        # Most recent memories should be first
        self.assertEqual(results[0]["memory_id"], 9)
        self.assertEqual(results[1]["memory_id"], 8)
        self.assertEqual(results[2]["memory_id"], 7)

    def test_hybrid_retrieval(self):
        """Test hybrid retrieval strategy."""
        # Create a query embedding similar to the first memory
        query_embedding = self.memory.memory_embeddings[0] * 0.9 + np.random.randn(768) * 0.1
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Retrieve memories
        results = self.hybrid_strategy.retrieve(query_embedding, 3, {"memory": self.memory})

        # Verify results
        self.assertEqual(len(results), 3)

        # Check that results include both similarity and recency factors
        self.assertIn("similarity", results[0])
        self.assertIn("recency", results[0])

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        # Create a query embedding dissimilar to all memories
        query_embedding = np.random.randn(768)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Initialize with high threshold
        self.similarity_strategy.initialize({"confidence_threshold": 0.9})

        # Retrieve memories
        results = self.similarity_strategy.retrieve(query_embedding, 3, {"memory": self.memory})

        # Verify that no results pass the threshold
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
