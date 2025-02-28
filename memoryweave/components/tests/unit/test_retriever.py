"""
Unit tests for the Retriever class.
"""

import unittest

from memoryweave.retriever import Retriever
from tests.utils.mock_models import MockEmbeddingModel, MockMemory


class RetrieverTest(unittest.TestCase):
    """Unit tests for the Retriever class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create mock memory and embedding model
        self.memory = MockMemory(embedding_dim=768)
        self.embedding_model = MockEmbeddingModel(embedding_dim=768)

        # Populate memory with test data
        self._populate_test_memory()

        # Create retriever
        self.retriever = Retriever(memory=self.memory, embedding_model=self.embedding_model)

    def _populate_test_memory(self):
        """Populate memory with test data."""
        # Create test data
        test_data = [
            ("apple", "Apples are red fruit", {"type": "food", "category": "fruit"}),
            ("banana", "Bananas are yellow fruit", {"type": "food", "category": "fruit"}),
            ("carrot", "Carrots are orange vegetables", {"type": "food", "category": "vegetable"}),
            ("dog", "Dogs are common pets", {"type": "animal", "category": "pet"}),
            ("cat", "Cats are independent pets", {"type": "animal", "category": "pet"}),
        ]

        # Add memories
        for text, content, metadata in test_data:
            embedding = self.embedding_model.encode(text)
            metadata_with_content = metadata.copy()
            metadata_with_content["content"] = content
            self.memory.add_memory(embedding, content, metadata_with_content)

    def test_retriever_initialization(self):
        """Test retriever initialization."""
        # Verify components are initialized
        self.assertIsNotNone(self.retriever.query_analyzer)
        self.assertIsNotNone(self.retriever.retrieval_strategy)
        self.assertTrue(len(self.retriever.post_processors) > 0)

    def test_retrieve_default(self):
        """Test default retrieval."""
        # Retrieve memories
        results = self.retriever.retrieve("apple")

        # Verify results
        self.assertTrue(len(results) > 0)
        self.assertIn("memory_id", results[0])
        self.assertIn("relevance_score", results[0])

    def test_retrieve_with_strategy(self):
        """Test retrieval with specific strategy."""
        # Test similarity strategy
        similarity_results = self.retriever.retrieve("apple", strategy="similarity")
        self.assertTrue(len(similarity_results) > 0)

        # Test temporal strategy
        temporal_results = self.retriever.retrieve("apple", strategy="temporal")
        self.assertTrue(len(temporal_results) > 0)

        # First temporal result should be the last added memory
        self.assertEqual(temporal_results[0]["memory_id"], 4)  # 5th memory, 0-indexed

    def test_retrieve_top_k(self):
        """Test top_k parameter."""
        # Retrieve with top_k=2
        results = self.retriever.retrieve("fruit", top_k=2)
        self.assertEqual(len(results), 2)

        # Retrieve with top_k=3
        results = self.retriever.retrieve("fruit", top_k=3)
        self.assertEqual(len(results), 3)

    def test_retrieve_minimum_relevance(self):
        """Test minimum_relevance parameter."""
        # Set a high minimum relevance
        results = self.retriever.retrieve("unrelated query", minimum_relevance=0.8)

        # Should return no results as relevance is too low
        self.assertEqual(len(results), 0)

    def test_pipeline_configuration(self):
        """Test pipeline configuration."""
        # Create a custom pipeline configuration
        pipeline_config = [
            {"component": "query_analyzer", "config": {}},
            {"component": "similarity_retrieval", "config": {"confidence_threshold": 0.2}},
        ]

        # Configure the pipeline
        self.retriever.configure_pipeline(pipeline_config)

        # Retrieve memories
        results = self.retriever.retrieve("apple")

        # Verify results
        self.assertTrue(len(results) > 0)

    def test_conversation_state_tracking(self):
        """Test conversation state tracking."""
        # Retrieve with conversation history
        self.retriever.retrieve("What is an apple?")
        self.retriever.retrieve("Tell me about bananas")

        # Check conversation history
        self.assertEqual(len(self.retriever.conversation_history), 2)
        self.assertEqual(self.retriever.conversation_history[0]["content"], "What is an apple?")
        self.assertEqual(self.retriever.conversation_history[1]["content"], "Tell me about bananas")

    def test_dynamic_threshold_adjustment(self):
        """Test dynamic threshold adjustment."""
        # Enable dynamic threshold adjustment
        self.retriever.enable_dynamic_threshold_adjustment(window_size=2)

        # Initial threshold
        initial_threshold = self.retriever.minimum_relevance

        # Make several retrievals to trigger adjustment
        self.retriever.retrieve("apple")
        self.retriever.retrieve("banana")
        self.retriever.retrieve("unrelated query")

        # Check if threshold was adjusted
        self.assertEqual(len(self.retriever.recent_retrieval_metrics), 2)  # Limited by window size


if __name__ == "__main__":
    unittest.main()
