"""
Integration tests for the MemoryWeave retriever.
"""

import unittest

from memoryweave.core.refactored_retrieval import RefactoredRetriever
# ContextualRetriever has been migrated to components architecture

from tests.utils.mock_models import MockEmbeddingModel, MockMemory


class RetrieverIntegrationTest(unittest.TestCase):
    """
    Integration tests for the ContextualRetriever and RefactoredRetriever.

    These tests verify that both retrievers produce similar results for
    various types of queries, ensuring that the refactored implementation
    maintains compatibility with the original.
    """

    def setUp(self):
        """Set up test environment before each test."""
        # Create embedding model
        self.embedding_model = MockEmbeddingModel(embedding_dim=768)

        # Create memory
        self.memory = MockMemory(embedding_dim=768)

        # Populate memory with test data
        self._populate_test_memory()

        # Original retriever is no longer needed as we've migrated to components

        # Create refactored retriever
        self.refactored_retriever = RefactoredRetriever(
            memory=self.memory,
            embedding_model=self.embedding_model,
            retrieval_strategy="hybrid",
            confidence_threshold=0.3,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
        )

    def _populate_test_memory(self):
        """Populate memory with test data."""
        # Add personal information
        self._add_personal_memory(
            "My favorite color is blue", {"type": "interaction", "speaker": "user"}
        )
        self._add_personal_memory("I live in Seattle", {"type": "interaction", "speaker": "user"})
        self._add_personal_memory(
            "I work as a software engineer", {"type": "interaction", "speaker": "user"}
        )
        self._add_personal_memory(
            "My wife's name is Sarah", {"type": "interaction", "speaker": "user"}
        )

        # Add factual information
        self._add_factual_memory(
            "Python",
            "A high-level programming language known for readability",
            ["Programming", "Scripting"],
        )
        self._add_factual_memory(
            "Machine Learning",
            "A field of AI that enables systems to learn from data",
            ["AI", "Data Science"],
        )
        self._add_factual_memory(
            "Memory Management",
            "Techniques for efficiently allocating computer memory",
            ["Computing", "Programming"],
        )

        # Add conversation history
        self._add_conversation_memory(
            "Tell me about Python",
            "Python is a high-level programming language known for its readability and versatility.",
        )
        self._add_conversation_memory(
            "How does it handle memory?",
            "Python uses automatic memory management with garbage collection.",
        )

    def _add_personal_memory(self, content, metadata):
        """Add a personal memory."""
        embedding = self.embedding_model.encode(content)
        self.memory.add_memory(embedding, content, metadata)

    def _add_factual_memory(self, concept, description, related_concepts):
        """Add a factual memory."""
        embedding = self.embedding_model.encode(description)
        metadata = {
            "type": "concept",
            "name": concept,
            "description": description,
            "related": related_concepts,
        }
        self.memory.add_memory(embedding, description, metadata)

    def _add_conversation_memory(self, message, response):
        """Add a conversation memory."""
        content = f"User: {message}\nAI: {response}"
        embedding = self.embedding_model.encode(content)
        metadata = {
            "type": "interaction",
            "speaker": "user",
            "message": message,
            "response": response,
        }
        self.memory.add_memory(embedding, content, metadata)

    def _verify_retrieval_results(self, results):
        """Verify that the retrieval results meet basic quality criteria."""
        # Check that we got some results
        self.assertTrue(len(results) > 0, "No results returned from retrieval")

        # Check that results have required fields
        for result in results:
            self.assertIn("memory_id", result, "Result missing memory_id field")
            self.assertIn("relevance_score", result, "Result missing relevance_score field")
            self.assertIn("content", result, "Result missing content field")

        # Check that results are sorted by relevance
        scores = [r.get("relevance_score", 0) for r in results]
        self.assertEqual(
            scores, sorted(scores, reverse=True), "Results not sorted by relevance score"
        )

    def test_personal_query(self):
        """Test retrieval for personal queries."""
        query = "What's my favorite color?"
        conversation_history = []

        # Get results from refactored retriever
        refactored_results = self.refactored_retriever.retrieve_for_context(
            query, conversation_history
        )

        # Verify basic result quality
        self._verify_retrieval_results(refactored_results)

        # Verify that the color information is retrieved
        color_found = False
        for result in refactored_results:
            if "blue" in str(result.get("content", "")).lower():
                color_found = True
                break

        self.assertTrue(color_found, "Failed to retrieve color information")

    def test_factual_query(self):
        """Test retrieval for factual queries."""
        query = "Tell me about programming languages"
        conversation_history = []

        # Get results from refactored retriever
        refactored_results = self.refactored_retriever.retrieve_for_context(
            query, conversation_history
        )

        # Verify basic result quality
        self._verify_retrieval_results(refactored_results)

        # Verify that programming information is retrieved
        programming_found = False
        for result in refactored_results:
            content = str(result.get("content", "")).lower()
            if "python" in content or "programming" in content:
                programming_found = True
                break

        self.assertTrue(programming_found, "Failed to retrieve programming information")

    def test_contextual_followup(self):
        """Test retrieval for contextual follow-up queries."""
        # First query to establish context
        first_query = "Tell me about Python"
        conversation_history = []

        first_results = self.refactored_retriever.retrieve_for_context(
            first_query, conversation_history
        )

        # Create conversation history
        conversation_history = [
            {
                "speaker": "user",
                "message": first_query,
                "response": "Python is a high-level programming language.",
            }
        ]

        # Follow-up query
        followup_query = "How does it handle memory management?"

        # Get results from refactored retriever
        refactored_results = self.refactored_retriever.retrieve_for_context(
            followup_query, conversation_history
        )

        # Verify basic result quality
        self._verify_retrieval_results(refactored_results)

        # Verify that memory management or Python information is retrieved
        relevant_found = False
        for result in refactored_results:
            content = str(result.get("content", "")).lower()
            if "memory" in content or "python" in content:
                relevant_found = True
                break

        self.assertTrue(relevant_found, "Failed to retrieve contextually relevant information")


if __name__ == "__main__":
    unittest.main()
