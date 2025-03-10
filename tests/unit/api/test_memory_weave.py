import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from memoryweave.api.config import MemoryWeaveConfig
from memoryweave.api.memory_weave import MemoryWeaveAPI
from memoryweave.interfaces.retrieval import QueryType


class TestMemoryWeaveAPI(unittest.TestCase):
    """Test the MemoryWeaveAPI class."""

    @patch("memoryweave.api.memory_weave.LLMProvider")
    @patch("memoryweave.api.memory_weave._get_embedder")
    @patch("memoryweave.api.memory_weave.create_memory_store_and_adapter")
    def setUp(self, mock_create_store, mock_get_embedder, mock_llm_provider):
        """Set up test environment before each test method."""
        # Setup mock embedding model
        self.mock_embedding_model = MagicMock()
        self.mock_embedding_model.get_sentence_embedding_dimension.return_value = 768
        self.mock_embedding_model.encode.return_value = np.ones(768)
        mock_get_embedder.return_value = self.mock_embedding_model

        # Setup mock memory store and adapter
        self.mock_memory_store = MagicMock()
        self.mock_memory_adapter = MagicMock()
        mock_create_store.return_value = (self.mock_memory_store, self.mock_memory_adapter)

        # Setup mock LLM provider
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = "This is a test response."
        mock_llm_provider.return_value = self.mock_llm

        # Setup API with mocks
        self.config = MemoryWeaveConfig(debug=True)
        self.api = MemoryWeaveAPI(config=self.config)

        # Mock the retrieval orchestrator for testing
        self.api.retrieval_orchestrator = MagicMock()
        self.api.retrieval_orchestrator.retrieve.return_value = [
            {
                "memory_id": "0",
                "content": "Test memory content",
                "relevance_score": 0.9,
                "metadata": {"type": "test"},
            }
        ]

    def test_initialization(self):
        """Test that the API initializes correctly with all components."""
        self.assertIsNotNone(self.api.memory_store)
        self.assertIsNotNone(self.api.memory_adapter)
        self.assertIsNotNone(self.api.embedding_model)
        self.assertIsNotNone(self.api.llm_provider)
        self.assertIsNotNone(self.api.strategy)
        self.assertIsNotNone(self.api.prompt_builder)
        self.assertEqual(self.api.conversation_history, [])

    def test_add_memory(self):
        """Test adding a memory."""
        # Setup
        self.mock_memory_adapter.add.return_value = "123"

        # Execute
        memory_id = self.api.add_memory("Test memory content")

        # Assert
        self.assertEqual(memory_id, "123")
        self.mock_embedding_model.encode.assert_called_once()
        self.mock_memory_adapter.add.assert_called_once()

    def test_add_memories(self):
        """Test adding multiple memories."""
        # Setup
        self.mock_memory_adapter.add.side_effect = ["1", "2", "3"]
        texts = ["Memory 1", "Memory 2", "Memory 3"]

        # Execute
        memory_ids = self.api.add_memories(texts)

        # Assert
        self.assertEqual(memory_ids, ["1", "2", "3"])
        self.assertEqual(self.mock_embedding_model.encode.call_count, 3)
        self.assertEqual(self.mock_memory_adapter.add.call_count, 3)

    def test_add_memory_with_metadata(self):
        """Test adding a memory with metadata."""
        # Setup
        self.mock_memory_adapter.add.return_value = "123"
        metadata = {"type": "test", "importance": 0.8}

        # Execute
        memory_id = self.api.add_memory("Test memory content", metadata)

        # Assert
        self.assertEqual(memory_id, "123")
        self.mock_memory_adapter.add.assert_called_once()
        # Check that metadata was passed correctly
        _, _, passed_metadata = self.mock_memory_adapter.add.call_args[0]
        self.assertEqual(passed_metadata["type"], "test")
        self.assertEqual(passed_metadata["importance"], 0.8)
        self.assertIn("created_at", passed_metadata)

    @patch("memoryweave.api.memory_weave.MemoryWeaveAPI._analyze_query")
    def test_chat(self, mock_analyze_query):
        """Test the chat method."""
        # Setup
        mock_analyze_query.return_value = (
            {"text": "test query"},  # query_obj
            {"max_results": 5},  # adapted_params
            ["test", "query"],  # expanded_keywords
            QueryType.FACTUAL,  # query_type
            [],  # entities
        )
        self.api.prompt_builder = MagicMock()
        self.api.prompt_builder.build_chat_prompt.return_value = "Test prompt"

        # Execute
        response = self.api.chat("test query")

        # Assert
        self.assertEqual(response, "This is a test response.")
        self.api.retrieval_orchestrator.retrieve.assert_called_once()
        self.api.prompt_builder.build_chat_prompt.assert_called_once()
        self.mock_llm.generate.assert_called_once_with(prompt="Test prompt", max_new_tokens=512)
        self.assertEqual(len(self.api.conversation_history), 2)  # User and assistant messages

    def test_retrieve(self):
        """Test the retrieve method."""
        # Execute
        results = self.api.retrieve("test query", top_k=5)

        # Assert
        self.api.retrieval_orchestrator.retrieve.assert_called_once()
        self.assertEqual(results, self.api.retrieval_orchestrator.retrieve.return_value)

    def test_analyze_query(self):
        """Test the query analysis method."""
        # Setup
        self.api.query_analyzer = MagicMock()
        self.api.query_analyzer.analyze.return_value = QueryType.PERSONAL
        self.api.query_analyzer.extract_keywords.return_value = ["test", "query"]
        self.api.query_analyzer.extract_entities.return_value = []

        self.api.query_adapter = MagicMock()
        self.api.query_adapter.adapt_parameters.return_value = {"confidence_threshold": 0.2}

        # Execute
        query_obj, adapted_params, expanded_keywords, query_type, entities = (
            self.api._analyze_query("test query")
        )

        # Assert
        self.assertEqual(query_type, QueryType.PERSONAL)
        self.assertEqual(expanded_keywords, ["test", "query"])
        self.assertEqual(entities, [])
        self.assertEqual(adapted_params["confidence_threshold"], 0.2)

    def test_store_interaction(self):
        """Test storing conversation interactions in memory."""
        # Execute
        self.api._store_interaction("User message", "Assistant reply", time.time())

        # Assert
        self.assertEqual(self.mock_embedding_model.encode.call_count, 2)  # Once for each message
        self.assertEqual(self.mock_memory_adapter.add.call_count, 2)  # Once for each message

    def test_update_conversation_history(self):
        """Test updating conversation history."""
        # Execute
        self.api._update_conversation_history("User message", "Assistant reply")

        # Assert
        self.assertEqual(len(self.api.conversation_history), 2)
        self.assertEqual(self.api.conversation_history[0]["role"], "user")
        self.assertEqual(self.api.conversation_history[0]["content"], "User message")
        self.assertEqual(self.api.conversation_history[1]["role"], "assistant")
        self.assertEqual(self.api.conversation_history[1]["content"], "Assistant reply")

    def test_clear_memories(self):
        """Test clearing memories."""
        # Setup
        self.mock_memory_adapter.get_all.return_value = [
            MagicMock(id="1", metadata={"type": "normal"}),
            MagicMock(id="2", metadata={"type": "personal_attribute"}),
        ]

        # Execute - keep personal attributes
        removed = self.api.clear_memories(keep_personal_attributes=True)

        # Assert
        self.mock_memory_adapter.clear.assert_called_once()
        self.assertEqual(removed, 1)  # Only one normal memory removed

        # Reset mocks
        self.mock_memory_adapter.clear.reset_mock()

        # Execute - clear all
        self.api.clear_memories(keep_personal_attributes=False)

        # Assert
        self.mock_memory_adapter.clear.assert_called_once()

    def test_extract_personal_attributes(self):
        """Test extracting personal attributes from messages."""
        # Setup
        self.api.personal_attribute_manager = MagicMock()
        self.api.personal_attribute_manager.extract_attributes.return_value = {
            "name": "John",
            "age": "30",
        }

        # Patch add_memory to track calls
        original_add_memory = self.api.add_memory
        self.api.add_memory = MagicMock()

        # Execute
        self.api._extract_personal_attributes("My name is John and I am 30 years old", time.time())

        # Assert
        self.api.personal_attribute_manager.extract_attributes.assert_called_once()
        self.assertEqual(self.api.add_memory.call_count, 2)  # One call for each attribute

        # Restore original method
        self.api.add_memory = original_add_memory


if __name__ == "__main__":
    unittest.main()
