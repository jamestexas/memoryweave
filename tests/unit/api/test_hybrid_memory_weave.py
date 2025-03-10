import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from memoryweave.api.config import MemoryStoreConfig, MemoryWeaveConfig
from memoryweave.api.hybrid_memory_weave import HybridMemoryWeaveAPI
from memoryweave.interfaces.retrieval import QueryType


class TestHybridMemoryWeaveAPI(unittest.TestCase):
    """Test the HybridMemoryWeaveAPI class."""

    @patch("memoryweave.api.hybrid_memory_weave.LLMProvider")
    @patch("memoryweave.api.hybrid_memory_weave._get_embedder")
    @patch("memoryweave.api.hybrid_memory_weave.create_memory_store_and_adapter")
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
        self.config = MemoryWeaveConfig(
            debug=True, memory_store=MemoryStoreConfig(store_type="hybrid")
        )
        self.api = HybridMemoryWeaveAPI(config=self.config)

        # Mock components
        self.api.text_chunker = MagicMock()
        self.api.text_chunker.create_chunks.return_value = [
            {"text": "Chunk 1", "metadata": {}},
            {"text": "Chunk 2", "metadata": {}},
        ]

        # Mock retrieval orchestrator
        self.api.retrieval_orchestrator = MagicMock()
        self.api.retrieval_orchestrator.retrieve.return_value = [
            {
                "memory_id": "0",
                "content": "Test memory content",
                "relevance_score": 0.9,
                "metadata": {"type": "test"},
            }
        ]

        # Setup hybrid-specific references
        self.api.hybrid_memory_store = self.mock_memory_store
        self.api.hybrid_memory_adapter = self.mock_memory_adapter

        # Mock HybridFabricStrategy
        self.api.hybrid_strategy = MagicMock()
        self.api.strategy = self.api.hybrid_strategy

    def test_initialization(self):
        """Test that the HybridMemoryWeaveAPI initializes correctly with all components."""
        self.assertIsNotNone(self.api.memory_store)
        self.assertIsNotNone(self.api.memory_adapter)
        self.assertIsNotNone(self.api.embedding_model)
        self.assertIsNotNone(self.api.llm_provider)
        self.assertIsNotNone(self.api.text_chunker)
        self.assertIsNotNone(self.api.hybrid_strategy)
        self.assertEqual(self.api.chunked_memory_ids, set())
        self.assertTrue(hasattr(self.api, "adaptive_chunk_threshold"))

    def test_create_memory_store(self):
        """Test the override of _create_memory_store method."""
        # Setup
        config = MemoryStoreConfig(store_type="standard")

        # Execute - should force to hybrid type
        with patch(
            "memoryweave.api.hybrid_memory_weave.create_memory_store_and_adapter"
        ) as mock_create:
            mock_memory_store = MagicMock()
            mock_memory_adapter = MagicMock()
            mock_create.return_value = (mock_memory_store, mock_memory_adapter)

            memory_store, memory_adapter = self.api._create_memory_store(config)

        # Assert
        self.assertEqual(config.store_type, "hybrid")  # Should force hybrid type

    def test_add_memory_without_chunking(self):
        """Test adding a small memory that doesn't need chunking."""
        # Setup
        small_text = "This is a small text that doesn't need chunking."
        self.api.adaptive_chunk_threshold = 100  # Set threshold higher than text length
        self.mock_memory_adapter.add.return_value = "123"

        # Execute
        memory_id = self.api.add_memory(small_text)

        # Assert
        self.assertEqual(memory_id, "123")
        self.mock_embedding_model.encode.assert_called_once()
        self.mock_memory_adapter.add.assert_called_once()
        self.assertNotIn("123", self.api.chunked_memory_ids)  # Should not be tracked as chunked

    def test_add_memory_with_chunking(self):
        """Test adding a large memory that needs chunking."""
        # Setup
        long_text = "This is a " + "very long " * 50 + "text that needs chunking."
        self.api.adaptive_chunk_threshold = 20  # Set threshold lower than text length
        self.mock_memory_adapter.add_hybrid.return_value = "456"

        # Setup mock for _select_important_chunks
        with patch.object(self.api, "_select_important_chunks") as mock_select_chunks:
            mock_select_chunks.return_value = (
                [
                    {"text": "Chunk 1", "metadata": {}},
                    {"text": "Chunk 2", "metadata": {}},
                ],  # Selected chunks
                [np.ones(768), np.ones(768)],  # Chunk embeddings
            )

            # Execute
            memory_id = self.api.add_memory(long_text)

        # Assert
        self.assertEqual(memory_id, "456")
        self.mock_embedding_model.encode.assert_called_once()
        self.api.text_chunker.create_chunks.assert_called_once()
        mock_select_chunks.assert_called_once()
        self.mock_memory_adapter.add_hybrid.assert_called_once()
        self.assertIn("456", self.api.chunked_memory_ids)  # Should be tracked as chunked

    def test_add_conversation_memory(self):
        """Test adding a conversation memory."""
        # Setup
        turns = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm fine, thank you!"},
        ]
        self.mock_memory_adapter.add_hybrid.return_value = "789"

        # Setup mock for _select_important_chunks
        with patch.object(self.api, "_select_important_chunks") as mock_select_chunks:
            mock_select_chunks.return_value = (
                [
                    {"text": "Chunk 1", "metadata": {}},
                    {"text": "Chunk 2", "metadata": {}},
                ],  # Selected chunks
                [np.ones(768), np.ones(768)],  # Chunk embeddings
            )

            # Execute
            memory_id = self.api.add_conversation_memory(turns)

        # Assert
        self.assertEqual(memory_id, "789")
        self.mock_embedding_model.encode.assert_called()
        self.assertIn("789", self.api.chunked_memory_ids)

        # Check metadata contains conversation-specific fields
        _, _, _, _, metadata = self.mock_memory_adapter.add_hybrid.call_args[0]
        self.assertEqual(metadata["type"], "conversation")
        self.assertEqual(metadata["turn_count"], 2)

    def test_store_hybrid_interaction(self):
        """Test storing conversation as hybrid memories."""
        # Setup
        user_message = "This is a " + "very long " * 50 + "user message"
        assistant_message = "This is a short assistant message"
        timestamp = time.time()

        # Set thresholds
        self.api.adaptive_chunk_threshold = 20  # Lower than user message, higher than assistant

        # Setup mocks for add and add_hybrid
        self.mock_memory_adapter.add.return_value = "user_mem_id"
        self.mock_memory_adapter.add_hybrid.return_value = "assistant_mem_id"

        # Execute with patch for _select_important_chunks
        with patch.object(self.api, "_select_important_chunks") as mock_select_chunks:
            mock_select_chunks.return_value = (
                [
                    {"text": "Chunk 1", "metadata": {}},
                    {"text": "Chunk 2", "metadata": {}},
                ],  # Selected chunks
                [np.ones(768), np.ones(768)],  # Chunk embeddings
            )

            self.api._store_hybrid_interaction(user_message, assistant_message, timestamp)

        # Assert
        # User message is long, should be chunked
        self.mock_memory_adapter.add_hybrid.assert_called_once()
        self.assertIn("assistant_mem_id", self.api.chunked_memory_ids)

        # Assistant message is short, should not be chunked
        self.mock_memory_adapter.add.assert_called_once()

    def test_chat(self):
        """Test the chat method with hybrid memory retrieval."""
        # Setup
        self.api._analyze_query = MagicMock(
            return_value=(
                {"text": "test query"},  # query_obj
                {"max_results": 5},  # adapted_params
                ["test", "query"],  # expanded_keywords
                QueryType.FACTUAL,  # query_type
                [],  # entities
            )
        )
        self.api.prompt_builder = MagicMock()
        self.api.prompt_builder.build_chat_prompt.return_value = "Test prompt"

        # Mock _store_hybrid_interaction to verify it's called
        self.api._store_hybrid_interaction = MagicMock()

        # Execute
        response = self.api.chat("test query")

        # Assert
        self.assertEqual(response, "This is a test response.")
        self.api.retrieval_orchestrator.retrieve.assert_called_once()
        self.api.prompt_builder.build_chat_prompt.assert_called_once()
        self.mock_llm.generate.assert_called_once_with(prompt="Test prompt", max_new_tokens=512)
        self.api._store_hybrid_interaction.assert_called_once()

    def test_select_important_chunks(self):
        """Test the chunk selection logic."""
        # Setup chunks and embeddings
        chunks = [
            {"text": "Chunk 1 with some important content", "metadata": {}},
            {"text": "Chunk 2 with less relevant content", "metadata": {}},
            {"text": "Chunk 3 with very important keywords here", "metadata": {}},
        ]
        full_embedding = np.ones(768)

        # Setup encode to return different values for different chunks
        chunk_embeddings = [
            np.ones(768) * 0.8,  # Good similarity to full embedding
            np.ones(768) * 0.5,  # Medium similarity
            np.ones(768) * 0.9,  # Best similarity
        ]
        self.mock_embedding_model.encode.side_effect = chunk_embeddings

        # Execute
        with patch.object(self.api, "_extract_simple_keywords") as mock_extract:
            mock_extract.return_value = ["important", "keywords"]
            selected_chunks, selected_embeddings = self.api._select_important_chunks(
                chunks, full_embedding
            )

        # Assert
        self.assertEqual(len(selected_chunks), min(len(chunks), self.api.max_chunks_per_memory))
        # Should select chunks 3 and 1 as they have highest similarity
        self.assertEqual(selected_chunks[0]["text"], "Chunk 3 with very important keywords here")

    def test_extract_simple_keywords(self):
        """Test the keyword extraction logic."""
        # Execute
        keywords = self.api._extract_simple_keywords(
            "This is a test sentence with some important keywords to extract."
        )

        # Assert
        self.assertIsInstance(keywords, list)
        self.assertTrue(all(isinstance(k, str) for k in keywords))
        # Should extract words longer than 3 chars that aren't stopwords
        self.assertIn("test", keywords)
        self.assertIn("sentence", keywords)
        self.assertIn("important", keywords)
        self.assertIn("keywords", keywords)
        self.assertIn("extract", keywords)
        # Shouldn't include stopwords or short words
        self.assertNotIn("this", keywords)
        self.assertNotIn("is", keywords)
        self.assertNotIn("a", keywords)
        self.assertNotIn("to", keywords)

    def test_analyze_chunking_needs(self):
        """Test the adaptive chunking analysis."""
        # Test short text
        short_text = "This is a short text."
        should_chunk, threshold = self.api._analyze_chunking_needs(short_text)
        self.assertFalse(should_chunk)

        # Test long text
        long_text = "This is a very long text. " * 50
        should_chunk, threshold = self.api._analyze_chunking_needs(long_text)
        self.assertTrue(should_chunk)

        # Test text with code blocks
        code_text = "```python\ndef function():\n    return True\n```"
        should_chunk, threshold = self.api._analyze_chunking_needs(code_text)
        self.assertTrue(should_chunk)
        self.assertLess(threshold, self.api.adaptive_chunk_threshold)

        # Test text with lists
        list_text = "My list:\n- Item 1\n- Item 2\n- Item 3\n"
        should_chunk, threshold = self.api._analyze_chunking_needs(list_text)
        self.assertTrue(should_chunk)

        # Test with auto chunking disabled
        self.api.enable_auto_chunking = False
        should_chunk, threshold = self.api._analyze_chunking_needs(long_text)
        self.assertEqual(threshold, self.api.adaptive_chunk_threshold)
        self.assertEqual(should_chunk, len(long_text) > self.api.adaptive_chunk_threshold)

    def test_configure_two_stage_retrieval(self):
        """Test configuring two-stage retrieval."""
        # Execute
        self.api.configure_two_stage_retrieval(
            enable=True, first_stage_k=40, first_stage_threshold_factor=0.6
        )

        # Assert
        self.api.hybrid_strategy.configure_two_stage.assert_called_once_with(True, 40, 0.6)

    def test_configure_chunking(self):
        """Test configuring chunking parameters."""
        # Setup
        self.api.text_chunker.initialize = MagicMock()

        # Execute
        self.api.configure_chunking(
            adaptive_chunk_threshold=1000,
            enable_auto_chunking=False,
            max_chunks_per_memory=5,
            chunk_size=400,
            chunk_overlap=50,
            min_chunk_size=100,
            keyword_boost_factor=0.4,
        )

        # Assert
        self.assertEqual(self.api.adaptive_chunk_threshold, 1000)
        self.assertEqual(self.api.enable_auto_chunking, False)
        self.assertEqual(self.api.max_chunks_per_memory, 5)

        # Should initialize text chunker with subset of params
        self.api.text_chunker.initialize.assert_called_once()
        chunker_params = self.api.text_chunker.initialize.call_args[0][0]
        self.assertEqual(chunker_params["chunk_size"], 400)
        self.assertEqual(chunker_params["chunk_overlap"], 50)
        self.assertEqual(chunker_params["min_chunk_size"], 100)

        # Should initialize strategy with keyword-related params
        self.api.strategy.initialize.assert_called_once()
        strategy_params = self.api.strategy.initialize.call_args[0][0]
        self.assertEqual(strategy_params["keyword_boost_factor"], 0.4)

    def test_get_chunking_statistics(self):
        """Test getting chunking statistics."""
        # Setup
        self.mock_memory_adapter.get_all.return_value = [MagicMock(), MagicMock()]
        self.mock_memory_adapter.get_chunk_count.return_value = 5
        self.mock_memory_adapter.get_average_chunks_per_memory.return_value = 2.5
        self.api.chunked_memory_ids = set(["1", "2"])

        # Execute
        stats = self.api.get_chunking_statistics()

        # Assert
        self.assertEqual(stats["total_memories"], 2)
        self.assertEqual(stats["chunked_memories"], 2)
        self.assertEqual(stats["total_chunks"], 5)
        self.assertEqual(stats["avg_chunks_per_memory"], 2.5)
        self.assertEqual(stats["adaptive_chunk_threshold"], self.api.adaptive_chunk_threshold)
        self.assertEqual(stats["enable_auto_chunking"], self.api.enable_auto_chunking)

    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        # Setup
        self.mock_memory_adapter.get_all.return_value = [MagicMock(), MagicMock()]
        self.mock_memory_adapter.get_chunk_count.return_value = 5
        self.api.chunked_memory_ids = set(["1", "2"])

        # Execute
        stats = self.api.get_performance_stats()

        # Assert
        self.assertIn("memory_usage", stats)
        self.assertIn("total_memories", stats["memory_usage"])
        self.assertIn("chunked_memories", stats["memory_usage"])
        self.assertIn("total_chunks", stats["memory_usage"])

    def test_search_by_keywords(self):
        """Test keyword-based memory search."""
        # Setup
        keywords = ["python", "code", "memory"]
        self.api.bm25_index = MagicMock()
        self.api.bm25_index.get_scores.return_value = np.array([0.8, 0.5, 0.3])
        self.api.bm25_documents = ["doc1", "doc2", "doc3"]
        self.api.bm25_doc_ids = ["id1", "id2", "id3"]

        # Mock memory retrieval
        self.mock_memory_adapter.get.side_effect = [
            MagicMock(content="Python code", metadata={"type": "code"}),
            MagicMock(content="Memory management", metadata={"type": "article"}),
        ]

        # Execute
        results = self.api.search_by_keywords(keywords, limit=2)

        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["memory_id"], "id1")  # Highest score
        self.assertEqual(results[0]["retrieval_method"], "bm25")
        self.assertIn("relevance_score", results[0])


if __name__ == "__main__":
    unittest.main()
