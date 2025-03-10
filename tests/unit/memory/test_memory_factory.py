import unittest
from unittest.mock import MagicMock, patch

from memoryweave.api.config import VectorSearchConfig
from memoryweave.components.memory_encoding import MemoryEncoder
from memoryweave.factory.memory_factory import (
    MemoryStoreConfig,
    create_memory_encoder,
    create_memory_store_and_adapter,
)
from memoryweave.storage.adapter import MemoryAdapter
from memoryweave.storage.chunked_adapter import ChunkedMemoryAdapter
from memoryweave.storage.chunked_store import ChunkedMemoryStore
from memoryweave.storage.hybrid_adapter import HybridMemoryAdapter
from memoryweave.storage.hybrid_store import HybridMemoryStore
from memoryweave.storage.memory_store import StandardMemoryStore


class TestMemoryFactory(unittest.TestCase):
    """Test the memory_factory module functions."""

    def setUp(self):
        self.mock_sentence_transformer_patcher = patch(
            "memoryweave.factory.memory_factory.SentenceTransformer"
        )
        self.mock_sentence_transformer = self.mock_sentence_transformer_patcher.start()

        self.mock_device_patcher = patch("memoryweave.utils._get_device", return_value="cpu")
        self.mock_get_device = self.mock_device_patcher.start()

        self.mock_embedding_model = MagicMock()
        self.mock_embedding_model.get_sentence_embedding_dimension.return_value = 768
        self.mock_sentence_transformer.return_value = self.mock_embedding_model

    def tearDown(self):
        self.mock_sentence_transformer_patcher.stop()
        self.mock_device_patcher.stop()

    @patch("memoryweave.factory.memory_factory.create_vector_search_provider")
    def test_create_memory_store_and_adapter_standard(self, mock_create_vector_search):
        """Test creating a standard memory store and adapter."""
        # Setup
        mock_vector_search = MagicMock()
        mock_create_vector_search.return_value = mock_vector_search

        # Execute
        config = MemoryStoreConfig(
            store_type="standard", vector_search=VectorSearchConfig(type="numpy")
        )
        memory_store, memory_adapter = create_memory_store_and_adapter(config)

        # Assert
        self.assertIsInstance(memory_store, StandardMemoryStore)
        self.assertIsInstance(memory_adapter, MemoryAdapter)
        self.assertEqual(memory_adapter.memory_store, memory_store)
        mock_create_vector_search.assert_called_once()

    @patch("memoryweave.factory.memory_factory.create_vector_search_provider")
    def test_create_memory_store_and_adapter_hybrid(self, mock_create_vector_search):
        """Test creating a hybrid memory store and adapter."""
        # Setup
        mock_vector_search = MagicMock()
        mock_create_vector_search.return_value = mock_vector_search

        # Execute
        config = MemoryStoreConfig(
            store_type="hybrid", vector_search=VectorSearchConfig(type="numpy")
        )
        memory_store, memory_adapter = create_memory_store_and_adapter(config)

        # Assert
        self.assertIsInstance(memory_store, HybridMemoryStore)
        self.assertIsInstance(memory_adapter, HybridMemoryAdapter)
        self.assertEqual(memory_adapter.memory_store, memory_store)
        self.assertEqual(memory_adapter.hybrid_store, memory_store)
        mock_create_vector_search.assert_called_once()

    @patch("memoryweave.factory.memory_factory.create_vector_search_provider")
    def test_create_memory_store_and_adapter_chunked(self, mock_create_vector_search):
        """Test creating a chunked memory store and adapter."""
        # Setup
        mock_vector_search = MagicMock()
        mock_create_vector_search.return_value = mock_vector_search

        # Execute
        config = MemoryStoreConfig(
            store_type="chunked", vector_search=VectorSearchConfig(type="numpy")
        )
        memory_store, memory_adapter = create_memory_store_and_adapter(config)

        # Assert
        self.assertIsInstance(memory_store, ChunkedMemoryStore)
        self.assertIsInstance(memory_adapter, ChunkedMemoryAdapter)
        self.assertEqual(memory_adapter.memory_store, memory_store)
        self.assertEqual(memory_adapter.chunked_store, memory_store)

    def test_create_memory_store_and_adapter_no_vector_search(self):
        """Test creating a memory store without vector search configuration."""
        # Execute
        config = MemoryStoreConfig(store_type="standard", vector_search=None)
        memory_store, memory_adapter = create_memory_store_and_adapter(config)

        # Assert
        self.assertIsInstance(memory_store, StandardMemoryStore)
        self.assertIsInstance(memory_adapter, MemoryAdapter)
        self.assertEqual(memory_adapter.memory_store, memory_store)

    @patch("memoryweave.factory.memory_factory._get_device", return_value="cpu")
    def test_create_memory_encoder_with_name(self, mock_get_device):
        """Test creating a memory encoder using embedding_model_name."""
        # Execute
        encoder = create_memory_encoder(embedding_model_name="test-model")

        # Assert
        self.assertIsInstance(encoder, MemoryEncoder)
        self.mock_sentence_transformer.assert_called_once_with("test-model", device="cpu")
        mock_get_device.assert_called_once_with("auto")

    @patch("memoryweave.factory.memory_factory._get_device", return_value="cpu")
    def test_create_memory_encoder_with_default_name(self, mock_get_device):
        """Test creating a memory encoder using the default embedding_model_name."""
        # Execute

        encoder = create_memory_encoder()

        # Assert
        print(f"CONSOLE IS TYPE: {type(encoder)}")
        self.assertIsInstance(encoder, MemoryEncoder)
        self.mock_sentence_transformer.assert_called_once_with(
            "sentence-transformers/paraphrase-MiniLM-L6-v2", device="cpu"
        )
        mock_get_device.assert_called_once_with("auto")

    @patch("memoryweave.factory.memory_factory._get_device", return_value="cuda")
    def test_create_memory_encoder_with_provided_model(self, mock_get_device):
        """Test creating a memory encoder with a provided embedding_model."""
        # Setup
        provided_model = MagicMock()
        provided_model.get_sentence_embedding_dimension.return_value = 768

        # Execute
        encoder = create_memory_encoder(embedding_model=provided_model, device="cuda")

        # Assert
        self.assertIsInstance(encoder, MemoryEncoder)
        self.mock_sentence_transformer.assert_not_called()  # SentenceTransformer should not be called
        mock_get_device.assert_called_once_with("cuda")

    @patch("memoryweave.factory.memory_factory._get_device", return_value="mps")
    def test_create_memory_encoder_initialization(self, mock_get_device):
        """Test the initialization of the MemoryEncoder."""
        # Setup
        mock_embedding_model = MagicMock()
        mock_embedding_model.get_sentence_embedding_dimension.return_value = 768
        encoder = create_memory_encoder(
            embedding_model=mock_embedding_model,
            context_window_size=5,
            use_episodic_markers=False,
            context_enhancer_config={"test": "value"},
        )

        # Assert
        self.assertEqual(encoder.context_window_size, 5)
        self.assertFalse(encoder.use_episodic_markers)
        print(f"CONTEXT ENHANCER IS TYPE: {type(encoder.context_enhancer)}")
        self.assertEqual(encoder.context_enhancer, {"test": "value"})
        mock_get_device.assert_not_called()  # _get_device should not be called due to us making an embedding model


if __name__ == "__main__":
    unittest.main()
