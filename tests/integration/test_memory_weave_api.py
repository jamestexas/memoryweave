# tests/integration/test_memory_weave_api.py
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memoryweave.api import MemoryWeaveAPI


class TestMemoryWeaveAPI:
    """Integration tests for the MemoryWeaveAPI class."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model that returns fixed embeddings."""
        model = MagicMock()
        model.encode.return_value = np.ones(384) / np.sqrt(384)  # Unit vector
        return model

    @pytest.fixture
    def api(self, mock_embedding_model):
        """Create a test instance of MemoryWeaveAPI with mocked components."""
        with patch("memoryweave.api.memory_weave._get_embedder") as mock_get_embedder:
            mock_get_embedder.return_value = mock_embedding_model

            # Mock the LLM to avoid loading real models
            with patch("memoryweave.api.memory_weave.get_llm"):
                with patch("memoryweave.api.memory_weave.get_tokenizer"):
                    api = MemoryWeaveAPI(
                        model_name="test-model", embedding_model_name="test-embeddings", debug=True
                    )
                    return api

    def test_add_memory(self, api, mock_embedding_model):
        """Test adding a memory to the system."""
        # Test adding a memory
        memory_text = "This is a test memory"
        memory_id = api.add_memory(memory_text)

        # Verify embedding model was called
        mock_embedding_model.encode.assert_called_with(
            memory_text, show_progress_bar=api.show_progress_bar
        )

        # Verify memory was added
        assert len(api.memory_store.get_all()) == 1

        # Verify we can retrieve the memory
        memory = api.memory_store.get(memory_id)
        assert str(memory.content) == memory_text

    def test_retrieve(self, api):
        """Test retrieving memories."""
        # Add some test memories
        api.add_memory("I live in New York")
        api.add_memory("My favorite color is blue")
        api.add_memory("I enjoy hiking on weekends")

        # Test basic retrieval
        results = api.retrieve("Where do I live?")
        assert len(results) > 0

        # Test with custom parameters
        results_custom = api.retrieve(
            "Where do I live?",
            top_k=2,
            confidence_threshold=0.05,
            weights={"similarity_weight": 0.8, "temporal_weight": 0.1},
        )

        # Should respect top_k
        assert len(results_custom) <= 2

    def test_chat(self, api):
        """Test the chat functionality."""
        # Add memories
        api.add_memory("My name is Alex")
        api.add_memory("I live in Boston")

        # Mock the LLM generation
        with patch.object(api, "_generate_response") as mock_generate:
            mock_generate.return_value = "Your name is Alex and you live in Boston."

            # Test chat
            response = api.chat("What's my name and where do I live?")

            # Verify response
            assert "Alex" in response
            assert "Boston" in response

            # Verify conversation history updated
            assert len(api.conversation_history) == 2
            assert api.conversation_history[0]["role"] == "user"
            assert api.conversation_history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chat_stream(self, api):
        """Test streaming chat functionality."""
        # Add memories
        api.add_memory("My name is Alex")

        # Mock the streaming function
        with patch.object(api, "_stream_generate_response") as mock_stream:

            async def mock_stream_generator():
                for token in ["Your ", "name ", "is ", "Alex", "."]:
                    yield token

            mock_stream.return_value = mock_stream_generator()

            # Test streaming
            response_chunks = []
            async for chunk in api.chat_stream("What's my name?"):
                response_chunks.append(chunk)

            # Verify streaming response
            assert len(response_chunks) == 5
            assert "".join(response_chunks) == "Your name is Alex."

    def test_memory_activation(self, api):
        """Test memory activation during retrieval."""
        # Add memories
        mem_id = api.add_memory("I have a dog named Max")

        # Track activation before
        activation_before = api.activation_manager.get_activation(mem_id)

        # Retrieve related content
        api.retrieve("What's my dog's name?")

        # Check activation increased
        activation_after = api.activation_manager.get_activation(mem_id)
        assert activation_after > activation_before
