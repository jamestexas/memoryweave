"""
Tests for the memory adapter components.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from memoryweave.adapters.memory_adapter import (
    LegacyMemoryAdapter, 
    LegacyVectorStoreAdapter, 
    LegacyActivationManagerAdapter
)


class TestLegacyMemoryAdapter:
    """Test suite for the LegacyMemoryAdapter class."""
    
    @pytest.fixture
    def mock_legacy_memory(self):
        """Create a mock legacy ContextualMemory."""
        mock_memory = MagicMock()
        
        # Configure mock to return values for testing
        mock_memory.memory_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        mock_memory.memory_metadata = [
            {"text": "Memory 0", "source": "test", "importance": 0.8},
            {"text": "Memory 1", "source": "test", "importance": 0.7}
        ]
        mock_memory.activation_levels = np.array([0.5, 0.3])
        
        # Configure add_memory method
        def mock_add_memory(embedding, text, metadata=None):
            # Return the index of the "new" memory
            return len(mock_memory.memory_metadata) - 1
        
        mock_memory.add_memory.side_effect = mock_add_memory
        
        return mock_memory
    
    def test_init(self, mock_legacy_memory):
        """Test initialization of the adapter."""
        adapter = LegacyMemoryAdapter(mock_legacy_memory)
        
        assert adapter._legacy_memory == mock_legacy_memory
        assert adapter._memory_id_map == {}
        assert adapter._next_id == 0
    
    def test_add(self, mock_legacy_memory):
        """Test adding a memory through the adapter."""
        adapter = LegacyMemoryAdapter(mock_legacy_memory)
        
        # Add a memory
        embedding = np.array([0.7, 0.8, 0.9])
        content = "Test memory"
        metadata = {"source": "test", "importance": 0.9}
        
        memory_id = adapter.add(embedding, content, metadata)
        
        # Verify legacy method was called
        mock_legacy_memory.add_memory.assert_called_once()
        call_args = mock_legacy_memory.add_memory.call_args
        assert np.array_equal(call_args[0][0], embedding)
        assert call_args[0][1] == content
        
        # Verify ID mapping
        assert memory_id in adapter._memory_id_map
        assert adapter._memory_id_map[memory_id] == mock_legacy_memory.add_memory.return_value
    
    def test_get(self, mock_legacy_memory):
        """Test retrieving a memory through the adapter."""
        adapter = LegacyMemoryAdapter(mock_legacy_memory)
        
        # Add a memory to create an ID mapping
        memory_id = adapter.add(np.array([0.7, 0.8, 0.9]), "Test memory")
        
        # Get the memory
        memory = adapter.get(memory_id)
        
        # Verify memory fields
        assert memory["id"] == memory_id
        assert memory["content"]["text"] == "Memory 1"  # From the mock
        assert memory["metadata"]["source"] == "test"
        assert memory["metadata"]["importance"] == 0.7
        
        # Test non-existent memory
        with pytest.raises(KeyError):
            adapter.get("non_existent_id")
    
    def test_get_all(self, mock_legacy_memory):
        """Test retrieving all memories through the adapter."""
        adapter = LegacyMemoryAdapter(mock_legacy_memory)
        
        # Add two memories to create ID mappings
        memory_id1 = adapter.add(np.array([0.7, 0.8, 0.9]), "Test memory 1")
        memory_id2 = adapter.add(np.array([0.4, 0.5, 0.6]), "Test memory 2")
        
        # Get all memories
        memories = adapter.get_all()
        
        # Verify memories
        assert len(memories) == 2
        assert memories[0]["id"] == memory_id1
        assert memories[1]["id"] == memory_id2
    
    def test_get_embeddings(self, mock_legacy_memory):
        """Test retrieving embeddings through the adapter."""
        adapter = LegacyMemoryAdapter(mock_legacy_memory)
        
        # Get embeddings
        embeddings = adapter.get_embeddings()
        
        # Verify embeddings
        assert np.array_equal(embeddings, mock_legacy_memory.memory_embeddings)
    
    def test_get_ids(self, mock_legacy_memory):
        """Test retrieving all memory IDs through the adapter."""
        adapter = LegacyMemoryAdapter(mock_legacy_memory)
        
        # Add memories to create ID mappings
        memory_id1 = adapter.add(np.array([0.7, 0.8, 0.9]), "Test memory 1")
        memory_id2 = adapter.add(np.array([0.4, 0.5, 0.6]), "Test memory 2")
        
        # Get IDs
        ids = adapter.get_ids()
        
        # Verify IDs
        assert len(ids) == 2
        assert memory_id1 in ids
        assert memory_id2 in ids
    
    def test_update_activation(self, mock_legacy_memory):
        """Test updating activation through the adapter."""
        adapter = LegacyMemoryAdapter(mock_legacy_memory)
        
        # Add a memory to create an ID mapping
        memory_id = adapter.add(np.array([0.7, 0.8, 0.9]), "Test memory")
        
        # Set up the _update_activation method on the mock
        mock_legacy_memory._update_activation = MagicMock()
        
        # Update activation
        adapter.update_activation(memory_id, 0.5)
        
        # Verify legacy method was called
        mock_legacy_memory._update_activation.assert_called_once_with(
            adapter._memory_id_map[memory_id]
        )
        
        # Test non-existent memory
        with pytest.raises(KeyError):
            adapter.update_activation("non_existent_id", 0.5)


class TestLegacyVectorStoreAdapter:
    """Test suite for the LegacyVectorStoreAdapter class."""
    
    @pytest.fixture
    def mock_legacy_memory(self):
        """Create a mock legacy ContextualMemory with retrieve_memories method."""
        mock_memory = MagicMock()
        
        # Configure retrieve_memories method
        def mock_retrieve_memories(query_embedding, top_k=5, activation_boost=False, confidence_threshold=0.0, **kwargs):
            # Return mock results
            return [
                (0, 0.95, {"text": "Memory 0"}),
                (1, 0.85, {"text": "Memory 1"}),
                (2, 0.75, {"text": "Memory 2"})
            ][:top_k]
        
        mock_memory.retrieve_memories.side_effect = mock_retrieve_memories
        
        return mock_memory
    
    @pytest.fixture
    def memory_adapter(self, mock_legacy_memory):
        """Create a memory adapter with ID mappings."""
        adapter = LegacyMemoryAdapter(mock_legacy_memory)
        
        # Create ID mappings
        adapter._memory_id_map = {
            "memory0": 0,
            "memory1": 1,
            "memory2": 2
        }
        
        return adapter
    
    def test_init(self, mock_legacy_memory, memory_adapter):
        """Test initialization of the adapter."""
        adapter = LegacyVectorStoreAdapter(mock_legacy_memory, memory_adapter)
        
        assert adapter._legacy_memory == mock_legacy_memory
        assert adapter._memory_adapter == memory_adapter
    
    def test_search(self, mock_legacy_memory, memory_adapter):
        """Test searching through the adapter."""
        adapter = LegacyVectorStoreAdapter(mock_legacy_memory, memory_adapter)
        
        # Create query embedding
        query_embedding = np.array([0.1, 0.2, 0.3])
        
        # Search for similar vectors
        results = adapter.search(query_embedding, k=2, threshold=0.8)
        
        # Verify legacy method was called
        mock_legacy_memory.retrieve_memories.assert_called_once()
        call_args = mock_legacy_memory.retrieve_memories.call_args
        assert np.array_equal(call_args[1]["query_embedding"], query_embedding)
        assert call_args[1]["top_k"] == 2
        assert call_args[1]["confidence_threshold"] == 0.8
        
        # Verify results
        assert len(results) == 2
        assert results[0][0] == "memory0"  # ID mapping from memory_adapter
        assert results[0][1] == 0.95
        assert results[1][0] == "memory1"
        assert results[1][1] == 0.85
    
    def test_search_with_threshold_filter(self, mock_legacy_memory, memory_adapter):
        """Test searching with threshold filtering."""
        # Modify mock to return unfiltered results
        def mock_retrieve_memories(query_embedding, top_k=5, activation_boost=False, confidence_threshold=0.0, **kwargs):
            # Return mock results without filtering by threshold
            return [
                (0, 0.95, {"text": "Memory 0"}),
                (1, 0.85, {"text": "Memory 1"}),
                (2, 0.65, {"text": "Memory 2"})  # Below threshold
            ][:top_k]
        
        mock_legacy_memory.retrieve_memories.side_effect = mock_retrieve_memories
        
        adapter = LegacyVectorStoreAdapter(mock_legacy_memory, memory_adapter)
        
        # Search with high threshold
        results = adapter.search(np.array([0.1, 0.2, 0.3]), k=3, threshold=0.8)
        
        # Verify only results above threshold are returned
        assert len(results) == 2  # memory2 should be filtered out
        assert results[0][0] == "memory0"
        assert results[1][0] == "memory1"


class TestLegacyActivationManagerAdapter:
    """Test suite for the LegacyActivationManagerAdapter class."""
    
    @pytest.fixture
    def mock_legacy_memory(self):
        """Create a mock legacy ContextualMemory."""
        mock_memory = MagicMock()
        
        # Configure mock to return values for testing
        mock_memory.activation_levels = np.array([0.5, 0.8, 0.3])
        
        # Set up the _update_activation method on the mock
        mock_memory._update_activation = MagicMock()
        
        return mock_memory
    
    @pytest.fixture
    def memory_adapter(self, mock_legacy_memory):
        """Create a memory adapter with ID mappings."""
        adapter = LegacyMemoryAdapter(mock_legacy_memory)
        
        # Create ID mappings
        adapter._memory_id_map = {
            "memory0": 0,
            "memory1": 1,
            "memory2": 2
        }
        
        return adapter
    
    def test_init(self, mock_legacy_memory, memory_adapter):
        """Test initialization of the adapter."""
        adapter = LegacyActivationManagerAdapter(mock_legacy_memory, memory_adapter)
        
        assert adapter._legacy_memory == mock_legacy_memory
        assert adapter._memory_adapter == memory_adapter
    
    def test_update_activation(self, mock_legacy_memory, memory_adapter):
        """Test updating activation through the adapter."""
        adapter = LegacyActivationManagerAdapter(mock_legacy_memory, memory_adapter)
        
        # Update activation
        adapter.update_activation("memory1", 0.5)
        
        # Verify memory_adapter method was called
        memory_adapter.update_activation.assert_called_once_with("memory1", 0.5)
    
    def test_get_activation(self, mock_legacy_memory, memory_adapter):
        """Test getting activation through the adapter."""
        adapter = LegacyActivationManagerAdapter(mock_legacy_memory, memory_adapter)
        
        # Get activation
        activation = adapter.get_activation("memory1")
        
        # Verify correct activation is returned
        assert activation == 0.8  # From mock_legacy_memory.activation_levels[1]
        
        # Test non-existent memory
        with pytest.raises(KeyError):
            adapter.get_activation("non_existent_id")
    
    def test_decay_activations(self, mock_legacy_memory, memory_adapter):
        """Test decaying activations through the adapter."""
        adapter = LegacyActivationManagerAdapter(mock_legacy_memory, memory_adapter)
        
        # Original activations
        original_activations = mock_legacy_memory.activation_levels.copy()
        
        # Decay activations
        adapter.decay_activations(0.5)
        
        # Verify activations were decayed
        expected = original_activations * 0.5
        assert np.array_equal(mock_legacy_memory.activation_levels, expected)
    
    def test_get_most_active(self, mock_legacy_memory, memory_adapter):
        """Test getting most active memories through the adapter."""
        adapter = LegacyActivationManagerAdapter(mock_legacy_memory, memory_adapter)
        
        # Get most active memories
        most_active = adapter.get_most_active(2)
        
        # Verify most active memories
        assert len(most_active) == 2
        assert most_active[0][0] == "memory1"  # Highest activation (0.8)
        assert most_active[0][1] == 0.8
        assert most_active[1][0] == "memory0"  # Second highest (0.5)
        assert most_active[1][1] == 0.5