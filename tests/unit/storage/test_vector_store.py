"""
Tests for the VectorStore components.
"""

import numpy as np
import pytest

from memoryweave.storage.vector_store import SimpleVectorStore, ActivationVectorStore


class TestSimpleVectorStore:
    """Test suite for the SimpleVectorStore class."""

    def test_init(self):
        """Test initialization of vector store."""
        store = SimpleVectorStore()
        assert hasattr(store, "_vectors")
        assert len(store._vectors) == 0

    def test_add_vector(self):
        """Test adding a vector to the store."""
        store = SimpleVectorStore()
        vector = np.array([0.1, 0.2, 0.3])
        
        # Add vector
        store.add("memory1", vector)
        
        # Verify vector was added
        assert "memory1" in store._vectors
        assert np.array_equal(store._vectors["memory1"], vector)
        assert store._dirty  # Index should be marked as dirty

    def test_search_basic(self):
        """Test basic search functionality."""
        store = SimpleVectorStore()
        
        # Add vectors
        store.add("memory1", np.array([1.0, 0.0, 0.0]))
        store.add("memory2", np.array([0.0, 1.0, 0.0]))
        store.add("memory3", np.array([0.0, 0.0, 1.0]))
        
        # Search for similar vector
        query = np.array([0.9, 0.1, 0.0])
        results = store.search(query, k=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0][0] == "memory1"  # Most similar should be memory1
        assert results[0][1] > 0.9  # Similarity should be high
        assert results[1][0] == "memory2"  # Second most similar is memory2

    def test_search_with_threshold(self):
        """Test search with similarity threshold."""
        store = SimpleVectorStore()
        
        # Add vectors
        store.add("memory1", np.array([1.0, 0.0, 0.0]))
        store.add("memory2", np.array([0.0, 1.0, 0.0]))
        store.add("memory3", np.array([0.0, 0.0, 1.0]))
        
        # Search with high threshold
        query = np.array([0.7, 0.7, 0.0])
        results = store.search(query, k=3, threshold=0.8)
        
        # Verify results - none should meet threshold
        assert len(results) == 0
        
        # Search with lower threshold
        results = store.search(query, k=3, threshold=0.6)
        
        # Verify results - memory1 and memory2 should meet threshold
        assert len(results) == 2

    def test_remove_vector(self):
        """Test removing a vector from the store."""
        store = SimpleVectorStore()
        
        # Add vectors
        store.add("memory1", np.array([1.0, 0.0, 0.0]))
        store.add("memory2", np.array([0.0, 1.0, 0.0]))
        
        # Verify vectors were added
        assert "memory1" in store._vectors
        assert "memory2" in store._vectors
        
        # Remove vector
        store.remove("memory1")
        
        # Verify vector was removed
        assert "memory1" not in store._vectors
        assert "memory2" in store._vectors
        assert store._dirty  # Index should be marked as dirty

    def test_clear(self):
        """Test clearing all vectors from the store."""
        store = SimpleVectorStore()
        
        # Add vectors
        store.add("memory1", np.array([1.0, 0.0, 0.0]))
        store.add("memory2", np.array([0.0, 1.0, 0.0]))
        
        # Verify vectors were added
        assert len(store._vectors) == 2
        
        # Clear store
        store.clear()
        
        # Verify store is empty
        assert len(store._vectors) == 0
        assert store._dirty  # Index should be marked as dirty

    def test_index_building(self):
        """Test that the search index is built correctly."""
        store = SimpleVectorStore()
        
        # Add vectors
        store.add("memory1", np.array([1.0, 0.0, 0.0]))
        store.add("memory2", np.array([0.0, 1.0, 0.0]))
        
        # Force index building
        store._build_index()
        
        # Verify index was built
        assert not store._dirty
        assert store._matrix is not None
        assert store._matrix.shape == (2, 3)
        assert len(store._id_to_index) == 2
        assert len(store._index_to_id) == 2


class TestActivationVectorStore:
    """Test suite for the ActivationVectorStore class."""

    def test_init(self):
        """Test initialization of activation vector store."""
        store = ActivationVectorStore(activation_weight=0.3)
        assert hasattr(store, "_vector_store")
        assert hasattr(store, "_activations")
        assert store._activation_weight == 0.3

    def test_add_vector(self):
        """Test adding a vector to the store."""
        store = ActivationVectorStore()
        vector = np.array([0.1, 0.2, 0.3])
        
        # Add vector
        store.add("memory1", vector)
        
        # Verify vector was added
        assert "memory1" in store._vector_store._vectors
        assert "memory1" in store._activations
        assert store._activations["memory1"] == 0.0  # Initial activation

    def test_search_with_activation_boost(self):
        """Test search with activation boost."""
        store = ActivationVectorStore(activation_weight=0.5)
        
        # Add vectors with different activations
        store.add("memory1", np.array([0.8, 0.0, 0.0]))  # Lower similarity
        store.add("memory2", np.array([0.7, 0.0, 0.0]))  # Even lower similarity
        
        # Update activations
        store._activations["memory1"] = 0.2  # Low activation
        store._activations["memory2"] = 1.0  # High activation
        
        # Search without activation boost (activation_weight=0)
        store._activation_weight = 0.0
        query = np.array([1.0, 0.0, 0.0])
        results_no_boost = store.search(query, k=2)
        
        # Verify results - should be ordered by similarity only
        assert results_no_boost[0][0] == "memory1"  # memory1 has higher similarity
        
        # Search with activation boost
        store._activation_weight = 0.5
        results_with_boost = store.search(query, k=2)
        
        # Verify results - memory2 should be boosted to first position
        assert results_with_boost[0][0] == "memory2"  # memory2 boosted by activation
        assert results_with_boost[1][0] == "memory1"

    def test_update_activation(self):
        """Test updating activation level."""
        store = ActivationVectorStore()
        
        # Add vector
        store.add("memory1", np.array([0.1, 0.2, 0.3]))
        
        # Verify initial activation
        assert store._activations["memory1"] == 0.0
        
        # Update activation
        store.update_activation("memory1", 0.5)
        
        # Verify updated activation
        assert store._activations["memory1"] == 0.5
        
        # Update again
        store.update_activation("memory1", 0.3)
        
        # Verify cumulative update
        assert store._activations["memory1"] == 0.8
        
        # Test non-existent memory
        with pytest.raises(KeyError):
            store.update_activation("non_existent", 0.5)

    def test_remove_vector(self):
        """Test removing a vector from the store."""
        store = ActivationVectorStore()
        
        # Add vectors
        store.add("memory1", np.array([1.0, 0.0, 0.0]))
        store.add("memory2", np.array([0.0, 1.0, 0.0]))
        
        # Verify vectors were added
        assert "memory1" in store._vector_store._vectors
        assert "memory1" in store._activations
        
        # Remove vector
        store.remove("memory1")
        
        # Verify vector was removed
        assert "memory1" not in store._vector_store._vectors
        assert "memory1" not in store._activations

    def test_clear(self):
        """Test clearing all vectors from the store."""
        store = ActivationVectorStore()
        
        # Add vectors
        store.add("memory1", np.array([1.0, 0.0, 0.0]))
        store.add("memory2", np.array([0.0, 1.0, 0.0]))
        
        # Verify vectors were added
        assert len(store._vector_store._vectors) == 2
        assert len(store._activations) == 2
        
        # Clear store
        store.clear()
        
        # Verify store is empty
        assert len(store._vector_store._vectors) == 0
        assert len(store._activations) == 0