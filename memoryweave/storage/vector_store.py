"""Vector store implementation for MemoryWeave.

This module provides implementations for vector storage and similarity search,
optimized for memory retrieval operations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, IVectorStore, MemoryID


class SimpleVectorStore(IVectorStore):
    """Simple in-memory vector store implementation using numpy.

    This implementation is suitable for small to medium-sized memory sets.
    For larger memory sets, consider using an optimized vector database.
    """

    def __init__(self):
        """Initialize the vector store."""
        self._vectors: Dict[MemoryID, EmbeddingVector] = {}
        self._id_to_index: Dict[MemoryID, int] = {}
        self._index_to_id: Dict[int, MemoryID] = {}
        self._dirty: bool = True
        self._matrix: Optional[np.ndarray] = None
        self.component_id = "simple_vector_store"
        
    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        return self.component_id
        
    def get_type(self):
        """Get the type of this component."""
        from memoryweave.interfaces.pipeline import ComponentType
        return ComponentType.VECTOR_STORE

    def add(self, id: MemoryID, vector: EmbeddingVector) -> None:
        """Add a vector to the store."""
        self._vectors[id] = vector
        self._dirty = True

    def search(
        self, query_vector: EmbeddingVector, k: int, threshold: Optional[float] = None
    ) -> List[Tuple[MemoryID, float]]:
        """Search for similar vectors."""
        if not self._vectors:
            return []

        # Build the search index if necessary
        if self._dirty:
            self._build_index()

        # Compute similarities
        similarities = self._compute_similarities(query_vector)

        # Sort by similarity (descending)
        indices = np.argsort(-similarities)

        # Filter by threshold if provided
        if threshold is not None:
            indices = indices[similarities[indices] >= threshold]

        # Limit to k results
        indices = indices[:k]

        # Convert indices to IDs and scores
        results = []
        for idx in indices:
            memory_id = self._index_to_id[idx]
            score = float(similarities[idx])
            results.append((memory_id, score))

        return results

    def remove(self, id: MemoryID) -> None:
        """Remove a vector from the store."""
        if id in self._vectors:
            del self._vectors[id]
            self._dirty = True

    def clear(self) -> None:
        """Clear all vectors from the store."""
        self._vectors.clear()
        self._id_to_index.clear()
        self._index_to_id.clear()
        self._dirty = True
        self._matrix = None

    def _build_index(self) -> None:
        """Build the search index."""
        # Map IDs to indices
        self._id_to_index = {id: i for i, id in enumerate(self._vectors.keys())}
        self._index_to_id = {i: id for id, i in self._id_to_index.items()}

        # Build the matrix
        if not self._vectors:
            self._matrix = np.array([])
        else:
            self._matrix = np.stack([self._vectors[id] for id in self._id_to_index.keys()])

        # Normalize vectors for cosine similarity
        if self._matrix.size > 0:
            norm = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            # Avoid division by zero
            norm[norm == 0] = 1e-10
            self._matrix = self._matrix / norm

        self._dirty = False

    def _compute_similarities(self, query_vector: EmbeddingVector) -> np.ndarray:
        """Compute similarities between query vector and all vectors."""
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_vector / query_norm

        # Compute dot products (cosine similarities)
        if self._matrix is not None and self._matrix.size > 0:
            return np.dot(self._matrix, normalized_query)
        else:
            return np.array([])


class ActivationVectorStore(IVectorStore):
    """Vector store that combines similarity with activation levels.

    This implementation enhances similarity search with activation levels,
    making recently accessed or important memories more likely to be retrieved.
    """

    def __init__(self, activation_weight: float = 0.2):
        """Initialize the vector store.

        Args:
            activation_weight: Weight of activation in final similarity score (0-1)
        """
        self._vector_store = SimpleVectorStore()
        self._activations: Dict[MemoryID, float] = {}
        self._activation_weight = activation_weight
        self.component_id = "activation_vector_store"
        
    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        return self.component_id
        
    def get_type(self):
        """Get the type of this component."""
        from memoryweave.interfaces.pipeline import ComponentType
        return ComponentType.VECTOR_STORE

    def add(self, id: MemoryID, vector: EmbeddingVector) -> None:
        """Add a vector to the store."""
        self._vector_store.add(id, vector)
        self._activations[id] = 0.0

    def search(
        self, query_vector: EmbeddingVector, k: int, threshold: Optional[float] = None
    ) -> List[Tuple[MemoryID, float]]:
        """Search for similar vectors with activation boost."""
        # Get similarity results
        similarity_results = self._vector_store.search(
            query_vector, len(self._activations), threshold
        )

        # Apply activation boost
        boosted_results = []
        for memory_id, similarity in similarity_results:
            activation = self._activations.get(memory_id, 0.0)
            # Normalize activation to 0-1 scale
            max_activation = max(self._activations.values()) if self._activations else 1.0
            if max_activation == 0:
                max_activation = 1.0
            normalized_activation = activation / max_activation

            # Combine similarity and activation
            combined_score = (
                1 - self._activation_weight
            ) * similarity + self._activation_weight * normalized_activation

            boosted_results.append((memory_id, combined_score))

        # Sort by combined score and take top k
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:k]

    def remove(self, id: MemoryID) -> None:
        """Remove a vector from the store."""
        self._vector_store.remove(id)
        if id in self._activations:
            del self._activations[id]

    def clear(self) -> None:
        """Clear all vectors from the store."""
        self._vector_store.clear()
        self._activations.clear()

    def update_activation(self, id: MemoryID, activation_delta: float) -> None:
        """Update activation level for a memory."""
        if id not in self._activations:
            raise KeyError(f"Memory with ID {id} not found")

        self._activations[id] += activation_delta
