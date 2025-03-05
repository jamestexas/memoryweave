"""NumPy-based vector search implementation."""

from typing import Any

import numpy as np

from memoryweave.storage.vector_search.base import IVectorSearchProvider


class NumpyVectorSearch(IVectorSearchProvider):
    """Simple vector search implementation using NumPy.

    This is a reference implementation that provides exact search using
    cosine similarity computed with NumPy. It's suitable for small to
    medium-sized collections but will not scale well to very large datasets.
    """

    def __init__(self, dimension: int = 768, metric: str = "cosine", **kwargs):
        """
        Initialize the NumPy vector search.

        Args:
            dimension: Dimension of vectors
            metric: Similarity metric ("cosine", "dot", or "l2")
            **kwargs: Additional arguments (ignored)
        """
        self._dimension = dimension
        self._metric = metric
        self._vectors = None
        self._ids = []
        self._id_to_index = {}
        self._dirty = True

    def index(self, vectors: np.ndarray, ids: list[Any]) -> None:
        """
        Index vectors with associated IDs.

        Args:
            vectors: Matrix of vectors to index (each row is a vector)
            ids: list of IDs corresponding to each vector
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")

        self._vectors = vectors.copy()
        self._ids = list(ids)
        self._id_to_index = {id_val: i for i, id_val in enumerate(ids)}

        # Normalize vectors for cosine similarity
        if self._metric == "cosine":
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1e-10
            self._vectors = self._vectors / norms

        self._dirty = False

    def search(
        self, query_vector: np.ndarray, k: int, threshold: float | None = None
    ) -> list[tuple[Any, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return
            threshold: Optional similarity threshold

        Returns:
            list of (id, similarity_score) tuples
        """
        if self._vectors is None or len(self._vectors) == 0:
            return []

        # Normalize query vector
        if self._metric == "cosine":
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                query_norm = 1e-10
            query_vector = query_vector / query_norm

        # Compute similarities based on metric
        if self._metric == "cosine" or self._metric == "dot":
            similarities = np.dot(self._vectors, query_vector)
        elif self._metric == "l2":
            distances = np.linalg.norm(self._vectors - query_vector, axis=1)
            # Convert distances to similarities (1 / (1 + distance))
            similarities = 1 / (1 + distances)
        else:
            raise ValueError(f"Unsupported metric: {self._metric}")

        # Filter by threshold if provided
        if threshold is not None:
            valid_indices = np.where(similarities >= threshold)[0]
            if len(valid_indices) == 0:
                return []
            top_indices = valid_indices[np.argsort(-similarities[valid_indices])[:k]]
        else:
            # Get top k indices
            top_indices = np.argsort(-similarities)[:k]

        # Return (id, score) pairs
        return [(self._ids[i], float(similarities[i])) for i in top_indices]

    def update(self, vector_id: Any, vector: np.ndarray) -> None:
        """
        Update a vector in the index.

        Args:
            vector_id: ID of the vector to update
            vector: New vector
        """
        if vector_id not in self._id_to_index:
            raise KeyError(f"Vector ID {vector_id} not found")

        idx = self._id_to_index[vector_id]

        # Update vector
        if self._metric == "cosine":
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm == 0:
                norm = 1e-10
            normalized_vector = vector / norm
            self._vectors[idx] = normalized_vector
        else:
            self._vectors[idx] = vector

    def delete(self, vector_id: Any) -> None:
        """
        Delete a vector from the index.

        Args:
            vector_id: ID of the vector to delete
        """
        if vector_id not in self._id_to_index:
            raise KeyError(f"Vector ID {vector_id} not found")

        idx = self._id_to_index[vector_id]

        # Remove vector and update indices
        self._vectors = np.delete(self._vectors, idx, axis=0)
        del self._ids[idx]

        # Update id_to_index mapping
        self._id_to_index = {id_val: i for i, id_val in enumerate(self._ids)}

    def clear(self) -> None:
        """Clear the index."""
        self._vectors = None
        self._ids = []
        self._id_to_index = {}
        self._dirty = True

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            dictionary of statistics
        """
        return {
            "type": "numpy",
            "metric": self._metric,
            "dimension": self._dimension,
            "size": len(self._ids) if self._ids else 0,
            "memory_usage_mb": (
                self._vectors.nbytes / (1024 * 1024) if self._vectors is not None else 0
            ),
        }

    @property
    def dimension(self) -> int:
        """Get the dimension of vectors in the index."""
        return self._dimension

    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        return len(self._ids) if self._ids else 0
