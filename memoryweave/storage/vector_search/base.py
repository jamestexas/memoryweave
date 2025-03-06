"""Base interface for vector search providers."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class IVectorSearchProvider(ABC):
    """Interface for vector search providers.

    This interface defines the common functionality that all vector search
    providers must implement, regardless of the underlying technology.
    """

    @abstractmethod
    def index(self, vectors: np.ndarray, ids: list[Any]) -> None:
        """
        Index vectors with associated IDs.

        Args:
            vectors: Matrix of vectors to index (each row is a vector)
            ids: List of IDs corresponding to each vector
        """
        pass

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, k: int, threshold: Optional[float] = None
    ) -> list[tuple[Any, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return
            threshold: Optional similarity threshold

        Returns:
            List of (id, similarity_score) tuples
        """
        pass

    @abstractmethod
    def update(self, vector_id: Any, vector: np.ndarray) -> None:
        """
        Update a vector in the index.

        Args:
            vector_id: ID of the vector to update
            vector: New vector
        """
        pass

    @abstractmethod
    def delete(self, vector_id: Any) -> None:
        """
        Delete a vector from the index.

        Args:
            vector_id: ID of the vector to delete
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the index."""
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary of statistics
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimension of vectors in the index."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Get the number of vectors in the index."""
        pass
