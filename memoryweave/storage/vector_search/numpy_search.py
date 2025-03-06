"""NumPy-based vector search implementation."""

import time
from typing import Any

import numpy as np

from memoryweave.storage.vector_search.base import IVectorSearchProvider


class NumpyVectorSearch(IVectorSearchProvider):
    """
    Optimized vector search with caching and batch processing.

    This implementation enhances the basic NumpyVectorSearch with:
    1. Query caching to avoid redundant calculations
    2. Batch processing for large vector sets
    3. Pre-normalization for faster similarity computation
    4. Memory-efficient processing
    """

    def __init__(
        self,
        dimension: int = 768,
        metric: str = "cosine",
        batch_size: int = 1000,
        enable_cache: bool = True,
        **kwargs,
    ):
        """
        Initialize the optimized vector search.

        Args:
            dimension: Dimension of vectors
            metric: Similarity metric ("cosine", "dot", or "l2")
            batch_size: Size of batches for processing large datasets
            enable_cache: Whether to enable query caching
            **kwargs: Additional arguments (ignored)
        """
        self._dimension = dimension
        self._metric = metric
        self._batch_size = batch_size
        self._enable_cache = enable_cache

        # Index storage
        self._vectors = None
        self._normalized_vectors = None  # Cache for normalized vectors
        self._ids = []
        self._id_to_index = {}

        # State tracking
        self._dirty = True
        self._last_update_time = time.time()

        # Query cache
        self._query_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 20  # Store last 20 query results

        # Statistics
        self._stats = {
            "last_query_time": 0,
            "average_query_time": 0,
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def index(self, vectors: np.ndarray, ids: list[Any]) -> None:
        """
        Index vectors with associated IDs and pre-normalize.

        Args:
            vectors: Matrix of vectors to index (each row is a vector)
            ids: List of IDs corresponding to each vector
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")

        self._vectors = vectors.copy()
        self._ids = list(ids)
        self._id_to_index = {id_val: i for i, id_val in enumerate(ids)}

        # Pre-normalize vectors for cosine similarity
        if self._metric == "cosine":
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1e-10
            self._normalized_vectors = self._vectors / norms
        else:
            self._normalized_vectors = self._vectors

        # Clear cache on reindexing
        self._query_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._dirty = False
        self._last_update_time = time.time()

    def search(
        self, query_vector: np.ndarray, k: int, threshold: float | None = None
    ) -> list[tuple[Any, float]]:
        """
        Search for similar vectors with optimized processing.

        Args:
            query_vector: Query vector
            k: Number of results to return
            threshold: Optional similarity threshold

        Returns:
            List of (id, similarity_score) tuples
        """
        start_time = time.time()

        if self._vectors is None or len(self._vectors) == 0:
            return []

        # Generate cache key for this query if caching is enabled
        if self._enable_cache:
            # Use hash of query features rather than full vector to avoid memory issues
            # Important query parameters that affect results
            query_hash = hash(tuple(np.round(query_vector[:5], 2)))  # Hash first 5 dims rounded
            query_key = (query_hash, k, threshold)

            # Check cache
            if query_key in self._query_cache:
                self._cache_hits += 1
                self._stats["cache_hits"] += 1

                # Log query time
                query_time = time.time() - start_time
                self._stats["last_query_time"] = query_time

                return self._query_cache[query_key]

            self._cache_misses += 1
            self._stats["cache_misses"] += 1

        # Normalize query vector for cosine similarity
        if self._metric == "cosine":
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                query_norm = 1e-10
            query_vector = query_vector / query_norm

        # If dataset is small, do direct calculation
        if len(self._normalized_vectors) <= self._batch_size:
            if self._metric == "cosine" or self._metric == "dot":
                similarities = np.dot(self._normalized_vectors, query_vector)
            elif self._metric == "l2":
                distances = np.linalg.norm(self._normalized_vectors - query_vector, axis=1)
                # Convert distances to similarities (1 / (1 + distance))
                similarities = 1 / (1 + distances)
            else:
                raise ValueError(f"Unsupported metric: {self._metric}")

            # Apply threshold if provided
            if threshold is not None:
                valid_indices = np.where(similarities >= threshold)[0]
                if len(valid_indices) == 0:
                    # Log query time
                    query_time = time.time() - start_time
                    self._update_stats(query_time)
                    return []
                top_indices = valid_indices[np.argsort(-similarities[valid_indices])[:k]]
            else:
                # Get top k indices
                top_indices = np.argsort(-similarities)[:k]

            # Format results
            results = [(self._ids[i], float(similarities[i])) for i in top_indices]

            # Cache results if enabled
            if self._enable_cache:
                # Manage cache size
                if len(self._query_cache) >= self._max_cache_size:
                    # Remove oldest entry (first key)
                    self._query_cache.pop(next(iter(self._query_cache)))
                self._query_cache[query_key] = results

            # Log query time
            query_time = time.time() - start_time
            self._update_stats(query_time)

            return results

        # For larger datasets, process in batches
        top_indices = []
        top_scores = []

        # Process the dataset in batches to reduce memory usage
        for i in range(0, len(self._normalized_vectors), self._batch_size):
            end_idx = min(i + self._batch_size, len(self._normalized_vectors))
            batch = self._normalized_vectors[i:end_idx]

            # Calculate similarities for this batch
            if self._metric == "cosine" or self._metric == "dot":
                batch_similarities = np.dot(batch, query_vector)
            elif self._metric == "l2":
                batch_distances = np.linalg.norm(batch - query_vector, axis=1)
                batch_similarities = 1 / (1 + batch_distances)

            # Apply threshold filtering
            if threshold is not None:
                valid_batch_indices = np.where(batch_similarities >= threshold)[0]
                if len(valid_batch_indices) == 0:
                    continue

                # Adjust indices to global scale
                valid_indices = valid_batch_indices + i
                valid_scores = batch_similarities[valid_batch_indices]
            else:
                valid_indices = np.arange(i, end_idx)
                valid_scores = batch_similarities

            # Collect all candidates
            top_indices.extend(valid_indices)
            top_scores.extend(valid_scores)

        # Ensure we have results
        if not top_indices:
            # Log query time
            query_time = time.time() - start_time
            self._update_stats(query_time)
            return []

        # Convert to arrays for efficient operations
        top_indices = np.array(top_indices)
        top_scores = np.array(top_scores)

        # Get top k overall
        if len(top_indices) > k:
            top_k_indices = np.argsort(-top_scores)[:k]
            final_indices = top_indices[top_k_indices]
            final_scores = top_scores[top_k_indices]
        else:
            final_indices = top_indices
            final_scores = top_scores

        # Format results
        results = [(self._ids[int(i)], float(s)) for i, s in zip(final_indices, final_scores)]

        # Cache results if enabled
        if self._enable_cache:
            # Manage cache size
            if len(self._query_cache) >= self._max_cache_size:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query_key] = results

        # Log query time
        query_time = time.time() - start_time
        self._update_stats(query_time)

        return results

    def _update_stats(self, query_time: float) -> None:
        """Update search statistics."""
        self._stats["last_query_time"] = query_time
        self._stats["total_queries"] += 1

        # Update average query time
        n = self._stats["total_queries"]
        prev_avg = self._stats.get("average_query_time", 0)
        self._stats["average_query_time"] = ((n - 1) * prev_avg + query_time) / n

        # Update cache stats
        self._stats["cache_hits"] = self._cache_hits
        self._stats["cache_misses"] = self._cache_misses

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
        self._vectors[idx] = vector

        # Update normalized vector
        if self._metric == "cosine":
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm == 0:
                norm = 1e-10
            normalized_vector = vector / norm
            self._normalized_vectors[idx] = normalized_vector
        else:
            self._normalized_vectors[idx] = vector

        # Clear cache on update
        self._query_cache = {}
        self._last_update_time = time.time()

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
        if self._normalized_vectors is not None:
            self._normalized_vectors = np.delete(self._normalized_vectors, idx, axis=0)

        del self._ids[idx]

        # Update id_to_index mapping
        self._id_to_index = {id_val: i for i, id_val in enumerate(self._ids)}

        # Clear cache on delete
        self._query_cache = {}
        self._last_update_time = time.time()

    def clear(self) -> None:
        """Clear the index."""
        self._vectors = None
        self._normalized_vectors = None
        self._ids = []
        self._id_to_index = {}
        self._query_cache = {}
        self._dirty = True
        self._last_update_time = time.time()

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "type": "optimized_numpy",
            "metric": self._metric,
            "dimension": self._dimension,
            "size": len(self._ids) if self._ids else 0,
            "memory_usage_mb": (
                (
                    self._vectors.nbytes
                    + (
                        self._normalized_vectors.nbytes
                        if self._normalized_vectors is not None
                        else 0
                    )
                )
                / (1024 * 1024)
                if self._vectors is not None
                else 0
            ),
            "batch_size": self._batch_size,
            "cache_enabled": self._enable_cache,
            "cache_size": len(self._query_cache),
            "last_update_time": self._last_update_time,
        }

        # Add search performance stats
        stats.update(self._stats)

        return stats

    @property
    def dimension(self) -> int:
        """Get the dimension of vectors in the index."""
        return self._dimension

    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        return len(self._ids) if self._ids else 0
