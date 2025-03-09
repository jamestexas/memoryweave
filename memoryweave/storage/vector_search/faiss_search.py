"""FAISS-based vector search implementation for efficient similarity search."""

import logging
from importlib.util import find_spec
from typing import Any

import numpy as np

from memoryweave.storage.vector_search.base import IVectorSearchProvider

logger = logging.getLogger(__name__)

# Check if FAISS is available
if find_spec("faiss") is not None:
    import faiss

    FAISS_AVAILABLE = True
else:
    FAISS_AVAILABLE = False
    logger.warning(
        "[bold red]FAISS is not installed. Install with: pip install faiss-cpu (or faiss-gpu)[/]",
    )


class FaissVectorSearch(IVectorSearchProvider):
    """
    FAISS-based vector search implementation for efficient similarity search.

    This implementation uses Facebook AI Similarity Search (FAISS) for
    high-performance vector similarity search, supporting both exact and
    approximate nearest neighbor algorithms.
    """

    def __init__(
        self,
        dimension: int = 768,
        metric: str = "cosine",
        index_type: str = "Flat",
        nprobe: int = 8,
        use_quantization: bool = False,
        build_threshold: int = 50,
        **kwargs,
    ):
        """
        Initialize the FAISS vector search.

        Args:
            dimension: Dimension of vectors
            metric: Similarity metric ("cosine", "l2", or "ip")
            index_type: FAISS index type ("Flat", "IVF", "IVFPQ", "HNSW")
            nprobe: Number of clusters to probe (for IVF indices)
            use_quantization: Whether to use scalar quantization
            build_threshold: Minimum number of vectors before building index
            **kwargs: Additional arguments
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu (or faiss-gpu)"
            )

        self._dimension = dimension
        self._metric = metric
        self._index_type = index_type
        self._nprobe = nprobe
        self._use_quantization = use_quantization
        self._build_threshold = build_threshold

        # Parse IVF params if using IVF indices
        self._nlist = 100  # Default
        if "IVF" in index_type:
            ivf_parts = index_type.split(",")
            if len(ivf_parts) > 0 and "IVF" in ivf_parts[0]:
                try:
                    # Extract number after IVF (e.g., IVF100 -> 100)
                    self._nlist = int("".join(filter(str.isdigit, ivf_parts[0])))
                except ValueError:
                    self._nlist = 100  # Default if parsing fails

        # Internal state
        self._index = None
        self._ids = []
        self._id_to_idx = {}
        self._is_trained = False
        self._dirty = True
        self._stats = {
            "size": 0,
            "last_build_time": 0,
            "last_search_time": 0,
        }

    def index(self, vectors: np.ndarray, ids: list[Any]) -> None:
        """
        Index vectors with associated IDs.

        Args:
            vectors: Matrix of vectors to index (each row is a vector)
            ids: list of IDs corresponding to each vector
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")

        if len(vectors) == 0:
            return

        # Store IDs and create mapping
        self._ids = list(ids)
        self._id_to_idx = {id_val: i for i, id_val in enumerate(ids)}

        # Convert to float32 for FAISS
        vectors_f32 = vectors.astype(np.float32)

        # Normalize vectors for cosine similarity if needed
        if self._metric == "cosine":
            faiss.normalize_L2(vectors_f32)

        # Determine if we need to build the index
        needs_index_build = self._index is None or (
            self._dirty and len(vectors) >= self._build_threshold
        )

        if needs_index_build:
            self._build_index(vectors_f32)
        else:
            # Add to existing index
            if hasattr(self._index, "add"):
                # Direct add for standard indices
                self._index.add(vectors_f32)
            elif hasattr(self._index, "add_with_ids"):
                # Add with IDs for ID mapping indices
                int_ids = np.arange(len(ids), dtype=np.int64)
                self._index.add_with_ids(vectors_f32, int_ids)
            else:
                # Fallback - rebuild the index
                self._build_index(vectors_f32)

        self._dirty = False
        self._stats["size"] = len(self._ids)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        threshold: float | None = None,
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
        if self._index is None or len(self._ids) == 0:
            return []

        # Ensure vector is the right shape and type
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        # Normalize for cosine similarity if needed
        if self._metric == "cosine":
            faiss.normalize_L2(query_vector)

        # Perform search
        import time

        start_time = time.time()

        # Get more results than needed if we'll apply threshold filtering
        search_k = k * 2 if threshold is not None else k
        search_k = min(search_k, len(self._ids))

        # Fixed k to at least 1
        search_k = max(1, search_k)

        # Search the index
        distances, indices = self._index.search(query_vector, search_k)
        self._stats["last_search_time"] = time.time() - start_time

        # Flatten results
        distances = distances[0]
        indices = indices[0]

        # Convert distances to similarities based on metric
        if self._metric == "l2":
            # For L2, smaller is better, so convert to similarity score
            # Use exponential decay: similarity = exp(-distance)
            similarities = np.exp(-distances)
        else:
            # For IP and cosine, larger is better
            similarities = distances

        # Apply threshold if provided
        results = []
        for idx, similarity in zip(indices, similarities):
            # Ignore padding indices (-1)
            if idx == -1:
                continue

            # Apply threshold
            if threshold is not None and similarity < threshold:
                continue

            # Get original ID
            original_id = self._ids[idx]

            results.append((original_id, float(similarity)))

        # Sort by similarity and limit to k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def update(self, vector_id: Any, vector: np.ndarray) -> None:
        """
        Update a vector in the index.

        Note: FAISS doesn't support efficient updates, so this rebuilds the index.
        For frequent updates, consider batching them.

        Args:
            vector_id: ID of the vector to update
            vector: New vector
        """
        if vector_id not in self._id_to_idx:
            raise KeyError(f"Vector ID {vector_id} not found")

        # Get vectors
        if not hasattr(self._index, "reconstruct"):
            # If we can't reconstruct vectors, we need to rebuild the index
            logger.warning("FAISS index doesn't support reconstruction, rebuilding index")
            self._dirty = True
            # We'd need all vectors to rebuild, which we don't have
            raise NotImplementedError("Update operation not supported for this index type")

        # Get all vectors
        idx = self._id_to_idx[vector_id]
        vectors = np.zeros((len(self._ids), self._dimension), dtype=np.float32)

        # Reconstruct all vectors
        for i in range(len(self._ids)):
            vectors[i] = self._index.reconstruct(i)

        # Update the vector
        vectors[idx] = vector

        # Rebuild the index
        self._build_index(vectors)

    def delete(self, vector_id: Any) -> None:
        """
        Delete a vector from the index.

        Note: FAISS requires special index types for deletion. For basic indices,
        this rebuilds the entire index.

        Args:
            vector_id: ID of the vector to delete
        """
        if vector_id not in self._id_to_idx:
            raise KeyError(f"Vector ID {vector_id} not found")

        idx = self._id_to_idx[vector_id]

        # Check if index supports removal
        if hasattr(self._index, "remove_ids"):
            # For indices that support removal
            id_array = np.array([idx], dtype=np.int64)
            self._index.remove_ids(id_array)
        else:
            # Otherwise, we need to rebuild the index
            logger.warning("FAISS index doesn't support removal, rebuilding index")

            # Get all vectors except the one to delete
            vectors = []
            ids = []

            # Check if we can reconstruct vectors
            if hasattr(self._index, "reconstruct"):
                for i, id_val in enumerate(self._ids):
                    if i != idx:
                        vector = self._index.reconstruct(i)
                        vectors.append(vector)
                        ids.append(id_val)
            else:
                # Can't reconstruct or remove, so we can't delete
                raise NotImplementedError("Delete operation not supported for this index type")

            # Rebuild the index with remaining vectors
            if vectors:
                vectors_array = np.stack(vectors)
                self._ids = ids
                self._id_to_idx = {id_val: i for i, id_val in enumerate(ids)}
                self._build_index(vectors_array)
            else:
                # No vectors left
                self.clear()

    def clear(self) -> None:
        """Clear the index."""
        self._index = None
        self._ids = []
        self._id_to_idx = {}
        self._is_trained = False
        self._dirty = True
        self._stats["size"] = 0

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            dictionary of statistics
        """
        stats = {
            "type": "faiss",
            "index_type": self._index_type,
            "metric": self._metric,
            "dimension": self._dimension,
            "size": len(self._ids),
            "is_trained": self._is_trained,
            "last_build_time": self._stats.get("last_build_time", 0),
            "last_search_time": self._stats.get("last_search_time", 0),
        }

        # Add FAISS-specific info if available
        if self._index is not None:
            stats["faiss_type"] = type(self._index).__name__

        return stats

    @property
    def dimension(self) -> int:
        """Get the dimension of vectors in the index."""
        return self._dimension

    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        return len(self._ids)

    def _build_index(self, vectors: np.ndarray) -> None:
        """
        Build the FAISS index for the given vectors.

        Args:
            vectors: Matrix of vectors (float32)
        """
        import time

        start_time = time.time()

        if len(vectors) == 0:
            self._index = None
            self._is_trained = False
            return

        # Determine index type to use
        index = self._create_index(vectors)

        # Train if needed
        if "IVF" in self._index_type and not self._is_trained:
            try:
                # IVF indices need training
                index.train(vectors)
                self._is_trained = True
            except Exception as e:
                logger.error(f"Error training FAISS index: {e}")
                # Fall back to flat index
                index = faiss.IndexFlatL2(self._dimension)
                self._index_type = "Flat"
                self._is_trained = False

        # Add vectors to index
        index.add(vectors)

        # Wrap with ID mapping if needed
        if not isinstance(index, faiss.IndexIDMap) and not isinstance(index, faiss.IndexIDMap2):
            index = faiss.IndexIDMap(index)

        # Set nprobe for IVF indices
        if hasattr(index, "nprobe"):
            index.nprobe = self._nprobe

        self._index = index
        self._stats["last_build_time"] = time.time() - start_time

    def _create_index(self, vectors: np.ndarray) -> Any:
        """
        Create the appropriate FAISS index based on configuration.

        Args:
            vectors: Matrix of vectors to index

        Returns:
            FAISS index
        """
        # Determine the base index
        if self._metric == "cosine" or self._metric == "ip":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:  # l2
            metric_type = faiss.METRIC_L2

        # Create appropriate index
        if self._index_type == "Flat":
            # Exact nearest neighbor search
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                return faiss.IndexFlatIP(self._dimension)
            else:
                return faiss.IndexFlatL2(self._dimension)

        elif self._index_type.startswith("IVF"):
            # Parse index type (e.g., "IVF100,Flat" or "IVF100,PQ16")
            index_parts = self._index_type.split(",")

            # Create base quantizer
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                quantizer = faiss.IndexFlatIP(self._dimension)
            else:
                quantizer = faiss.IndexFlatL2(self._dimension)

            # Determine subindex type
            if len(index_parts) > 1 and "PQ" in index_parts[1]:
                # For IVF+PQ (product quantization for memory efficiency)
                try:
                    # Extract the number after PQ (e.g., PQ16 -> 16)
                    m = int("".join(filter(str.isdigit, index_parts[1])))
                    # Create IVF-PQ index
                    return faiss.IndexIVFPQ(
                        quantizer, self._dimension, self._nlist, m, 8, metric_type
                    )
                except ValueError:
                    # Default to IVF with 8 subquantizers
                    return faiss.IndexIVFPQ(
                        quantizer, self._dimension, self._nlist, 8, 8, metric_type
                    )
            else:
                # Standard IVF with flat storage
                return faiss.IndexIVFFlat(quantizer, self._dimension, self._nlist, metric_type)

        elif self._index_type.startswith("HNSW"):
            # Hierarchical Navigable Small World graphs
            try:
                # Extract M parameter (e.g., HNSW32 -> 32)
                _M = int("".join(filter(str.isdigit, self._index_type)))  # noqa: N806
            except ValueError:
                _M = 32  # Default M # noqa: N806

            # Create HNSW index
            return faiss.IndexHNSWFlat(self._dimension, _M, metric_type)

        else:
            # Default to flat index for unknown types
            logger.warning(f"Unknown index type: {self._index_type}, using Flat index")
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                return faiss.IndexFlatIP(self._dimension)
            else:
                return faiss.IndexFlatL2(self._dimension)
