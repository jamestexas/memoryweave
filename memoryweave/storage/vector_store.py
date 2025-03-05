"""Vector store implementation for MemoryWeave.

This module provides implementations for vector storage and similarity search,
optimized for memory retrieval operations. Includes implementations optimized for
scaling to large memory stores using approximate nearest neighbor search.
"""

import time
from typing import Any, Callable, Literal, Optional

import faiss
import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, IVectorStore, MemoryID


class SimpleVectorStore(IVectorStore):
    """Simple in-memory vector store implementation using numpy.

    This implementation is suitable for small to medium-sized memory sets.
    For larger memory sets, consider using an optimized vector database.
    """

    def __init__(self):
        """Initialize the vector store."""
        self._vectors: dict[MemoryID, EmbeddingVector] = {}
        self._id_to_index: dict[MemoryID, int] = {}
        self._index_to_id: dict[int, MemoryID] = {}
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
    ) -> list[tuple[MemoryID, float]]:
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
        self._activations: dict[MemoryID, float] = {}
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
    ) -> list[tuple[MemoryID, float]]:
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


class ANNVectorStore(IVectorStore):
    """Vector store implementation using Approximate Nearest Neighbor search with FAISS.

    This implementation is optimized for large memory stores (500+ memories) and provides
    significantly better performance compared to the SimpleVectorStore for large datasets.
    It uses FAISS for efficient approximate nearest neighbor search.
    """

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "IVF100,Flat",
        metric: str = "cosine",
        nprobe: int = 10,
        build_threshold: int = 50,
        quantize: bool = False,
    ):
        """Initialize the ANN vector store.

        Args:
            dimension: Dimensionality of the embedding vectors
            index_type: FAISS index type (e.g., "Flat", "IVF100,Flat", "IVF100,PQ32")
            metric: Distance metric to use ("cosine", "l2", "ip")
            nprobe: Number of clusters to visit during search (higher = more accurate but slower)
            build_threshold: Minimum number of vectors before building the index
            quantize: Whether to use scalar quantization to reduce memory usage
        """
        self._vectors: dict[MemoryID, EmbeddingVector] = {}
        self._id_to_idx: dict[MemoryID, int] = {}
        self._idx_to_id: dict[int, MemoryID] = {}
        self._faiss_ids = np.array([], dtype=np.int64)

        self._dimension = dimension
        self._index_type = index_type
        self._metric = metric
        self._nprobe = nprobe
        self._build_threshold = build_threshold
        self._quantize = quantize

        self._index = None
        self._initialized = False
        self._dirty = True
        self._count = 0
        self.component_id = "ann_vector_store"

        # Performance tracking
        self._last_build_time = 0
        self._last_search_time = 0

    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        return self.component_id

    def get_type(self):
        """Get the type of this component."""
        from memoryweave.interfaces.pipeline import ComponentType

        return ComponentType.VECTOR_STORE

    def add(self, id: MemoryID, vector: EmbeddingVector) -> None:
        """Add a vector to the store."""
        # Store the vector
        if isinstance(id, str):
            # Convert string IDs to integers for FAISS
            if id not in self._id_to_idx:
                self._id_to_idx[id] = len(self._id_to_idx)
                self._idx_to_id[self._id_to_idx[id]] = id
        else:
            # If id is already an integer, use it directly
            if id not in self._id_to_idx:
                self._id_to_idx[id] = id
                self._idx_to_id[id] = id

        self._vectors[id] = vector
        self._dirty = True
        self._count += 1

        # Rebuild index if we've reached the threshold
        if self._dirty and self._count >= self._build_threshold:
            self._build_index()

    def search(
        self, query_vector: EmbeddingVector, k: int, threshold: Optional[float] = None
    ) -> list[tuple[MemoryID, float]]:
        """Search for similar vectors using approximate nearest neighbor search."""
        if not self._vectors:
            return []

        # If we have fewer vectors than the threshold, use exact search
        if len(self._vectors) < self._build_threshold:
            return self._exact_search(query_vector, k, threshold)

        # Build the index if necessary
        if self._dirty:
            self._build_index()

        # Normalize query vector for cosine similarity if needed
        search_vector = query_vector
        if self._metric == "cosine" or self._metric == "ip":
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                search_vector = query_vector / norm

        # Search the index
        start_time = time.time()
        k_search = min(k * 2, len(self._vectors))  # Get more results for filtering
        D, I = self._index.search(np.array([search_vector]), k_search)  # noqa: E741, N806
        self._last_search_time = time.time() - start_time

        # Convert scores based on the metric
        scores = D[0]
        if self._metric == "cosine" or self._metric == "ip":
            # For inner product, higher is better
            scores = D[0]
        elif self._metric == "l2":
            # For L2 distance, lower is better, so convert to similarity
            max_dist = float(max(D[0])) if len(D[0]) > 0 else 1.0
            if max_dist == 0:
                max_dist = 1.0
            scores = np.array([1.0 - (d / max_dist) for d in D[0]])

        # Filter by threshold if provided
        results = []
        for idx, score in zip(I[0], scores):
            if idx == -1:  # FAISS returns -1 for padded results
                continue

            memory_id = self._idx_to_id.get(int(idx))
            if memory_id is None:
                continue

            if threshold is None or score >= threshold:
                results.append((memory_id, float(score)))

        # Sort by score (descending) and limit to k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def remove(self, id: MemoryID) -> None:
        """Remove a vector from the store."""
        if id in self._vectors:
            del self._vectors[id]
            if id in self._id_to_idx:
                idx = self._id_to_idx[id]
                del self._id_to_idx[id]
                if idx in self._idx_to_id:
                    del self._idx_to_id[idx]
            self._dirty = True
            self._count -= 1

    def clear(self) -> None:
        """Clear all vectors from the store."""
        self._vectors.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._index = None
        self._initialized = False
        self._dirty = True
        self._count = 0

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the vector store."""
        return {
            "vector_count": len(self._vectors),
            "last_build_time": self._last_build_time,
            "last_search_time": self._last_search_time,
            "index_type": self._index_type,
            "metric": self._metric,
            "nprobe": self._nprobe,
            "initialized": self._initialized,
        }

    def progressive_filtering(
        self,
        query_vector: EmbeddingVector,
        initial_k: int = 100,
        final_k: int = 10,
        filter_fn: Optional[
            Callable[[list[tuple[MemoryID, float]]], list[tuple[MemoryID, float]]]
        ] = None,
    ) -> list[tuple[MemoryID, float]]:
        """Perform two-stage retrieval with progressive filtering.

        Args:
            query_vector: The query vector
            initial_k: Number of initial candidates to retrieve
            final_k: Final number of results to return
            filter_fn: Optional function to filter/rerank candidates

        Returns:
            Filtered list of (memory_id, score) tuples
        """
        # First stage: Get initial candidates
        candidates = self.search(query_vector, initial_k, threshold=None)

        if not candidates:
            return []

        # Second stage: Apply filter function if provided
        if filter_fn:
            filtered_results = filter_fn(candidates)
        else:
            filtered_results = candidates

        # Return top k results
        return filtered_results[:final_k]

    def _build_index(self) -> None:
        """Build the FAISS index for fast approximate search."""
        if not self._vectors:
            self._initialized = False
            self._dirty = False
            return

        start_time = time.time()
        vector_dim = next(iter(self._vectors.values())).shape[0]

        # Create ID mapping for original memory IDs
        ids = np.array(list(self._id_to_idx.values()), dtype=np.int64)
        self._faiss_ids = ids

        # Prepare vectors for indexing
        vectors = np.vstack([self._vectors[self._idx_to_id[idx]] for idx in ids])

        # Normalize vectors for cosine similarity if needed
        if self._metric == "cosine" or self._metric == "ip":
            faiss.normalize_L2(vectors)

        # Determine the FAISS index type based on our configuration
        if self._index_type == "Flat":
            if self._metric == "l2":
                index = faiss.IndexFlatL2(vector_dim)
            else:  # cosine or inner product
                index = faiss.IndexFlatIP(vector_dim)
        elif "IVF" in self._index_type:
            # For IVF indexes, we need a training step
            if self._metric == "l2":
                quantizer = faiss.IndexFlatL2(vector_dim)
                base_index = self._index_type.replace("IVF", "")
                nlist = int(base_index.split(",")[0])

                if "PQ" in self._index_type:
                    # Product Quantization for memory efficiency
                    pq_param = base_index.split(",")[1]
                    m = int(pq_param.replace("PQ", ""))
                    index = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, 8)
                else:
                    # IVF with flat storage
                    index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_L2)
            else:  # cosine or inner product
                quantizer = faiss.IndexFlatIP(vector_dim)
                base_index = self._index_type.replace("IVF", "")
                nlist = int(base_index.split(",")[0])

                if "PQ" in self._index_type:
                    m = int(base_index.split("PQ")[1])
                    index = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, 8)
                else:
                    index = faiss.IndexIVFFlat(
                        quantizer, vector_dim, nlist, faiss.METRIC_INNER_PRODUCT
                    )

            # Train the index with our vectors
            index.train(vectors)
        else:
            # Default to flat index if not recognized
            if self._metric == "l2":
                index = faiss.IndexFlatL2(vector_dim)
            else:  # cosine or inner product
                index = faiss.IndexFlatIP(vector_dim)

        # Apply scalar quantization if requested (reduces memory usage)
        if self._quantize and hasattr(index, "quantizer"):
            index.quantizer = faiss.IndexScalarQuantizer(vector_dim, faiss.ScalarQuantizer.QT_fp16)

        # Map the vectors to their original IDs
        index = faiss.IndexIDMap(index)
        index.add_with_ids(vectors, ids)

        # Set the number of clusters to probe during search
        if hasattr(index, "nprobe"):
            index.nprobe = self._nprobe

        self._index = index
        self._initialized = True
        self._dirty = False
        self._last_build_time = time.time() - start_time

    def _exact_search(
        self, query_vector: EmbeddingVector, k: int, threshold: Optional[float] = None
    ) -> list[tuple[MemoryID, float]]:
        """Perform exact search when we have few vectors."""
        if not self._vectors:
            return []

        # Normalize query vector for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_vector / query_norm

        # Compute similarities for all vectors
        results = []
        for memory_id, vector in self._vectors.items():
            # Normalize vector
            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0:
                vector_norm = 1e-10
            normalized_vector = vector / vector_norm

            # Compute similarity
            if self._metric == "cosine" or self._metric == "ip":
                similarity = float(np.dot(normalized_query, normalized_vector))
            else:  # l2
                distance = float(np.linalg.norm(normalized_query - normalized_vector))
                max_dist = 2.0  # Maximum L2 distance between normalized vectors
                similarity = 1.0 - (distance / max_dist)

            # Filter by threshold
            if threshold is None or similarity >= threshold:
                results.append((memory_id, similarity))

        # Sort by similarity and take top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# Define vector store type and scale options
VectorStoreScaleType = Literal["small", "medium", "large", "auto"]
IndexTypeOptions = Literal["Flat", "IVF", "IVFPQ", "HNSW"]


def get_optimal_faiss_config(scale: VectorStoreScaleType, dimension: int = 768) -> dict[str, Any]:
    """Get optimal FAISS configuration based on memory store scale.

    Args:
        scale: The scale of the memory store ("small", "medium", "large", "auto")
        dimension: Dimensionality of the embedding vectors

    Returns:
        Dictionary with optimal FAISS configuration parameters
    """
    if scale == "small":  # < 100 memories
        return {
            "index_type": "Flat",  # Exact search
            "nprobe": 1,
            "build_threshold": 50,
            "quantize": False,
        }
    elif scale == "medium":  # 100-500 memories
        return {
            "index_type": "IVF100,Flat",  # IVF with 100 clusters
            "nprobe": 10,  # 10% of clusters
            "build_threshold": 50,
            "quantize": False,
        }
    elif scale == "large":  # > 500 memories
        return {
            "index_type": "IVF256,Flat",  # More clusters for larger datasets
            "nprobe": 16,  # Fewer % of clusters but more absolute clusters
            "build_threshold": 50,
            "quantize": False,
        }
    else:  # "auto" - will adjust dynamically
        return {
            "index_type": "IVF100,Flat",
            "nprobe": 10,
            "build_threshold": 50,
            "quantize": False,
        }


class ANNActivationVectorStore(IVectorStore):
    """Vector store that combines ANN search with activation levels.

    This implementation enhances approximate nearest neighbor search with activation levels,
    making recently accessed or important memories more likely to be retrieved.
    It is optimized for large memory stores (500+ memories) and provides significantly
    better performance compared to the ActivationVectorStore.
    """

    def __init__(
        self,
        activation_weight: float = 0.2,
        dimension: int = 768,
        index_type: str = "IVF100,Flat",
        metric: str = "cosine",
        nprobe: int = 10,
        build_threshold: int = 50,
        quantize: bool = False,
    ):
        """Initialize the vector store with ANN and activation.

        Args:
            activation_weight: Weight of activation in final similarity score (0-1)
            dimension: Dimensionality of the embedding vectors
            index_type: FAISS index type (e.g., "Flat", "IVF100,Flat", "IVF100,PQ32")
            metric: Distance metric to use ("cosine", "l2", "ip")
            nprobe: Number of clusters to visit during search (higher = more accurate but slower)
            build_threshold: Minimum number of vectors before building the index
            quantize: Whether to use scalar quantization to reduce memory usage
        """
        self._vector_store = ANNVectorStore(
            dimension=dimension,
            index_type=index_type,
            metric=metric,
            nprobe=nprobe,
            build_threshold=build_threshold,
            quantize=quantize,
        )
        self._activations: dict[MemoryID, float] = {}
        self._activation_weight = activation_weight
        self.component_id = "ann_activation_vector_store"

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
    ) -> list[tuple[MemoryID, float]]:
        """Search for similar vectors with activation boost."""
        # Determine how many candidates to fetch for activation boosting
        initial_k = min(k * 3, len(self._activations)) if self._activations else k

        # Get similarity results
        similarity_results = self._vector_store.search(query_vector, initial_k, threshold)

        # Apply activation boost
        boosted_results = []
        for memory_id, similarity in similarity_results:
            activation = self._activations.get(memory_id, 0.0)
            # Normalize activation to 0-1 scale
            max_activation = max(self._activations.values()) if self._activations else 1.0
            if max_activation == 0:
                max_activation = 1.0
            normalized_activation = activation / max_activation

            # Combine similarity and activation with a smooth function that avoids
            # overprioritizing activation for large memory sets
            memory_count = len(self._activations)
            # Gradually reduce activation weight as memory store grows
            adjusted_weight = self._activation_weight
            if memory_count > 100:
                # For large memory stores, reduce the activation weight
                adjusted_weight = self._activation_weight * (100 / memory_count) ** 0.5
                adjusted_weight = max(0.05, min(self._activation_weight, adjusted_weight))

            # Calculate combined score with adjusted weight
            combined_score = (
                1 - adjusted_weight
            ) * similarity + adjusted_weight * normalized_activation

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

    def progressive_filtering(
        self,
        query_vector: EmbeddingVector,
        initial_k: int = 100,
        final_k: int = 10,
        filter_fn: Optional[
            Callable[[list[tuple[MemoryID, float]]], list[tuple[MemoryID, float]]]
        ] = None,
    ) -> list[tuple[MemoryID, float]]:
        """Perform two-stage retrieval with progressive filtering and activation boost.

        Args:
            query_vector: The query vector
            initial_k: Number of initial candidates to retrieve
            final_k: Final number of results to return
            filter_fn: Optional function to filter/rerank candidates

        Returns:
            Filtered list of (memory_id, score) tuples with activation boost
        """
        # Get more candidates than needed for better activation boosting
        effective_initial_k = (
            min(initial_k * 2, len(self._activations)) if self._activations else initial_k
        )

        # First stage: Get initial candidates with activation boost
        candidates = self.search(query_vector, effective_initial_k, threshold=None)

        if not candidates:
            return []

        # Second stage: Apply filter function if provided
        if filter_fn:
            filtered_results = filter_fn(candidates)
        else:
            filtered_results = candidates

        # Return top k results
        return filtered_results[:final_k]

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the vector store."""
        stats = self._vector_store.get_performance_stats()
        stats.update(
            {
                "activation_weight": self._activation_weight,
                "activation_count": len(self._activations),
            }
        )
        return stats
