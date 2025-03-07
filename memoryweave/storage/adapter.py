"""Memory adapter for consistent access to memory stores."""

import logging
from typing import Any

import numpy as np
from rich.logging import RichHandler

from memoryweave.interfaces.memory import EmbeddingVector, Memory, MemoryID
from memoryweave.storage.base_store import BaseMemoryStore
from memoryweave.storage.vector_search.base import IVectorSearchProvider

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)
logger = logging.getLogger(__name__)


class MemoryAdapter:
    """
    Adapter class for memory stores that provides consistent access patterns
    and handles ID translation transparently.
    """

    def __init__(
        self,
        memory_store: BaseMemoryStore,
        vector_search: IVectorSearchProvider | None = None,
    ):
        """
        Initialize the adapter with a memory store and optional vector search provider.

        Args:
            memory_store: Base memory store implementation
            vector_search: Optional vector search provider (will create a default one if None)
        """
        self.memory_store = memory_store
        self._vector_search = vector_search
        self._embeddings_matrix = None
        self._metadata_dict = None
        self._ids_list = None
        self._index_to_id_map = {}
        self._id_to_index_map = {}
        self._invalidated = True

    def set_vector_search(self, vector_search: IVectorSearchProvider) -> None:
        """
        Set the vector search provider.

        Args:
            vector_search: Vector search provider
        """
        self._vector_search = vector_search
        self.invalidate_cache()

    @property
    def memory_embeddings(self) -> np.ndarray:
        """Return all embeddings as a NumPy matrix for similarity computations."""
        if self._invalidated or self._embeddings_matrix is None:
            self._build_cache()
        return self._embeddings_matrix

    @property
    def memory_metadata(self) -> list[dict[str, Any]]:
        """Return metadata for each memory, in the same order as memory_embeddings."""
        if self._invalidated or self._metadata_dict is None:
            self._build_cache()
        return self._metadata_dict

    @property
    def memory_ids(self) -> list[MemoryID]:
        """Return all memory IDs in the same order as memory_embeddings."""
        if self._invalidated or self._ids_list is None:
            self._build_cache()
        return self._ids_list

    def _build_cache(self) -> None:
        """Build internal cache of embeddings and metadata."""
        try:
            logger.debug("Building memory adapter cache")
            memories = self.memory_store.get_all()

            if not memories:
                # Initialize empty cache
                self._embeddings_matrix = np.zeros((0, 768))  # Default embedding dimension
                self._metadata_dict = []
                self._ids_list = []
                self._index_to_id_map = {}
                self._id_to_index_map = {}
                self._invalidated = False
                logger.debug("No memories found in store")
                return

            # Extract data from memories
            embeddings = []
            metadata_list = []
            ids_list = []

            # Reset ID mappings
            self._index_to_id_map = {}
            self._id_to_index_map = {}

            for idx, memory in enumerate(memories):
                # Store embedding
                embeddings.append(memory.embedding)

                # Process metadata
                if memory.metadata is not None and isinstance(memory.metadata, dict):
                    metadata = memory.metadata.copy()
                else:
                    metadata = {}

                # Add adapter-specific metadata
                metadata["memory_id"] = idx  # Internal index ID for matrix operations
                metadata["original_id"] = memory.id  # Original ID from store

                # Extract content
                if isinstance(memory.content, dict) and "text" in memory.content:
                    metadata["content"] = memory.content["text"]
                else:
                    metadata["content"] = str(memory.content)

                metadata_list.append(metadata)
                ids_list.append(memory.id)

                # Update ID mappings
                self._index_to_id_map[idx] = memory.id
                self._id_to_index_map[memory.id] = idx

            # Create embeddings matrix
            if embeddings:
                self._embeddings_matrix = np.stack(embeddings)
            else:
                self._embeddings_matrix = np.zeros((0, 768))  # Default embedding dimension

            # Store metadata and IDs
            self._metadata_dict = metadata_list
            self._ids_list = ids_list
            self._invalidated = False

            logger.debug(f"Built cache with {len(embeddings)} memories")
            logger.debug(f"ID map has {len(self._index_to_id_map)} entries")

        except Exception as e:
            logger.error(f"Error building memory cache: {e}")
            import traceback

            traceback.print_exc()

            # Initialize empty cache on error
            self._embeddings_matrix = np.zeros((0, 768))
            self._metadata_dict = []
            self._ids_list = []
            self._index_to_id_map = {}
            self._id_to_index_map = {}

    def invalidate_cache(self) -> None:
        """Invalidate the cache after store changes."""
        self._invalidated = True
        self._embeddings_matrix = None
        self._metadata_dict = None
        self._ids_list = None

    def _resolve_id(self, memory_id: str | int) -> MemoryID:
        """
        Resolve a memory ID by handling both index-based and original IDs.

        This is a key method that resolves the ID mapping issues that caused
        problems in the original implementation.
        """
        # Handle integer or string-integer IDs (likely adapter-internal indices)
        if isinstance(memory_id, int) or (isinstance(memory_id, str) and memory_id.isdigit()):
            idx = int(memory_id)
            if idx in self._index_to_id_map:
                original_id = self._index_to_id_map[idx]
                logger.debug(f"Translated index {idx} to memory ID {original_id}")
                return original_id

        # Check if the ID exists in ID-to-index mapping (it's an original ID)
        if memory_id in self._id_to_index_map:
            return memory_id

        # Return the ID as is (it will be validated by the store)
        return memory_id

    def get(self, memory_id: str | int) -> Memory:
        """Get a memory by ID, handling ID translation."""
        resolved_id = self._resolve_id(memory_id)
        return self.memory_store.get(resolved_id)

    def add(
        self, embedding: EmbeddingVector, content: Any, metadata: dict[str, Any] | None = None
    ) -> MemoryID:
        """Add a memory to the store and invalidate cache."""
        memory_id = self.memory_store.add(embedding, content, metadata)
        self.invalidate_cache()
        return memory_id

    def get_all(self) -> list[Memory]:
        """Get all memories from the store."""
        return self.memory_store.get_all()

    def search_by_vector(
        self, query_vector: np.ndarray, limit: int = 10, threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for memories by vector similarity.

        Args:
            query_vector: Query embedding
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            list of result dictionaries with memory info and scores
        """
        if (
            self._invalidated
            or self._embeddings_matrix is None
            or len(self._embeddings_matrix) == 0
        ):
            self._build_cache()
            if len(self._embeddings_matrix) == 0:
                return []

        # If vector search provider is available, use it
        if self._vector_search is not None:
            try:
                # Use vector search provider
                search_results = self._vector_search.search(query_vector, limit, threshold)

                # Format results
                results = []
                for idx, score in search_results:
                    # Translate internal index to original ID
                    original_id = self._index_to_id_map.get(idx)
                    if original_id is None:
                        logger.warning(f"Could not resolve index {idx} to memory ID")
                        continue

                    try:
                        # Get the full memory
                        memory = self.memory_store.get(original_id)

                        # Create result object
                        result = {
                            "id": original_id,
                            "memory_id": original_id,  # For backward compatibility
                            "content": memory.content
                            if not isinstance(memory.content, dict)
                            else memory.content.get("text", str(memory.content)),
                            "metadata": memory.metadata if memory.metadata else {},
                            "score": score,
                            "relevance_score": score,  # For backward compatibility
                        }

                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error retrieving memory {original_id}: {e}")

                return results

            except Exception as e:
                logger.error(
                    f"Error using vector search provider: {e}, falling back to direct search"
                )
                # Fall back to direct search

        # Direct search using NumPy (fallback)
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_vector / query_norm

        # Compute similarities
        similarities = np.dot(self._embeddings_matrix, normalized_query)

        # Apply threshold filtering if needed
        if threshold is not None:
            valid_indices = np.where(similarities >= threshold)[0]
            if len(valid_indices) == 0:
                return []

            # Get top indices among valid ones
            top_indices = valid_indices[np.argsort(-similarities[valid_indices])][:limit]
        else:
            # Get top indices without threshold
            top_indices = np.argsort(-similarities)[:limit]

        # Format results
        results = []
        for idx in top_indices:
            original_id = self._index_to_id_map[int(idx)]
            similarity_score = float(similarities[idx])

            try:
                # Get the full memory
                memory = self.memory_store.get(original_id)

                # Create result object
                result = {
                    "id": original_id,
                    "memory_id": original_id,  # For backward compatibility
                    "content": memory.content
                    if not isinstance(memory.content, dict)
                    else memory.content.get("text", str(memory.content)),
                    "metadata": memory.metadata if memory.metadata else {},
                    "score": similarity_score,
                    "relevance_score": similarity_score,  # For backward compatibility
                }

                results.append(result)
            except Exception as e:
                logger.error(f"Error retrieving memory {original_id}: {e}")

        return results

    def update_metadata(self, memory_id: MemoryID, metadata: dict[str, Any]) -> None:
        """Update metadata for a memory."""
        resolved_id = self._resolve_id(memory_id)
        self.memory_store.update_metadata(resolved_id, metadata)
        self.invalidate_cache()

    def remove(self, memory_id: MemoryID) -> None:
        """Remove a memory from the store."""
        resolved_id = self._resolve_id(memory_id)
        self.memory_store.remove(resolved_id)
        self.invalidate_cache()

    def clear(self) -> None:
        """Clear all memories from the store."""
        self.memory_store.clear()
        self.invalidate_cache()
