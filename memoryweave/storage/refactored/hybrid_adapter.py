"""Adapter for hybrid memory stores."""

import logging
from typing import Any

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, MemoryID
from memoryweave.storage.refactored.adapter import MemoryAdapter
from memoryweave.storage.refactored.hybrid_store import HybridMemoryStore

logger = logging.getLogger(__name__)


class HybridMemoryAdapter(MemoryAdapter):
    """
    Adapter for hybrid memory stores that provides both full embeddings and chunk access.

    This adapter extends the base MemoryAdapter with hybrid-specific operations
    including efficient access to both full embeddings and selective chunks.
    """

    def __init__(self, memory_store: HybridMemoryStore):
        """
        Initialize the hybrid memory adapter.

        Args:
            memory_store: HybridMemoryStore instance
        """
        super().__init__(memory_store)
        self.hybrid_store = memory_store
        self._chunk_embeddings_cache = None
        self._chunk_metadata_cache = None
        self._chunk_ids_cache = None

    @property
    def chunk_embeddings(self) -> np.ndarray:
        """
        Get matrix of all chunk embeddings.

        Returns:
            Matrix with each row being one chunk embedding
        """
        if self._invalidated or self._chunk_embeddings_cache is None:
            self._build_chunk_cache()
        return self._chunk_embeddings_cache

    @property
    def chunk_metadata(self) -> list[dict[str, Any]]:
        """
        Get metadata for all chunks.

        Returns:
            list of metadata dictionaries for each chunk
        """
        if self._invalidated or self._chunk_metadata_cache is None:
            self._build_chunk_cache()
        return self._chunk_metadata_cache

    @property
    def chunk_ids(self) -> list[tuple[MemoryID, int]]:
        """
        Get all (memory_id, chunk_index) pairs.

        Returns:
            list of (memory_id, chunk_index) tuples
        """
        if self._invalidated or self._chunk_ids_cache is None:
            self._build_chunk_cache()
        return self._chunk_ids_cache

    def _build_chunk_cache(self):
        """Build cache of chunk embeddings and metadata."""
        # Get all hybrid memories
        all_memories = self.memory_store.get_all()

        chunk_embeddings = []
        chunk_metadata = []
        chunk_ids = []

        # Process each memory and its chunks
        for memory_idx, memory in enumerate(all_memories):
            memory_id = memory.id

            # Skip non-hybrid memories
            if not self.hybrid_store.is_hybrid(memory_id):
                continue

            chunks = self.hybrid_store.get_chunks(memory_id)

            for chunk in chunks:
                # Add embedding
                chunk_embeddings.append(chunk.embedding)

                # Create metadata entry
                meta = {}

                # Add memory metadata
                if memory.metadata:
                    meta.update(memory.metadata)

                # Add chunk metadata
                meta.update(chunk.metadata)

                # Add chunk identifiers
                meta["memory_id"] = memory_id
                meta["chunk_index"] = chunk.chunk_index
                meta["chunk_text"] = chunk.text
                meta["internal_memory_idx"] = memory_idx  # Add this for mapping
                meta["is_hybrid"] = True

                chunk_metadata.append(meta)
                chunk_ids.append((memory_id, chunk.chunk_index))

        # Convert to numpy array for embeddings
        if chunk_embeddings:
            self._chunk_embeddings_cache = np.stack(chunk_embeddings)
        else:
            # Return empty array with proper shape
            dim = 768  # Default dimension
            if all_memories and hasattr(all_memories[0], "embedding"):
                dim = all_memories[0].embedding.shape[0]
            self._chunk_embeddings_cache = np.zeros((0, dim))

        self._chunk_metadata_cache = chunk_metadata
        self._chunk_ids_cache = chunk_ids
        self._invalidated = False

    def invalidate_cache(self):
        """Invalidate all caches."""
        super().invalidate_cache()
        self._chunk_embeddings_cache = None
        self._chunk_metadata_cache = None
        self._chunk_ids_cache = None

    def search_chunks(
        self, query_vector: np.ndarray, limit: int = 10, threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for chunks similar to the query embedding.

        Args:
            query_vector: Query embedding
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            list of chunk result dictionaries
        """
        # Use the store's search_chunks method
        return self.hybrid_store.search_chunks(query_vector, limit, threshold)

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        threshold: float | None = None,
        keywords: list[str | None] = None,
    ) -> list[dict[str, Any]]:
        """
        Search using both full embeddings and chunks with optional keyword filtering.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            keywords: Optional list of keywords for filtering

        Returns:
            list of dictionaries with memory information and scores
        """
        # Use the store's hybrid search method
        return self.hybrid_store.search_hybrid(query_vector, limit, threshold, keywords)

    def add_hybrid(
        self,
        full_embedding: EmbeddingVector,
        chunks: list[dict[str, Any]],
        chunk_embeddings: list[EmbeddingVector],
        original_content: str,
        metadata: dict[str, Any | None] = None,
    ) -> MemoryID:
        """
        Add a memory with both full and chunk embeddings.

        Args:
            full_embedding: Embedding of the full content
            chunks: list of chunk dictionaries with text and metadata
            chunk_embeddings: list of embeddings for each chunk
            original_content: The original full text content
            metadata: Optional metadata for the memory

        Returns:
            Memory ID
        """
        memory_id = self.hybrid_store.add_hybrid(
            full_embedding, chunks, chunk_embeddings, original_content, metadata
        )
        self.invalidate_cache()
        return memory_id

    def is_hybrid(self, memory_id: MemoryID) -> bool:
        """
        Check if a memory is stored in hybrid format.

        Args:
            memory_id: ID of the memory

        Returns:
            True if the memory has both full and chunk embeddings
        """
        resolved_id = self._resolve_id(memory_id)
        return self.hybrid_store.is_hybrid(resolved_id)

    def get_chunks(self, memory_id: MemoryID) -> list[Any]:
        """
        Get all chunks for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            list of ChunkInfo objects
        """
        resolved_id = self._resolve_id(memory_id)
        return self.hybrid_store.get_chunks(resolved_id)

    def get_chunk_count(self) -> int:
        """
        Get the total number of chunks across all memories.

        Returns:
            Total chunk count
        """
        return self.hybrid_store.get_chunk_count()

    def get_average_chunks_per_memory(self) -> float:
        """
        Get the average number of chunks per hybrid memory.

        Returns:
            Average chunk count
        """
        return self.hybrid_store.get_average_chunks_per_memory()

    def _build_cache(self):
        """Build cache of all memory embeddings and metadata."""
        try:
            if not hasattr(self.hybrid_store, "get_all"):
                # Fall back to memory_store if hybrid_store doesn't have get_all
                memories = self.memory_store.get_all()
            else:
                memories = self.hybrid_store.get_all()

            if not memories:
                self._embeddings_matrix = np.zeros((0, 768))  # Default embedding dim
                self._metadata_dict = []
                self._ids_list = []
                self._index_to_id_map = {}
                self._id_to_index_map = {}
                self._invalidated = False
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

                # Create metadata entry
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

            # Also build chunk cache if appropriate
            if hasattr(self, "_build_chunk_cache"):
                self._build_chunk_cache()

        except Exception as e:
            logging.error(f"Error building memory cache: {e}")
            import traceback

            traceback.print_exc()

            # Initialize empty cache on error
            self._embeddings_matrix = np.zeros((0, 768))
            self._metadata_dict = []
            self._ids_list = []
            self._index_to_id_map = {}
            self._id_to_index_map = {}
