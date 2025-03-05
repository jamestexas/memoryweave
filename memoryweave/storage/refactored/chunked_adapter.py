"""Adapter for chunked memory stores."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, MemoryID
from memoryweave.storage.refactored.adapter import MemoryAdapter
from memoryweave.storage.refactored.chunked_store import ChunkedMemoryStore

logger = logging.getLogger(__name__)


class ChunkedMemoryAdapter(MemoryAdapter):
    """
    Adapter for chunked memory stores that provides additional chunk-specific functionality.

    This adapter extends the base MemoryAdapter with chunk-specific operations
    while maintaining the core adapter functionality.
    """

    def __init__(self, memory_store: ChunkedMemoryStore):
        """
        Initialize the chunked memory adapter.

        Args:
            memory_store: ChunkedMemoryStore instance
        """
        super().__init__(memory_store)
        self.chunked_store = memory_store
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
    def chunk_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all chunks.

        Returns:
            List of metadata dictionaries for each chunk
        """
        if self._invalidated or self._chunk_metadata_cache is None:
            self._build_chunk_cache()
        return self._chunk_metadata_cache

    @property
    def chunk_ids(self) -> List[Tuple[MemoryID, int]]:
        """
        Get all (memory_id, chunk_index) pairs.

        Returns:
            List of (memory_id, chunk_index) tuples
        """
        if self._invalidated or self._chunk_ids_cache is None:
            self._build_chunk_cache()
        return self._chunk_ids_cache

    def _build_chunk_cache(self):
        """Build cache of chunk embeddings and metadata."""
        # Get all memories
        all_memories = self.memory_store.get_all()

        chunk_embeddings = []
        chunk_metadata = []
        chunk_ids = []

        # Process each memory and its chunks
        for memory_idx, memory in enumerate(all_memories):
            memory_id = memory.id
            chunks = self.chunked_store.get_chunks(memory_id)

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
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query embedding.

        Args:
            query_vector: Query embedding
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            List of chunk result dictionaries
        """
        # Use the store's search_chunks method
        return self.chunked_store.search_chunks(query_vector, limit, threshold)

    def add_chunked(
        self,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: List[EmbeddingVector],
        original_content: str,
        metadata: Dict[str, Any | None] = None,
    ) -> MemoryID:
        """
        Add a memory consisting of multiple chunks.

        Args:
            chunks: List of chunk dictionaries with text and metadata
            chunk_embeddings: List of embeddings for each chunk
            original_content: The original full text content
            metadata: Optional metadata for the memory

        Returns:
            Memory ID
        """
        memory_id = self.chunked_store.add_chunked(
            chunks, chunk_embeddings, original_content, metadata
        )
        self.invalidate_cache()
        return memory_id

    def get_chunks(self, memory_id: MemoryID) -> List[Any]:
        """
        Get all chunks for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            List of ChunkInfo objects
        """
        resolved_id = self._resolve_id(memory_id)
        return self.chunked_store.get_chunks(resolved_id)

    def get_chunk_count(self) -> int:
        """
        Get the total number of chunks across all memories.

        Returns:
            Total chunk count
        """
        return self.chunked_store.get_chunk_count()

    def get_average_chunks_per_memory(self) -> float:
        """
        Get the average number of chunks per memory.

        Returns:
            Average chunk count
        """
        return self.chunked_store.get_average_chunks_per_memory()
