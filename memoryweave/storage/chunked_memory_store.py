"""Chunked memory store implementation for MemoryWeave.

This module extends the basic memory store to support chunked text storage
with multiple embeddings per memory, enabling better representation of
large text contexts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, MemoryID
from memoryweave.storage.memory_store import MemoryStore


@dataclass
class ChunkInfo:
    """Information about a chunk within a memory."""

    chunk_index: int
    embedding: EmbeddingVector
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkedMemoryStore(MemoryStore):
    """
    Extended memory store supporting chunked text storage.

    This store maintains multiple embeddings per memory item,
    corresponding to chunks of the original text. This allows for
    better representation and retrieval of large text contexts.
    """

    def __init__(self):
        """Initialize the chunked memory store."""
        super().__init__()
        self._chunks: Dict[MemoryID, List[ChunkInfo]] = {}
        self.component_id = "chunked_memory_store"

    def add_chunked(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[EmbeddingVector],
        original_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryID:
        """
        Add a memory consisting of multiple chunks.

        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embeddings for each chunk
            original_content: The original full text content
            metadata: Optional metadata for the memory

        Returns:
            Memory ID of the added memory
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not chunks:
            raise ValueError("At least one chunk must be provided")

        # Create a combined embedding (average of chunk embeddings)
        combined_embedding = np.mean(embeddings, axis=0)

        # Add the memory with the combined embedding
        memory_id = self.add(combined_embedding, original_content, metadata)

        # Store individual chunks
        chunk_objects = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_text = chunk.get("text", "")
            chunk_metadata = chunk.get("metadata", {}).copy()

            # Add chunk index if not present
            if "chunk_index" not in chunk_metadata:
                chunk_metadata["chunk_index"] = i

            chunk_objects.append(
                ChunkInfo(
                    chunk_index=i, embedding=embedding, text=chunk_text, metadata=chunk_metadata
                )
            )

        # Store chunks
        self._chunks[memory_id] = chunk_objects

        return memory_id

    def get_chunks(self, memory_id: MemoryID) -> List[ChunkInfo]:
        """
        Get all chunks for a memory.

        Args:
            memory_id: ID of the memory

        Returns:
            List of ChunkInfo objects
        """
        if memory_id not in self._chunks:
            return []

        return self._chunks[memory_id]

    def get_chunk_embeddings(self, memory_id: MemoryID) -> List[EmbeddingVector]:
        """
        Get all chunk embeddings for a memory.

        Args:
            memory_id: ID of the memory

        Returns:
            List of embeddings for each chunk
        """
        chunks = self.get_chunks(memory_id)
        return [chunk.embedding for chunk in chunks]

    def search_chunks(
        self, query_embedding: EmbeddingVector, limit: int = 10, threshold: Optional[float] = None
    ) -> List[Tuple[MemoryID, int, float]]:
        """
        Search for individual chunks matching the query embedding.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of tuples (memory_id, chunk_index, similarity_score)
        """
        # Normalize query vector for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_embedding / query_norm

        # Collect all chunks with their memory IDs
        all_chunks = []
        for memory_id, chunks in self._chunks.items():
            for chunk in chunks:
                all_chunks.append((memory_id, chunk))

        if not all_chunks:
            return []

        # Calculate similarities
        results = []
        for memory_id, chunk in all_chunks:
            # Normalize chunk embedding
            embedding = chunk.embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                norm = 1e-10
            normalized_embedding = embedding / norm

            # Calculate cosine similarity
            similarity = float(np.dot(normalized_query, normalized_embedding))

            # Apply threshold
            if threshold is None or similarity >= threshold:
                results.append((memory_id, chunk.chunk_index, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[2], reverse=True)

        # Return top results
        return results[:limit]

    def remove(self, memory_id: MemoryID) -> None:
        """
        Remove a memory and its chunks from the store.

        Args:
            memory_id: ID of the memory to remove
        """
        super().remove(memory_id)
        if memory_id in self._chunks:
            del self._chunks[memory_id]

    def clear(self) -> None:
        """Clear all memories and chunks."""
        super().clear()
        self._chunks.clear()

    def get_chunk_count(self) -> int:
        """Get the total number of chunks across all memories."""
        return sum(len(chunks) for chunks in self._chunks.values())

    def get_average_chunks_per_memory(self) -> float:
        """Get the average number of chunks per memory."""
        if not self._chunks:
            return 0.0
        return self.get_chunk_count() / len(self._chunks)


class ChunkedMemoryAdapter:
    """
    Adapter for ChunkedMemoryStore to ensure compatibility with the ContextualFabricStrategy.

    This adapter provides methods to work with chunked memories while
    maintaining compatibility with the existing retrieval architecture.
    """

    def __init__(self, memory_store: ChunkedMemoryStore):
        """
        Initialize the chunked memory adapter.

        Args:
            memory_store: The chunked memory store to adapt
        """
        self.memory_store = memory_store
        self._chunk_embeddings_cache = None
        self._chunk_metadata_cache = None
        self._chunk_id_cache = None
        self._invalidated = True

    def invalidate_cache(self):
        """Invalidate the cache when the memory store changes."""
        self._invalidated = True
        self._chunk_embeddings_cache = None
        self._chunk_metadata_cache = None
        self._chunk_id_cache = None

    @property
    def chunk_embeddings(self) -> np.ndarray:
        """
        Get matrix of all chunk embeddings.

        Returns:
            Matrix with each row being one chunk embedding
        """
        if self._invalidated or self._chunk_embeddings_cache is None:
            self._build_cache()
        return self._chunk_embeddings_cache

    @property
    def chunk_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all chunks.

        Returns:
            List of metadata dictionaries for each chunk
        """
        if self._invalidated or self._chunk_metadata_cache is None:
            self._build_cache()
        return self._chunk_metadata_cache

    @property
    def chunk_ids(self) -> List[Tuple[MemoryID, int]]:
        """
        Get all (memory_id, chunk_index) pairs.

        Returns:
            List of (memory_id, chunk_index) tuples
        """
        if self._invalidated or self._chunk_id_cache is None:
            self._build_cache()
        return self._chunk_id_cache

    def _build_cache(self):
        """Build cache of chunk embeddings and metadata."""
        # Get all memories
        all_memories = self.memory_store.get_all()

        chunk_embeddings = []
        chunk_metadata = []
        chunk_ids = []

        # Process each memory and its chunks
        for memory in all_memories:
            memory_id = memory.id
            chunks = self.memory_store.get_chunks(memory_id)

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

                chunk_metadata.append(meta)
                chunk_ids.append((memory_id, chunk.chunk_index))

        # Convert to numpy array for embeddings
        if chunk_embeddings:
            self._chunk_embeddings_cache = np.stack(chunk_embeddings)
        else:
            # Return empty array with proper shape
            dim = self.memory_store.get_all()[0].embedding.shape[0] if all_memories else 0
            self._chunk_embeddings_cache = np.zeros((0, dim))

        self._chunk_metadata_cache = chunk_metadata
        self._chunk_id_cache = chunk_ids
        self._invalidated = False

    def search_chunks(
        self, query_embedding: EmbeddingVector, limit: int = 10, threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query embedding.

        Args:
            query_embedding: Query embedding
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            List of dictionaries with chunk information and scores
        """
        if self._invalidated or self._chunk_embeddings_cache is None:
            self._build_cache()

        if len(self._chunk_embeddings_cache) == 0:
            return []

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_embedding / query_norm

        # Calculate similarities
        similarities = np.dot(self._chunk_embeddings_cache, normalized_query)

        # Apply threshold
        if threshold is not None:
            valid_indices = np.where(similarities >= threshold)[0]
            if len(valid_indices) == 0:
                return []

            # Sort valid indices
            sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])]
        else:
            # Sort all indices
            sorted_indices = np.argsort(-similarities)

        # Limit results
        top_indices = sorted_indices[:limit]

        # Create result objects
        results = []
        for idx in top_indices:
            memory_id, chunk_index = self._chunk_id_cache[idx]
            metadata = self._chunk_metadata_cache[idx]

            result = {
                "memory_id": memory_id,
                "chunk_index": chunk_index,
                "relevance_score": float(similarities[idx]),
                "metadata": metadata.copy(),
                "content": metadata.get("chunk_text", ""),
            }

            results.append(result)

        return results

    def get_memory_chunks(self, memory_id: MemoryID) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific memory.

        Args:
            memory_id: Memory ID

        Returns:
            List of chunk information dictionaries
        """
        chunks = self.memory_store.get_chunks(memory_id)
        if not chunks:
            return []

        results = []
        for chunk in chunks:
            results.append({
                "memory_id": memory_id,
                "chunk_index": chunk.chunk_index,
                "embedding": chunk.embedding,
                "content": chunk.text,
                "metadata": chunk.metadata.copy(),
            })

        return results
