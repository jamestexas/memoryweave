"""Chunked memory store implementation with improved ID handling."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, Memory, MemoryID
from memoryweave.storage.base_store import BaseMemoryStore
from memoryweave.storage.memory_store import StandardMemoryStore


@dataclass
class ChunkInfo:
    """Information about a chunk within a memory."""

    chunk_index: int
    embedding: EmbeddingVector
    text: str
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if None."""
        if self.metadata is None:
            self.metadata = {}


class ChunkedMemoryStore(BaseMemoryStore):
    """
    Memory store that supports chunking of text content.

    This store maintains both the full memory and its chunks,
    enabling more precise retrieval from parts of large text documents.
    """

    def __init__(self):
        """Initialize the chunked memory store."""
        super().__init__()
        # Use a standard memory store for the base functionality
        self._base_store = StandardMemoryStore()
        # Store chunks for each memory
        self._chunks: dict[MemoryID, list[ChunkInfo]] = {}
        self.component_id = "chunked_memory_store"

    def add(
        self, embedding: EmbeddingVector, content: Any, metadata: dict[str, Any | None] = None
    ) -> MemoryID:
        """
        Add a memory without chunking.

        Args:
            embedding: The embedding vector
            content: The memory content
            metadata: Optional metadata

        Returns:
            Memory ID
        """
        # Add to base store
        memory_id = self._base_store.add(embedding, content, metadata)
        return memory_id

    def add_with_id(
        self,
        memory_id: MemoryID,
        embedding: EmbeddingVector,
        content: Any,
        metadata: dict[str, Any | None] = None,
    ) -> MemoryID:
        """Add a memory with a specific ID."""
        # Pass to base store
        return self._base_store.add_with_id(memory_id, embedding, content, metadata)

    def add_chunked(
        self,
        chunks: list[dict[str, Any]],
        chunk_embeddings: list[EmbeddingVector],
        original_content: str,
        metadata: dict[str, Any | None] = None,
    ) -> MemoryID:
        """
        Add a memory consisting of multiple chunks.

        Args:
            chunks: list of chunk dictionaries with text and metadata
            chunk_embeddings: list of embeddings for each chunk
            original_content: The original full text content
            metadata: Optional metadata for the memory

        Returns:
            Memory ID
        """
        if len(chunks) != len(chunk_embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not chunks:
            raise ValueError("At least one chunk must be provided")

        # Create a combined embedding (average of chunk embeddings)
        combined_embedding = np.mean(chunk_embeddings, axis=0)

        # Add the memory with the combined embedding
        memory_id = self._base_store.add(combined_embedding, original_content, metadata)

        # Store individual chunks
        chunk_objects = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_text = chunk.get("text", "")
            chunk_metadata = chunk.get("metadata", {}).copy() if "metadata" in chunk else {}

            # Add chunk index if not present
            if "chunk_index" not in chunk_metadata:
                chunk_metadata["chunk_index"] = i

            chunk_objects.append(
                ChunkInfo(
                    chunk_index=i,
                    embedding=embedding,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )

        # Store chunks
        self._chunks[memory_id] = chunk_objects

        return memory_id

    def get(self, memory_id: MemoryID) -> Memory:
        """
        Get a memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object
        """
        memory_id = self._resolve_id(memory_id)
        return self._base_store.get(memory_id)

    def get_all(self) -> list[Memory]:
        """
        Get all memories.

        Returns:
            list of Memory objects
        """
        return self._base_store.get_all()

    def get_chunks(self, memory_id: MemoryID) -> list[ChunkInfo]:
        """
        Get all chunks for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            list of ChunkInfo objects
        """
        memory_id = self._resolve_id(memory_id)
        if memory_id not in self._chunks:
            return []
        return self._chunks[memory_id]

    def get_chunk_embeddings(self, memory_id: MemoryID) -> list[EmbeddingVector]:
        """
        Get all chunk embeddings for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            list of embeddings for each chunk
        """
        chunks = self.get_chunks(memory_id)
        return [chunk.embedding for chunk in chunks]

    def search_chunks(
        self, query_embedding: EmbeddingVector, limit: int = 10, threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for chunks similar to the query embedding.

        Args:
            query_embedding: Query embedding
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            list of chunk result dictionaries
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
        similarities = []
        for memory_id, chunk in all_chunks:
            # Normalize chunk embedding
            embedding = chunk.embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                norm = 1e-10
            normalized_embedding = embedding / norm

            # Calculate cosine similarity
            similarity = float(np.dot(normalized_query, normalized_embedding))
            if threshold is None or similarity >= threshold:
                similarities.append((memory_id, chunk, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)

        # Format results
        results = []
        for memory_id, chunk, similarity in similarities[:limit]:
            result = {
                "memory_id": memory_id,
                "chunk_index": chunk.chunk_index,
                "chunk_similarity": similarity,
                "content": chunk.text,
                "metadata": chunk.metadata.copy(),
            }
            results.append(result)

        return results

    def update_metadata(self, memory_id: MemoryID, metadata: dict[str, Any]) -> None:
        """
        Update metadata for a memory.

        Args:
            memory_id: Memory ID
            metadata: New metadata to update
        """
        memory_id = self._resolve_id(memory_id)
        self._base_store.update_metadata(memory_id, metadata)

    def remove(self, memory_id: MemoryID) -> None:
        """
        Remove a memory and its chunks.

        Args:
            memory_id: Memory ID
        """
        memory_id = self._resolve_id(memory_id)
        self._base_store.remove(memory_id)
        if memory_id in self._chunks:
            del self._chunks[memory_id]

    def clear(self) -> None:
        """Clear all memories and chunks."""
        self._base_store.clear()
        self._chunks.clear()

    def get_chunk_count(self) -> int:
        """
        Get the total number of chunks across all memories.

        Returns:
            Total chunk count
        """
        return sum(len(chunks) for chunks in self._chunks.values())

    def get_average_chunks_per_memory(self) -> float:
        """
        Get the average number of chunks per memory.

        Returns:
            Average chunk count
        """
        if not self._chunks:
            return 0.0
        return self.get_chunk_count() / len(self._chunks)

    def consolidate(self, max_memories: int) -> list[MemoryID]:
        """
        Consolidate memories to stay within capacity.

        Args:
            max_memories: Maximum number of memories to keep

        Returns:
            list of removed memory IDs
        """
        # Use the base store's consolidation logic
        removed_ids = self._base_store.consolidate(max_memories)

        # Also remove chunks for removed memories
        for memory_id in removed_ids:
            if memory_id in self._chunks:
                del self._chunks[memory_id]

        return removed_ids
