"""Hybrid memory store implementation with improved ID handling."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, Memory, MemoryID
from memoryweave.storage.refactored.base_store import BaseMemoryStore
from memoryweave.storage.refactored.memory_store import StandardMemoryStore


@dataclass
class ChunkInfo:
    """Information about a chunk within a memory."""

    chunk_index: int
    embedding: EmbeddingVector
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridMemoryInfo:
    """Information about a hybrid memory with full and chunk embeddings."""

    full_embedding: EmbeddingVector
    chunks: List[ChunkInfo]
    is_hybrid: bool = True


class HybridMemoryStore(BaseMemoryStore):
    """
    Memory-efficient hybrid memory store.

    This store maintains both full embeddings and selective chunks,
    optimizing memory usage while preserving context for retrieval.
    """

    def __init__(self):
        """Initialize the hybrid memory store."""
        super().__init__()
        # Use a standard memory store for the base functionality
        self._base_store = StandardMemoryStore()
        # Store hybrid info for memories
        self._hybrid_info: Dict[MemoryID, HybridMemoryInfo] = {}
        self.component_id = "hybrid_memory_store"

    def add(
        self, embedding: EmbeddingVector, content: Any, metadata: Dict[str, Any | None] = None
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
        metadata: Dict[str, Any | None] = None,
    ) -> MemoryID:
        """Add a memory with a specific ID."""
        # Pass to base store
        return self._base_store.add_with_id(memory_id, embedding, content, metadata)

    def add_hybrid(
        self,
        full_embedding: EmbeddingVector,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: List[EmbeddingVector],
        original_content: str,
        metadata: Dict[str, Any | None] = None,
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
        if len(chunks) != len(chunk_embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        # Add the memory with the full embedding
        memory_id = self._base_store.add(full_embedding, original_content, metadata)

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

        # Store hybrid information
        self._hybrid_info[memory_id] = HybridMemoryInfo(
            full_embedding=full_embedding,
            chunks=chunk_objects,
            is_hybrid=True,
        )

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

    def get_all(self) -> List[Memory]:
        """
        Get all memories.

        Returns:
            List of Memory objects
        """
        return self._base_store.get_all()

    def is_hybrid(self, memory_id: MemoryID) -> bool:
        """
        Check if a memory is stored in hybrid format.

        Args:
            memory_id: ID of the memory

        Returns:
            True if the memory has both full and chunk embeddings
        """
        memory_id = self._resolve_id(memory_id)
        return memory_id in self._hybrid_info

    def get_chunks(self, memory_id: MemoryID) -> List[ChunkInfo]:
        """
        Get all chunks for a memory.

        Args:
            memory_id: ID of the memory

        Returns:
            List of ChunkInfo objects
        """
        memory_id = self._resolve_id(memory_id)
        if memory_id not in self._hybrid_info:
            return []
        return self._hybrid_info[memory_id].chunks

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
        self, query_embedding: EmbeddingVector, limit: int = 10, threshold: float | None = None
    ) -> List[Dict[str, Any]]:
        """
        Search for individual chunks matching the query embedding.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of dictionaries with chunk information and scores
        """
        # Normalize query vector for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_embedding / query_norm

        # Collect all chunks with their memory IDs
        all_chunks = []
        for memory_id, hybrid_info in self._hybrid_info.items():
            for chunk in hybrid_info.chunks:
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
                results.append({
                    "memory_id": memory_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_similarity": similarity,
                    "content": chunk.text,
                    "metadata": chunk.metadata.copy(),
                })

        # Sort by similarity (descending)
        results.sort(key=lambda x: x["chunk_similarity"], reverse=True)

        # Return top results
        return results[:limit]

    def search_hybrid(
        self,
        query_embedding: EmbeddingVector,
        limit: int = 10,
        threshold: float | None = None,
        keywords: List[str | None] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search using both full embeddings and chunks with optional keyword filtering.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            keywords: Optional list of keywords for filtering

        Returns:
            List of dictionaries with memory information and scores
        """
        # Normalize query vector for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_embedding / query_norm

        # Calculate similarities for full embeddings
        full_results = []
        memories = self._base_store.get_all()

        for memory in memories:
            # Normalize memory embedding
            memory_embedding = memory.embedding
            memory_norm = np.linalg.norm(memory_embedding)
            if memory_norm == 0:
                memory_norm = 1e-10
            normalized_memory = memory_embedding / memory_norm

            # Calculate similarity
            similarity = float(np.dot(normalized_query, normalized_memory))

            # Apply threshold if provided
            if threshold is None or similarity >= threshold:
                # Determine if this is a hybrid memory
                is_hybrid = self.is_hybrid(memory.id)

                # Extract content
                if isinstance(memory.content, dict) and "text" in memory.content:
                    content = memory.content["text"]
                else:
                    content = str(memory.content)

                # Create result
                result = {
                    "memory_id": memory.id,
                    "content": content,
                    "relevance_score": similarity,
                    "is_hybrid": is_hybrid,
                }

                # Add metadata
                if memory.metadata:
                    for key, value in memory.metadata.items():
                        result[key] = value

                # Apply keyword boosting if keywords are provided
                if keywords and len(keywords) > 0:
                    content_lower = content.lower()
                    keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
                    keyword_boost = min(0.3, keyword_matches * 0.05)
                    result["relevance_score"] = min(1.0, similarity + keyword_boost)
                    result["keyword_matches"] = keyword_matches

                full_results.append(result)

        # Also search chunks if available
        chunk_results = self.search_chunks(query_embedding, limit=limit * 2, threshold=threshold)

        # Combine full and chunk results
        combined_results = []
        seen_memory_ids = set()

        # Add full results first (prioritize them)
        for result in full_results:
            memory_id = result["memory_id"]
            if memory_id in seen_memory_ids:
                continue

            seen_memory_ids.add(memory_id)
            combined_results.append(result)

        # Add unique chunk results
        for chunk_result in chunk_results:
            memory_id = chunk_result["memory_id"]
            if memory_id in seen_memory_ids:
                continue

            seen_memory_ids.add(memory_id)

            # Format chunk result to match full result format
            formatted_result = {
                "memory_id": memory_id,
                "content": chunk_result["content"],
                "relevance_score": chunk_result["chunk_similarity"],
                "is_hybrid": True,
                "chunk_index": chunk_result["chunk_index"],
            }

            # Apply keyword boosting if keywords are provided
            if keywords and len(keywords) > 0:
                content = formatted_result["content"].lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
                keyword_boost = min(0.3, keyword_matches * 0.05)
                formatted_result["relevance_score"] = min(
                    1.0, formatted_result["relevance_score"] + keyword_boost
                )
                formatted_result["keyword_matches"] = keyword_matches

            # Add metadata
            if "metadata" in chunk_result:
                for key, value in chunk_result["metadata"].items():
                    if key not in ("memory_id", "chunk_index"):
                        formatted_result[key] = value

            combined_results.append(formatted_result)

        # Sort by relevance and return top results
        combined_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return combined_results[:limit]

    def update_metadata(self, memory_id: MemoryID, metadata: Dict[str, Any]) -> None:
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
        Remove a memory and its hybrid information.

        Args:
            memory_id: Memory ID
        """
        memory_id = self._resolve_id(memory_id)
        self._base_store.remove(memory_id)
        if memory_id in self._hybrid_info:
            del self._hybrid_info[memory_id]

    def clear(self) -> None:
        """Clear all memories and hybrid information."""
        self._base_store.clear()
        self._hybrid_info.clear()

    def get_chunk_count(self) -> int:
        """
        Get the total number of chunks across all memories.

        Returns:
            Total chunk count
        """
        return sum(len(info.chunks) for info in self._hybrid_info.values())

    def get_average_chunks_per_memory(self) -> float:
        """
        Get the average number of chunks per hybrid memory.

        Returns:
            Average chunk count
        """
        if not self._hybrid_info:
            return 0.0
        return self.get_chunk_count() / len(self._hybrid_info)

    def consolidate(self, max_memories: int) -> List[MemoryID]:
        """
        Consolidate memories to stay within capacity.

        Args:
            max_memories: Maximum number of memories to keep

        Returns:
            List of removed memory IDs
        """
        # Use the base store's consolidation logic
        removed_ids = self._base_store.consolidate(max_memories)

        # Also remove hybrid info for removed memories
        for memory_id in removed_ids:
            if memory_id in self._hybrid_info:
                del self._hybrid_info[memory_id]

        return removed_ids
