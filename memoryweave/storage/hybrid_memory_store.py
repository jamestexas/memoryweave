"""Hybrid memory store implementation for MemoryWeave.

This module provides a memory-efficient hybrid approach that combines
full embeddings with selective chunking to optimize memory usage.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, MemoryID
from memoryweave.storage.memory_store import Memory, MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a chunk within a memory."""

    chunk_index: int
    embedding: EmbeddingVector
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridMemoryInfo:
    """Information about a hybrid memory with full and chunk embeddings."""

    full_embedding: EmbeddingVector
    chunks: list[ChunkInfo]
    is_hybrid: bool = True


class HybridMemoryStore(MemoryStore):
    """
    Memory-efficient hybrid memory store.

    This store maintains both full embeddings and selective chunks,
    optimizing memory usage while preserving context for retrieval.
    """

    def __init__(self):
        """Initialize the hybrid memory store."""
        super().__init__()
        self._hybrid_info: dict[MemoryID, HybridMemoryInfo] = {}
        self.component_id = "hybrid_memory_store"

    def add_hybrid(
        self,
        full_embedding: EmbeddingVector,
        chunks: list[dict[str, Any]],
        chunk_embeddings: list[EmbeddingVector],
        original_content: str,
        metadata: dict[str, Any] | None = None,
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
            Memory ID of the added memory
        """
        if len(chunks) != len(chunk_embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        # Add the memory with the full embedding
        memory_id = self.add(full_embedding, original_content, metadata)

        # Store individual chunks
        chunk_objects = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
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

        # Store hybrid information
        self._hybrid_info[memory_id] = HybridMemoryInfo(
            full_embedding=full_embedding, chunks=chunk_objects, is_hybrid=True
        )

        return memory_id

    def is_hybrid(self, memory_id: MemoryID) -> bool:
        """
        Check if a memory is stored in hybrid format.

        Args:
            memory_id: ID of the memory

        Returns:
            True if the memory has both full and chunk embeddings
        """
        return memory_id in self._hybrid_info

    def get_chunks(self, memory_id: MemoryID) -> list[ChunkInfo]:
        """
        Get all chunks for a memory.

        Args:
            memory_id: ID of the memory

        Returns:
            list of ChunkInfo objects
        """
        if memory_id not in self._hybrid_info:
            return []

        return self._hybrid_info[memory_id].chunks

    def get_chunk_embeddings(self, memory_id: MemoryID) -> list[EmbeddingVector]:
        """
        Get all chunk embeddings for a memory.

        Args:
            memory_id: ID of the memory

        Returns:
            list of embeddings for each chunk
        """
        chunks = self.get_chunks(memory_id)
        return [chunk.embedding for chunk in chunks]

    def search_chunks(
        self, query_embedding: EmbeddingVector, limit: int = 10, threshold: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """
        Search for individual chunks matching the query embedding.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            list of dictionaries with chunk information and scores
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
                results.append(
                    {
                        "memory_id": memory_id,
                        "chunk_index": chunk.chunk_index,
                        "chunk_similarity": similarity,
                        "content": chunk.text,
                        "metadata": chunk.metadata.copy(),
                    }
                )

        # Sort by similarity (descending)
        results.sort(key=lambda x: x["chunk_similarity"], reverse=True)

        # Return top results
        return results[:limit]

    def search_hybrid(
        self,
        query_embedding: EmbeddingVector,
        limit: int = 10,
        threshold: Optional[float] = None,
        keywords: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search using both full embeddings and chunks with optional keyword filtering.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            keywords: Optional list of keywords for filtering

        Returns:
            list of dictionaries with memory information and scores
        """
        # First get results from full embeddings
        full_results = self.search_by_vector(query_embedding, limit * 2, threshold)

        # Then get results from chunks
        chunk_results = self.search_chunks(query_embedding, limit * 2, threshold)

        # Combine the results with a preference for full embeddings
        combined_results = []

        # Track memory IDs to avoid duplicates
        seen_memory_ids = set()

        # Process full results first
        for result in full_results:
            memory_id = result["memory_id"]
            if memory_id in seen_memory_ids:
                continue

            seen_memory_ids.add(memory_id)

            # Check if this is a hybrid memory
            is_hybrid = self.is_hybrid(memory_id)
            result["is_hybrid"] = is_hybrid

            # Apply keyword boosting if keywords are provided
            if keywords and len(keywords) > 0:
                content = result.get("content", "").lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
                keyword_boost = min(0.3, keyword_matches * 0.05)
                result["relevance_score"] = min(1.0, result["relevance_score"] + keyword_boost)
                result["keyword_matches"] = keyword_matches

            combined_results.append(result)

        # Now process chunk results, avoiding duplicates
        for result in chunk_results:
            memory_id = result["memory_id"]
            if memory_id in seen_memory_ids:
                continue

            seen_memory_ids.add(memory_id)

            # Format chunk result to match full result format
            formatted_result = {
                "memory_id": memory_id,
                "content": result["content"],
                "relevance_score": result["chunk_similarity"],
                "is_hybrid": True,
                "chunk_index": result["chunk_index"],
            }

            # Apply keyword boosting if keywords are provided
            if keywords and len(keywords) > 0:
                content = formatted_result.get("content", "").lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
                keyword_boost = min(0.3, keyword_matches * 0.05)
                formatted_result["relevance_score"] = min(
                    1.0, formatted_result["relevance_score"] + keyword_boost
                )
                formatted_result["keyword_matches"] = keyword_matches

            # Add metadata
            if "metadata" in result:
                for key, value in result["metadata"].items():
                    if key not in ("memory_id", "chunk_index"):
                        formatted_result[key] = value

            combined_results.append(formatted_result)

        # Sort by relevance score
        combined_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Return top results
        return combined_results[:limit]

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for memories by vector similarity.

        Args:
            query_vector: Query vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            list of dictionaries with memory information and scores
        """
        if not self._memories:
            return []

        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_vector / query_norm

        # Calculate similarities using full embeddings
        similarities = {}
        for memory_id, embedding in self._memories.items():
            # Normalize memory embedding
            memory_norm = np.linalg.norm(embedding)
            if memory_norm == 0:
                memory_norm = 1e-10
            normalized_embedding = embedding / memory_norm

            # Calculate cosine similarity
            similarity = float(np.dot(normalized_query, normalized_embedding))
            similarities[memory_id] = similarity

        # Apply threshold
        filtered_similarities = {
            memory_id: score
            for memory_id, score in similarities.items()
            if threshold is None or score >= threshold
        }

        # Sort by similarity (descending)
        sorted_memories = sorted(filtered_similarities.items(), key=lambda x: x[1], reverse=True)

        # Take top results
        top_memories = sorted_memories[:limit]

        # Format results
        results = []
        for memory_id, score in top_memories:
            try:
                memory = self.get(memory_id)

                # Determine if this is a hybrid memory
                is_hybrid = self.is_hybrid(memory_id)

                # Get content text
                if isinstance(memory.content, dict) and "text" in memory.content:
                    content_text = memory.content["text"]
                else:
                    content_text = str(memory.content)

                # Create result
                result = {
                    "memory_id": memory_id,
                    "content": content_text,
                    "relevance_score": score,
                    "is_hybrid": is_hybrid,
                }

                # Add metadata
                if memory.metadata:
                    for key, value in memory.metadata.items():
                        result[key] = value

                results.append(result)
            except Exception as e:
                logger.error(f"Error retrieving memory {memory_id}: {e}")

        return results

    def remove(self, memory_id: MemoryID) -> None:
        """
        Remove a memory and its chunks from the store.

        Args:
            memory_id: ID of the memory to remove
        """
        super().remove(memory_id)
        if memory_id in self._hybrid_info:
            del self._hybrid_info[memory_id]

    def clear(self) -> None:
        """Clear all memories and chunks."""
        super().clear()
        self._hybrid_info.clear()

    def get_chunk_count(self) -> int:
        """Get the total number of chunks across all memories."""
        return sum(len(info.chunks) for info in self._hybrid_info.values())

    def get_average_chunks_per_memory(self) -> float:
        """Get the average number of chunks per hybrid memory."""
        if not self._hybrid_info:
            return 0.0
        return self.get_chunk_count() / len(self._hybrid_info)


class HybridMemoryAdapter:
    """
    Adapter for HybridMemoryStore to make it compatible with retrieval strategies.

    This adapter provides methods for efficient access to both full embeddings
    and selective chunks, optimizing memory usage during retrieval.
    """

    def __init__(self, memory_store: HybridMemoryStore):
        """
        Initialize the hybrid memory adapter.

        Args:
            memory_store: The hybrid memory store to adapt
        """
        self.memory_store = memory_store
        self._memory_embeddings_cache = None
        self._memory_metadata_cache = None
        self._memory_ids_cache = None
        self._index_to_id_map = {}
        self._chunk_embeddings_cache = None
        self._chunk_metadata_cache = None
        self._chunk_ids_cache = None
        self._invalidated = True

    def invalidate_cache(self):
        """Invalidate the cache when the memory store changes."""
        self._invalidated = True
        self._memory_embeddings_cache = None
        self._memory_metadata_cache = None
        self._memory_ids_cache = None
        self._chunk_embeddings_cache = None
        self._chunk_metadata_cache = None
        self._chunk_ids_cache = None
        self._index_to_id_map = {}

    @property
    def memory_embeddings(self) -> np.ndarray:
        """
        Get matrix of all memory embeddings.

        Returns:
            Matrix with each row being one memory embedding
        """
        if self._invalidated or self._memory_embeddings_cache is None:
            self._build_memory_cache()
        return self._memory_embeddings_cache

    @property
    def memory_metadata(self) -> list[dict[str, Any]]:
        """
        Get metadata for all memories.

        Returns:
            list of metadata dictionaries for each memory
        """
        if self._invalidated or self._memory_metadata_cache is None:
            self._build_memory_cache()
        return self._memory_metadata_cache

    @property
    def memory_ids(self) -> list[MemoryID]:
        """
        Get all memory IDs.

        Returns:
            list of memory IDs
        """
        if self._invalidated or self._memory_ids_cache is None:
            self._build_memory_cache()
        return self._memory_ids_cache

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

    def _build_memory_cache(self):
        """Build cache of memory embeddings and metadata."""
        # Get all memories
        all_memories = self.memory_store.get_all()

        memory_embeddings = []
        memory_metadata = []
        memory_ids = []

        # Process each memory
        for idx, memory in enumerate(all_memories):
            # Add embedding
            memory_embeddings.append(memory.embedding)

            # Create metadata entry with memory_id
            metadata = {}
            if memory.metadata:
                metadata.update(memory.metadata)

            metadata["memory_id"] = idx
            metadata["original_id"] = memory.id
            metadata["is_hybrid"] = self.memory_store.is_hybrid(memory.id)

            memory_metadata.append(metadata)
            memory_ids.append(memory.id)
            self._index_to_id_map[idx] = memory.id

        # Convert to numpy array for embeddings
        if memory_embeddings:
            self._memory_embeddings_cache = np.stack(memory_embeddings)
        else:
            # Return empty array with proper shape
            dim = 768  # Default dimension
            self._memory_embeddings_cache = np.zeros((0, dim))

        self._memory_metadata_cache = memory_metadata
        self._memory_ids_cache = memory_ids

    def _build_chunk_cache(self):
        """Build cache of chunk embeddings and metadata."""
        # Get all memories that have chunks
        hybrid_memories = {}
        for memory_id in self.memory_store._hybrid_info:
            hybrid_memories[memory_id] = self.memory_store._hybrid_info[memory_id]

        chunk_embeddings = []
        chunk_metadata = []
        chunk_ids = []

        # Process each memory's chunks
        for memory_id, hybrid_info in hybrid_memories.items():
            for chunk in hybrid_info.chunks:
                # Add embedding
                chunk_embeddings.append(chunk.embedding)

                # Create metadata entry
                metadata = chunk.metadata.copy()

                # Add memory ID and chunk info
                metadata["memory_id"] = memory_id
                metadata["chunk_index"] = chunk.chunk_index
                metadata["chunk_text"] = chunk.text

                chunk_metadata.append(metadata)
                chunk_ids.append((memory_id, chunk.chunk_index))

        # Convert to numpy array for embeddings
        if chunk_embeddings:
            self._chunk_embeddings_cache = np.stack(chunk_embeddings)
        else:
            # Return empty array with proper shape
            dim = 768  # Default dimension
            self._chunk_embeddings_cache = np.zeros((0, dim))

        self._chunk_metadata_cache = chunk_metadata
        self._chunk_ids_cache = chunk_ids
        self._invalidated = False

    def get(self, memory_id: MemoryID) -> Memory:
        """
        Retrieve a single memory by ID, translating integer indexes to real memory IDs if needed.
        """
        if isinstance(memory_id, int) or (isinstance(memory_id, str) and memory_id.isdigit()):
            idx = int(memory_id)
            if idx in self._index_to_id_map:
                actual_id = self._index_to_id_map[idx]
                logger.debug(f"Translated index {idx} to ID {actual_id}")
                return self.memory_store.get(actual_id)
            # fallback
            return self.memory_store.get(str(memory_id))
        else:
            return self.memory_store.get(memory_id)

    def add(self, embedding: np.ndarray, content: Any, metadata: dict[str, Any] = None) -> MemoryID:
        """Add a new memory, then invalidate our cache."""
        mem_id = self.memory_store.add(embedding, content, metadata)
        self.invalidate_cache()
        return mem_id

    def get_all(self) -> list[Memory]:
        """Forward to the underlying memory store."""
        return self.memory_store.get_all()

    def search_by_vector(
        self, query_vector: np.ndarray, limit: int = 10, threshold: float = None
    ) -> list[dict]:
        """Forward to the underlying memory store."""
        return self.memory_store.search_by_vector(query_vector, limit, threshold)

    def search_chunks(
        self, query_vector: np.ndarray, limit: int = 10, threshold: float = None
    ) -> list[dict]:
        """Forward to the underlying memory store."""
        return self.memory_store.search_chunks(query_vector, limit, threshold)

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        threshold: float = None,
        keywords: list[str] = None,
    ) -> list[dict]:
        """Search using both full embeddings and chunks with keyword filtering."""
        return self.memory_store.search_hybrid(query_vector, limit, threshold, keywords)

    def _build_cache(self):
        """Build cache of memory embeddings and metadata."""
        try:
            logger.debug("Building hybrid memory adapter cache")

            # Get memories directly from the hybrid store
            hybrid_memories = self.memory_store.get_all()

            if not hybrid_memories:
                self._memory_embeddings_cache = np.zeros((0, 768))  # Default dimension
                self._metadata_dict = []
                self._memory_ids_cache = []
                self._index_to_id_map = {}
                logger.debug("No hybrid memories found")
                return

            # Process memories into cache
            memory_embeddings = []
            memory_metadata = []
            memory_ids = []

            # Process each memory
            for idx, memory in enumerate(hybrid_memories):
                # Extract the full embedding (hybrid stores maintain both full and chunk embeddings)
                if (
                    hasattr(self.memory_store, "_hybrid_info")
                    and memory.id in self.memory_store._hybrid_info
                ):
                    embedding = self.memory_store._hybrid_info[memory.id].full_embedding
                else:
                    embedding = memory.embedding

                memory_embeddings.append(embedding)

                # Create metadata entry with memory_id
                metadata = {}
                if memory.metadata:
                    metadata.update(memory.metadata)

                metadata["memory_id"] = idx
                metadata["original_id"] = memory.id
                metadata["is_hybrid"] = self.memory_store.is_hybrid(memory.id)

                memory_metadata.append(metadata)
                memory_ids.append(memory.id)
                self._index_to_id_map[idx] = memory.id

            # Convert to numpy array for embeddings
            if memory_embeddings:
                self._memory_embeddings_cache = np.stack(memory_embeddings)
            else:
                self._memory_embeddings_cache = np.zeros((0, 768))  # Default dimension

            self._metadata_dict = memory_metadata
            self._memory_ids_cache = memory_ids
            self._invalidated = False

            logger.debug(f"Built hybrid cache with {len(memory_embeddings)} memories")

        except Exception as e:
            logger.error(f"Error building hybrid cache: {e}")
            import traceback

            traceback.print_exc()

            # Initialize empty cache on error
            self._memory_embeddings_cache = np.zeros((0, 768))  # Default dimension
            self._metadata_dict = []
            self._memory_ids_cache = []
            self._index_to_id_map = {}
