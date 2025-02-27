"""
Core implementation of the MemoryWeave contextual fabric.
"""

from typing import Optional

import numpy as np


class ContextualMemory:
    """
    Implements a contextual fabric approach to memory management.
    Rather than storing discrete memory nodes, this captures rich
    contextual signatures of information with associative patterns.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_memories: int = 1000,
        activation_threshold: float = 0.7,
    ):
        """
        Initialize the contextual memory system.

        Args:
            embedding_dim: Dimension of the contextual embeddings
            max_memories: Maximum number of memory traces to maintain
            activation_threshold: Threshold for memory activation
        """
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.activation_threshold = activation_threshold

        # Memory fabric stores both the embeddings and their associated metadata
        self.memory_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
        self.memory_metadata = []

        # Activation levels track recent access/relevance
        self.activation_levels = np.zeros(0, dtype=np.float32)

        # Temporal markers to capture sequence and episodic structure
        self.temporal_markers = np.zeros(0, dtype=np.int64)
        self.current_time = 0

    def add_memory(
        self,
        embedding: np.ndarray,
        text: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Add a new memory trace to the contextual fabric.

        Args:
            embedding: The contextual embedding of the memory
            text: The text content of the memory
            metadata: Additional metadata for the memory

        Returns:
            Index of the newly added memory
        """
        if metadata is None:
            metadata = {}

        # Update time counter
        self.current_time += 1

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Add new memory
        self.memory_embeddings = np.vstack([self.memory_embeddings, embedding])

        # Store metadata and text
        full_metadata = {
            "text": text,
            "created_at": self.current_time,
            "access_count": 0,
            **metadata,
        }
        self.memory_metadata.append(full_metadata)

        # Initialize activation and temporal marker
        self.activation_levels = np.append(self.activation_levels, 1.0)
        self.temporal_markers = np.append(self.temporal_markers, self.current_time)

        # Manage memory capacity if needed
        if len(self.memory_metadata) > self.max_memories:
            self._consolidate_memories()

        return len(self.memory_metadata) - 1

    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        activation_boost: bool = True,
    ) -> list[tuple[int, float, dict]]:
        """
        Retrieve relevant memories based on contextual similarity.

        Args:
            query_embedding: Embedding of the query context
            top_k: Number of memories to retrieve
            activation_boost: Whether to boost by activation level

        Returns:
            list of (memory_idx, similarity_score, metadata) tuples
        """
        if len(self.memory_metadata) == 0:
            return []

        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compute similarities
        similarities = np.dot(self.memory_embeddings, query_embedding)

        # Apply activation boosting if enabled
        if activation_boost:
            similarities = similarities * self.activation_levels

        # Get top-k indices
        if top_k >= len(similarities):
            top_indices = np.argsort(-similarities)
        else:
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Update activation levels for retrieved memories
        for idx in top_indices:
            self._update_activation(idx)

        # Return results with metadata
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.activation_threshold:
                results.append((int(idx), float(similarities[idx]), self.memory_metadata[idx]))

        return results

    def _update_activation(self, memory_idx: int) -> None:
        """
        Update activation level for a memory that's been accessed.

        Args:
            memory_idx: Index of the memory to update
        """
        # Increase activation for accessed memory
        self.activation_levels[memory_idx] = min(1.0, self.activation_levels[memory_idx] + 0.2)

        # Update access metadata
        self.memory_metadata[memory_idx]["access_count"] += 1
        self.memory_metadata[memory_idx]["last_accessed"] = self.current_time

        # Decay other activations slightly
        decay_mask = np.ones_like(self.activation_levels, dtype=bool)
        decay_mask[memory_idx] = False
        self.activation_levels[decay_mask] *= 0.95

    def _consolidate_memories(self) -> None:
        """
        Consolidate memories when capacity is reached,
        using activation levels and temporal factors.
        """
        # Compute a combined score for memory importance
        # This considers both activation and recency
        importance = self.activation_levels + 0.2 * (self.temporal_markers / self.current_time)

        # Find the least important memory
        least_important_idx = np.argmin(importance)

        # Remove the least important memory
        self.memory_embeddings = np.delete(self.memory_embeddings, least_important_idx, axis=0)
        self.activation_levels = np.delete(self.activation_levels, least_important_idx)
        self.temporal_markers = np.delete(self.temporal_markers, least_important_idx)
        del self.memory_metadata[least_important_idx]
