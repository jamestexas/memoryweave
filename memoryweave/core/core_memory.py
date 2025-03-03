"""
DEPRECATED: Core memory storage implementation for MemoryWeave.

This module is deprecated. Please use the component-based architecture instead:
- Use memoryweave.storage.memory_store.MemoryStore for memory storage
- Use memoryweave.storage.vector_store.VectorStore for vector storage
- Use memoryweave.storage.activation.ActivationManager for activation management
"""

import warnings
from typing import Any, Optional

import numpy as np

warnings.warn(
    "memoryweave.core.core_memory is deprecated. "
    "Use memoryweave.storage.memory_store and memoryweave.storage.vector_store instead.",
    DeprecationWarning,
    stacklevel=2,
)


class CoreMemory:
    """
    DEPRECATED: Implements core memory storage and basic operations.

    This class is deprecated and will be removed in a future version.
    Please use memoryweave.storage.memory_store.MemoryStore instead.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_memories: int = 1000,
    ):
        """
        Initialize the core memory system.

        Args:
            embedding_dim: Dimension of the contextual embeddings
            max_memories: Maximum number of memory traces to maintain
        """
        warnings.warn(
            "CoreMemory is deprecated and will be removed in a future version. "
            "Use memoryweave.storage.memory_store.MemoryStore instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories

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
        Add a new memory trace to the memory storage.

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

    def update_activation(self, memory_idx: int) -> None:
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

        # Remove metadata
        del self.memory_metadata[least_important_idx]

        return least_important_idx

    def get_memory_count(self) -> int:
        """Get the current number of memories stored."""
        return len(self.memory_metadata)

    def get_memory(self, idx: int) -> tuple[np.ndarray, dict]:
        """
        Get memory embedding and metadata by index.

        Args:
            idx: Index of the memory to retrieve

        Returns:
            Tuple of (embedding, metadata)
        """
        if idx < 0 or idx >= len(self.memory_metadata):
            raise IndexError(f"Memory index {idx} out of range")

        return self.memory_embeddings[idx], self.memory_metadata[idx]

    def get_all_memories(self) -> list[dict[str, Any]]:
        """
        Get all memories with their metadata and indices.

        Returns:
            list of dictionaries containing memory information
        """
        memories = []
        for i in range(len(self.memory_metadata)):
            memories.append({
                "index": i,
                "embedding": self.memory_embeddings[i],
                "metadata": self.memory_metadata[i],
                "activation": float(self.activation_levels[i]),
                "temporal_marker": int(self.temporal_markers[i]),
            })
        return memories
