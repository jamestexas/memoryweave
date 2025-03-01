"""Memory store implementation for MemoryWeave.

This module provides the base implementation of the memory store,
which handles storage and retrieval of memories.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from memoryweave.interfaces.memory import (
    EmbeddingVector,
    IMemoryStore,
    Memory,
    MemoryContent,
    MemoryID,
)


@dataclass
class MemoryMetadata:
    """Metadata for a memory in the store."""

    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    activation: float = 0.0
    access_count: int = 0
    user_metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore(IMemoryStore):
    """Basic implementation of a memory store."""

    def __init__(self):
        """Initialize the memory store."""
        self._memories: Dict[MemoryID, EmbeddingVector] = {}
        self._contents: Dict[MemoryID, MemoryContent] = {}
        self._metadata: Dict[MemoryID, MemoryMetadata] = {}
        self._next_id: int = 0

    def add(
        self, embedding: EmbeddingVector, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryID:
        """Add a memory and return its ID."""
        memory_id = self._generate_id()
        self._memories[memory_id] = embedding
        self._contents[memory_id] = {"text": content, "metadata": {}}

        # Initialize metadata
        memory_metadata = MemoryMetadata()
        if metadata:
            memory_metadata.user_metadata.update(metadata)
        self._metadata[memory_id] = memory_metadata

        return memory_id

    def get(self, memory_id: MemoryID) -> Memory:
        """Retrieve a memory by ID."""
        if memory_id not in self._memories:
            raise KeyError(f"Memory with ID {memory_id} not found")

        # Update access metadata
        metadata = self._metadata[memory_id]
        metadata.last_accessed = time.time()
        metadata.access_count += 1

        return Memory(
            id=memory_id,
            embedding=self._memories[memory_id],
            content=self._contents[memory_id],
            metadata=metadata.user_metadata,
        )

    def get_all(self) -> List[Memory]:
        """Retrieve all memories."""
        return [self.get(memory_id) for memory_id in self._memories.keys()]

    def get_embeddings(self) -> np.ndarray:
        """Get all embeddings as a matrix."""
        if not self._memories:
            # Return empty array with correct shape
            return np.array([]).reshape(0, 0)

        # Stack all embeddings into a matrix
        return np.stack(list(self._memories.values()))

    def get_ids(self) -> List[MemoryID]:
        """Get all memory IDs."""
        return list(self._memories.keys())

    def update_activation(self, memory_id: MemoryID, activation_delta: float) -> None:
        """Update activation level of a memory."""
        if memory_id not in self._metadata:
            raise KeyError(f"Memory with ID {memory_id} not found")

        self._metadata[memory_id].activation += activation_delta

    def update_metadata(self, memory_id: MemoryID, metadata: Dict[str, Any]) -> None:
        """Update metadata of a memory."""
        if memory_id not in self._metadata:
            raise KeyError(f"Memory with ID {memory_id} not found")

        self._metadata[memory_id].user_metadata.update(metadata)

    def remove(self, memory_id: MemoryID) -> None:
        """Remove a memory from the store."""
        if memory_id not in self._memories:
            raise KeyError(f"Memory with ID {memory_id} not found")

        del self._memories[memory_id]
        del self._contents[memory_id]
        del self._metadata[memory_id]

    def clear(self) -> None:
        """Clear all memories from the store."""
        self._memories.clear()
        self._contents.clear()
        self._metadata.clear()
        self._next_id = 0

    def consolidate(self, max_memories: int) -> List[MemoryID]:
        """Consolidate memories to stay within capacity."""
        if len(self._memories) <= max_memories:
            # No consolidation needed
            return []

        # Sort memories by activation level (lowest first)
        memories_by_activation = sorted(self._metadata.items(), key=lambda x: x[1].activation)

        # Determine how many memories to remove
        num_to_remove = len(self._memories) - max_memories
        memories_to_remove = [memory_id for memory_id, _ in memories_by_activation[:num_to_remove]]

        # Remove the memories
        for memory_id in memories_to_remove:
            self.remove(memory_id)

        return memories_to_remove

    def _generate_id(self) -> MemoryID:
        """Generate a unique ID for a new memory."""
        memory_id = str(self._next_id)
        self._next_id += 1
        return memory_id
