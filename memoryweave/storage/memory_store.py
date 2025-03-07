"""Standard memory store implementation with improved ID handling."""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, Memory, MemoryContent, MemoryID
from memoryweave.storage.base_store import StandardMemoryStore


@dataclass
class MemoryMetadata:
    """Metadata for a memory in the store."""

    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    activation: float = 0.0
    access_count: int = 0
    user_metadata: dict[str, Any] = field(default_factory=dict)


class StandardMemoryStore(StandardMemoryStore):
    """Standard memory store implementation with improved ID handling."""

    def __init__(self):
        """Initialize the memory store."""
        super().__init__()
        self._memories: dict[MemoryID, EmbeddingVector] = {}
        self._contents: dict[MemoryID, MemoryContent] = {}
        self._metadata: dict[MemoryID, MemoryMetadata] = {}
        self.component_id = "standard_memory_store"

    def add(
        self, embedding: EmbeddingVector, content: Any, metadata: dict[str, Any] | None = None
    ) -> MemoryID:
        """Add a memory and return its ID."""
        memory_id = self._generate_id()
        return self._add_internal(memory_id, embedding, content, metadata)

    def add_with_id(
        self,
        memory_id: MemoryID,
        embedding: EmbeddingVector,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryID:
        """Add a memory with a specific ID."""
        return self._add_internal(memory_id, embedding, content, metadata)

    def _add_internal(
        self,
        memory_id: MemoryID,
        embedding: EmbeddingVector,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryID:
        """Internal method to add a memory with the given ID."""
        # Store the embedding
        self._memories[memory_id] = embedding

        # Store the content in a standardized format
        if isinstance(content, dict) and "text" in content:
            self._contents[memory_id] = content
        else:
            self._contents[memory_id] = {"text": str(content), "metadata": {}}

        # Initialize metadata
        memory_metadata = MemoryMetadata()
        if metadata:
            memory_metadata.user_metadata.update(metadata)
        self._metadata[memory_id] = memory_metadata

        return memory_id

    def get(self, memory_id: MemoryID) -> Memory:
        """Retrieve a memory by ID."""
        memory_id = self._resolve_id(memory_id)

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

    def get_all(self) -> list[Memory]:
        """Retrieve all memories."""
        return [self.get(memory_id) for memory_id in self._memories.keys()]

    def get_embeddings(self) -> np.ndarray:
        """Get all embeddings as a matrix."""
        if not self._memories:
            # Return empty array with correct shape
            return np.array([]).reshape(0, 0)

        # Stack all embeddings into a matrix
        return np.stack(list(self._memories.values()))

    def get_ids(self) -> list[MemoryID]:
        """Get all memory IDs."""
        return list(self._memories.keys())

    def update_activation(self, memory_id: MemoryID, activation_delta: float) -> None:
        """Update activation level of a memory."""
        memory_id = self._resolve_id(memory_id)

        if memory_id not in self._metadata:
            raise KeyError(f"Memory with ID {memory_id} not found")

        self._metadata[memory_id].activation += activation_delta

    def update_metadata(self, memory_id: MemoryID, metadata: dict[str, Any]) -> None:
        """Update metadata of a memory."""
        memory_id = self._resolve_id(memory_id)

        if memory_id not in self._metadata:
            raise KeyError(f"Memory with ID {memory_id} not found")

        self._metadata[memory_id].user_metadata.update(metadata)

    def remove(self, memory_id: MemoryID) -> None:
        """Remove a memory from the store."""
        memory_id = self._resolve_id(memory_id)

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

    def consolidate(self, max_memories: int) -> list[MemoryID]:
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

    def add_multiple(self, memories: list[Memory]) -> list[MemoryID]:
        """Add multiple memories at once."""
        memory_ids = []

        for memory in memories:
            # If the memory already has an ID, use it
            if memory.id is not None:
                memory_id = self.add_with_id(
                    memory.id, memory.embedding, memory.content, memory.metadata
                )
            else:
                memory_id = self.add(memory.embedding, memory.content, memory.metadata)

            memory_ids.append(memory_id)

        return memory_ids
