"""Base memory store with improved ID handling."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from memoryweave.interfaces.memory import EmbeddingVector, Memory, MemoryID

logger = logging.getLogger(__name__)


class StandardMemoryStore(ABC):
    """Base class for all memory stores with consistent ID handling."""

    def __init__(self):
        """Initialize the memory store."""
        self._next_id = 0
        self._id_mapping = {}  # Maps external IDs to internal IDs
        self.component_id = "base_memory_store"

    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        return self.component_id

    def get_type(self):
        """Get the type of this component."""
        from memoryweave.interfaces.pipeline import ComponentType

        return ComponentType.MEMORY_STORE

    @abstractmethod
    def add(
        self,
        embedding: EmbeddingVector,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryID:
        """Add a memory to the store."""
        pass

    def add_with_id(
        self,
        memory_id: MemoryID,
        embedding: EmbeddingVector,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryID:
        """Add a memory with a specific ID."""
        # Default implementation - can be overridden by subclasses
        raise NotImplementedError("add_with_id not implemented by this store")

    @abstractmethod
    def get(self, memory_id: MemoryID) -> Memory:
        """Get a memory by ID."""
        pass

    @abstractmethod
    def get_all(self) -> list[Memory]:
        """Get all memories."""
        pass

    @abstractmethod
    def remove(self, memory_id: MemoryID) -> None:
        """Remove a memory from the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories from the store."""
        pass

    def update_metadata(self, memory_id: MemoryID, metadata: dict[str, Any]) -> None:
        """Update metadata for a memory."""
        # Default implementation - can be overridden by subclasses
        raise NotImplementedError("update_metadata not implemented by this store")

    def consolidate(self, max_memories: int) -> list[MemoryID]:
        """Consolidate memories to stay within capacity."""
        # Default implementation - can be overridden by subclasses
        return []

    def _generate_id(self) -> MemoryID:
        """Generate a new unique ID."""
        memory_id = str(self._next_id)
        self._next_id += 1
        return memory_id

    def _map_id(self, original_id: MemoryID, internal_id: Any) -> None:
        """Map an original ID to an internal ID."""
        self._id_mapping[original_id] = internal_id

    def _get_internal_id(self, external_id: MemoryID) -> Any:
        """Get internal ID from external ID."""
        if external_id in self._id_mapping:
            return self._id_mapping[external_id]
        return external_id  # Fallback to using the external ID directly

    def _resolve_id(self, memory_id: str | int) -> MemoryID:
        """Resolve a memory ID (handle both string and integer IDs)."""
        if not isinstance(memory_id, str):
            logging.debug(f"Memory ID {memory_id} is not a string, converting to string")
            memory_id = str(memory_id)
        if not memory_id.isdigit():
            logging.debug(f"Memory ID {memory_id} is not a number, using as is")
        return memory_id
