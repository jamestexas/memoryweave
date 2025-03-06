"""
Adapter components for integrating different memory systems.
"""

from typing import Any, Optional

import numpy as np

from memoryweave.components.base import Component


class CoreRetrieverAdapter(Component):
    """
    Adapter for core retriever to be used in the component architecture.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        pass

    def process_query(self, query_embedding: np.ndarray, **kwargs) -> dict[str, Any]:
        """Process a query embedding and return relevant memories."""
        # This is a placeholder for the actual implementation
        return {"memories": []}


class CategoryAdapter(Component):
    """
    Adapter for category-based memory operations.
    """

    def __init__(self, category_manager: Optional[Any] = None):
        """Initialize with optional existing category manager."""
        self.category_manager = category_manager

        # If no category manager provided, create a new one
        if self.category_manager is None:
            from memoryweave.components.category_manager import CategoryManager

            self.category_manager = CategoryManager()

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        if hasattr(self.category_manager, "initialize"):
            self.category_manager.initialize(config)

    def assign_to_category(self, embedding: np.ndarray) -> int:
        """Assign a memory embedding to a category."""
        if hasattr(self.category_manager, "assign_to_category"):
            return self.category_manager.assign_to_category(embedding)
        return -1

    def add_memory_category_mapping(self, memory_idx: int, category_idx: int) -> None:
        """Add a mapping between memory and category."""
        if hasattr(self.category_manager, "add_memory_category_mapping"):
            self.category_manager.add_memory_category_mapping(memory_idx, category_idx)

    def get_category_for_memory(self, memory_idx: int) -> int:
        """Get the category for a memory."""
        if hasattr(self.category_manager, "get_category_for_memory"):
            return self.category_manager.get_category_for_memory(memory_idx)
        return -1

    def get_memories_for_category(self, category_idx: int) -> list[int]:
        """Get all memories in a category."""
        if hasattr(self.category_manager, "get_memories_for_category"):
            return self.category_manager.get_memories_for_category(category_idx)
        return []

    def get_category_similarities(self, embedding: np.ndarray) -> list[tuple[int, float]]:
        """Get similarities between an embedding and all categories."""
        if hasattr(self.category_manager, "get_category_similarities"):
            return self.category_manager.get_category_similarities(embedding)
        return []

    def update_category_activation(self, category_idx: int, activation_delta: float) -> None:
        """Update the activation level of a category."""
        if hasattr(self.category_manager, "update_category_activation"):
            self.category_manager.update_category_activation(category_idx, activation_delta)

    def consolidate_categories(self, threshold: Optional[float] = None) -> int:
        """Consolidate similar categories."""
        if hasattr(self.category_manager, "consolidate_categories"):
            return self.category_manager.consolidate_categories(threshold)
        return 0

    def get_category_statistics(self) -> dict[str, Any]:
        """Get statistics about the current categories."""
        if hasattr(self.category_manager, "get_category_statistics"):
            return self.category_manager.get_category_statistics()
        return {"num_categories": 0}
