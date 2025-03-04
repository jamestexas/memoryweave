"""
Component for ART-inspired clustering of memories into categories.
"""

from typing import Any, Optional

import numpy as np

from memoryweave.components.base import Component
from memoryweave.core.category_manager import CategoryManager as CoreCategoryManager


class CategoryManager(Component):
    """
    Component for ART-inspired clustering of memories into categories.

    This component provides dynamic categorization of memories based on
    their embedding similarity, following principles from Adaptive
    Resonance Theory.
    """

    def __init__(self, core_category_manager: Optional[CoreCategoryManager] = None):
        """Initialize with optional existing category manager."""
        self.core_manager = core_category_manager
        self.vigilance_threshold = 0.8
        self.learning_rate = 0.2
        self.embedding_dim = 768

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.vigilance_threshold = config.get("vigilance_threshold", 0.8)
        self.learning_rate = config.get("learning_rate", 0.2)
        self.embedding_dim = config.get("embedding_dim", 768)

        # Create core manager if not provided
        if not self.core_manager:
            self.core_manager = CoreCategoryManager(
                embedding_dim=self.embedding_dim,
                vigilance_threshold=self.vigilance_threshold,
                learning_rate=self.learning_rate,
                dynamic_vigilance=config.get("dynamic_vigilance", False),
                vigilance_strategy=config.get("vigilance_strategy", "decreasing"),
                min_vigilance=config.get("min_vigilance", 0.5),
                max_vigilance=config.get("max_vigilance", 0.9),
                target_categories=config.get("target_categories", 5),
                enable_category_consolidation=config.get("enable_category_consolidation", False),
                consolidation_threshold=config.get("consolidation_threshold", 0.7),
                min_category_size=config.get("min_category_size", 3),
                consolidation_frequency=config.get("consolidation_frequency", 50),
                hierarchical_method=config.get("hierarchical_method", "average"),
            )
        elif hasattr(self.core_manager, "vigilance_threshold"):
            # Update existing core manager parameters if they exist
            self.core_manager.vigilance_threshold = self.vigilance_threshold
            self.core_manager.learning_rate = self.learning_rate

    def assign_to_category(self, embedding: np.ndarray) -> int:
        """
        Assign a memory embedding to a category.

        Args:
            embedding: The memory embedding to categorize

        Returns:
            Index of the assigned category
        """
        if not self.core_manager:
            self.initialize({})  # Initialize with default settings
        return self.core_manager.assign_to_category(embedding)

    def add_memory_category_mapping(self, memory_idx: int, category_idx: int) -> None:
        """
        Add a mapping between a memory and its category.

        Args:
            memory_idx: Index of the memory
            category_idx: Index of the category
        """
        if not self.core_manager:
            self.initialize({})  # Initialize with default settings
        self.core_manager.add_memory_category_mapping(memory_idx, category_idx)

    def update_category_activation(self, category_idx: int) -> None:
        """
        Update activation level for a category that's been accessed.

        Args:
            category_idx: Index of the category to update
        """
        if not self.core_manager:
            return
        self.core_manager.update_category_activation(category_idx)

    def get_category_for_memory(self, memory_idx: int) -> int:
        """
        Get the category index for a memory.

        Args:
            memory_idx: Index of the memory

        Returns:
            Category index for the memory
        """
        if not self.core_manager:
            raise IndexError(
                f"Memory index {memory_idx} out of range - no category manager initialized"
            )
        return self.core_manager.get_category_for_memory(memory_idx)

    def get_memories_for_category(self, category_idx: int) -> list[int]:
        """
        Get all memory indices for a category.

        Args:
            category_idx: Index of the category

        Returns:
            list of memory indices in the category
        """
        if not self.core_manager:
            return []
        return self.core_manager.get_memories_for_category(category_idx)

    def get_category_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between query and all category prototypes.

        This method provides optimized similarity calculation between
        a query embedding and all category prototypes, with activation
        weighting for better retrieval accuracy.

        Args:
            query_embedding: Query embedding vector

        Returns:
            Array of similarity scores for each category
        """
        if not self.core_manager:
            return np.array([], dtype=np.float32)
        return self.core_manager.get_category_similarities(query_embedding)

    def get_category_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the current categories.

        Returns:
            dictionary with category statistics
        """
        # Make sure we have a core manager
        if not self.core_manager:
            return {"num_categories": 0, "memories_per_category": {}, "category_activations": {}}
        return self.core_manager.get_category_statistics()

    def consolidate_categories(self, threshold: Optional[float] = None) -> int:
        """
        Trigger category consolidation with an optional custom threshold.

        Args:
            threshold: Custom similarity threshold

        Returns:
            Number of categories after consolidation
        """
        if not self.core_manager:
            return 0
        return self.core_manager.consolidate_categories_manually(threshold)

    def add_to_category(self, memory_id: str, embedding: np.ndarray) -> int:
        """
        Add a memory to a category based on its embedding.

        Args:
            memory_id: ID of the memory to add (as string)
            embedding: The memory embedding to categorize

        Returns:
            Index of the assigned category
        """
        if not self.core_manager:
            # Initialize with the correct embedding dimension from the provided embedding
            self.embedding_dim = embedding.shape[0]
            self.initialize({"embedding_dim": self.embedding_dim})

        # First assign to a category
        category_idx = self.core_manager.assign_to_category(embedding)

        # Convert string ID to integer for core manager
        # We'll use a hash of the string to get a reasonably unique integer
        # This is better than just using the string length or similar
        memory_idx = hash(memory_id) % (2**31 - 1)  # Keep within positive int range

        # Then add the mapping
        self.core_manager.add_memory_category_mapping(memory_idx, category_idx)

        # Store the mapping between string ID and integer ID if needed later
        if not hasattr(self, "_string_to_int_map"):
            self._string_to_int_map = {}
        self._string_to_int_map[memory_id] = memory_idx

        return category_idx
