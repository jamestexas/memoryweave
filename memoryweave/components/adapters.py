"""
Adapters for integrating core components with the pipeline architecture.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from memoryweave.components.base import Component, RetrievalComponent
from memoryweave.components.category_manager import CategoryManager
from memoryweave.core.category_manager import CategoryManager as CoreCategoryManager
from memoryweave.core.contextual_memory import ContextualMemory


class CoreRetrieverAdapter(RetrievalComponent):
    """
    Adapter for using the core MemoryRetriever in the pipeline architecture.

    This adapter wraps the core MemoryRetriever and exposes it through the
    component interface defined by the pipeline architecture.
    """

    def __init__(
        self,
        memory: ContextualMemory,
        default_top_k: int = 5,
        confidence_threshold: float = 0.0,
    ):
        """
        Initialize the adapter.

        Args:
            memory: The memory instance to use for retrieval
            default_top_k: Default number of results to retrieve
            confidence_threshold: Default confidence threshold for retrieval
        """
        self.memory = memory
        self.default_top_k = default_top_k
        self.confidence_threshold = confidence_threshold
        self.activation_boost = True
        self.use_categories = True

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        self.confidence_threshold = config.get("confidence_threshold", self.confidence_threshold)
        self.default_top_k = config.get("top_k", self.default_top_k)
        self.use_categories = config.get("use_categories", self.use_categories)
        self.activation_boost = config.get("activation_boost", self.activation_boost)

    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query to retrieve relevant memories.

        Args:
            query: The query string
            context: Context containing query_embedding, etc.

        Returns:
            Updated context with results
        """
        # Get query embedding from context
        query_embedding = context.get("query_embedding")
        if query_embedding is None:
            # Try to get embedding model and create embedding
            embedding_model = context.get("embedding_model")
            if embedding_model:
                query_embedding = embedding_model.encode(query)

        # If still no embedding, return empty results
        if query_embedding is None:
            return {"results": []}

        # Get top_k from context or use default
        top_k = context.get("top_k", self.default_top_k)

        # Apply query type adaptation if available
        adapted_params = context.get("adapted_retrieval_params", {})
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)

        # Retrieve memories
        results = self.memory.retrieve_memories(
            query_embedding=query_embedding,
            top_k=top_k,
            activation_boost=self.activation_boost,
            use_categories=self.use_categories,
            confidence_threshold=confidence_threshold,
        )

        # Format results as dictionaries
        formatted_results = []
        for idx, score, metadata in results:
            formatted_results.append({"memory_id": idx, "relevance_score": score, **metadata})

        return {"results": formatted_results}


class CategoryAdapter(Component):
    """
    Adapter for integrating CategoryManager components within the pipeline.

    This adapter provides bidirectional compatibility between the legacy
    and new category systems, ensuring proper memory ID mapping between
    the systems.
    """

    def __init__(
        self,
        core_category_manager: Optional[CoreCategoryManager] = None,
        component_category_manager: Optional[CategoryManager] = None,
    ):
        """
        Initialize the category adapter.

        Args:
            core_category_manager: Optional core category manager instance
            component_category_manager: Optional component category manager instance
        """
        self.core_manager = core_category_manager
        self.component_manager = component_category_manager

        # Create default managers if not provided
        if not self.component_manager and not self.core_manager:
            self.core_manager = CoreCategoryManager()
            self.component_manager = CategoryManager(self.core_manager)
        elif not self.component_manager and self.core_manager:
            self.component_manager = CategoryManager(self.core_manager)
        elif not self.core_manager and self.component_manager:
            self.core_manager = self.component_manager.core_manager

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the component with configuration.

        Args:
            config: Configuration dictionary
        """
        # Initialize component manager if it exists
        if self.component_manager:
            self.component_manager.initialize(config)

        # If core manager exists but component doesn't, create component
        if self.core_manager and not self.component_manager:
            self.component_manager = CategoryManager(self.core_manager)
            self.component_manager.initialize(config)

    def assign_to_category(self, embedding: np.ndarray) -> int:
        """
        Assign a memory embedding to a category.

        Args:
            embedding: The memory embedding to categorize

        Returns:
            Index of the assigned category
        """
        return self.component_manager.assign_to_category(embedding)

    def add_memory_category_mapping(self, memory_idx: int, category_idx: int) -> None:
        """
        Add a mapping between a memory and its category.

        Args:
            memory_idx: Index of the memory
            category_idx: Index of the category
        """
        self.component_manager.add_memory_category_mapping(memory_idx, category_idx)

    def get_category_for_memory(self, memory_idx: int) -> int:
        """
        Get the category index for a memory.

        Args:
            memory_idx: Index of the memory

        Returns:
            Category index for the memory
        """
        return self.component_manager.get_category_for_memory(memory_idx)

    def get_memories_for_category(self, category_idx: int) -> List[int]:
        """
        Get all memory indices for a category.

        Args:
            category_idx: Index of the category

        Returns:
            List of memory indices in the category
        """
        return self.component_manager.get_memories_for_category(category_idx)

    def get_category_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between query and all category prototypes.

        Args:
            query_embedding: Query embedding vector

        Returns:
            Array of similarity scores for each category
        """
        return self.component_manager.get_category_similarities(query_embedding)

    def update_category_activation(self, category_idx: int) -> None:
        """
        Update activation level for a category that's been accessed.

        Args:
            category_idx: Index of the category to update
        """
        self.component_manager.update_category_activation(category_idx)

    def consolidate_categories(self, threshold: Optional[float] = None) -> int:
        """
        Trigger category consolidation with an optional custom threshold.

        Args:
            threshold: Custom similarity threshold

        Returns:
            Number of categories after consolidation
        """
        return self.component_manager.consolidate_categories(threshold)

    def get_category_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current categories.

        Returns:
            Dictionary with category statistics
        """
        return self.component_manager.get_category_statistics()
