"""
DEPRECATED: Implementation of MemoryWeave's contextual memory system.

This module is deprecated. Please use the component-based architecture instead:
- Use memoryweave.components.memory_manager.MemoryManager for memory management
- Use memoryweave.components.retriever.Retriever for memory retrieval
- Use memoryweave.components.category_manager.CategoryManager for category management

See MIGRATION_GUIDE.md for detailed migration instructions.
"""

import warnings
from typing import Literal, Optional

import numpy as np

# These imports will trigger their own deprecation warnings
from memoryweave.core.category_manager import CategoryManager
from memoryweave.core.core_memory import CoreMemory
from memoryweave.core.memory_retriever import MemoryRetriever

warnings.warn(
    "memoryweave.core.contextual_memory is deprecated. "
    "Use memoryweave.components.memory_manager.MemoryManager instead.",
    DeprecationWarning,
    stacklevel=2,
)


class ContextualMemory:
    """
    DEPRECATED: Implements a contextual fabric approach to memory management.

    This class is deprecated and will be removed in a future version.
    Please use memoryweave.components.memory_manager.MemoryManager instead.
    
    This class combines CoreMemory, CategoryManager, and MemoryRetriever
    to provide a unified interface for memory operations while maintaining
    the original API of the monolithic implementation.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_memories: int = 1000,
        activation_threshold: float = 0.5,
        use_art_clustering: bool = False,
        vigilance_threshold: float = 0.85,
        learning_rate: float = 0.1,
        dynamic_vigilance: bool = False,
        vigilance_strategy: Literal[
            "decreasing", "increasing", "category_based", "density_based"
        ] = "decreasing",
        min_vigilance: float = 0.5,
        max_vigilance: float = 0.9,
        target_categories: int = 5,
        enable_category_consolidation: bool = False,
        consolidation_threshold: float = 0.7,
        min_category_size: int = 3,
        consolidation_frequency: int = 50,
        hierarchical_method: Literal["single", "complete", "average", "weighted"] = "average",
        default_confidence_threshold: float = 0.0,
        adaptive_retrieval: bool = False,
        semantic_coherence_check: bool = False,
        coherence_threshold: float = 0.2,
        use_ann: bool = True,  # Enable Approximate Nearest Neighbor by default
    ):
        """
        Initialize the contextual memory system.

        Args:
            embedding_dim: Dimension of the contextual embeddings
            max_memories: Maximum number of memory traces to maintain
            activation_threshold: Threshold for memory activation
            use_art_clustering: Whether to use ART-inspired clustering
            vigilance_threshold: Initial threshold for creating new categories (ART vigilance)
            learning_rate: Rate at which category prototypes are updated
            dynamic_vigilance: Whether to use dynamic vigilance adjustment
            vigilance_strategy: Strategy for adjusting vigilance
            min_vigilance: Minimum vigilance threshold for dynamic adjustment
            max_vigilance: Maximum vigilance threshold for dynamic adjustment
            target_categories: Target number of categories for category_based strategy
            enable_category_consolidation: Whether to enable periodic category consolidation
            consolidation_threshold: Similarity threshold for merging categories
            min_category_size: Minimum number of memories per category before consolidation
            consolidation_frequency: How often to run consolidation (every N memories added)
            hierarchical_method: Method for hierarchical clustering linkage
            default_confidence_threshold: Default minimum similarity score for memory retrieval
            adaptive_retrieval: Whether to use adaptive k selection
            semantic_coherence_check: Whether to check semantic coherence of retrieved memories
            coherence_threshold: Threshold for semantic coherence between memories
            use_ann: Whether to use Approximate Nearest Neighbor search for efficient retrieval at scale
        """
        warnings.warn(
            "ContextualMemory is deprecated and will be removed in a future version. "
            "Use memoryweave.components.memory_manager.MemoryManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Initialize the core memory storage
        self.core_memory = CoreMemory(
            embedding_dim=embedding_dim,
            max_memories=max_memories,
        )
        
        # Store key parameters
        self.use_ann = use_ann

        # Initialize the category manager if ART clustering is enabled
        self.category_manager = None
        if use_art_clustering:
            self.category_manager = CategoryManager(
                embedding_dim=embedding_dim,
                vigilance_threshold=vigilance_threshold,
                learning_rate=learning_rate,
                dynamic_vigilance=dynamic_vigilance,
                vigilance_strategy=vigilance_strategy,
                min_vigilance=min_vigilance,
                max_vigilance=max_vigilance,
                target_categories=target_categories,
                enable_category_consolidation=enable_category_consolidation,
                consolidation_threshold=consolidation_threshold,
                min_category_size=min_category_size,
                consolidation_frequency=consolidation_frequency,
                hierarchical_method=hierarchical_method,
            )

        # Initialize the memory retriever
        self.memory_retriever = MemoryRetriever(
            core_memory=self.core_memory,
            category_manager=self.category_manager,
            default_confidence_threshold=default_confidence_threshold,
            adaptive_retrieval=adaptive_retrieval,
            semantic_coherence_check=semantic_coherence_check,
            coherence_threshold=coherence_threshold,
            use_ann=self.use_ann,
        )

        # Store configuration
        self.embedding_dim = embedding_dim
        self.use_art_clustering = use_art_clustering
        self.activation_threshold = activation_threshold  # Store this for reference

        # Expose key attributes from component classes
        # to maintain the original API
        self._setup_property_proxies()

    def _setup_property_proxies(self):
        """Set up property proxies to maintain the original API."""
        # Core memory properties
        self.memory_embeddings = self.core_memory.memory_embeddings
        self.memory_metadata = self.core_memory.memory_metadata
        self.activation_levels = self.core_memory.activation_levels
        self.temporal_markers = self.core_memory.temporal_markers
        self.current_time = self.core_memory.current_time
        self.max_memories = self.core_memory.max_memories

        # Category manager properties
        if self.category_manager:
            self.category_prototypes = self.category_manager.category_prototypes
            self.memory_categories = self.category_manager.memory_categories
            self.category_activations = self.category_manager.category_activations
            self.vigilance_threshold = self.category_manager.vigilance_threshold

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
        # Add memory to core storage
        memory_idx = self.core_memory.add_memory(embedding, text, metadata)

        # If using ART clustering, assign to a category
        if self.use_art_clustering and self.category_manager:
            category_idx = self.category_manager.assign_to_category(embedding)
            self.category_manager.add_memory_category_mapping(memory_idx, category_idx)

        # Update property proxies
        self._setup_property_proxies()

        return memory_idx

    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        activation_boost: bool = True,
        use_categories: bool = None,
        confidence_threshold: float = None,
        max_k_override: bool = False,
    ) -> list[tuple[int, float, dict]]:
        """
        Retrieve relevant memories based on contextual similarity.

        Args:
            query_embedding: Embedding of the query context
            top_k: Number of memories to retrieve
            activation_boost: Whether to boost by activation level
            use_categories: Whether to use category-based retrieval
            confidence_threshold: Minimum similarity score threshold for inclusion
            max_k_override: Whether to return exactly top_k results

        Returns:
            list of (memory_idx, similarity_score, metadata) tuples
        """
        # Delegate to the memory retriever
        results = self.memory_retriever.retrieve_memories(
            query_embedding=query_embedding,
            top_k=top_k,
            activation_boost=activation_boost,
            use_categories=use_categories
            if use_categories is not None
            else self.use_art_clustering,
            confidence_threshold=confidence_threshold,
            max_k_override=max_k_override,
        )

        # Update property proxies after retrieval (activation levels may have changed)
        self._setup_property_proxies()

        return results

    def get_category_statistics(self) -> dict:
        """
        Get statistics about the current categories.

        Returns:
            dictionary with category statistics
        """
        if not self.use_art_clustering or not self.category_manager:
            return {"num_categories": 0}

        # Pass necessary data to category manager for statistics
        # This is needed because the category manager doesn't have direct
        # access to activation_levels
        self.category_manager.activation_levels = self.activation_levels
        self.category_manager.memory_embeddings = self.memory_embeddings

        return self.category_manager.get_category_statistics()

    def category_similarity_matrix(self) -> np.ndarray:
        """
        Get the similarity matrix between all category prototypes.

        Returns:
            2D numpy array of similarity scores
        """
        if (
            not self.use_art_clustering
            or not self.category_manager
            or len(self.category_manager.category_prototypes) < 2
        ):
            return np.array([])

        return np.dot(
            self.category_manager.category_prototypes, self.category_manager.category_prototypes.T
        )

    def consolidate_categories_manually(self, threshold: float = None) -> int:
        """
        Manually trigger category consolidation with an optional custom threshold.

        Args:
            threshold: Custom similarity threshold

        Returns:
            Number of categories after consolidation
        """
        if not self.use_art_clustering or not self.category_manager:
            return 0

        result = self.category_manager.consolidate_categories_manually(threshold)

        # Update property proxies after consolidation
        self._setup_property_proxies()

        return result

    def _update_activation(self, memory_idx: int) -> None:
        """
        Update activation level for a memory that's been accessed.

        Args:
            memory_idx: Index of the memory to update
        """
        self.core_memory.update_activation(memory_idx)

        # Update property proxies
        self._setup_property_proxies()
