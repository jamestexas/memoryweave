"""
Adapter that provides a ContextualMemory-compatible interface using the new component architecture.

This adapter allows for a smooth transition from the core.contextual_memory.ContextualMemory
to the new component-based architecture by providing the same interface.
"""

import logging
import warnings
from typing import Optional

import numpy as np

from memoryweave.components.category_manager import CategoryManager
from memoryweave.storage.adapter import MemoryAdapter
from memoryweave.storage.memory_store import StandardMemoryStore
from memoryweave.storage.vector_search import create_vector_search_provider

logger = logging.getLogger(__name__)


class ContextualMemoryAdapter:
    """
    Adapter that provides a ContextualMemory-compatible interface using the new component architecture.

    This class mimics the interface of core.contextual_memory.ContextualMemory but uses
    the new component-based architecture internally.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_memories: int = 1000,
        activation_threshold: float = 0.5,
        use_art_clustering: bool = False,
        vigilance_threshold: float = 0.85,
        learning_rate: float = 0.1,
        default_confidence_threshold: float = 0.0,
        adaptive_retrieval: bool = False,
        semantic_coherence_check: bool = False,
        coherence_threshold: float = 0.2,
        use_ann: bool = True,
        **kwargs,
    ):
        """
        Initialize the contextual memory adapter.

        Args:
            embedding_dim: Dimension of the contextual embeddings
            max_memories: Maximum number of memory traces to maintain
            activation_threshold: Threshold for memory activation
            use_art_clustering: Whether to use ART-inspired clustering
            vigilance_threshold: Initial threshold for creating new categories (ART vigilance)
            learning_rate: Rate at which category prototypes are updated
            default_confidence_threshold: Default minimum similarity score for memory retrieval
            adaptive_retrieval: Whether to use adaptive k selection
            semantic_coherence_check: Whether to check semantic coherence of retrieved memories
            coherence_threshold: Threshold for semantic coherence between memories
            use_ann: Whether to use Approximate Nearest Neighbor search
            **kwargs: Additional arguments for backward compatibility
        """
        warnings.warn(
            "ContextualMemoryAdapter is a transitional adapter. Consider using the component-based architecture directly.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Create memory store
        self.memory_store = StandardMemoryStore()

        # Create vector search provider
        vector_search = None
        if use_ann:
            vector_search = create_vector_search_provider(
                "faiss", dimension=embedding_dim, use_quantization=False
            )

        # Create memory adapter
        self.memory_adapter = MemoryAdapter(self.memory_store, vector_search)

        # Create category manager if ART clustering is enabled
        self.category_manager = None
        if use_art_clustering:
            self.category_manager = CategoryManager()
            self.category_manager.initialize(
                {
                    "vigilance_threshold": vigilance_threshold,
                    "learning_rate": learning_rate,
                    "embedding_dim": embedding_dim,
                }
            )

        # Store configuration
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.activation_threshold = activation_threshold
        self.use_art_clustering = use_art_clustering
        self.default_confidence_threshold = default_confidence_threshold
        self.adaptive_retrieval = adaptive_retrieval
        self.semantic_coherence_check = semantic_coherence_check
        self.coherence_threshold = coherence_threshold

        # Initialize properties for compatibility
        self._setup_property_proxies()

        # Initialize time counter
        self.current_time = 0

    def _setup_property_proxies(self):
        """Set up property proxies to maintain the original API."""
        # Memory properties
        self.memory_embeddings = self.memory_adapter.memory_embeddings
        self.memory_metadata = self.memory_adapter.memory_metadata

        # Initialize activation levels and temporal markers
        self.activation_levels = np.ones(len(self.memory_metadata), dtype=np.float32)
        self.temporal_markers = np.arange(len(self.memory_metadata), dtype=np.int64)

        # Category manager properties
        if self.category_manager:
            self.category_prototypes = getattr(
                self.category_manager, "category_prototypes", np.zeros((0, self.embedding_dim))
            )
            self.memory_categories = getattr(
                self.category_manager, "memory_categories", np.zeros(0, dtype=np.int64)
            )
            self.category_activations = getattr(
                self.category_manager, "category_activations", np.zeros(0, dtype=np.float32)
            )
            self.vigilance_threshold = getattr(self.category_manager, "vigilance_threshold", 0.85)

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
        # Update time counter
        self.current_time += 1

        # Prepare metadata
        if metadata is None:
            metadata = {}

        full_metadata = {
            "created_at": self.current_time,
            "access_count": 0,
            **metadata,
        }

        # Add memory to store
        memory_id = self.memory_adapter.add(embedding, text, full_metadata)

        # If using ART clustering, assign to a category
        if self.use_art_clustering and self.category_manager:
            category_idx = self.category_manager.assign_to_category(embedding)
            self.category_manager.add_memory_category_mapping(memory_id, category_idx)

        # Update property proxies
        self._setup_property_proxies()

        # Return the memory index (as int for compatibility)
        return int(memory_id) if isinstance(memory_id, str) else memory_id

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
        # Use default threshold if none provided
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold

        # Determine whether to use categories
        if use_categories is None:
            use_categories = self.use_art_clustering

        # Perform vector search
        results = self.memory_adapter.search_by_vector(
            query_vector=query_embedding,
            limit=top_k,
            threshold=confidence_threshold,
        )

        # Format results to match the expected return format
        formatted_results = []
        for result in results:
            memory_id = result.get("memory_id")
            score = result.get("score", result.get("relevance_score", 0))
            metadata = result.get("metadata", {})

            # Add content to metadata for compatibility
            if "content" in result and "text" not in metadata:
                metadata["text"] = result["content"]

            # Update activation for this memory
            self._update_activation(memory_id)

            # Convert memory_id to int if it's a string (for compatibility)
            if isinstance(memory_id, str):
                try:
                    memory_id = int(memory_id)
                except ValueError:
                    # If it can't be converted to int, use a hash
                    memory_id = hash(memory_id) % (2**31)

            formatted_results.append((memory_id, score, metadata))

        return formatted_results

    def _update_activation(self, memory_idx: int) -> None:
        """
        Update activation level for a memory that's been accessed.

        Args:
            memory_idx: Index of the memory to update
        """
        # Update access metadata
        memory = self.memory_adapter.get(memory_idx)
        metadata = memory.metadata or {}

        # Increment access count
        access_count = metadata.get("access_count", 0) + 1
        last_accessed = self.current_time

        # Update metadata
        updated_metadata = {
            **metadata,
            "access_count": access_count,
            "last_accessed": last_accessed,
        }

        self.memory_adapter.update_metadata(memory_idx, updated_metadata)

        # Update activation levels (for compatibility)
        if hasattr(self, "activation_levels") and len(self.activation_levels) > memory_idx:
            self.activation_levels[memory_idx] = min(1.0, self.activation_levels[memory_idx] + 0.2)

            # Decay other activations slightly
            decay_mask = np.ones_like(self.activation_levels, dtype=bool)
            decay_mask[memory_idx] = False
            self.activation_levels[decay_mask] *= 0.95

        # Update property proxies
        self._setup_property_proxies()

    def get_category_statistics(self) -> dict:
        """
        Get statistics about the current categories.

        Returns:
            dictionary with category statistics
        """
        if not self.use_art_clustering or not self.category_manager:
            return {"num_categories": 0}

        # Get statistics from category manager
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
            or not hasattr(self.category_manager, "category_prototypes")
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

        result = self.category_manager.consolidate_categories(threshold)

        # Update property proxies after consolidation
        self._setup_property_proxies()

        return result
