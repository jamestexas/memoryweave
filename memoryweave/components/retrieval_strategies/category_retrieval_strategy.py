"""
Category-based retrieval strategy for MemoryWeave.

This module implements a retrieval strategy that leverages category information
to improve retrieval results, focusing on semantic coherence within categories.
"""

import logging
from typing import Any, Optional

import numpy as np
from rich.logging import RichHandler

from memoryweave.components.base import RetrievalStrategy
from memoryweave.components.category_manager import CategoryManager
from memoryweave.components.retrieval_strategies_impl import SimilarityRetrievalStrategy

logger = logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)

# TODO: This is hacky but for now here we are:
# Patch the SimilarityRetrievalStrategy to add category_id to results


class CategoryRetrievalStrategy(RetrievalStrategy):
    """
    Retrieves memories based on ART category clustering.

    This strategy uses the ART-inspired clustering to first identify
    the most relevant categories, then retrieves memories from those
    categories.
    """

    def __init__(self, memory: Any, category_manager: Optional[CategoryManager] = None):
        """
        Initialize with memory and category manager.

        Args:
            memory: The memory to retrieve from
            category_manager: Optional category manager to use
        """
        self.memory = memory
        # Use provided category_manager or get it from memory
        self.category_manager = category_manager or getattr(memory, "category_manager", None)

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.confidence_threshold = config.get("confidence_threshold", 0.0)
        self.max_categories = config.get("max_categories", 3)
        self.activation_boost = config.get("activation_boost", True)
        self.category_selection_threshold = config.get("category_selection_threshold", 0.5)

        # For testing/benchmarking, set minimum results
        self.min_results = max(1, config.get("min_results", 5))

        # Initialize category manager if not provided
        if self.category_manager is None and hasattr(self.memory, "category_manager"):
            self.category_manager = self.memory.category_manager

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Retrieve memories using category-based retrieval."""

        # Helper function for similarity retrieval fallback
        def similarity_fallback():
            similarity_strategy = SimilarityRetrievalStrategy(memory)
            if hasattr(similarity_strategy, "initialize"):
                similarity_strategy.initialize({"confidence_threshold": self.confidence_threshold})

            fallback_results = similarity_strategy.retrieve(query_embedding, top_k, context)

            # Add category fields to each result
            for result in fallback_results:
                if "category_id" not in result:
                    result["category_id"] = -1  # Default category ID
                if "category_similarity" not in result:
                    result["category_similarity"] = 0.0  # Default similarity

            return fallback_results

        # Get memory from context or instance
        memory = context.get("memory", self.memory)

        # Check if memory has category_manager
        category_manager = self.category_manager or getattr(memory, "category_manager", None)
        if category_manager is None:
            # Fall back to similarity retrieval if no category manager
            logger.info(
                "CategoryRetrievalStrategy: No category manager found, falling back to similarity retrieval"
            )
            return similarity_fallback()

        # Apply query type adaptation if available
        adapted_params = context.get("adapted_retrieval_params", {})
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)
        max_categories = adapted_params.get("max_categories", self.max_categories)

        try:
            # Get category similarities
            category_similarities = category_manager.get_category_similarities(query_embedding)

            # If no categories, fall back to similarity retrieval
            if len(category_similarities) == 0:
                logger.info(
                    "CategoryRetrievalStrategy: No categories found, falling back to similarity retrieval"
                )
                return similarity_fallback()

            # Get category IDs
            category_ids = list(category_manager.categories.keys())

            # Select top categories with similarity above threshold
            selected_categories = []

            # Sort category indices by similarity (descending)
            sorted_indices = np.argsort(-category_similarities)

            for i in sorted_indices:
                if (
                    i < len(category_ids)
                    and category_similarities[i] >= self.category_selection_threshold
                ):
                    selected_categories.append(category_ids[i])
                if len(selected_categories) >= max_categories:
                    break

            # If no categories selected, use top N categories
            if not selected_categories and len(category_similarities) > 0:
                num_to_select = min(max_categories, len(category_similarities))
                selected_categories = [
                    category_ids[i] for i in sorted_indices[:num_to_select] if i < len(category_ids)
                ]

            # Get memories from selected categories
            candidate_indices = []
            for cat_idx in selected_categories:
                cat_memories = category_manager.get_memories_for_category(cat_idx)
                candidate_indices.extend(cat_memories)

            # If no candidates, fall back to similarity retrieval
            if not candidate_indices:
                logger.info(
                    "CategoryRetrievalStrategy: No candidate memories found, falling back to similarity retrieval"
                )
                return similarity_fallback()

            # Try to convert memory IDs to integers if they're stored as strings
            memory_indices = []
            for memory_id in candidate_indices:
                if isinstance(memory_id, str) and memory_id.isdigit():
                    memory_indices.append(int(memory_id))
                elif isinstance(memory_id, (int, np.integer)):
                    memory_indices.append(int(memory_id))
                else:
                    # Skip non-numeric IDs
                    continue

            # Calculate similarities for candidate memories
            # Access memory embeddings using integer indices
            embeddings = []
            valid_indices = []

            # Get embeddings for valid memory indices
            if hasattr(memory, "memory_embeddings"):
                for idx in memory_indices:
                    if 0 <= idx < len(memory.memory_embeddings):
                        embeddings.append(memory.memory_embeddings[idx])
                        valid_indices.append(idx)

            if not embeddings:
                # Fall back to similarity if we couldn't get embeddings
                logger.info(
                    "CategoryRetrievalStrategy: No valid embeddings found, falling back to similarity retrieval"
                )
                return similarity_fallback()

            # Calculate similarities
            embeddings_array = np.array(embeddings)
            candidate_similarities = np.dot(embeddings_array, query_embedding)

            # Apply activation boost if enabled
            if self.activation_boost and hasattr(memory, "activation_levels"):
                activation_levels = []
                for idx in valid_indices:
                    if idx < len(memory.activation_levels):
                        activation_levels.append(memory.activation_levels[idx])
                    else:
                        activation_levels.append(1.0)  # Default activation

                activation_array = np.array(activation_levels)
                candidate_similarities = candidate_similarities * activation_array

            # Filter by confidence threshold
            valid_candidates = np.where(candidate_similarities >= confidence_threshold)[0]

            # Apply minimum results guarantee if needed
            if len(valid_candidates) == 0 and hasattr(self, "min_results") and self.min_results > 0:
                # Sort all candidates by similarity
                sorted_idx = np.argsort(-candidate_similarities)
                # Take top min_results candidates regardless of threshold
                valid_candidates = sorted_idx[: min(self.min_results, len(candidate_similarities))]

            if len(valid_candidates) == 0:
                logger.info(
                    "CategoryRetrievalStrategy: No candidates passed threshold, falling back to similarity retrieval"
                )
                return similarity_fallback()

            # Get top-k memories
            top_k = min(top_k, len(valid_candidates))
            top_memory_indices = np.argsort(-candidate_similarities[valid_candidates])[:top_k]

            # Format results
            results = []
            for i in top_memory_indices:
                candidate_idx = valid_candidates[i]
                if candidate_idx >= len(valid_indices):
                    continue  # Skip invalid index

                memory_idx = valid_indices[candidate_idx]
                similarity = candidate_similarities[candidate_idx]

                # Get category for memory
                try:
                    memory_id = memory_idx  # Use the index directly as the ID
                    category_id = category_manager.get_category_for_memory(memory_id)

                    # Get category similarity if possible
                    cat_idx = category_ids.index(category_id) if category_id in category_ids else -1
                    if cat_idx >= 0 and cat_idx < len(category_similarities):
                        category_similarity = category_similarities[cat_idx]
                    else:
                        category_similarity = 0.0
                except (IndexError, ValueError):
                    # If we can't get the category, use defaults
                    category_id = -1
                    category_similarity = 0.0

                # Update memory activation
                if hasattr(memory, "update_activation"):
                    memory.update_activation(memory_idx)

                # Extract metadata
                metadata = {}
                if hasattr(memory, "memory_metadata") and memory_idx < len(memory.memory_metadata):
                    metadata = memory.memory_metadata[memory_idx]

                # Create result
                result = {
                    "memory_id": memory_idx,
                    "relevance_score": float(similarity),
                    "category_id": int(category_id),
                    "category_similarity": float(category_similarity),
                    "below_threshold": similarity < confidence_threshold,
                }

                # Add metadata
                if metadata:
                    result.update(metadata)

                # Add to results
                results.append(result)

            return results

        except Exception as e:
            # On any error, fall back to similarity retrieval
            logging.warning(
                f"Category retrieval failed with error: {str(e)}. Falling back to similarity retrieval."
            )
            return similarity_fallback()


# Create a subclass of SimilarityRetrievalStrategy that adds category fields
class CategoryAwareSimilarityStrategy(SimilarityRetrievalStrategy):
    def retrieve(self, query_embedding, top_k, context):
        results = super().retrieve(query_embedding, top_k, context)
        # Add category fields to results
        for result in results:
            result["category_id"] = -1
            result["category_similarity"] = 0.0
        return results
