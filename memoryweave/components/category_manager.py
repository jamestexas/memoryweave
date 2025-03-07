"""
Category management component for MemoryWeave.

This module implements a standalone component for dynamic category management,
inspired by Adaptive Resonance Theory (ART). It allows memories to self-organize
into categories based on similarity patterns.
"""

import logging
import time
from typing import Any, Optional

import numpy as np

from memoryweave.components.base import MemoryComponent
from memoryweave.interfaces.memory import EmbeddingVector, MemoryID
from memoryweave.storage.base_store import BaseMemoryStore

logger = logging.getLogger(__name__)


class CategoryManager(MemoryComponent):
    """
    Component that manages memory categorization using ART-inspired clustering.

    Features:
    1. Dynamic category creation based on similarity
    2. Vigilance parameter to control category granularity
    3. Prototype learning to refine categories over time
    4. Category consolidation to merge similar categories
    """

    def __init__(
        self,
        memory_store: Optional[BaseMemoryStore] = None,
        core_manager: Optional["CategoryManager"] = None,
        embedding_dim: int = 768,
        vigilance_threshold: float = 0.8,
        learning_rate: float = 0.2,
        enable_category_consolidation: bool = True,
        consolidation_threshold: float = 0.8,
    ):
        """Initialize the category manager."""
        self.memory_store = memory_store
        self.categories: dict[int, dict[str, Any]] = {}
        self.memory_to_category: dict[MemoryID, int] = {}
        self.next_category_id = 0

        # Default parameters
        self.vigilance_threshold = vigilance_threshold
        self.learning_rate = learning_rate
        self.consolidation_threshold = consolidation_threshold
        self.embedding_dim = embedding_dim
        self.min_category_size = 3
        self.component_id = "category_manager"
        self.enable_category_consolidation = enable_category_consolidation

        # For new instances, set core_manager to self
        # For instances created with a core_manager parameter, use that
        self.core_manager = core_manager or self

        # Statistics tracking
        self.stats = {
            "total_categorized": 0,
            "new_categories_created": 0,
            "memories_reassigned": 0,
            "categories_consolidated": 0,
            "last_consolidation": 0,
            "num_categories": 0,
        }

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the component with configuration.

        Args:
            config: Configuration dictionary with parameters
        """
        self.vigilance_threshold = config.get("vigilance_threshold", self.vigilance_threshold)
        self.consolidation_threshold = config.get(
            "consolidation_threshold", self.consolidation_threshold
        )
        self.embedding_dim = config.get("embedding_dim", self.embedding_dim)
        self.min_category_size = config.get("min_category_size", self.min_category_size)
        self.learning_rate = config.get("learning_rate", self.learning_rate)
        self.enable_category_consolidation = config.get(
            "enable_category_consolidation", self.enable_category_consolidation
        )

        if "memory_store" in config:
            self.memory_store = config["memory_store"]

        # Set core_manager to self after initialization
        self.core_manager = self

    def add_to_category(self, memory_id: MemoryID, embedding: EmbeddingVector) -> int:
        """
        Add a memory to a category, creating a new one if needed.

        Args:
            memory_id: ID of the memory to categorize
            embedding: Embedding vector of the memory

        Returns:
            Category ID that the memory was assigned to
        """
        # Check if memory is already categorized
        if memory_id in self.memory_to_category:
            return self.memory_to_category[memory_id]

        # Find best matching category
        best_category_id = -1
        best_similarity = -1.0

        for category_id, category in self.categories.items():
            prototype = category["prototype"]
            similarity = self._calculate_similarity(embedding, prototype)

            if similarity > best_similarity:
                best_similarity = similarity
                best_category_id = category_id

        # Check if the best match exceeds the vigilance threshold
        if best_similarity >= self.vigilance_threshold and best_category_id >= 0:
            # Add to existing category
            category = self.categories[best_category_id]
            category["members"].add(memory_id)
            self.memory_to_category[memory_id] = best_category_id

            # Update prototype (weighted average)
            self._update_category_prototype(best_category_id, embedding)

            self.stats["total_categorized"] += 1
            return best_category_id
        else:
            # Create new category
            new_id = self._create_new_category(memory_id, embedding)
            self.stats["new_categories_created"] += 1
            self.stats["total_categorized"] += 1
            return new_id

    # Alias for compatibility with tests
    def assign_to_category(self, embedding: EmbeddingVector) -> int:
        """
        Assign an embedding to a category, creating a new one if needed.

        This is an alias for backward compatibility.

        Args:
            embedding: Embedding vector to categorize

        Returns:
            Category ID that the embedding was assigned to
        """
        # Generate a temporary memory ID if not provided
        memory_id = f"temp_{time.time()}_{hash(str(embedding))}"
        return self.add_to_category(memory_id, embedding)

    def add_memory_category_mapping(self, memory_id: MemoryID, category_id: int) -> None:
        """
        Add a mapping between memory and category.

        Args:
            memory_id: ID of the memory
            category_id: ID of the category
        """
        # Ensure the category exists
        if category_id not in self.categories:
            # Create an empty category with this ID
            self.categories[category_id] = {
                "prototype": np.zeros(self.embedding_dim),
                "members": set(),
                "created_at": time.time(),
                "updated_at": time.time(),
            }

        # Add memory to the category's members
        self.categories[category_id]["members"].add(memory_id)

        # Update the memory-to-category mapping
        self.memory_to_category[memory_id] = category_id

    def get_category_for_memory(self, memory_id: MemoryID) -> int:
        """
        Get the category ID for a memory.

        Args:
            memory_id: Memory ID to look up

        Returns:
            Category ID or -1 if not categorized
        """
        if memory_id not in self.memory_to_category:
            raise IndexError(f"Memory ID {memory_id} not found in any category")
        return self.memory_to_category.get(memory_id, -1)

    def get_memories_for_category(self, category_id: int) -> list[MemoryID]:
        """
        Get all memories in a category.

        Args:
            category_id: Category ID

        Returns:
            list of memory IDs in the category
        """
        if category_id not in self.categories:
            return []

        return list(self.categories[category_id]["members"])

    def get_category_similarities(self, embedding: EmbeddingVector) -> np.ndarray:
        """
        Get similarities between an embedding and all category prototypes.

        Args:
            embedding: Query embedding

        Returns:
            Array of similarity scores for each category
        """
        if not self.categories:
            return np.array([])

        similarities = np.zeros(len(self.categories))

        for i, (_category_id, category) in enumerate(self.categories.items()):
            prototype = category["prototype"]
            similarity = self._calculate_similarity(embedding, prototype)
            similarities[i] = similarity

        return similarities

    def update_category_activation(self, category_id: int, activation_delta: float = 0.1) -> None:
        """
        Update the activation level of a category.

        Args:
            category_id: ID of the category
            activation_delta: Amount to increase activation by
        """
        if category_id in self.categories:
            current_activation = self.categories[category_id].get("activation", 0.0)
            self.categories[category_id]["activation"] = min(
                1.0, current_activation + activation_delta
            )
            self.categories[category_id]["last_activated"] = time.time()

    def consolidate_categories(self, threshold: Optional[float] = None) -> int:
        """
        Merge similar categories to keep the number manageable.

        Args:
            threshold: Override default threshold for merging

        Returns:
            Number of categories after consolidation
        """
        if not self.enable_category_consolidation:
            return len(self.categories)

        if threshold is None:
            threshold = self.consolidation_threshold

        # Only proceed if we have enough categories
        if len(self.categories) < 2:
            return len(self.categories)

        # Build similarity matrix between all category prototypes
        category_ids = list(self.categories.keys())
        num_categories = len(category_ids)
        similarity_matrix = np.zeros((num_categories, num_categories))

        for i in range(num_categories):
            for j in range(i + 1, num_categories):
                cat_i_id = category_ids[i]
                cat_j_id = category_ids[j]

                prototype_i = self.categories[cat_i_id]["prototype"]
                prototype_j = self.categories[cat_j_id]["prototype"]

                similarity = self._calculate_similarity(prototype_i, prototype_j)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        # Find pairs of categories to merge
        merged_categories = []
        to_merge = []

        for i in range(num_categories):
            for j in range(i + 1, num_categories):
                if similarity_matrix[i, j] >= threshold:
                    # Check if both categories are large enough
                    cat_i_size = len(self.categories[category_ids[i]]["members"])
                    cat_j_size = len(self.categories[category_ids[j]]["members"])

                    if (
                        cat_i_size >= self.min_category_size
                        and cat_j_size >= self.min_category_size
                    ):
                        to_merge.append((category_ids[i], category_ids[j]))
                        merged_categories.extend([category_ids[i], category_ids[j]])

        # Merge the identified categories
        for cat_i_id, cat_j_id in to_merge:
            if cat_i_id in self.categories and cat_j_id in self.categories:
                self._merge_categories(cat_i_id, cat_j_id)
                self.stats["categories_consolidated"] += 1

        self.stats["last_consolidation"] = time.time()
        # Update num_categories for stats
        self.stats["num_categories"] = len(self.categories)
        return len(self.categories)

    def recategorize(self, memory_id: MemoryID, embedding: EmbeddingVector) -> int:
        """
        Re-categorize a memory, possibly moving it to a better matching category.

        Args:
            memory_id: ID of the memory to recategorize
            embedding: Updated embedding vector

        Returns:
            New category ID
        """
        # Remove from current category if categorized
        if memory_id in self.memory_to_category:
            current_cat_id = self.memory_to_category[memory_id]
            if current_cat_id in self.categories:
                self.categories[current_cat_id]["members"].remove(memory_id)

                # If category is now empty, remove it
                if not self.categories[current_cat_id]["members"]:
                    del self.categories[current_cat_id]
                else:
                    # Update the prototype
                    self._update_category_prototype(current_cat_id)

        # Add to best matching category
        return self.add_to_category(memory_id, embedding)

    def get_all_categories(self) -> dict[int, dict[str, Any]]:
        """
        Get all categories with their properties.

        Returns:
            dictionary of category IDs to category properties
        """
        return self.categories

    def get_category_statistics(self) -> dict[str, Any]:
        """
        Get statistics about categorization.

        Returns:
            dictionary of statistics
        """
        stats = self.stats.copy()

        # Add dynamic statistics
        stats.update(
            {
                "num_categories": len(self.categories),
                "average_category_size": sum(
                    len(cat["members"]) for cat in self.categories.values()
                )
                / max(1, len(self.categories)),
                "largest_category_size": max(
                    (len(cat["members"]) for cat in self.categories.values()), default=0
                ),
                "category_activations": {
                    cat_id: cat.get("activation", 0.0) for cat_id, cat in self.categories.items()
                },
                "memories_per_category": {
                    cat_id: len(cat["members"]) for cat_id, cat in self.categories.items()
                },
            }
        )

        return stats

    def _create_new_category(self, memory_id: MemoryID, embedding: EmbeddingVector) -> int:
        """Create a new category with the given memory as its first member."""
        category_id = self.next_category_id
        self.next_category_id += 1

        self.categories[category_id] = {
            "prototype": embedding.copy(),
            "members": {memory_id},
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        self.memory_to_category[memory_id] = category_id
        # Update num_categories for stats
        self.stats["num_categories"] = len(self.categories)
        return category_id

    def _update_category_prototype(
        self, category_id: int, new_embedding: Optional[EmbeddingVector] = None
    ) -> None:
        """Update a category's prototype vector."""
        category = self.categories[category_id]
        members = category["members"]

        if not members:
            return

        # If we have a memory store, use it to get all embeddings
        if self.memory_store is not None:
            embeddings = []
            for memory_id in members:
                try:
                    memory = self.memory_store.get(memory_id)
                    embeddings.append(memory.embedding)
                except Exception:  # noqa: S110
                    # Skip memories that can't be retrieved
                    pass

            if embeddings:
                # Compute new prototype as average of embeddings
                prototype = np.mean(embeddings, axis=0)
                category["prototype"] = prototype

        # If no memory store or no embeddings retrieved, use new embedding if provided
        elif new_embedding is not None:
            # Use learning rate for weighted average
            category["prototype"] = (1 - self.learning_rate) * category[
                "prototype"
            ] + self.learning_rate * new_embedding
            # Normalize
            norm = np.linalg.norm(category["prototype"])
            if norm > 0:
                category["prototype"] /= norm

        category["updated_at"] = time.time()

    def _merge_categories(self, cat_id_1: int, cat_id_2: int) -> int:
        """Merge two categories."""
        # Get the larger category as primary
        if len(self.categories[cat_id_1]["members"]) >= len(self.categories[cat_id_2]["members"]):
            primary_id, secondary_id = cat_id_1, cat_id_2
        else:
            primary_id, secondary_id = cat_id_2, cat_id_1

        # Get categories
        primary = self.categories[primary_id]
        secondary = self.categories[secondary_id]

        # Merge members
        for memory_id in secondary["members"]:
            primary["members"].add(memory_id)
            self.memory_to_category[memory_id] = primary_id

        # Update prototype (weighted average based on category sizes)
        primary_size = len(primary["members"])
        secondary_size = len(secondary["members"])
        total_size = primary_size + secondary_size

        primary["prototype"] = (primary_size / total_size) * primary["prototype"] + (
            secondary_size / total_size
        ) * secondary["prototype"]

        # Normalize
        norm = np.linalg.norm(primary["prototype"])
        if norm > 0:
            primary["prototype"] /= norm

        # Remove secondary category
        del self.categories[secondary_id]

        # Update num_categories for stats
        self.stats["num_categories"] = len(self.categories)
        return primary_id

    def _calculate_similarity(
        self, embedding1: EmbeddingVector, embedding2: EmbeddingVector
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def process(self, data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a memory for categorization.

        Args:
            data: Memory data including ID and embedding
            context: Context information

        Returns:
            Updated data with category information
        """
        memory_id = data.get("id")
        embedding = data.get("embedding")

        if memory_id is None or embedding is None:
            return data

        # Add to appropriate category
        category_id = self.add_to_category(memory_id, embedding)

        # Add category information to result
        result = dict(data)
        result["category_id"] = category_id
        result["category_members"] = (
            len(self.categories[category_id]["members"]) if category_id in self.categories else 0
        )
        result["category_prototype_similarity"] = self._calculate_similarity(
            embedding, self.get_category_prototype(category_id)
        )

        return result

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query to add category-related information.

        Args:
            query: Query string
            context: Query context including embeddings

        Returns:
            Updated context with category information
        """
        # Get query embedding
        query_embedding = context.get("query_embedding")
        if query_embedding is None:
            return {}

        # Find the closest category to the query
        best_category_id = -1
        best_similarity = -1.0

        for category_id, category in self.categories.items():
            prototype = category["prototype"]
            similarity = self._calculate_similarity(query_embedding, prototype)

            if similarity > best_similarity:
                best_similarity = similarity
                best_category_id = category_id

        # Add category information to context
        result = {
            "category_match": best_category_id,
            "category_similarity": best_similarity,
        }

        # If we found a good category match, add member information
        if best_category_id >= 0 and best_similarity >= self.vigilance_threshold * 0.8:
            result["category_members"] = list(self.categories[best_category_id]["members"])
            result["use_category_filter"] = True
        else:
            result["use_category_filter"] = False

        return result

    def get_category_prototype(self, category_id: int) -> EmbeddingVector:
        """
        Get the prototype vector for a category.

        Args:
            category_id: Category ID

        Returns:
            Prototype embedding vector
        """
        if category_id not in self.categories:
            # Return zero vector of proper dimension
            return np.zeros(self.embedding_dim)

        return self.categories[category_id]["prototype"]

    def filter_by_category(
        self,
        results: list[dict[str, Any]],
        query_embedding: EmbeddingVector,
        similarity_boost: float = 0.2,
        min_similarity: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Filter and boost results based on category membership.

        Args:
            results: list of retrieval results
            query_embedding: Query embedding vector
            similarity_boost: How much to boost results in similar categories
            min_similarity: Minimum similarity for category matching

        Returns:
            Filtered and boosted results
        """
        if not results or not self.categories:
            return results

        # Find most relevant category for the query
        best_category_id = -1
        best_similarity = -1.0

        for category_id, category in self.categories.items():
            prototype = category["prototype"]
            similarity = self._calculate_similarity(query_embedding, prototype)

            if similarity > best_similarity:
                best_similarity = similarity
                best_category_id = category_id

        # Get category members
        if best_category_id in self.categories and best_similarity >= min_similarity:
            category_members = self.categories[best_category_id]["members"]
        else:
            category_members = set()

        # No filtering if no members or low similarity
        if not category_members or best_similarity < min_similarity:
            return results

        # Boost results in the same category
        boosted_results = []
        for result in results:
            memory_id = result.get("memory_id")
            if memory_id is not None:
                # Check if memory belongs to the same category
                memory_category = self.memory_to_category.get(memory_id, -1)
                is_same_category = memory_category == best_category_id

                # Apply boosting if in same category
                if is_same_category:
                    # Deep copy the result to avoid modifying the original
                    boosted_result = dict(result)
                    # Boost the relevance score
                    base_score = boosted_result.get("relevance_score", 0.0)
                    boosted_score = min(1.0, base_score * (1.0 + similarity_boost))
                    boosted_result["relevance_score"] = boosted_score
                    boosted_result["category_boosted"] = True
                    boosted_results.append(boosted_result)
                else:
                    boosted_results.append(result)
            else:
                boosted_results.append(result)

        # Sort by relevance score
        boosted_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return boosted_results
