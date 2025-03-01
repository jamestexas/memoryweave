"""Category management for MemoryWeave.

This module provides implementations for memory categorization,
using ART-inspired clustering to organize memories into categories.
"""

from typing import Dict, List, Set

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector, ICategoryManager, MemoryID


class CategoryManager(ICategoryManager):
    """Implementation of ART-inspired memory categorization."""

    def __init__(self, vigilance: float = 0.85):
        """Initialize the category manager.
        
        Args:
            vigilance: Threshold for category matching (0-1)
                Higher values create more specific categories
        """
        self._vigilance = vigilance
        self._categories: Dict[int, EmbeddingVector] = {}  # Category ID -> Prototype
        self._members: Dict[int, Set[MemoryID]] = {}  # Category ID -> Members
        self._memory_category: Dict[MemoryID, int] = {}  # Memory ID -> Category ID
        self._next_category_id = 0

    def add_to_category(self, memory_id: MemoryID, embedding: EmbeddingVector) -> int:
        """Add a memory to a category and return the category ID."""
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            norm = 1e-10
        normalized_embedding = embedding / norm

        # Find best matching category
        best_category_id = None
        best_similarity = -1.0

        for category_id, prototype in self._categories.items():
            similarity = np.dot(normalized_embedding, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_category_id = category_id

        # Check if similarity exceeds vigilance threshold
        if best_category_id is not None and best_similarity >= self._vigilance:
            # Add to existing category
            category_id = best_category_id
            self._update_category(category_id, memory_id, normalized_embedding)
        else:
            # Create new category
            category_id = self._create_category(memory_id, normalized_embedding)

        return category_id

    def get_category(self, memory_id: MemoryID) -> int:
        """Get the category ID for a memory."""
        if memory_id not in self._memory_category:
            raise KeyError(f"Memory with ID {memory_id} not found")

        return self._memory_category[memory_id]

    def get_category_members(self, category_id: int) -> List[MemoryID]:
        """Get all memories in a category."""
        if category_id not in self._members:
            raise KeyError(f"Category with ID {category_id} not found")

        return list(self._members[category_id])

    def get_category_prototype(self, category_id: int) -> EmbeddingVector:
        """Get the prototype vector for a category."""
        if category_id not in self._categories:
            raise KeyError(f"Category with ID {category_id} not found")

        return self._categories[category_id]

    def consolidate_categories(self, similarity_threshold: float) -> List[int]:
        """Merge similar categories."""
        if len(self._categories) <= 1:
            return []

        # Find pairs of similar categories
        category_ids = list(self._categories.keys())
        categories_to_merge = []

        for i in range(len(category_ids)):
            for j in range(i + 1, len(category_ids)):
                id1 = category_ids[i]
                id2 = category_ids[j]

                prototype1 = self._categories[id1]
                prototype2 = self._categories[id2]

                similarity = np.dot(prototype1, prototype2)

                if similarity >= similarity_threshold:
                    categories_to_merge.append((id1, id2))

        # Merge categories
        merged_categories = []

        for id1, id2 in categories_to_merge:
            # Skip if either category has already been merged
            if id1 not in self._categories or id2 not in self._categories:
                continue

            # Merge id2 into id1
            self._merge_categories(id1, id2)
            merged_categories.append(id2)

        return merged_categories

    def _create_category(self, memory_id: MemoryID, embedding: EmbeddingVector) -> int:
        """Create a new category."""
        category_id = self._next_category_id
        self._next_category_id += 1

        # Store category prototype
        self._categories[category_id] = embedding

        # Initialize members
        self._members[category_id] = {memory_id}

        # Associate memory with category
        self._memory_category[memory_id] = category_id

        return category_id

    def _update_category(self,
                        category_id: int,
                        memory_id: MemoryID,
                        embedding: EmbeddingVector) -> None:
        """Update an existing category with a new memory."""
        # Add memory to category
        self._members[category_id].add(memory_id)

        # Associate memory with category
        self._memory_category[memory_id] = category_id

        # Update prototype
        current_prototype = self._categories[category_id]
        learning_rate = 1.0 / len(self._members[category_id])
        new_prototype = (1 - learning_rate) * current_prototype + learning_rate * embedding

        # Normalize prototype
        norm = np.linalg.norm(new_prototype)
        if norm == 0:
            norm = 1e-10
        normalized_prototype = new_prototype / norm

        self._categories[category_id] = normalized_prototype

    def _merge_categories(self, target_id: int, source_id: int) -> None:
        """Merge source category into target category."""
        # Update prototype
        target_prototype = self._categories[target_id]
        source_prototype = self._categories[source_id]

        target_size = len(self._members[target_id])
        source_size = len(self._members[source_id])
        total_size = target_size + source_size

        # Weighted average of prototypes
        new_prototype = (
            (target_size / total_size) * target_prototype +
            (source_size / total_size) * source_prototype
        )

        # Normalize prototype
        norm = np.linalg.norm(new_prototype)
        if norm == 0:
            norm = 1e-10
        normalized_prototype = new_prototype / norm

        self._categories[target_id] = normalized_prototype

        # Update members
        for memory_id in self._members[source_id]:
            self._members[target_id].add(memory_id)
            self._memory_category[memory_id] = target_id

        # Remove source category
        del self._members[source_id]
        del self._categories[source_id]
