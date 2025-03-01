"""
Implementation of the ART-inspired category management for MemoryWeave.
"""

from typing import List, Literal

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


class CategoryManager:
    """
    Implements Adaptive Resonance Theory (ART) inspired category management.

    This class handles the categorization of memories, including dynamic
    category formation, prototype updates, and category consolidation.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
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
    ):
        """
        Initialize the category manager.

        Args:
            embedding_dim: Dimension of the memory embeddings
            vigilance_threshold: Initial threshold for creating new categories (ART vigilance)
            learning_rate: Rate at which category prototypes are updated
            dynamic_vigilance: Whether to use dynamic vigilance adjustment
            vigilance_strategy: Strategy for adjusting vigilance ("decreasing", "increasing",
                               "category_based", or "density_based")
            min_vigilance: Minimum vigilance threshold for dynamic adjustment
            max_vigilance: Maximum vigilance threshold for dynamic adjustment
            target_categories: Target number of categories for category_based strategy
            enable_category_consolidation: Whether to enable periodic category consolidation
            consolidation_threshold: Similarity threshold for merging categories in hierarchical clustering
            min_category_size: Minimum number of memories per category before considering consolidation
            consolidation_frequency: How often to run consolidation (every N memories added)
            hierarchical_method: Method for hierarchical clustering linkage
        """
        self.embedding_dim = embedding_dim
        self.initial_vigilance = vigilance_threshold
        self.vigilance_threshold = vigilance_threshold
        self.learning_rate = learning_rate

        # Dynamic vigilance parameters
        self.dynamic_vigilance = dynamic_vigilance
        self.vigilance_strategy = vigilance_strategy
        self.min_vigilance = min_vigilance
        self.max_vigilance = max_vigilance
        self.target_categories = target_categories

        # Category structures
        self.category_prototypes = np.zeros((0, embedding_dim), dtype=np.float32)
        self.memory_categories = np.zeros(0, dtype=np.int64)
        self.category_activations = np.zeros(0, dtype=np.float32)

        # Category consolidation parameters
        self.enable_category_consolidation = enable_category_consolidation
        self.consolidation_threshold = consolidation_threshold
        self.min_category_size = min_category_size
        self.consolidation_frequency = consolidation_frequency
        self.hierarchical_method = hierarchical_method

        # Tracking variables
        self.memories_added = 0
        self.last_consolidation = 0

        # For compatibility with tests
        self.activation_levels = None
        self.memory_embeddings = None

    def assign_to_category(self, embedding: np.ndarray) -> int:
        """
        Assign a memory to a category using ART-inspired resonance.

        Args:
            embedding: The memory embedding to categorize

        Returns:
            Index of the assigned category
        """
        self.memories_added += 1

        # Update vigilance if using dynamic vigilance
        if self.dynamic_vigilance:
            self._update_vigilance()

        if len(self.category_prototypes) == 0:
            # Create the first category
            self.category_prototypes = np.vstack([self.category_prototypes, embedding])
            self.category_activations = np.append(self.category_activations, 1.0)
            return 0

        # Calculate resonance with existing categories
        similarities = np.dot(self.category_prototypes, embedding)
        best_match = np.argmax(similarities)

        # Check if best match exceeds vigilance threshold
        if similarities[best_match] >= self.vigilance_threshold:
            # Update existing category prototype
            self._update_category_prototype(best_match, embedding)
            return best_match
        else:
            # Create new category
            self.category_prototypes = np.vstack([self.category_prototypes, embedding])
            self.category_activations = np.append(self.category_activations, 1.0)

            # Check if it's time to run category consolidation
            if (
                self.enable_category_consolidation
                and self.memories_added >= self.last_consolidation + self.consolidation_frequency
                and len(self.category_prototypes) > 1
            ):
                self._consolidate_categories()
                self.last_consolidation = self.memories_added

            return len(self.category_prototypes) - 1

    def add_memory_category_mapping(self, memory_idx: int, category_idx: int) -> None:
        """
        Add a mapping between a memory and its category.

        Args:
            memory_idx: Index of the memory
            category_idx: Index of the category
        """
        # Ensure the memory_categories array is large enough
        if memory_idx >= len(self.memory_categories):
            # Create new array with appropriate size
            new_size = memory_idx + 1
            new_categories = np.zeros(new_size, dtype=np.int64)
            # Copy existing data
            if len(self.memory_categories) > 0:
                new_categories[: len(self.memory_categories)] = self.memory_categories
            self.memory_categories = new_categories

        # Set the category for the memory
        self.memory_categories[memory_idx] = category_idx

        # Update category activation
        self.update_category_activation(category_idx)

    def update_category_activation(self, category_idx: int) -> None:
        """
        Update activation level for a category that's been accessed.

        Args:
            category_idx: Index of the category to update
        """
        if category_idx >= len(self.category_activations):
            return

        # Increase activation for accessed category
        self.category_activations[category_idx] = min(
            1.0, self.category_activations[category_idx] + 0.2
        )

        # Decay other category activations slightly
        decay_mask = np.ones_like(self.category_activations, dtype=bool)
        decay_mask[category_idx] = False
        self.category_activations[decay_mask] *= 0.95

    def _update_category_prototype(self, category_idx: int, embedding: np.ndarray) -> None:
        """
        Update a category prototype with a new embedding.

        Args:
            category_idx: Index of the category to update
            embedding: New embedding to incorporate
        """
        # Adaptive Resonance Theory inspired update
        # Gradually move the prototype toward the new embedding
        self.category_prototypes[category_idx] = (
            1 - self.learning_rate
        ) * self.category_prototypes[category_idx] + self.learning_rate * embedding

        # Normalize the updated prototype
        self.category_prototypes[category_idx] /= np.linalg.norm(
            self.category_prototypes[category_idx]
        )

    def _update_vigilance(self) -> None:
        """
        Update vigilance threshold based on the selected dynamic strategy.
        """
        if not self.dynamic_vigilance:
            return

        if self.vigilance_strategy == "decreasing":
            # Start high, gradually decrease - encourages more merging over time
            decay_factor = min(
                1.0, self.memories_added / 100
            )  # Adjust the divisor to control decay rate
            self.vigilance_threshold = max(
                self.min_vigilance, self.initial_vigilance * (1 - decay_factor * 0.5)
            )

        elif self.vigilance_strategy == "increasing":
            # Start low, gradually increase - creates broader categories first
            growth_factor = min(
                1.0, self.memories_added / 100
            )  # Adjust the divisor to control growth rate
            self.vigilance_threshold = min(
                self.max_vigilance,
                self.min_vigilance + (self.max_vigilance - self.min_vigilance) * growth_factor,
            )

        elif self.vigilance_strategy == "category_based":
            # Adjust based on number of categories
            if len(self.category_prototypes) > 0:
                current_ratio = len(self.category_prototypes) / max(1, self.memories_added)
                target_ratio = self.target_categories / max(1000, self.memories_added)

                if current_ratio > target_ratio * 1.2:  # Too many categories
                    # Decrease vigilance to encourage merging
                    self.vigilance_threshold = max(
                        self.min_vigilance, self.vigilance_threshold - 0.01
                    )
                elif current_ratio < target_ratio * 0.8:  # Too few categories
                    # Increase vigilance to encourage new categories
                    self.vigilance_threshold = min(
                        self.max_vigilance, self.vigilance_threshold + 0.01
                    )

        elif self.vigilance_strategy == "density_based":
            # Adjust based on density in embedding space
            if len(self.category_prototypes) > 10:  # Need enough categories to compute density
                # Compute pairwise similarities as a measure of density
                similarities = np.dot(self.category_prototypes, self.category_prototypes.T)

                # Calculate average similarity (excluding self-similarity)
                total_sim = similarities.sum() - len(self.category_prototypes)  # Subtract diagonal
                avg_sim = total_sim / (
                    len(self.category_prototypes) * (len(self.category_prototypes) - 1)
                )

                # In dense regions (high similarity), increase vigilance for finer distinctions
                # In sparse regions (low similarity), decrease vigilance to group more
                density_factor = avg_sim * 2  # Scale factor
                self.vigilance_threshold = self.min_vigilance + density_factor * (
                    self.max_vigilance - self.min_vigilance
                )

                # Ensure boundaries
                self.vigilance_threshold = max(
                    self.min_vigilance, min(self.max_vigilance, self.vigilance_threshold)
                )

    def _consolidate_categories(self) -> None:
        """
        Consolidate similar categories using hierarchical clustering.
        This reduces category fragmentation by merging similar category prototypes.
        """
        num_categories = len(self.category_prototypes)
        if num_categories <= 1:
            return

        # Identify small categories that should be merged
        category_counts = {}
        for cat_idx in self.memory_categories:
            category_counts[int(cat_idx)] = category_counts.get(int(cat_idx), 0) + 1

        # Compute pairwise distances for hierarchical clustering
        # (using 1-similarity as distance)
        similarities = np.dot(self.category_prototypes, self.category_prototypes.T)
        distances = 1.0 - similarities

        # Replace diagonal with zeros (self-distance)
        np.fill_diagonal(distances, 0.0)

        # Convert distance matrix to condensed form for linkage
        condensed_distances = []
        for i in range(num_categories):
            for j in range(i + 1, num_categories):
                condensed_distances.append(distances[i, j])

        # If we don't have enough categories to consolidate, return
        if len(condensed_distances) == 0:
            return

        # Perform hierarchical clustering on category prototypes
        Z = linkage(condensed_distances, method=self.hierarchical_method)

        # Form flat clusters using the consolidation threshold
        # Higher threshold = more merging
        cluster_labels = fcluster(Z, t=1.0 - self.consolidation_threshold, criterion="distance")

        # Count the number of final clusters
        num_final_clusters = len(np.unique(cluster_labels))

        # If no consolidation happened, no need to proceed
        if num_final_clusters == num_categories:
            return

        # Create new prototypes for the consolidated categories
        new_prototypes = np.zeros((num_final_clusters, self.embedding_dim), dtype=np.float32)
        cluster_sizes = np.zeros(num_final_clusters, dtype=np.int64)

        # For each original category, contribute to its new cluster's prototype
        for old_cat_idx, cluster_idx in enumerate(cluster_labels):
            # Convert to 0-based indexing
            cluster_idx = cluster_idx - 1

            # Weight by number of memories in the category
            weight = category_counts.get(old_cat_idx, 0)
            if weight == 0:  # Skip empty categories
                continue

            # Add weighted prototype to the new cluster's prototype
            new_prototypes[cluster_idx] += self.category_prototypes[old_cat_idx] * weight
            cluster_sizes[cluster_idx] += weight

        # Normalize the new prototypes
        for cluster_idx in range(num_final_clusters):
            if cluster_sizes[cluster_idx] > 0:
                new_prototypes[cluster_idx] /= cluster_sizes[cluster_idx]
                # Ensure unit length
                norm = np.linalg.norm(new_prototypes[cluster_idx])
                if norm > 0:
                    new_prototypes[cluster_idx] /= norm

        # Create mapping from old categories to new ones
        old_to_new_category = {}
        for old_cat_idx, cluster_idx in enumerate(cluster_labels):
            old_to_new_category[old_cat_idx] = cluster_idx - 1  # Convert to 0-based indexing

        # Create new category activations by taking maximum of merged categories
        new_activations = np.zeros(num_final_clusters, dtype=np.float32)
        for old_cat_idx, new_cat_idx in old_to_new_category.items():
            new_activations[new_cat_idx] = max(
                new_activations[new_cat_idx], self.category_activations[old_cat_idx]
            )

        # Remap memory categories
        for i in range(len(self.memory_categories)):
            old_cat = self.memory_categories[i]
            self.memory_categories[i] = old_to_new_category[old_cat]

        # Update category prototypes and activations
        self.category_prototypes = new_prototypes
        self.category_activations = new_activations

    def get_category_statistics(self) -> dict:
        """
        Get statistics about the current categories.

        Returns:
            Dictionary with category statistics
        """
        if len(self.category_prototypes) == 0:
            return {"num_categories": 0}

        # Count memories per category
        category_counts = {}
        for cat_idx in self.memory_categories:
            category_counts[int(cat_idx)] = category_counts.get(int(cat_idx), 0) + 1

        # Calculate average activation per category
        category_avg_activation = {}
        for cat_idx in range(len(self.category_prototypes)):
            mask = self.memory_categories == cat_idx
            if np.any(mask):
                category_avg_activation[cat_idx] = (
                    float(np.mean(self.activation_levels[mask]))
                    if hasattr(self, "activation_levels") and self.activation_levels is not None
                    else 0.0
                )
            else:
                category_avg_activation[cat_idx] = 0.0

        # Calculate pairwise similarity between category prototypes
        category_similarities = {}
        if len(self.category_prototypes) > 1:
            sim_matrix = np.dot(self.category_prototypes, self.category_prototypes.T)
            for i in range(len(self.category_prototypes)):
                for j in range(i + 1, len(self.category_prototypes)):
                    if i != j:
                        category_similarities[(i, j)] = float(sim_matrix[i, j])

        # Calculate category density (average intra-category similarity)
        category_density = {}
        for cat_idx in range(len(self.category_prototypes)):
            mask = self.memory_categories == cat_idx
            cat_memories = np.where(mask)[0]

            if (
                len(cat_memories) > 1
                and hasattr(self, "memory_embeddings")
                and self.memory_embeddings is not None
            ):
                # Calculate pairwise similarities within category
                cat_embeddings = self.memory_embeddings[cat_memories]
                similarities = np.dot(cat_embeddings, cat_embeddings.T)

                # Get upper triangle (excluding diagonal)
                upper_tri = similarities[np.triu_indices(len(cat_memories), k=1)]

                if len(upper_tri) > 0:
                    category_density[cat_idx] = float(np.mean(upper_tri))
                else:
                    category_density[cat_idx] = 0.0
            else:
                category_density[cat_idx] = 1.0  # Single-memory category is perfectly dense

        return {
            "num_categories": len(self.category_prototypes),
            "memories_per_category": category_counts,
            "category_activations": {
                i: float(act) for i, act in enumerate(self.category_activations)
            },
            "average_memory_activation_per_category": category_avg_activation,
            "category_similarities": category_similarities,
            "category_density": category_density,
            "current_vigilance": float(self.vigilance_threshold)
            if self.dynamic_vigilance
            else None,
        }

    def get_category_for_memory(self, memory_idx: int) -> int:
        """
        Get the category index for a memory.

        Args:
            memory_idx: Index of the memory

        Returns:
            Category index for the memory
        """
        if memory_idx < 0 or memory_idx >= len(self.memory_categories):
            raise IndexError(f"Memory index {memory_idx} out of range")

        return int(self.memory_categories[memory_idx])

    def get_memories_for_category(self, category_idx: int) -> List[int]:
        """
        Get all memory indices for a category.

        Args:
            category_idx: Index of the category

        Returns:
            List of memory indices in the category
        """
        return np.where(self.memory_categories == category_idx)[0].tolist()
    
    def get_category_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Get similarities between a query and all category prototypes.
        
        Args:
            query_embedding: Embedding vector of the query
            
        Returns:
            Array of similarity scores for each category
        """
        if len(self.category_prototypes) == 0:
            return np.array([], dtype=np.float32)
            
        # Normalize the query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
            
        # Calculate dot product similarity with all category prototypes
        similarities = np.dot(self.category_prototypes, query_embedding)
        
        # Weight similarities by category activation levels
        if self.category_activations is not None and len(self.category_activations) > 0:
            similarities = similarities * self.category_activations
            
        return similarities

    def consolidate_categories_manually(self, threshold: float = None) -> int:
        """
        Manually trigger category consolidation with an optional custom threshold.

        Args:
            threshold: Custom similarity threshold (overrides self.consolidation_threshold if provided)

        Returns:
            Number of categories after consolidation
        """
        if threshold is not None:
            # Store original threshold
            original_threshold = self.consolidation_threshold
            # Set custom threshold
            self.consolidation_threshold = threshold

        # Perform consolidation
        self._consolidate_categories()

        # Restore original threshold if needed
        if threshold is not None:
            self.consolidation_threshold = original_threshold

        return len(self.category_prototypes)
