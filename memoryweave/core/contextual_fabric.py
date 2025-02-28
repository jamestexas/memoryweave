"""
Core implementation of the MemoryWeave contextual fabric.
"""

from typing import Literal, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


class ContextualMemory:
    """
    Implements a contextual fabric approach to memory management.
    Rather than storing discrete memory nodes, this captures rich
    contextual signatures of information with associative patterns.
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
            default_confidence_threshold: Default minimum similarity score for memory retrieval
            adaptive_retrieval: Whether to use adaptive k selection based on relevance distribution
            semantic_coherence_check: Whether to check semantic coherence of retrieved memories
            coherence_threshold: Threshold for semantic coherence between memories
        """
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.activation_threshold = activation_threshold

        # ART-related parameters
        self.use_art_clustering = use_art_clustering
        self.initial_vigilance = vigilance_threshold
        self.vigilance_threshold = vigilance_threshold
        self.learning_rate = learning_rate

        # Dynamic vigilance parameters
        self.dynamic_vigilance = dynamic_vigilance
        self.vigilance_strategy = vigilance_strategy
        self.min_vigilance = min_vigilance
        self.max_vigilance = max_vigilance
        self.target_categories = target_categories
        self.memories_added = 0

        # Memory fabric stores both the embeddings and their associated metadata
        self.memory_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
        self.memory_metadata = []

        # Activation levels track recent access/relevance
        self.activation_levels = np.zeros(0, dtype=np.float32)

        # Temporal markers to capture sequence and episodic structure
        self.temporal_markers = np.zeros(0, dtype=np.int64)
        self.current_time = 0

        # ART-inspired category structures
        if use_art_clustering:
            # Category prototypes (centroids)
            self.category_prototypes = np.zeros((0, embedding_dim), dtype=np.float32)
            # Memory to category mappings
            self.memory_categories = np.zeros(0, dtype=np.int64)
            # Category activation levels
            self.category_activations = np.zeros(0, dtype=np.float32)

            # Category consolidation parameters
            self.enable_category_consolidation = enable_category_consolidation
            self.consolidation_threshold = consolidation_threshold
            self.min_category_size = min_category_size
            self.consolidation_frequency = consolidation_frequency
            self.hierarchical_method = hierarchical_method
            self.last_consolidation = 0

        # Confidence thresholding parameters
        self.default_confidence_threshold = default_confidence_threshold
        self.adaptive_retrieval = adaptive_retrieval
        self.semantic_coherence_check = semantic_coherence_check
        self.coherence_threshold = coherence_threshold

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
        if metadata is None:
            metadata = {}

        # Update time counter
        self.current_time += 1
        self.memories_added += 1

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Add new memory
        self.memory_embeddings = np.vstack([self.memory_embeddings, embedding])

        # Store metadata and text
        full_metadata = {
            "text": text,
            "created_at": self.current_time,
            "access_count": 0,
            **metadata,
        }
        self.memory_metadata.append(full_metadata)

        # Initialize activation and temporal marker
        self.activation_levels = np.append(self.activation_levels, 1.0)
        self.temporal_markers = np.append(self.temporal_markers, self.current_time)

        # If using ART clustering, assign to a category
        if self.use_art_clustering:
            # Update vigilance if using dynamic vigilance
            if self.dynamic_vigilance:
                self._update_vigilance()

            category_idx = self._assign_to_category(embedding)
            self.memory_categories = np.append(self.memory_categories, category_idx)
            # Update category activation
            self.category_activations[category_idx] = 1.0

            # Check if it's time to run category consolidation
            if (
                self.enable_category_consolidation
                and self.memories_added >= self.last_consolidation + self.consolidation_frequency
                and len(self.category_prototypes) > 1
            ):
                self._consolidate_categories()
                self.last_consolidation = self.memories_added

        # Manage memory capacity if needed
        if len(self.memory_metadata) > self.max_memories:
            self._consolidate_memories()

        return len(self.memory_metadata) - 1

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
            use_categories: Whether to use category-based retrieval (defaults to self.use_art_clustering)
            confidence_threshold: Minimum similarity score threshold for inclusion (overrides default if provided)
            max_k_override: Whether to return exactly top_k results even if fewer meet the threshold

        Returns:
            list of (memory_idx, similarity_score, metadata) tuples
        """
        if len(self.memory_metadata) == 0:
            return []

        # Use default confidence threshold if none provided
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold

        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Determine whether to use categories
        if use_categories is None:
            use_categories = self.use_art_clustering

        # Use category-based retrieval if enabled
        if use_categories and len(self.category_prototypes) > 0:
            results = self._retrieve_with_categories(
                query_embedding,
                top_k,
                activation_boost,
                confidence_threshold,
                max_k_override,
            )
        else:
            # Standard similarity-based retrieval
            results = self._retrieve_with_similarity(
                query_embedding,
                top_k,
                activation_boost,
                confidence_threshold,
                max_k_override,
            )

        # Apply semantic coherence check if enabled
        if self.semantic_coherence_check and len(results) > 1:
            results = self._apply_coherence_check(results, query_embedding)

        # Apply adaptive k selection if enabled
        if self.adaptive_retrieval and not max_k_override:
            results = self._adaptive_k_selection(results)

        return results

    def _retrieve_with_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        activation_boost: bool,
        confidence_threshold: float,
        max_k_override: bool,
    ) -> list[tuple[int, float, dict]]:
        """
        Retrieve memories based on direct similarity comparison.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of memories to retrieve
            activation_boost: Whether to boost by activation level
            confidence_threshold: Minimum similarity threshold
            max_k_override: Whether to return exactly top_k results

        Returns:
            list of (memory_idx, similarity_score, metadata) tuples
        """
        # Compute similarities
        similarities = np.dot(self.memory_embeddings, query_embedding)

        # Apply activation boosting if enabled
        if activation_boost:
            similarities = similarities * self.activation_levels

        # Filter by confidence threshold if specified
        valid_indices = np.where(similarities >= confidence_threshold)[0]
        if len(valid_indices) == 0:
            return []

        valid_similarities = similarities[valid_indices]

        # Get top-k indices
        if top_k >= len(valid_similarities):
            top_relative_indices = np.argsort(-valid_similarities)
        else:
            top_relative_indices = np.argpartition(-valid_similarities, top_k)[:top_k]
            top_relative_indices = top_relative_indices[
                np.argsort(-valid_similarities[top_relative_indices])
            ]

        # Convert back to original indices
        top_indices = valid_indices[top_relative_indices]

        # Update activation levels for retrieved memories
        for idx in top_indices:
            self._update_activation(idx)

        # Return results with metadata
        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarities[idx]), self.memory_metadata[idx]))
            # If not using max_k_override, we'll respect the confidence threshold
            # Otherwise, we'll keep adding results until we hit top_k
            if len(results) >= top_k and not max_k_override:
                break

        return results

    def _retrieve_with_categories(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        activation_boost: bool,
        confidence_threshold: float = 0.0,
        max_k_override: bool = False,
    ) -> list[tuple[int, float, dict]]:
        """
        Retrieve memories using ART-inspired category-based approach.

        Args:
            query_embedding: Embedding of the query context
            top_k: Number of memories to retrieve
            activation_boost: Whether to boost by activation level
            confidence_threshold: Minimum similarity score threshold
            max_k_override: Whether to return exactly top_k results

        Returns:
            list of (memory_idx, similarity_score, metadata) tuples
        """
        # First, find resonating categories
        category_similarities = np.dot(self.category_prototypes, query_embedding)

        # Apply category activation boost if enabled
        if activation_boost:
            category_similarities = category_similarities * self.category_activations

        # Get top categories (more than we need to ensure enough memories)
        # Filter by confidence threshold if specified
        valid_categories = np.where(category_similarities >= confidence_threshold)[0]
        if len(valid_categories) == 0:
            return []

        valid_category_similarities = category_similarities[valid_categories]

        num_categories = min(3, len(valid_category_similarities))
        if num_categories == 0:
            return []

        top_category_indices_rel = np.argpartition(-valid_category_similarities, num_categories)[
            :num_categories
        ]
        top_category_indices = valid_categories[top_category_indices_rel]

        # Collect candidate memories from top categories
        candidate_indices = []
        for cat_idx in top_category_indices:
            # Find memories in this category
            cat_memories = np.where(self.memory_categories == cat_idx)[0]
            candidate_indices.extend(cat_memories)

        if not candidate_indices:
            return []

        # Calculate similarities for candidate memories
        candidate_similarities = np.dot(self.memory_embeddings[candidate_indices], query_embedding)

        # Apply memory activation boost if enabled
        if activation_boost:
            candidate_similarities = (
                candidate_similarities * self.activation_levels[candidate_indices]
            )

        # Filter by confidence threshold
        valid_candidates = np.where(candidate_similarities >= confidence_threshold)[0]
        if len(valid_candidates) == 0:
            return []

        valid_candidate_indices = [candidate_indices[i] for i in valid_candidates]
        valid_candidate_similarities = candidate_similarities[valid_candidates]

        # Get top-k memories from candidates
        if top_k >= len(valid_candidate_similarities):
            top_memory_indices = np.argsort(-valid_candidate_similarities)
        else:
            top_memory_indices = np.argpartition(-valid_candidate_similarities, top_k)[:top_k]
            top_memory_indices = top_memory_indices[
                np.argsort(-valid_candidate_similarities[top_memory_indices])
            ]

        # Map back to original indices
        top_indices = [valid_candidate_indices[i] for i in top_memory_indices]

        # Update activation levels for retrieved memories
        for idx in top_indices:
            self._update_activation(idx)

            # Also update category activations
            if self.use_art_clustering:
                cat_idx = self.memory_categories[idx]
                self._update_category_activation(cat_idx)

        # Return results with metadata
        results = []
        for i, idx in enumerate(top_indices):
            similarity = valid_candidate_similarities[top_memory_indices[i]]
            results.append((int(idx), float(similarity), self.memory_metadata[idx]))
            if len(results) >= top_k and not max_k_override:
                break

        return results

    def _apply_coherence_check(
        self,
        retrieved_memories: list[tuple[int, float, dict]],
        query_embedding: np.ndarray,
    ) -> list[tuple[int, float, dict]]:
        """
        Apply semantic coherence check to retrieved memories.

        Args:
            retrieved_memories: List of (memory_idx, similarity_score, metadata) tuples
            query_embedding: Query embedding vector

        Returns:
            Filtered list of memories that form a coherent set
        """
        if len(retrieved_memories) <= 1:
            return retrieved_memories

        # Extract indices and embeddings
        indices = [idx for idx, _, _ in retrieved_memories]
        embeddings = self.memory_embeddings[indices]

        # Calculate pairwise similarities between retrieved memories
        pairwise_similarities = np.dot(embeddings, embeddings.T)

        # Find outliers (memories with low average similarity to other memories)
        avg_similarities = (pairwise_similarities.sum(axis=1) - 1) / (
            len(indices) - 1
        )  # Exclude self-similarity

        # Identify coherent memories (those with avg similarity above threshold)
        coherent_mask = avg_similarities >= self.coherence_threshold

        # If all memories are outliers, keep the most relevant one
        if not np.any(coherent_mask):
            best_idx = np.argmax([score for _, score, _ in retrieved_memories])
            return [retrieved_memories[best_idx]]

        # Filter the retrieved memories
        coherent_memories = [m for i, m in enumerate(retrieved_memories) if coherent_mask[i]]

        # If we filtered out too many memories, add back some of the highest scoring ones
        # to ensure we don't lose too much recall
        if len(coherent_memories) < min(3, len(retrieved_memories) // 2):
            # Sort remaining memories by score
            remaining = [(i, m) for i, m in enumerate(retrieved_memories) if not coherent_mask[i]]
            remaining.sort(key=lambda x: x[1][1], reverse=True)  # Sort by similarity score

            # Add back top memories until we have at least 3 or half of original
            while len(coherent_memories) < min(3, len(retrieved_memories) // 2) and remaining:
                _, memory = remaining.pop(0)
                coherent_memories.append(memory)

        return coherent_memories

    def _adaptive_k_selection(
        self,
        retrieved_memories: list[tuple[int, float, dict]],
    ) -> list[tuple[int, float, dict]]:
        """
        Adaptively select the number of memories to return based on relevance distribution.

        Args:
            retrieved_memories: List of (memory_idx, similarity_score, metadata) tuples

        Returns:
            Filtered list with adaptively selected number of memories
        """
        if len(retrieved_memories) <= 1:
            return retrieved_memories

        # Get similarity scores
        scores = np.array([score for _, score, _ in retrieved_memories])

        # Calculate differences between consecutive scores (after sorting)
        sorted_indices = np.argsort(-scores)
        sorted_scores = scores[sorted_indices]

        if len(sorted_scores) <= 1:
            return retrieved_memories

        diffs = np.diff(sorted_scores)

        # Find the largest drop in similarity
        largest_drop_idx = np.argmin(diffs)

        # Only use the cut point if there's a significant drop (>10% of the max score)
        # and we're not cutting off too many memories
        if -diffs[largest_drop_idx] > 0.1 * sorted_scores[0] and largest_drop_idx + 1 >= min(
            3, len(retrieved_memories) // 2
        ):
            # Keep memories up to the largest drop
            selected_indices = sorted_indices[: largest_drop_idx + 1]
            return [retrieved_memories[idx] for idx in selected_indices]

        # Otherwise return all memories
        return retrieved_memories

    def _update_vigilance(self):
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
                target_ratio = self.target_categories / max(self.max_memories, self.memories_added)

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
            if len(self.memory_embeddings) > 10:  # Need enough memories to compute density
                # Compute pairwise similarities as a measure of density
                sample_size = min(
                    50, len(self.memory_embeddings)
                )  # Limit computation for large memories
                indices = np.random.choice(len(self.memory_embeddings), sample_size, replace=False)
                sample_embeddings = self.memory_embeddings[indices]
                similarities = np.dot(sample_embeddings, sample_embeddings.T)

                # Calculate average similarity (excluding self-similarity)
                total_sim = similarities.sum() - sample_size  # Subtract diagonal
                avg_sim = total_sim / (sample_size * (sample_size - 1))

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

    def _update_activation(self, memory_idx: int) -> None:
        """
        Update activation level for a memory that's been accessed.

        Args:
            memory_idx: Index of the memory to update
        """
        # Increase activation for accessed memory
        self.activation_levels[memory_idx] = min(1.0, self.activation_levels[memory_idx] + 0.2)

        # Update access metadata
        self.memory_metadata[memory_idx]["access_count"] += 1
        self.memory_metadata[memory_idx]["last_accessed"] = self.current_time

        # Decay other activations slightly
        decay_mask = np.ones_like(self.activation_levels, dtype=bool)
        decay_mask[memory_idx] = False
        self.activation_levels[decay_mask] *= 0.95

    def _update_category_activation(self, category_idx: int) -> None:
        """
        Update activation level for a category that's been accessed.

        Args:
            category_idx: Index of the category to update
        """
        if not self.use_art_clustering or category_idx >= len(self.category_activations):
            return

        # Increase activation for accessed category
        self.category_activations[category_idx] = min(
            1.0, self.category_activations[category_idx] + 0.2
        )

        # Decay other category activations slightly
        decay_mask = np.ones_like(self.category_activations, dtype=bool)
        decay_mask[category_idx] = False
        self.category_activations[decay_mask] *= 0.95

    def _consolidate_memories(self) -> None:
        """
        Consolidate memories when capacity is reached,
        using activation levels and temporal factors.
        """
        # Compute a combined score for memory importance
        # This considers both activation and recency
        importance = self.activation_levels + 0.2 * (self.temporal_markers / self.current_time)

        # Find the least important memory
        least_important_idx = np.argmin(importance)

        # Remove the least important memory
        self.memory_embeddings = np.delete(self.memory_embeddings, least_important_idx, axis=0)
        self.activation_levels = np.delete(self.activation_levels, least_important_idx)
        self.temporal_markers = np.delete(self.temporal_markers, least_important_idx)

        # Update category mappings if using ART
        if self.use_art_clustering:
            removed_category = self.memory_categories[least_important_idx]
            self.memory_categories = np.delete(self.memory_categories, least_important_idx)

            # Check if this was the last memory in its category
            if removed_category not in self.memory_categories:
                # Remove the category prototype
                self._remove_category(removed_category)

                # Update category indices for memories with higher indices
                self.memory_categories[self.memory_categories > removed_category] -= 1

        del self.memory_metadata[least_important_idx]

    def _assign_to_category(self, embedding: np.ndarray) -> int:
        """
        Assign a memory to a category using ART-inspired resonance.

        Args:
            embedding: The memory embedding to categorize

        Returns:
            Index of the assigned category
        """
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
            return len(self.category_prototypes) - 1

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

    def _remove_category(self, category_idx: int) -> None:
        """
        Remove a category and its prototype.

        Args:
            category_idx: Index of the category to remove
        """
        self.category_prototypes = np.delete(self.category_prototypes, category_idx, axis=0)
        self.category_activations = np.delete(self.category_activations, category_idx)

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
        if not self.use_art_clustering or len(self.category_prototypes) == 0:
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
                category_avg_activation[cat_idx] = float(np.mean(self.activation_levels[mask]))
            else:
                category_avg_activation[cat_idx] = 0.0

        # Calculate pairwise similarity between category prototypes
        category_similarities = {}
        if len(self.category_prototypes) > 1:
            sim_matrix = np.dot(self.category_prototypes, self.category_prototypes.T)
            for i in range(len(self.category_prototypes)):
                for j in range(i + 1, len(self.category_prototypes)):
                    # Only include non-self similarities
                    if i != j:
                        category_similarities[(i, j)] = float(sim_matrix[i, j])

        # Calculate category density (average intra-category similarity)
        category_density = {}
        for cat_idx in range(len(self.category_prototypes)):
            mask = self.memory_categories == cat_idx
            cat_memories = np.where(mask)[0]

            if len(cat_memories) > 1:
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

    def category_similarity_matrix(self) -> np.ndarray:
        """
        Get the similarity matrix between all category prototypes.

        Returns:
            2D numpy array of similarity scores
        """
        if not self.use_art_clustering or len(self.category_prototypes) < 2:
            return np.array([])

        return np.dot(self.category_prototypes, self.category_prototypes.T)

    def consolidate_categories_manually(self, threshold: float = None) -> int:
        """
        Manually trigger category consolidation with an optional custom threshold.

        Args:
            threshold: Custom similarity threshold (overrides self.consolidation_threshold if provided)

        Returns:
            Number of categories after consolidation
        """
        if not self.use_art_clustering:
            return 0

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
