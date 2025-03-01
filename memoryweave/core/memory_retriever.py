"""
Memory retriever implementation for MemoryWeave.
"""

from typing import Any, Optional

import numpy as np


class MemoryRetriever:
    """
    Retrieves memories from the memory system based on various strategies.
    """

    def __init__(
        self,
        core_memory: Any,
        category_manager: Optional[Any] = None,
        default_confidence_threshold: float = 0.0,
        adaptive_retrieval: bool = False,
        semantic_coherence_check: bool = False,
        coherence_threshold: float = 0.2,
    ):
        """
        Initialize the memory retriever.

        Args:
            core_memory: The core memory to retrieve from
            category_manager: Optional category manager for category-based retrieval
            default_confidence_threshold: Default minimum similarity score for memory inclusion
            adaptive_retrieval: Whether to use adaptive k selection based on relevance distribution
            semantic_coherence_check: Whether to check semantic coherence of retrieved memories
            coherence_threshold: Threshold for semantic coherence between memories
        """
        self.core_memory = core_memory
        self.category_manager = category_manager
        self.default_confidence_threshold = default_confidence_threshold
        self.adaptive_retrieval = adaptive_retrieval
        self.semantic_coherence_check = semantic_coherence_check
        self.coherence_threshold = coherence_threshold

    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        activation_boost: bool = True,
        use_categories: bool = None,
        confidence_threshold: float = None,
        max_k_override: bool = False,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """
        Retrieve relevant memories based on contextual similarity.

        Args:
            query_embedding: Embedding of the query context
            top_k: Number of memories to retrieve
            activation_boost: Whether to boost by activation level
            use_categories: Whether to use category-based retrieval
            confidence_threshold: Minimum similarity score threshold for inclusion
            max_k_override: Whether to return exactly top_k results even if fewer meet the threshold

        Returns:
            list of (memory_idx, similarity_score, metadata) tuples
        """
        # Check if memory is empty
        if hasattr(self.core_memory, "get_memory_count"):
            if self.core_memory.get_memory_count() == 0:
                return []
        elif hasattr(self.core_memory, "memory_metadata"):
            if len(self.core_memory.memory_metadata) == 0:
                return []
        else:
            # Try to determine if memory is empty in some other way
            try:
                if len(self.core_memory.memory_embeddings) == 0:
                    return []
            except (AttributeError, TypeError):
                # If we can't determine, assume it's not empty
                pass

        # Use default confidence threshold if none provided
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold

        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Determine whether to use categories
        if use_categories is None:
            use_categories = self.category_manager is not None

        # Use category-based retrieval if enabled
        if use_categories and self.category_manager is not None:
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
    ) -> list[tuple[int, float, dict[str, Any]]]:
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
        # Handle different memory implementations
        if hasattr(self.core_memory, "retrieve_memories"):
            # If core_memory has its own retrieve_memories method, use it
            return self.core_memory.retrieve_memories(
                query_embedding=query_embedding,
                top_k=top_k,
                activation_boost=activation_boost,
                confidence_threshold=confidence_threshold,
            )

        # Otherwise, implement the retrieval logic here
        # Compute similarities
        similarities = np.dot(self.core_memory.memory_embeddings, query_embedding)

        # Apply activation boosting if enabled
        if activation_boost:
            similarities = similarities * self.core_memory.activation_levels

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
            results.append((
                int(idx),
                float(similarities[idx]),
                self.core_memory.memory_metadata[idx],
            ))
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
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """
        Retrieve memories using category-based approach.

        Args:
            query_embedding: Embedding of the query context
            top_k: Number of memories to retrieve
            activation_boost: Whether to boost by activation level
            confidence_threshold: Minimum similarity score threshold
            max_k_override: Whether to return exactly top_k results

        Returns:
            list of (memory_idx, similarity_score, metadata) tuples
        """
        # If the core_memory has its own category-based retrieval, use it
        if hasattr(self.core_memory, "retrieve_memories") and hasattr(
            self.core_memory, "use_art_clustering"
        ):
            if self.core_memory.use_art_clustering:
                return self.core_memory.retrieve_memories(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    activation_boost=activation_boost,
                    use_categories=True,
                    confidence_threshold=confidence_threshold,
                )

        # Otherwise, implement category-based retrieval here
        # This is a simplified implementation that would need to be expanded
        # based on the actual category manager implementation
        if self.category_manager is None:
            return self._retrieve_with_similarity(
                query_embedding, top_k, activation_boost, confidence_threshold, max_k_override
            )

        # Get category similarities
        category_similarities = self.category_manager.get_category_similarities(query_embedding)

        # Get top categories
        num_categories = min(3, len(category_similarities))
        if num_categories == 0:
            return []

        top_category_indices = np.argpartition(-category_similarities, num_categories)[
            :num_categories
        ]

        # Get memories from top categories
        candidate_indices = []
        for cat_idx in top_category_indices:
            cat_memories = self.category_manager.get_memories_for_category(cat_idx)
            candidate_indices.extend(cat_memories)

        if not candidate_indices:
            return []

        # Calculate similarities for candidate memories
        candidate_similarities = np.dot(
            self.core_memory.memory_embeddings[candidate_indices], query_embedding
        )

        # Apply activation boost
        if activation_boost:
            candidate_similarities = (
                candidate_similarities * self.core_memory.activation_levels[candidate_indices]
            )

        # Filter by confidence threshold
        valid_candidates = np.where(candidate_similarities >= confidence_threshold)[0]
        if len(valid_candidates) == 0:
            return []

        valid_candidate_indices = [candidate_indices[i] for i in valid_candidates]
        valid_candidate_similarities = candidate_similarities[valid_candidates]

        # Get top-k memories
        if top_k >= len(valid_candidate_similarities):
            top_memory_indices = np.argsort(-valid_candidate_similarities)
        else:
            top_memory_indices = np.argpartition(-valid_candidate_similarities, top_k)[:top_k]
            top_memory_indices = top_memory_indices[
                np.argsort(-valid_candidate_similarities[top_memory_indices])
            ]

        # Map back to original indices
        top_indices = [valid_candidate_indices[i] for i in top_memory_indices]

        # Update activations
        for idx in top_indices:
            self._update_activation(idx)

        # Return results
        results = []
        for i, idx in enumerate(top_indices):
            similarity = valid_candidate_similarities[top_memory_indices[i]]
            results.append((int(idx), float(similarity), self.core_memory.memory_metadata[idx]))
            if len(results) >= top_k and not max_k_override:
                break

        return results

    def _apply_coherence_check(
        self,
        retrieved_memories: list[tuple[int, float, dict[str, Any]]],
        query_embedding: np.ndarray,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """
        Apply semantic coherence check to retrieved memories.

        Args:
            retrieved_memories: list of (memory_idx, similarity_score, metadata) tuples
            query_embedding: Query embedding vector

        Returns:
            Filtered list of memories that form a coherent set
        """
        if len(retrieved_memories) <= 1:
            return retrieved_memories

        # Extract indices and embeddings
        indices = [idx for idx, _, _ in retrieved_memories]
        embeddings = self.core_memory.memory_embeddings[indices]

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
        retrieved_memories: list[tuple[int, float, dict[str, Any]]],
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """
        Adaptively select the number of memories to return based on relevance distribution.

        Args:
            retrieved_memories: list of (memory_idx, similarity_score, metadata) tuples

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

    def _update_activation(self, memory_idx: int) -> None:
        """
        Update activation level for a memory that's been accessed.

        Args:
            memory_idx: Index of the memory to update
        """
        # Check if core_memory has an update_activation method
        if hasattr(self.core_memory, "update_activation"):
            self.core_memory.update_activation(memory_idx)
        else:
            # Otherwise, implement a basic activation update
            # Increase activation for accessed memory
            self.core_memory.activation_levels[memory_idx] = min(
                1.0, self.core_memory.activation_levels[memory_idx] + 0.2
            )

            # Update access metadata if available
            if hasattr(self.core_memory, "memory_metadata"):
                if "access_count" in self.core_memory.memory_metadata[memory_idx]:
                    self.core_memory.memory_metadata[memory_idx]["access_count"] += 1
                if hasattr(self.core_memory, "current_time"):
                    self.core_memory.memory_metadata[memory_idx]["last_accessed"] = (
                        self.core_memory.current_time
                    )

            # Decay other activations slightly
            decay_mask = np.ones_like(self.core_memory.activation_levels, dtype=bool)
            decay_mask[memory_idx] = False
            self.core_memory.activation_levels[decay_mask] *= 0.95
