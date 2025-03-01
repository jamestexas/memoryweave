"""
Implementation of memory retrieval strategies for MemoryWeave.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from memoryweave.core.core_memory import CoreMemory
from memoryweave.core.category_manager import CategoryManager


class MemoryRetriever:
    """
    Implements memory retrieval strategies.
    
    This class provides methods for retrieving memories based on similarity,
    category-based approaches, and applies post-processing like semantic
    coherence checking and adaptive k selection.
    """

    def __init__(
        self,
        core_memory: CoreMemory,
        category_manager: Optional[CategoryManager] = None,
        activation_threshold: float = 0.5,
        default_confidence_threshold: float = 0.0,
        adaptive_retrieval: bool = False,
        semantic_coherence_check: bool = False,
        coherence_threshold: float = 0.2,
    ):
        """
        Initialize the memory retriever.

        Args:
            core_memory: Core memory instance
            category_manager: Optional category manager for category-based retrieval
            activation_threshold: Threshold for memory activation
            default_confidence_threshold: Default minimum similarity score for memory retrieval
            adaptive_retrieval: Whether to use adaptive k selection
            semantic_coherence_check: Whether to check semantic coherence of retrieved memories
            coherence_threshold: Threshold for semantic coherence between memories
        """
        self.core_memory = core_memory
        self.category_manager = category_manager
        self.activation_threshold = activation_threshold
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
    ) -> List[Tuple[int, float, Dict]]:
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
            List of (memory_idx, similarity_score, metadata) tuples
        """
        if self.core_memory.get_memory_count() == 0:
            return []

        # Use default confidence threshold if none provided
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold

        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Determine whether to use categories
        if use_categories is None:
            use_categories = self.category_manager is not None

        # Use category-based retrieval if enabled and category manager is available
        if use_categories and self.category_manager and len(self.category_manager.category_prototypes) > 0:
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
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve memories based on direct similarity comparison.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of memories to retrieve
            activation_boost: Whether to boost by activation level
            confidence_threshold: Minimum similarity threshold
            max_k_override: Whether to return exactly top_k results

        Returns:
            List of (memory_idx, similarity_score, metadata) tuples
        """
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
            top_relative_indices = np.argpartition(-valid_similarities, min(top_k-1, len(valid_similarities)-1))[:top_k]
            top_relative_indices = top_relative_indices[
                np.argsort(-valid_similarities[top_relative_indices])
            ]

        # Convert back to original indices
        top_indices = valid_indices[top_relative_indices]

        # Update activation levels for retrieved memories
        for idx in top_indices:
            self.core_memory.update_activation(idx)

        # Return results with metadata
        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarities[idx]), self.core_memory.memory_metadata[idx]))
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
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve memories using ART-inspired category-based approach.

        Args:
            query_embedding: Embedding of the query context
            top_k: Number of memories to retrieve
            activation_boost: Whether to boost by activation level
            confidence_threshold: Minimum similarity score threshold
            max_k_override: Whether to return exactly top_k results

        Returns:
            List of (memory_idx, similarity_score, metadata) tuples
        """
        # First, find resonating categories
        category_similarities = np.dot(self.category_manager.category_prototypes, query_embedding)

        # Apply category activation boost if enabled
        if activation_boost:
            category_similarities = category_similarities * self.category_manager.category_activations

        # Get top categories (more than we need to ensure enough memories)
        # Filter by confidence threshold if specified
        valid_categories = np.where(category_similarities >= confidence_threshold)[0]
        if len(valid_categories) == 0:
            return []

        valid_category_similarities = category_similarities[valid_categories]

        # Fix: Ensure num_categories doesn't exceed the length of valid_category_similarities
        num_categories = min(3, len(valid_category_similarities))
        if num_categories == 0:
            return []

        # Fix: Use min(num_categories-1, len(valid_category_similarities)-1) to avoid out of bounds
        if num_categories > 1:
            top_category_indices_rel = np.argpartition(
                -valid_category_similarities, 
                min(num_categories-1, len(valid_category_similarities)-1)
            )[:num_categories]
        else:
            # If we only need one category, just take the argmax
            top_category_indices_rel = np.array([np.argmax(valid_category_similarities)])
            
        top_category_indices = valid_categories[top_category_indices_rel]

        # Collect candidate memories from top categories
        candidate_indices = []
        for cat_idx in top_category_indices:
            # Find memories in this category
            cat_memories = np.where(self.category_manager.memory_categories == cat_idx)[0]
            candidate_indices.extend(cat_memories)

        if not candidate_indices:
            return []

        # Calculate similarities for candidate memories
        candidate_similarities = np.dot(self.core_memory.memory_embeddings[candidate_indices], query_embedding)

        # Apply memory activation boost if enabled
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

        # Get top-k memories from candidates
        if top_k >= len(valid_candidate_similarities):
            top_memory_indices = np.argsort(-valid_candidate_similarities)
        else:
            # Fix: Use min(top_k-1, len(valid_candidate_similarities)-1) to avoid out of bounds
            top_memory_indices = np.argpartition(
                -valid_candidate_similarities, 
                min(top_k-1, len(valid_candidate_similarities)-1)
            )[:top_k]
            top_memory_indices = top_memory_indices[
                np.argsort(-valid_candidate_similarities[top_memory_indices])
            ]

        # Map back to original indices
        top_indices = [valid_candidate_indices[i] for i in top_memory_indices]

        # Update activation levels for retrieved memories
        for idx in top_indices:
            self.core_memory.update_activation(idx)

            # Also update category activations
            cat_idx = self.category_manager.memory_categories[idx]
            self.category_manager.update_category_activation(cat_idx)

        # Return results with metadata
        results = []
        for i, idx in enumerate(top_indices):
            similarity = valid_candidate_similarities[top_memory_indices[i]]
            results.append((int(idx), float(similarity), self.core_memory.memory_metadata[idx]))
            if len(results) >= top_k and not max_k_override:
                break

        return results

    def _apply_coherence_check(
        self,
        retrieved_memories: List[Tuple[int, float, Dict]],
        query_embedding: np.ndarray,
    ) -> List[Tuple[int, float, Dict]]:
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
        retrieved_memories: List[Tuple[int, float, Dict]],
    ) -> List[Tuple[int, float, Dict]]:
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
