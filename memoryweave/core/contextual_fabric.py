"""
Core implementation of the MemoryWeave contextual fabric.
"""

from typing import Optional, List, Tuple

import numpy as np


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
    ):
        """
        Initialize the contextual memory system.

        Args:
            embedding_dim: Dimension of the contextual embeddings
            max_memories: Maximum number of memory traces to maintain
            activation_threshold: Threshold for memory activation
            use_art_clustering: Whether to use ART-inspired clustering
            vigilance_threshold: Threshold for creating new categories (ART vigilance)
            learning_rate: Rate at which category prototypes are updated
        """
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.activation_threshold = activation_threshold
        
        # ART-related parameters
        self.use_art_clustering = use_art_clustering
        self.vigilance_threshold = vigilance_threshold
        self.learning_rate = learning_rate

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
            category_idx = self._assign_to_category(embedding)
            self.memory_categories = np.append(self.memory_categories, category_idx)
            # Update category activation
            self.category_activations[category_idx] = 1.0

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
    ) -> list[tuple[int, float, dict]]:
        """
        Retrieve relevant memories based on contextual similarity.

        Args:
            query_embedding: Embedding of the query context
            top_k: Number of memories to retrieve
            activation_boost: Whether to boost by activation level
            use_categories: Whether to use category-based retrieval (defaults to self.use_art_clustering)

        Returns:
            list of (memory_idx, similarity_score, metadata) tuples
        """
        if len(self.memory_metadata) == 0:
            return []

        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Determine whether to use categories
        if use_categories is None:
            use_categories = self.use_art_clustering

        # Use category-based retrieval if enabled
        if use_categories and len(self.category_prototypes) > 0:
            return self._retrieve_with_categories(query_embedding, top_k, activation_boost)
        
        # Standard similarity-based retrieval
        # Compute similarities
        similarities = np.dot(self.memory_embeddings, query_embedding)

        # Apply activation boosting if enabled
        if activation_boost:
            similarities = similarities * self.activation_levels

        # Get top-k indices
        if top_k >= len(similarities):
            top_indices = np.argsort(-similarities)
        else:
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Update activation levels for retrieved memories
        for idx in top_indices:
            self._update_activation(idx)

        # Return results with metadata
        results = []
        for idx in top_indices:
            # Always include at least top_k results, regardless of threshold
            # This ensures we always get some results
            results.append((int(idx), float(similarities[idx]), self.memory_metadata[idx]))
            if len(results) >= top_k:
                break

        return results
    
    def _retrieve_with_categories(
        self, 
        query_embedding: np.ndarray, 
        top_k: int, 
        activation_boost: bool
    ) -> list[tuple[int, float, dict]]:
        """
        Retrieve memories using ART-inspired category-based approach.
        
        Args:
            query_embedding: Embedding of the query context
            top_k: Number of memories to retrieve
            activation_boost: Whether to boost by activation level
            
        Returns:
            list of (memory_idx, similarity_score, metadata) tuples
        """
        # First, find resonating categories
        category_similarities = np.dot(self.category_prototypes, query_embedding)
        
        # Apply category activation boost if enabled
        if activation_boost:
            category_similarities = category_similarities * self.category_activations
        
        # Get top categories (more than we need to ensure enough memories)
        num_categories = min(3, len(category_similarities))
        if num_categories == 0:
            return []
            
        top_category_indices = np.argpartition(-category_similarities, num_categories)[:num_categories]
        
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
            candidate_similarities = candidate_similarities * self.activation_levels[candidate_indices]
        
        # Get top-k memories from candidates
        if top_k >= len(candidate_similarities):
            top_memory_indices = np.argsort(-candidate_similarities)
        else:
            top_memory_indices = np.argpartition(-candidate_similarities, top_k)[:top_k]
            top_memory_indices = top_memory_indices[np.argsort(-candidate_similarities[top_memory_indices])]
        
        # Map back to original indices
        top_indices = [candidate_indices[i] for i in top_memory_indices]
        
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
            similarity = candidate_similarities[top_memory_indices[i]]
            # Always include results regardless of threshold
            results.append((int(idx), float(similarity), self.memory_metadata[idx]))
        
        return results

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
        self.category_activations[category_idx] = min(1.0, self.category_activations[category_idx] + 0.2)
        
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
            (1 - self.learning_rate) * self.category_prototypes[category_idx] + 
            self.learning_rate * embedding
        )
        
        # Normalize the updated prototype
        self.category_prototypes[category_idx] /= np.linalg.norm(self.category_prototypes[category_idx])
    
    def _remove_category(self, category_idx: int) -> None:
        """
        Remove a category and its prototype.
        
        Args:
            category_idx: Index of the category to remove
        """
        self.category_prototypes = np.delete(self.category_prototypes, category_idx, axis=0)
        self.category_activations = np.delete(self.category_activations, category_idx)
    
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
                
        return {
            "num_categories": len(self.category_prototypes),
            "memories_per_category": category_counts,
            "category_activations": {i: float(act) for i, act in enumerate(self.category_activations)},
            "average_memory_activation_per_category": category_avg_activation
        }
