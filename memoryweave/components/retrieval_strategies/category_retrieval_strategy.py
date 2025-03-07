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
from memoryweave.interfaces.memory import EmbeddingVector

logger = logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("memoryweave")


class CategoryRetrievalStrategy(RetrievalStrategy):
    """
    Retrieval strategy that utilizes category information for better results.

    This strategy:
    1. Identifies the most relevant category for a query
    2. Prioritizes memories from that category
    3. Boosts semantically related memories
    """

    def __init__(self, memory, category_manager: Optional[CategoryManager] = None):
        """
        Initialize the category retrieval strategy.

        Args:
            memory: Memory store to retrieve from
            category_manager: CategoryManager component to use
        """
        super().__init__(memory)
        self.category_manager = category_manager
        self.confidence_threshold = 0.0
        self.primary_boost_factor = 1.5
        self.fallback_threshold_factor = 0.8
        self.max_category_results = 20

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the strategy with configuration.

        Args:
            config: Configuration dictionary with parameters:
                - confidence_threshold: Minimum relevance threshold (default: 0.0)
                - primary_boost_factor: How much to boost primary category results (default: 1.5)
                - fallback_threshold_factor: Factor for fallback threshold (default: 0.8)
                - max_category_results: Maximum results to consider from primary category (default: 20)
                - category_manager: CategoryManager instance
        """
        self.confidence_threshold = config.get("confidence_threshold", 0.0)
        self.primary_boost_factor = config.get("primary_boost_factor", 1.5)
        self.fallback_threshold_factor = config.get("fallback_threshold_factor", 0.8)
        self.max_category_results = config.get("max_category_results", 20)

        if "category_manager" in config:
            self.category_manager = config["category_manager"]

    def retrieve(
        self, query_embedding: EmbeddingVector, top_k: int, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories based on category matching.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of results to return
            **kwargs: Additional retrieval arguments

        Returns:
            list of retrieval results
        """
        # Handle case where category_manager is missing
        if self.category_manager is None:
            # Fall back to base similarity search
            return super().retrieve(query_embedding, top_k, **kwargs)

        # Find the best category for this query
        best_category_id = -1
        best_similarity = -1.0

        for category_id, category in self.category_manager.categories.items():
            prototype = category.get("prototype")
            if prototype is not None:
                similarity = self._calculate_similarity(query_embedding, prototype)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category_id = category_id

        # Get results
        results = []

        # If we found a good category match
        if best_category_id >= 0 and best_similarity >= self.confidence_threshold:
            # Get category members
            category_members = self.category_manager.get_category_members(best_category_id)

            # Retrieve memories from this category
            category_results = []

            # For each member, calculate similarity to query
            for memory_id in category_members:
                try:
                    memory = self.memory.get(memory_id)
                    similarity = self._calculate_similarity(query_embedding, memory.embedding)

                    if similarity >= self.confidence_threshold:
                        category_results.append(
                            {
                                "memory_id": memory_id,
                                "content": memory.content,
                                "metadata": memory.metadata,
                                "relevance_score": similarity
                                * self.primary_boost_factor,  # Boost score
                                "category_id": best_category_id,
                                "retrieval_type": "category",
                            }
                        )
                except Exception as e:
                    logger.debug(f"Error retrieving memory {memory_id}: {e}")
                    continue

            # Sort by relevance
            category_results = sorted(
                category_results, key=lambda x: x["relevance_score"], reverse=True
            )[: self.max_category_results]

            # Add to results
            results.extend(category_results)

        # If we didn't get enough results, add general similarity results
        if len(results) < top_k:
            # Lower threshold for fallback search
            fallback_threshold = self.confidence_threshold * self.fallback_threshold_factor

            # Check which IDs we already have to avoid duplicates
            existing_ids = {r["memory_id"] for r in results}

            # General search with memory adapter
            general_results = self._retrieve_by_similarity(
                query_embedding, top_k=top_k * 2, confidence_threshold=fallback_threshold, **kwargs
            )

            # Filter out duplicates
            for result in general_results:
                if result["memory_id"] not in existing_ids:
                    results.append(result)
                    existing_ids.add(result["memory_id"])

                    # Stop when we have enough
                    if len(results) >= top_k:
                        break

        # Sort by relevance and limit to top_k
        results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_k]

        return results

    def _retrieve_by_similarity(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        confidence_threshold: float = 0.0,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Retrieve by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of results to return
            confidence_threshold: Minimum similarity threshold
            **kwargs: Additional retrieval arguments

        Returns:
            list of retrieval results
        """
        # Use memory store's retrieve_memories method if available
        if hasattr(self.memory, "retrieve_memories"):
            # Retrieve memories with embedding
            retrieval_results = self.memory.retrieve_memories(
                query_embedding, top_k=top_k, confidence_threshold=confidence_threshold, **kwargs
            )

            # Format results
            results = []
            for memory_id, score, _content in retrieval_results:
                if score >= confidence_threshold:
                    try:
                        memory = self.memory.get(memory_id)
                        results.append(
                            {
                                "memory_id": memory_id,
                                "content": memory.content,
                                "metadata": memory.metadata,
                                "relevance_score": score,
                                "retrieval_type": "similarity",
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Error retrieving memory {memory_id}: {e}")
                        continue

            return results

        # Fallback: Direct similarity calculation
        else:
            # Get all memories
            all_memories = self.memory.get_all()

            # Calculate similarities
            results = []
            for memory in all_memories:
                similarity = self._calculate_similarity(query_embedding, memory.embedding)

                if similarity >= confidence_threshold:
                    results.append(
                        {
                            "memory_id": memory.id,
                            "content": memory.content,
                            "metadata": memory.metadata,
                            "relevance_score": similarity,
                            "retrieval_type": "similarity",
                        }
                    )

            # Sort by relevance and limit to top_k
            results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_k]

            return results

    def _calculate_similarity(self, vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
