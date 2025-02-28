# memoryweave/components/retrieval_strategies.py
from typing import Any

import numpy as np

from memoryweave.components.base import RetrievalStrategy
from memoryweave.core import ContextualMemory


class SimilarityRetrievalStrategy(RetrievalStrategy):
    """
    Retrieves memories based purely on similarity to query embedding.
    """

    def __init__(self, memory: ContextualMemory):
        self.memory = memory

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.confidence_threshold = config.get("confidence_threshold", 0.0)
        self.activation_boost = config.get("activation_boost", True)

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Retrieve memories based on similarity to query embedding."""
        # Use memory's retrieve_memories with similarity approach
        results = self.memory.retrieve_memories(
            query_embedding,
            top_k=top_k,
            activation_boost=self.activation_boost,
            confidence_threshold=self.confidence_threshold,
        )

        # Format results
        formatted_results = []
        for idx, score, metadata in results:
            formatted_results.append({"memory_id": idx, "relevance_score": score, **metadata})

        return formatted_results


class TemporalRetrievalStrategy(RetrievalStrategy):
    """
    Retrieves memories based on recency and activation.
    """

    def __init__(self, memory: ContextualMemory):
        self.memory = memory

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        pass

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Retrieve memories based on temporal factors."""
        # Get memories sorted by temporal markers (most recent first)
        temporal_order = np.argsort(-self.memory.temporal_markers)[:top_k]

        results = []
        for idx in temporal_order:
            results.append({
                "memory_id": int(idx),
                "relevance_score": float(self.memory.activation_levels[idx]),
                **self.memory.memory_metadata[idx],
            })

        return results


class HybridRetrievalStrategy(RetrievalStrategy):
    """
    Hybrid retrieval combining similarity, recency, and keyword matching.
    """

    def __init__(self, memory: ContextualMemory):
        self.memory = memory

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.relevance_weight = config.get("relevance_weight", 0.7)
        self.recency_weight = config.get("recency_weight", 0.3)
        self.confidence_threshold = config.get("confidence_threshold", 0.0)

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Retrieve memories using hybrid approach."""
        # Get similarity scores
        similarities = np.dot(self.memory.memory_embeddings, query_embedding)

        # Normalize temporal factors
        max_time = float(self.memory.current_time)
        temporal_factors = self.memory.temporal_markers / max_time if max_time > 0 else 0

        # Combine scores
        combined_scores = (
            self.relevance_weight * similarities + self.recency_weight * temporal_factors
        )

        # Apply activation boost
        combined_scores = combined_scores * self.memory.activation_levels

        # Apply confidence threshold filtering
        valid_indices = np.where(combined_scores >= self.confidence_threshold)[0]
        if len(valid_indices) == 0:
            return []

        # Get top-k indices from valid indices
        array_size = len(valid_indices)
        if top_k >= array_size:
            top_relative_indices = np.argsort(-combined_scores[valid_indices])
        else:
            top_relative_indices = np.argpartition(-combined_scores[valid_indices], top_k)[:top_k]
            top_relative_indices = top_relative_indices[
                np.argsort(-combined_scores[valid_indices][top_relative_indices])
            ]

        # Format results
        results = []
        for idx in valid_indices[top_relative_indices]:
            score = float(combined_scores[idx])
            results.append({
                "memory_id": int(idx),
                "relevance_score": score,
                "similarity": float(similarities[idx]),
                "recency": float(temporal_factors[idx]),
                **self.memory.memory_metadata[idx],
            })

        return results[:top_k]
