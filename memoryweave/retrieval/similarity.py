"""Similarity-based retrieval strategy for MemoryWeave.

This module provides implementations of retrieval strategies based on
vector similarity between query and memory embeddings.
"""

from typing import Any, Dict, List, Optional

from memoryweave.interfaces.memory import EmbeddingVector, IMemoryStore, IVectorStore
from memoryweave.interfaces.retrieval import (
    IRetrievalStrategy,
    RetrievalParameters,
    RetrievalResult,
)


class SimilarityRetrievalStrategy(IRetrievalStrategy):
    """Retrieval strategy based on pure vector similarity."""

    def __init__(self, memory_store: IMemoryStore, vector_store: IVectorStore):
        """Initialize the similarity retrieval strategy.

        Args:
            memory_store: Memory store to retrieve memory content
            vector_store: Vector store for similarity search
        """
        self._memory_store = memory_store
        self._vector_store = vector_store
        self._default_params = {"similarity_threshold": 0.6, "max_results": 10}

    def retrieve(
        self, query_embedding: EmbeddingVector, parameters: Optional[RetrievalParameters] = None
    ) -> List[RetrievalResult]:
        """Retrieve memories based on a query embedding."""
        # Merge parameters with defaults
        params = self._default_params.copy()
        if parameters:
            params.update(parameters)

        # Get similarity threshold and max results
        similarity_threshold = params.get("similarity_threshold", 0.6)
        max_results = params.get("max_results", 10)

        # Search for similar vectors
        similar_vectors = self._vector_store.search(
            query_vector=query_embedding, k=max_results, threshold=similarity_threshold
        )

        # Convert results to RetrievalResult format
        results = []
        for memory_id, similarity_score in similar_vectors:
            # Retrieve memory content
            memory = self._memory_store.get(memory_id)

            # Create result
            result = RetrievalResult(
                memory_id=memory_id,
                content=memory.content["text"],
                metadata=memory.metadata,
                relevance_score=float(similarity_score),
            )

            results.append(result)

        return results

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the retrieval strategy."""
        if "similarity_threshold" in config:
            self._default_params["similarity_threshold"] = config["similarity_threshold"]

        if "max_results" in config:
            self._default_params["max_results"] = config["max_results"]
