"""Similarity-based retrieval strategy for MemoryWeave.

This module provides implementations of retrieval strategies based on
vector similarity between query and memory embeddings.
"""

from typing import Any, Optional

import numpy as np

from memoryweave.interfaces.memory import EmbeddingVector
from memoryweave.interfaces.retrieval import (
    IRetrievalStrategy,
    RetrievalParameters,
    RetrievalResult,
)
from memoryweave.storage.base_store import BaseMemoryStore
from memoryweave.storage.vector_search.base import IVectorSearchProvider


class SimilarityRetrievalStrategy(IRetrievalStrategy):
    """Retrieval strategy based on pure vector similarity."""

    def process(self, input_data: Any) -> Any:
        """Process the input data as a pipeline stage.

        This method implements IPipelineStage.process to make the component
        usable in a pipeline.
        """
        # Handle different types of input
        if isinstance(input_data, dict):
            # Check if this is a structured query
            if "embedding" in input_data:
                # Get query embedding and parameters
                query_embedding = input_data["embedding"]
                parameters = input_data.get("parameters", {})

                # Retrieve memories based on the query
                memories = self.retrieve(query_embedding, parameters)

                # Return the memories
                return memories

            # Check if this is just a vector
            elif "query_embedding" in input_data:
                query_embedding = input_data["query_embedding"]
                parameters = input_data.get("parameters", {})

                # Retrieve memories based on the query
                memories = self.retrieve(query_embedding, parameters)

                # Return the memories
                return memories

        # Pass through for unsupported input types
        return input_data

    def __init__(self, memory_store: BaseMemoryStore, vector_store: IVectorSearchProvider):
        """Initialize the similarity retrieval strategy.

        Args:
            memory_store: Memory store to retrieve memory content
            vector_store: Vector store for similarity search
        """
        self._memory_store = memory_store
        self._vector_store = vector_store
        self._default_params = {"similarity_threshold": 0.6, "max_results": 10}
        self.component_id = "similarity_retrieval_strategy"

    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        return self.component_id

    def get_type(self):
        """Get the type of this component."""
        from memoryweave.interfaces.pipeline import ComponentType

        return ComponentType.RETRIEVAL_STRATEGY

    def retrieve(
        self, query_embedding: EmbeddingVector, parameters: Optional[RetrievalParameters] = None
    ) -> list[RetrievalResult]:
        """Retrieve memories based on a query embedding."""
        # Merge parameters with defaults
        params = self._default_params.copy()
        if parameters:
            params.update(parameters)

        # Get similarity threshold and max results
        similarity_threshold = params.get("similarity_threshold", 0.6)
        max_results = params.get("max_results", 10)
        min_results = params.get("min_results", 0)

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

        # If no results meet the threshold but min_results is set, return at least that many
        if len(results) < min_results and min_results > 0:
            # Special case when memory store has no memories
            if len(self._memory_store._memories) == 0:
                return []

            # Just get all memories if we need to ensure minimum results
            all_memory_ids = list(self._memory_store._memories.keys())

            # Calculate similarity scores for all memories
            similarity_scores = {}
            for memory_id in all_memory_ids:
                memory = self._memory_store.get(memory_id)
                try:
                    embedding = memory.embedding  # This is the standard way to access it
                    # Calculate cosine similarity
                    query_norm = np.linalg.norm(query_embedding)
                    memory_norm = np.linalg.norm(embedding)
                    if query_norm > 0 and memory_norm > 0:
                        similarity = np.dot(query_embedding, embedding) / (query_norm * memory_norm)
                        similarity_scores[memory_id] = similarity
                except Exception:  # noqa: S110
                    pass  # Skip memories with missing or invalid embeddings

            # Sort by similarity
            sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

            # Get top min_results
            more_vectors = sorted_scores[:min_results]

            # Only add memories not already in results
            existing_ids = {r["memory_id"] for r in results}
            for memory_id, similarity_score in more_vectors:
                if memory_id not in existing_ids and len(results) < min_results:
                    memory = self._memory_store.get(memory_id)
                    result = RetrievalResult(
                        memory_id=memory_id,
                        content=memory.content["text"],
                        metadata=memory.metadata,
                        relevance_score=float(similarity_score),
                    )
                    results.append(result)
                    existing_ids.add(memory_id)

            # Re-sort the results
            results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)

        return results

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the retrieval strategy."""
        if "similarity_threshold" in config:
            self._default_params["similarity_threshold"] = config["similarity_threshold"]

        if "max_results" in config:
            self._default_params["max_results"] = config["max_results"]
