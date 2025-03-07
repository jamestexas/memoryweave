"""Two-stage retrieval strategy for MemoryWeave.

This module provides implementations of a two-stage retrieval approach
where an initial broader search is followed by more refined filtering.
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


class TwoStageRetrievalStrategy(IRetrievalStrategy):
    """Two-stage retrieval strategy with initial broad search and refinement."""

    def __init__(
        self,
        memory_store: BaseMemoryStore,
        vector_store: IVectorSearchProvider,
        first_stage_strategy: Optional[IRetrievalStrategy] = None,
        second_stage_strategy: Optional[IRetrievalStrategy] = None,
    ):
        """Initialize the two-stage retrieval strategy.

        Args:
            memory_store: Memory store to retrieve memory content
            vector_store: Vector store for similarity search
            first_stage_strategy: Strategy for initial broad search
            second_stage_strategy: Strategy for refinement
        """
        self._memory_store = memory_store
        self._vector_store = vector_store
        self._first_stage_strategy = first_stage_strategy
        self._second_stage_strategy = second_stage_strategy
        self._default_params = {
            "first_stage_threshold": 0.5,
            "second_stage_threshold": 0.7,
            "first_stage_max": 30,
            "final_max_results": 10,
            "keyword_boost": 0.2,
        }
        self.component_id = "two_stage_retrieval_strategy"

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
        """Retrieve memories using a two-stage approach."""
        # Merge parameters with defaults
        params = self._default_params.copy()
        if parameters:
            params.update(parameters)

        # Get parameters
        first_stage_threshold = params.get("first_stage_threshold", 0.5)
        second_stage_threshold = params.get("second_stage_threshold", 0.7)
        first_stage_max = params.get("first_stage_max", 30)
        final_max_results = params.get("final_max_results", 10)
        keyword_boost = params.get("keyword_boost", 0.2)

        # Use keywords if provided
        keywords = params.get("keywords", [])

        # FIRST STAGE: Broad retrieval
        if self._first_stage_strategy:
            # Use provided first stage strategy
            first_stage_params = {
                "similarity_threshold": first_stage_threshold,
                "max_results": first_stage_max,
            }
            candidates = self._first_stage_strategy.retrieve(query_embedding, first_stage_params)
        else:
            # Default to vector similarity
            similar_vectors = self._vector_store.search(
                query_vector=query_embedding, k=first_stage_max, threshold=first_stage_threshold
            )

            # Convert to RetrievalResult format
            candidates = []
            for memory_id, similarity_score in similar_vectors:
                memory = self._memory_store.get(memory_id)

                result = RetrievalResult(
                    memory_id=memory_id,
                    content=memory.content["text"],
                    metadata=memory.metadata,
                    relevance_score=float(similarity_score),
                )

                candidates.append(result)

        # Early return if no candidates
        if not candidates:
            return []

        # SECOND STAGE: Refinement
        if self._second_stage_strategy:
            # Use provided second stage strategy
            second_stage_params = {
                "similarity_threshold": second_stage_threshold,
                "max_results": final_max_results,
            }

            # We need to extract just the memory IDs from candidates
            candidate_ids = [result["memory_id"] for result in candidates]
            second_stage_params["candidate_ids"] = candidate_ids

            # Use the second stage strategy for refinement
            results = self._second_stage_strategy.retrieve(query_embedding, second_stage_params)
        else:
            # Default refinement: re-rank using cosine similarity and keyword boost
            results = self._refine_candidates(
                candidates, query_embedding, keywords, keyword_boost, second_stage_threshold
            )

        # Sort by relevance score and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:final_max_results]

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the retrieval strategy."""
        if "first_stage_threshold" in config:
            self._default_params["first_stage_threshold"] = config["first_stage_threshold"]

        if "second_stage_threshold" in config:
            self._default_params["second_stage_threshold"] = config["second_stage_threshold"]

        if "first_stage_max" in config:
            self._default_params["first_stage_max"] = config["first_stage_max"]

        if "final_max_results" in config:
            self._default_params["final_max_results"] = config["final_max_results"]

        if "keyword_boost" in config:
            self._default_params["keyword_boost"] = config["keyword_boost"]

        # Configure sub-strategies if provided
        if self._first_stage_strategy and "first_stage_config" in config:
            self._first_stage_strategy.configure(config["first_stage_config"])

        if self._second_stage_strategy and "second_stage_config" in config:
            self._second_stage_strategy.configure(config["second_stage_config"])

    def _refine_candidates(
        self,
        candidates: list[RetrievalResult],
        query_embedding: EmbeddingVector,
        keywords: list[str],
        keyword_boost: float,
        threshold: float,
    ) -> list[RetrievalResult]:
        """Refine candidates using similarity and keyword matching."""
        refined_results = []

        for result in candidates:
            memory_id = result["memory_id"]
            memory = self._memory_store.get(memory_id)

            # Get embedding and compute cosine similarity
            embedding = memory.embedding

            # Normalize embeddings for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            embedding_norm = np.linalg.norm(embedding)

            if query_norm == 0 or embedding_norm == 0:
                similarity = 0.0
            else:
                normalized_query = query_embedding / query_norm
                normalized_embedding = embedding / embedding_norm
                similarity = np.dot(normalized_query, normalized_embedding)

            # Skip if below threshold
            if similarity < threshold:
                continue

            # Apply keyword boost if applicable
            content_text = memory.content["text"].lower()
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_text)
            boost = keyword_matches * keyword_boost

            # Cap boost to avoid extreme values
            boost = min(boost, 0.5)

            # Calculate final score
            final_score = min(1.0, similarity + boost)

            # Update result with new score
            result["relevance_score"] = float(final_score)
            refined_results.append(result)

        return refined_results
