"""Temporal-based retrieval strategy for MemoryWeave.

This module provides implementations of retrieval strategies based on
temporal factors such as recency and frequency of access.
"""

import time
from typing import Any, Dict, List, Optional

from memoryweave.interfaces.memory import EmbeddingVector, IActivationManager, IMemoryStore
from memoryweave.interfaces.retrieval import (
    IRetrievalStrategy,
    RetrievalParameters,
    RetrievalResult,
)


class TemporalRetrievalStrategy(IRetrievalStrategy):
    """Retrieval strategy based on memory recency and activation."""

    def process(self, input_data: Any) -> Any:
        """Process the input data as a pipeline stage.

        This method implements IPipelineStage.process to make the component
        usable in a pipeline.
        """
        # Handle different types of input
        if isinstance(input_data, dict):
            # For temporal retrieval, we might not even need the embedding
            # but we'll check for it for consistency
            if "embedding" in input_data:
                # Get query embedding and parameters
                query_embedding = input_data["embedding"]
                parameters = input_data.get("parameters", {})

                # Retrieve memories based on the query
                memories = self.retrieve(query_embedding, parameters)

                # Return the memories
                return memories

            # Check if this is just a vector or parameters
            elif "query_embedding" in input_data:
                query_embedding = input_data["query_embedding"]
                parameters = input_data.get("parameters", {})

                # Retrieve memories based on the query
                memories = self.retrieve(query_embedding, parameters)

                # Return the memories
                return memories
            else:
                # Temporal retrieval can work without an embedding
                parameters = input_data.get("parameters", {})
                memories = self.retrieve(None, parameters)
                return memories

        # Pass through for unsupported input types
        return input_data

    def __init__(self, memory_store: IMemoryStore, activation_manager: IActivationManager):
        """Initialize the temporal retrieval strategy.

        Args:
            memory_store: Memory store to retrieve memory content
            activation_manager: Activation manager for memory activations
        """
        self._memory_store = memory_store
        self._activation_manager = activation_manager
        self._default_params = {"max_results": 10, "recency_window_days": 7.0}
        self.component_id = "temporal_retrieval_strategy"

    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        return self.component_id

    def get_type(self):
        """Get the type of this component."""
        from memoryweave.interfaces.pipeline import ComponentType

        return ComponentType.RETRIEVAL_STRATEGY

    def retrieve(
        self,
        query_embedding: Optional[EmbeddingVector] = None,
        parameters: Optional[RetrievalParameters] = None,
    ) -> List[RetrievalResult]:
        """Retrieve memories based on temporal factors.

        Note:
            The query_embedding is optional for this strategy, as it relies
            on temporal factors rather than vector similarity.
        """
        # Merge parameters with defaults
        params = self._default_params.copy()
        if parameters:
            params.update(parameters)

        # Get parameters
        max_results = params.get("max_results", 10)
        recency_window_days = params.get("recency_window_days", 7.0)

        # Get most active memories
        active_memories = self._activation_manager.get_most_active(max_results)

        # Convert results to RetrievalResult format
        results = []
        for memory_id, activation in active_memories:
            # Retrieve memory content
            memory = self._memory_store.get(memory_id)

            # Use creation time as additional metadata
            creation_time = memory.metadata.get("created_at", 0)
            current_time = time.time()

            # Convert activation to a relevance score (0-1)
            recency_factor = self._calculate_recency_factor(
                creation_time, current_time, recency_window_days
            )

            # Combine activation and recency
            relevance_score = 0.5 * (activation / 10.0 + 0.5) + 0.5 * recency_factor

            # Create result
            result = RetrievalResult(
                memory_id=memory_id,
                content=memory.content["text"],
                metadata=memory.metadata,
                relevance_score=float(relevance_score),
            )

            results.append(result)

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return results

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the retrieval strategy."""
        if "max_results" in config:
            self._default_params["max_results"] = config["max_results"]

        if "recency_window_days" in config:
            self._default_params["recency_window_days"] = config["recency_window_days"]

    def _calculate_recency_factor(
        self, creation_time: float, current_time: float, recency_window_days: float
    ) -> float:
        """Calculate a recency factor (0-1) based on memory age."""
        # Calculate age in seconds
        age_seconds = current_time - creation_time

        # Convert window to seconds
        window_seconds = recency_window_days * 24 * 60 * 60

        # Calculate recency factor
        # 1.0 for brand new, approaching 0.0 as age approaches window
        if age_seconds <= 0:
            return 1.0
        elif age_seconds >= window_seconds:
            return 0.0
        else:
            return 1.0 - (age_seconds / window_seconds)
