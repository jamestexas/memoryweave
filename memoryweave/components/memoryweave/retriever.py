"""
Retriever module for MemoryWeave.

This module provides memory retrieval functionality using a component-based
architecture for more modular and testable memory retrieval.
"""

from typing import Any

import numpy as np

from memoryweave.components.base import RetrievalStrategy
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.post_processors import (
    AdaptiveKProcessor,
    KeywordBoostProcessor,
    SemanticCoherenceProcessor,
)
from memoryweave.components.query_analysis import QueryAnalyzer
from memoryweave.components.retrieval_strategies import (
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
)


class Retriever:
    """
    Memory retrieval system that integrates multiple components for comprehensive retrieval.

    This class acts as the main interface for memory retrieval operations, integrating
    components for query analysis, retrieval strategies, and post-processing.
    """

    def __init__(self, memory=None, embedding_model=None):
        """
        Initialize the retriever.

        Args:
            memory: Memory instance to use for retrieval
            embedding_model: Model for generating embeddings from queries
        """
        self.memory = memory
        self.embedding_model = embedding_model
        self.memory_manager = MemoryManager()

        # Default pipeline components
        self.query_analyzer = None
        self.retrieval_strategy = None
        self.post_processors = []

        # Default settings
        self.top_k = 5
        self.minimum_relevance = 0.0

        # Advanced features
        self.use_two_stage_retrieval = True
        self.query_type_adaptation = True
        self.dynamic_threshold_adjustment = False
        self.threshold_adjustment_window = 5
        self.recent_retrieval_metrics = []

        # Conversation state tracking
        self.conversation_history = []
        self.conversation_context = {}

        # Initialize if memory and embedding_model are provided
        if memory and embedding_model:
            self.initialize_components()

    def initialize_components(self):
        """Initialize default components for the retrieval pipeline."""
        # Create and initialize query analyzer
        self.query_analyzer = QueryAnalyzer()
        self.memory_manager.register_component("query_analyzer", self.query_analyzer)

        # Create and initialize retrieval strategies
        self.retrieval_strategy = HybridRetrievalStrategy(self.memory)
        self.memory_manager.register_component("hybrid_retrieval", self.retrieval_strategy)

        similarity_retrieval = SimilarityRetrievalStrategy(self.memory)
        self.memory_manager.register_component("similarity_retrieval", similarity_retrieval)

        temporal_retrieval = TemporalRetrievalStrategy(self.memory)
        self.memory_manager.register_component("temporal_retrieval", temporal_retrieval)

        # Create and initialize post-processors
        keyword_processor = KeywordBoostProcessor()
        self.memory_manager.register_component("keyword_boost", keyword_processor)
        self.post_processors.append(keyword_processor)

        coherence_processor = SemanticCoherenceProcessor()
        self.memory_manager.register_component("coherence", coherence_processor)
        self.post_processors.append(coherence_processor)

        adaptive_k = AdaptiveKProcessor()
        self.memory_manager.register_component("adaptive_k", adaptive_k)
        self.post_processors.append(adaptive_k)

        # Build default pipeline
        self._build_default_pipeline()

    def _build_default_pipeline(self):
        """Build the default retrieval pipeline."""
        pipeline_config = [
            {"component": "query_analyzer", "config": {}},
            {
                "component": "hybrid_retrieval",
                "config": {
                    "relevance_weight": 0.7,
                    "recency_weight": 0.3,
                    "confidence_threshold": self.minimum_relevance,
                },
            },
            {"component": "keyword_boost", "config": {"keyword_boost_weight": 0.5}},
            {"component": "coherence", "config": {"coherence_threshold": 0.2}},
            {"component": "adaptive_k", "config": {"adaptive_k_factor": 0.3}},
        ]

        self.memory_manager.build_pipeline(pipeline_config)

    def configure_pipeline(self, pipeline_config: list[dict[str, Any]]):
        """
        Configure the retrieval pipeline.

        Args:
            pipeline_config: List of pipeline step configurations
        """
        self.memory_manager.build_pipeline(pipeline_config)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
        minimum_relevance: float | None = None,
        include_metadata: bool = True,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories relevant to the query.

        Args:
            query: The query string
            top_k: Number of memories to retrieve (overrides self.top_k)
            strategy: Retrieval strategy to use ("hybrid", "similarity", or "temporal")
            minimum_relevance: Minimum relevance score for results
            include_metadata: Whether to include memory metadata in results
            conversation_history: Optional conversation history for context

        Returns:
            List of retrieved memory dicts with memory_id, content, relevance_score, etc.
        """
        if top_k is None:
            top_k = self.top_k

        # Update conversation state
        self._update_conversation_state(query, conversation_history)

        # Update parameters if provided
        if minimum_relevance is not None and minimum_relevance != self.minimum_relevance:
            self.minimum_relevance = minimum_relevance
            # Update configuration for retrieval strategies
            for step in self.memory_manager.pipeline:
                if isinstance(step["component"], RetrievalStrategy):
                    step["config"]["confidence_threshold"] = minimum_relevance

        # Use specified strategy or default
        if strategy:
            # Temporarily switch strategy for this query
            original_pipeline = self.memory_manager.pipeline

            # Create a modified pipeline with the requested strategy
            modified_pipeline = list(original_pipeline)
            for i, step in enumerate(modified_pipeline):
                if isinstance(step["component"], RetrievalStrategy):
                    # Replace with the requested strategy
                    if strategy == "similarity":
                        component_name = "similarity_retrieval"
                    elif strategy == "temporal":
                        component_name = "temporal_retrieval"
                    else:  # Default to hybrid
                        component_name = "hybrid_retrieval"

                    modified_pipeline[i] = {
                        "component": self.memory_manager.components[component_name],
                        "config": {"confidence_threshold": self.minimum_relevance, "top_k": top_k},
                    }

            # Use the modified pipeline for this query
            self.memory_manager.pipeline = modified_pipeline

            # Execute retrieval
            result = self._execute_retrieval(query, top_k)

            # Restore the original pipeline
            self.memory_manager.pipeline = original_pipeline

            return result
        else:
            # Use the configured pipeline
            return self._execute_retrieval(query, top_k)

    def _execute_retrieval(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Execute the retrieval pipeline.

        Args:
            query: The query string
            top_k: Number of memories to retrieve

        Returns:
            List of retrieved memory dicts
        """
        # Generate query embedding if needed
        query_embedding = None
        if self.embedding_model:
            query_embedding = self.embedding_model.encode(query)

        # Prepare context for pipeline
        context = {
            "query": query,
            "query_embedding": query_embedding,
            "memory": self.memory,
            "top_k": top_k,
            "conversation_history": self.conversation_history,
            "conversation_context": self.conversation_context,
        }

        # Execute pipeline
        pipeline_result = self.memory_manager.execute_pipeline(query, context)

        # Extract results from pipeline context
        results = pipeline_result.get("results", [])

        # Apply dynamic threshold adjustment if enabled
        if self.dynamic_threshold_adjustment:
            self._adjust_thresholds(results)

        # Filter based on minimum relevance if not already done
        results = [r for r in results if r.get("relevance_score", 0) >= self.minimum_relevance]

        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Update retrieval metrics
        self._update_retrieval_metrics(results)

        return results[:top_k]

    def _update_conversation_state(
        self,
        current_input: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Update internal conversation state tracking.

        Args:
            current_input: Current user input
            conversation_history: Optional external conversation history
        """
        # Use provided conversation history if available
        if conversation_history is not None:
            self.conversation_history = conversation_history

        # Add current input to conversation history
        self.conversation_history.append(
            {"role": "user", "content": current_input, "timestamp": np.datetime64("now")}
        )

        # Limit conversation history length
        max_history = 10
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

    def _update_retrieval_metrics(self, results: list[dict[str, Any]]) -> None:
        """
        Update retrieval metrics for dynamic threshold adjustment.

        Args:
            results: Retrieved results
        """
        if not self.dynamic_threshold_adjustment:
            return

        # Calculate metrics
        result_count = len(results)
        avg_score = np.mean([r.get("relevance_score", 0) for r in results]) if results else 0

        # Add to recent metrics
        self.recent_retrieval_metrics.append({"result_count": result_count, "avg_score": avg_score})

        # Limit metrics history
        if len(self.recent_retrieval_metrics) > self.threshold_adjustment_window:
            self.recent_retrieval_metrics.pop(0)

    def _adjust_thresholds(self, results: list[dict[str, Any]]) -> None:
        """
        Dynamically adjust thresholds based on recent retrieval metrics.

        Args:
            results: Current retrieval results
        """
        if len(self.recent_retrieval_metrics) < self.threshold_adjustment_window:
            return

        # Calculate average metrics
        avg_result_count = np.mean([m["result_count"] for m in self.recent_retrieval_metrics])
        avg_score = np.mean([m["avg_score"] for m in self.recent_retrieval_metrics])

        # Adjust minimum relevance threshold
        if avg_result_count < 1:
            # Too few results, lower threshold
            self.minimum_relevance = max(0.0, self.minimum_relevance - 0.05)
        elif avg_result_count > 10 and avg_score < 0.3:
            # Too many low-quality results, raise threshold
            self.minimum_relevance = min(0.9, self.minimum_relevance + 0.05)

        # Update retrieval strategy thresholds
        for step in self.memory_manager.pipeline:
            if isinstance(step["component"], RetrievalStrategy):
                step["config"]["confidence_threshold"] = self.minimum_relevance

    def set_embedding_model(self, embedding_model):
        """Set the embedding model."""
        self.embedding_model = embedding_model

    def set_memory(self, memory):
        """Set the memory instance."""
        self.memory = memory

        # Update memory reference in retrieval strategies
        for step in self.memory_manager.pipeline:
            component = step["component"]
            if isinstance(component, RetrievalStrategy):
                component.memory = memory

    def enable_dynamic_threshold_adjustment(
        self, enable: bool = True, window_size: int = 5
    ) -> None:
        """
        Enable or disable dynamic threshold adjustment.

        Args:
            enable: Whether to enable dynamic threshold adjustment
            window_size: Size of the window for calculating metrics
        """
        self.dynamic_threshold_adjustment = enable
        self.threshold_adjustment_window = window_size
        self.recent_retrieval_metrics = []
