"""
Retriever module for MemoryWeave.

This module provides memory retrieval functionality using a component-based
architecture for more modular and testable memory retrieval.
"""

from typing import Any

import numpy as np

from memoryweave.components import dynamic_threshold_adjuster
from memoryweave.components.base import RetrievalStrategy
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.post_processors import (
    AdaptiveKProcessor,
    KeywordBoostProcessor,
    SemanticCoherenceProcessor,
)
from memoryweave.components.query_adapter import QueryTypeAdapter
from memoryweave.components.query_analysis import QueryAnalyzer
from memoryweave.components.retrieval_strategies import (
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
    TwoStageRetrievalStrategy,
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
        self.query_adapter = None
        self.retrieval_strategy = None
        self.post_processors = []
        self.two_stage_strategy = None

        # Default settings
        self.top_k = 5
        self.minimum_relevance = 0.0

        # Advanced features
        self.use_two_stage_retrieval = True
        self.first_stage_k = 20  # Number of candidates in first stage
        self.first_stage_threshold_factor = 0.7  # Lower threshold for first stage
        self.query_type_adaptation = True
        self.adaptation_strength = 1.0  # How strongly to adapt based on query type (0.0-1.0)
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

        # Create and initialize query adapter
        self.query_adapter = QueryTypeAdapter()
        self.memory_manager.register_component("query_adapter", self.query_adapter)

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
        # Register the new dynamic threshold adjuster if dynamic thresholding is enabled
        if self.dynamic_threshold_adjustment:
            self.dynamic_threshold_adjuster = dynamic_threshold_adjuster()
            self.memory_manager.register_component(
                "dynamic_threshold", self.dynamic_threshold_adjuster
            )

        # Initialize two-stage retrieval strategy
        self.two_stage_strategy = TwoStageRetrievalStrategy(
            self.memory, base_strategy=self.retrieval_strategy, post_processors=self.post_processors
        )
        self.memory_manager.register_component("two_stage_retrieval", self.two_stage_strategy)

        # Build default pipeline
        self._build_default_pipeline()

    def _build_default_pipeline(self):
        """Build the default retrieval pipeline."""
        # Configure query adapter
        query_adapter_config = dict(
            # Enable query type adaptation if enabled
            adaptation_strength=(self.adaptation_strength if self.query_type_adaptation else 0.0),
            confidence_threshold=self.minimum_relevance,
            first_stage_k=self.first_stage_k,
            first_stage_threshold_factor=self.first_stage_threshold_factor,
        )
        pipeline_steps = [
            dict(component="query_analyzer", config={}),
            dict(component="query_adapter", config=query_adapter_config),
        ]

        if self.use_two_stage_retrieval:
            # Use two-stage retrieval in the pipeline with query type adaptation
            pipeline_steps.append(
                dict(
                    component="two_stage_retrieval",
                    config=dict(
                        confidence_threshold=self.minimum_relevance,
                        first_stage_k=self.first_stage_k,
                        first_stage_threshold_factor=self.first_stage_threshold_factor,
                        post_processor_config=dict(
                            keyword_boost_weight=0.5,
                            coherence_threshold=0.2,
                            adaptive_k_factor=0.3,
                        ),
                    ),
                ),
            )
        else:
            # Use standard pipeline without two-stage retrieval
            # This adds each component separately to the pipeline
            pipeline_steps.extend(
                [
                    dict(
                        component="hybrid_retrieval",
                        config=dict(
                            relevance_weight=0.7,
                            recency_weight=0.3,
                            confidence_threshold=self.minimum_relevance,
                        ),
                    ),
                    dict(
                        component="keyword_boost",
                        config=dict(keyword_boost_weight=0.5),
                    ),
                    dict(
                        component="coherence",
                        config=dict(coherence_threshold=0.2),
                    ),
                    dict(
                        component="adaptive_k",
                        config=dict(adaptive_k_factor=0.3),
                    ),
                ],
            )
        missing_components = [
            step["component"]
            for step in pipeline_steps
            if step["component"] not in self.memory_manager.components
        ]
        if missing_components:
            raise ValueError(f"Missing registered components: {missing_components}")
        self.memory_manager.build_pipeline(pipeline_steps)

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
            strategy: Retrieval strategy to use ("hybrid", "similarity", "temporal", or "two_stage")
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

        # Generate query embedding
        query_embedding = None
        if self.embedding_model:
            query_embedding = self.embedding_model.encode(query)

        # Prepare context for retrieval
        query_context = {
            "query": query,
            "query_embedding": query_embedding,
            "memory": self.memory,
            "top_k": top_k,
            "conversation_history": self.conversation_history,
            "conversation_context": self.conversation_context,
        }

        # Run query analyzer to get query type
        query_analysis = self.memory_manager.components["query_analyzer"].process_query(
            query, query_context
        )
        query_context.update(query_analysis)

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
                    elif strategy == "two_stage":
                        component_name = "two_stage_retrieval"
                    else:  # Default to hybrid
                        component_name = "hybrid_retrieval"

                    # Get the appropriate component
                    component = self.memory_manager.components[component_name]

                    # If we're switching to two-stage and the existing isn't,
                    # ensure it has references to post-processors
                    if strategy == "two_stage" and not isinstance(
                        component, TwoStageRetrievalStrategy
                    ):
                        # Create a new two-stage strategy with appropriate references
                        component = TwoStageRetrievalStrategy(
                            self.memory,
                            base_strategy=self.retrieval_strategy,
                            post_processors=self.post_processors,
                        )

                    modified_pipeline[i] = {
                        "component": component,
                        "config": {
                            "confidence_threshold": self.minimum_relevance,
                            "top_k": top_k,
                            "first_stage_k": self.first_stage_k,
                            "first_stage_threshold_factor": self.first_stage_threshold_factor,
                        },
                    }

            # Use the modified pipeline for this query
            self.memory_manager.pipeline = modified_pipeline

            # Execute pipeline
            pipeline_result = self.memory_manager.execute_pipeline(query, query_context)

            # Extract results
            results = pipeline_result.get("results", [])

            # Apply dynamic threshold adjustment if enabled
            if self.dynamic_threshold_adjustment:
                self._adjust_thresholds(results)

            # Update retrieval metrics
            self._update_retrieval_metrics(results)

            # Restore the original pipeline
            self.memory_manager.pipeline = original_pipeline

            return results
        else:
            # Use the configured pipeline
            pipeline_result = self.memory_manager.execute_pipeline(query, query_context)

            # Extract results
            results = pipeline_result.get("results", [])

            # Apply dynamic threshold adjustment if enabled
            if self.dynamic_threshold_adjustment:
                self._adjust_thresholds(results)

            # Update retrieval metrics
            self._update_retrieval_metrics(results)

            return results

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
        self.conversation_history.append({
            "role": "user",
            "content": current_input,
            "timestamp": np.datetime64("now"),
        })

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

    def configure_two_stage_retrieval(
        self,
        enable: bool = True,
        first_stage_k: int = 20,
        first_stage_threshold_factor: float = 0.7,
    ) -> None:
        """
        Configure two-stage retrieval.

        Args:
            enable: Whether to enable two-stage retrieval
            first_stage_k: Number of candidates to retrieve in first stage
            first_stage_threshold_factor: Factor to multiply minimum_relevance by for first stage
        """
        self.use_two_stage_retrieval = enable
        self.first_stage_k = first_stage_k
        self.first_stage_threshold_factor = first_stage_threshold_factor

        # If we already have a two-stage strategy, update its configuration
        if self.two_stage_strategy:
            # Update the existing strategy with the new parameters
            self.two_stage_strategy.initialize({
                "confidence_threshold": self.minimum_relevance,
                "first_stage_k": first_stage_k,
                "first_stage_threshold_factor": first_stage_threshold_factor,
            })

        # Rebuild pipeline with updated configuration
        self._build_default_pipeline()

    def configure_query_type_adaptation(
        self, enable: bool = True, adaptation_strength: float = 1.0
    ) -> None:
        """
        Configure query type adaptation.

        Args:
            enable: Whether to enable query type adaptation
            adaptation_strength: How strongly to adapt parameters (0.0-1.0)
        """
        self.query_type_adaptation = enable
        self.adaptation_strength = adaptation_strength

        # Update query adapter configuration
        if self.query_adapter:
            self.query_adapter.initialize({
                "adaptation_strength": adaptation_strength if enable else 0.0,
                "confidence_threshold": self.minimum_relevance,
                "first_stage_k": self.first_stage_k,
                "first_stage_threshold_factor": self.first_stage_threshold_factor,
            })

        # Rebuild pipeline with updated configuration
        self._build_default_pipeline()
