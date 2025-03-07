"""
Retriever module for MemoryWeave.

This module provides memory retrieval functionality using a component-based
architecture for more modular and testable memory retrieval.
"""

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from memoryweave.components.activation import ActivationManager
from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.base import RetrievalStrategy
from memoryweave.components.category_manager import CategoryManager
from memoryweave.components.dynamic_threshold_adjuster import DynamicThresholdAdjuster
from memoryweave.components.keyword_expander import KeywordExpander
from memoryweave.components.memory_decay import MemoryDecayComponent
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.personal_attributes import PersonalAttributeManager
from memoryweave.components.post_processors import (
    AdaptiveKProcessor,
    KeywordBoostProcessor,
    MinimumResultGuaranteeProcessor,
    PersonalAttributeProcessor,
    SemanticCoherenceProcessor,
)
from memoryweave.components.query_adapter import QueryTypeAdapter
from memoryweave.components.query_analysis import QueryAnalyzer

# Import and register Contextual Fabric Strategy
from memoryweave.components.retrieval_strategies import (
    CategoryRetrievalStrategy,  # Add this import
    ContextualFabricStrategy,
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
    TwoStageRetrievalStrategy,
)
from memoryweave.components.retrieval_strategies.hybrid_bm25_vector_strategy import (
    HybridBM25VectorStrategy,
)
from memoryweave.components.temporal_context import TemporalContextBuilder
from memoryweave.storage.refactored.memory_store import StandardMemoryStore

logger = logging.getLogger(__name__)

_MEMORY_STORE = StandardMemoryStore()
_EMBEDDER = None
_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
_RETRIEVER = None


def _get_memory_store():
    global _MEMORY_STORE
    if _MEMORY_STORE is None:
        _MEMORY_STORE = StandardMemoryStore()
    return _MEMORY_STORE


def _get_embedder(model_name: str = _EMBEDDING_MODEL_NAME, **kwargs) -> Any:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name, **kwargs)
    return _EMBEDDER


def _get_retriever(
    memory_store: None | StandardMemoryStore = None,
    embedding_model: None | SentenceTransformer | Any = None,
    embedding_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
):
    global _RETRIEVER
    if _RETRIEVER is None:
        # Get embedder if not provided
        if embedding_model is None:
            embedding_model = _get_embedder(embedding_model_name)

        # Create memory encoder if needed
        from memoryweave.components.memory_encoding import MemoryEncoder

        memory_encoder = MemoryEncoder(embedding_model)
        memory_encoder.initialize({})

        _RETRIEVER = Retriever(
            memory=_get_memory_store() if memory_store is None else memory_store,
            embedding_model=embedding_model,
            memory_encoder=memory_encoder,
        )
        _RETRIEVER.initialize_components()
        # Configure it all once:
    return _RETRIEVER


class Retriever:
    """
    Memory retrieval system that integrates multiple components for comprehensive retrieval.

    This class acts as the main interface for memory retrieval operations, integrating
    components for query analysis, retrieval strategies, and post-processing.
    """

    def __init__(
        self, memory: StandardMemoryStore | None = None, embedding_model=None, memory_encoder=None
    ) -> None:
        """
        Initialize the retriever.

        Args:
            memory: Memory instance to use for retrieval
            embedding_model: Model for generating embeddings from queries
            memory_encoder: Component for encoding memory content
        """
        # If memory is not provided, create a new memory manager
        self.memory_manager = MemoryManager(memory)  # Pass along if you have one
        if memory is None:
            self.memory = self.memory_manager.memory_store  # Use the created StandardMemoryStore

        # Else use the provided memory instance
        else:
            self.memory = memory
        self.embedding_model = embedding_model
        self.memory_encoder = memory_encoder

        # Default pipeline components
        self.query_analyzer = None
        self.query_adapter = None
        self.retrieval_strategy = None
        self.post_processors = []
        self.two_stage_strategy = None
        self.personal_attribute_manager = None
        self.keyword_expander = None

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

        # Memory decay settings
        self.memory_decay_enabled = True
        self.memory_decay_rate = 0.99
        self.memory_decay_interval = 10

        # Minimum result guarantee
        self.min_results_guarantee = 1

        # Conversation state tracking
        self.conversation_history = []
        self.conversation_context = {}

        # Initialize if memory and embedding_model are provided
        if memory and embedding_model:
            self.initialize_components()

    def initialize_components(self):
        """
        Initialize default components for the retrieval pipeline, making Contextual Fabric
        the main retrieval strategy (either single-stage or two-stage).
        """
        # Create and initialize query analyzer
        self.query_analyzer = QueryAnalyzer()
        self.memory_manager.register_component("query_analyzer", self.query_analyzer)

        # Register memory encoder if available
        if self.memory_encoder:
            self.memory_manager.register_component("memory_encoder", self.memory_encoder)

        # Create and initialize category manager
        self.category_manager = CategoryManager(self.memory)
        self.memory_manager.register_component("category_manager", self.category_manager)

        # Create and initialize personal attribute manager
        self.personal_attribute_manager = PersonalAttributeManager()
        self.memory_manager.register_component(
            "personal_attributes", self.personal_attribute_manager
        )

        # Create and initialize keyword expander
        self.keyword_expander = KeywordExpander()
        self.memory_manager.register_component("keyword_expander", self.keyword_expander)

        # Create and initialize query adapter
        self.query_adapter = QueryTypeAdapter()
        self.memory_manager.register_component("query_adapter", self.query_adapter)

        #
        # We still register your other retrieval strategies in case you want them,
        # but we'll focus the pipeline on contextual_fabric below.
        #

        # Similarity strategy
        similarity_retrieval = SimilarityRetrievalStrategy(self.memory)
        self.memory_manager.register_component("similarity_retrieval", similarity_retrieval)

        # Hybrid retrieval (BM25+similarity)
        hybrid_retrieval = HybridRetrievalStrategy(self.memory)
        self.memory_manager.register_component("hybrid_retrieval", hybrid_retrieval)

        # Temporal retrieval
        temporal_retrieval = TemporalRetrievalStrategy(self.memory)
        self.memory_manager.register_component("temporal_retrieval", temporal_retrieval)

        # Hybrid BM25+Vector
        hybrid_bm25_vector_retrieval = HybridBM25VectorStrategy(self.memory)
        hybrid_bm25_vector_retrieval.initialize(
            {
                "vector_weight": 0.2,
                "bm25_weight": 0.8,
                "confidence_threshold": self.minimum_relevance,
                "activation_boost": True,
                "enable_dynamic_weighting": True,
                "keyword_weight_bias": 0.7,
            }
        )
        self.memory_manager.register_component(
            "hybrid_bm25_vector_retrieval", hybrid_bm25_vector_retrieval
        )

        # Create required helper components for contextual fabric
        activation_manager = ActivationManager()
        associative_linker = AssociativeMemoryLinker()
        temporal_context = TemporalContextBuilder()

        self.memory_manager.register_component("activation_manager", activation_manager)
        self.memory_manager.register_component("associative_linker", associative_linker)
        self.memory_manager.register_component("temporal_context", temporal_context)

        # Create and initialize the contextual fabric strategy
        contextual_fabric = ContextualFabricStrategy(
            memory_store=self.memory,
            associative_linker=associative_linker,
            temporal_context=temporal_context,
            activation_manager=activation_manager,
        )
        contextual_fabric.initialize(
            {
                "confidence_threshold": self.minimum_relevance,
                "similarity_weight": 0.5,
                "associative_weight": 0.3,
                "temporal_weight": 0.1,
                "activation_weight": 0.1,
                "activation_boost_factor": 1.5,
                "memory_store": self.memory,
            }
        )

        self.memory_manager.register_component("contextual_fabric_strategy", contextual_fabric)

        # Create category-based retrieval strategy
        category_retrieval = CategoryRetrievalStrategy(
            self.memory, category_manager=self.category_manager
        )
        self.memory_manager.register_component("category_retrieval", category_retrieval)

        #
        # Create and initialize post-processors
        #
        keyword_processor = KeywordBoostProcessor()
        self.memory_manager.register_component("keyword_boost", keyword_processor)
        self.post_processors.append(keyword_processor)

        coherence_processor = SemanticCoherenceProcessor()
        self.memory_manager.register_component("coherence", coherence_processor)
        self.post_processors.append(coherence_processor)

        attribute_processor = PersonalAttributeProcessor()
        self.memory_manager.register_component("attribute_processor", attribute_processor)
        self.post_processors.append(attribute_processor)

        memory_decay = MemoryDecayComponent()
        memory_decay.initialize(
            {
                "memory_decay_enabled": self.memory_decay_enabled,
                "memory_decay_rate": self.memory_decay_rate,
                "memory_decay_interval": self.memory_decay_interval,
                "memory": self.memory,
            }
        )
        self.memory_manager.register_component("memory_decay", memory_decay)

        min_result_processor = MinimumResultGuaranteeProcessor()
        min_result_processor.initialize(
            {
                "min_results": self.min_results_guarantee,
                "fallback_threshold_factor": 0.5,
                "min_fallback_threshold": 0.05,
                "memory": self.memory,
            }
        )
        self.memory_manager.register_component("min_result_guarantee", min_result_processor)
        self.post_processors.append(min_result_processor)

        adaptive_k = AdaptiveKProcessor()
        self.memory_manager.register_component("adaptive_k", adaptive_k)
        self.post_processors.append(adaptive_k)

        # If dynamic thresholding is enabled, add that component
        if self.dynamic_threshold_adjustment:
            self.dynamic_threshold_adjuster = DynamicThresholdAdjuster()
            self.memory_manager.register_component(
                "dynamic_threshold", self.dynamic_threshold_adjuster
            )

        #
        # Here is the key difference: we set the base retrieval to "contextual_fabric_strategy".
        #
        base_strategy = self.memory_manager.components["contextual_fabric_strategy"]

        # Create two-stage that uses Contextual Fabric as the “base”
        self.two_stage_strategy = TwoStageRetrievalStrategy(
            self.memory,
            base_strategy=base_strategy,  # <--- contextual_fabric
            post_processors=self.post_processors,
        )
        self.memory_manager.register_component("two_stage_retrieval", self.two_stage_strategy)

        #
        # We'll store self.retrieval_strategy as either the single-stage
        # “contextual_fabric_strategy” (if you later set use_two_stage_retrieval=False)
        # or the “two_stage_retrieval” object (if use_two_stage_retrieval=True).
        # The final pipeline build will pick one of them.
        #
        # But for now, let's just set it to None; _build_default_pipeline will choose.
        #
        self.retrieval_strategy = None

        # Finally, build the pipeline with the above components
        self._build_default_pipeline()

    def _build_default_pipeline(self):
        """
        Build the default retrieval pipeline so that:
        - If self.use_two_stage_retrieval is True, we do two_stage with base=ContextualFabric
        - Otherwise, we just do single-stage "contextual_fabric_strategy."
        """
        self._ensure_components_registered()

        adaptation_strength = self.adaptation_strength if self.query_type_adaptation else 0.0
        logger.info(
            f"Retriever._build_default_pipeline: query_type_adaptation={self.query_type_adaptation}, "
            f"adaptation_strength={adaptation_strength}"
        )

        # Configure the query adapter in case we need it
        query_adapter_config = dict(
            adaptation_strength=adaptation_strength,
            confidence_threshold=self.minimum_relevance,
            first_stage_k=self.first_stage_k,
            first_stage_threshold_factor=self.first_stage_threshold_factor,
        )

        pipeline_steps = [
            dict(component="query_analyzer", config={}),
            dict(component="personal_attributes", config={}),
            dict(
                component="memory_decay",
                config={
                    "memory_decay_enabled": self.memory_decay_enabled,
                    "memory_decay_rate": self.memory_decay_rate,
                    "memory_decay_interval": self.memory_decay_interval,
                    "memory": self.memory,
                },
            ),
            dict(
                component="keyword_expander",
                config={"enable_expansion": True, "max_expansions_per_keyword": 5},
            ),
            dict(component="query_adapter", config=query_adapter_config),
        ]

        if self.use_two_stage_retrieval:
            # If using two-stage, the final retrieval step is "two_stage_retrieval",
            # which has "contextual_fabric_strategy" as the base inside.
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
                )
            )
            # We'll store self.retrieval_strategy as the actual "two_stage_retrieval" component
            self.retrieval_strategy = self.memory_manager.components["two_stage_retrieval"]
        else:
            # Single‐stage: we directly add "contextual_fabric_strategy" to the pipeline
            pipeline_steps.append(
                dict(
                    component="contextual_fabric_strategy",
                    config=dict(
                        confidence_threshold=self.minimum_relevance,
                        # Could set or override these if you want
                    ),
                )
            )
            # We'll store self.retrieval_strategy as the "contextual_fabric_strategy" component
            self.retrieval_strategy = self.memory_manager.components["contextual_fabric_strategy"]

        # Filter out any steps that are missing
        missing = [
            step["component"]
            for step in pipeline_steps
            if step["component"] not in self.memory_manager.components
        ]
        if missing:
            print(f"Warning: Missing registered components: {missing}")

        pipeline_steps = [
            step for step in pipeline_steps if step["component"] in self.memory_manager.components
        ]

        if not pipeline_steps:
            print("Warning: No components available for pipeline")
            return

        self.memory_manager.build_pipeline(pipeline_steps)
        for i, step in enumerate(self.memory_manager.pipeline):
            comp = step["component"]
            logger.debug(
                f"Step {i}: {comp.__class__.__name__}, id={id(comp)} config={step.get('config')}"
            )

    def _ensure_components_registered(self):
        """Ensure all required components are registered properly."""
        # Check for missing core components and register them if needed
        required_components = {
            "query_analyzer": self.query_analyzer,
            "personal_attributes": self.personal_attribute_manager,
            "memory_decay": self.memory_manager.components.get("memory_decay"),
            "keyword_expander": self.keyword_expander,
            "query_adapter": self.query_adapter,
            "two_stage_retrieval": self.two_stage_strategy,
            "hybrid_retrieval": self.retrieval_strategy,
            "hybrid_bm25_vector_retrieval": self.memory_manager.components.get(
                "hybrid_bm25_vector_retrieval"
            ),
            "similarity_retrieval": self.memory_manager.components.get("similarity_retrieval"),
            "temporal_retrieval": self.memory_manager.components.get("temporal_retrieval"),
            "keyword_boost": self.memory_manager.components.get("keyword_boost"),
            "coherence": self.memory_manager.components.get("coherence"),
            "adaptive_k": self.memory_manager.components.get("adaptive_k"),
            "attribute_processor": self.memory_manager.components.get("attribute_processor"),
            "min_result_guarantee": self.memory_manager.components.get("min_result_guarantee"),
        }

        # Initialize any components that are not initialized yet
        for name, component in list(required_components.items()):
            if component is None and hasattr(self, name):
                required_components[name] = getattr(self, name)

        # If we still don't have the two_stage_retrieval and it's needed, create it
        if required_components["two_stage_retrieval"] is None and self.use_two_stage_retrieval:
            # Initialize it from scratch with proper dependencies
            if self.retrieval_strategy and self.post_processors:
                self.two_stage_strategy = TwoStageRetrievalStrategy(
                    self.memory,
                    base_strategy=self.retrieval_strategy,
                    post_processors=self.post_processors,
                )
                required_components["two_stage_retrieval"] = self.two_stage_strategy

        # Register all missing components at once using the new bulk registration method
        missing_components = {
            name: component
            for name, component in required_components.items()
            if component is not None and name not in self.memory_manager.components
        }

        if missing_components:
            logger.debug(f"Registering missing components: {list(missing_components.keys())}")
            self.memory_manager.register_components(missing_components)

    def configure_pipeline(self, pipeline_config: list[dict[str, Any]]):
        """
        Configure the retrieval pipeline.

        Args:
            pipeline_config: List of pipeline step configurations
        """
        self.memory_manager.build_pipeline(pipeline_config)

    def configure_semantic_coherence(self, enable: bool = True):
        """
        Configure semantic coherence checking.

        Args:
            enable: Whether to enable semantic coherence checking
        """

        logger = logging.getLogger(__name__)
        logger.info(f"Retriever.configure_semantic_coherence: Setting enable={enable}")

        # Create semantic coherence processor if it doesn't exist
        if not hasattr(self, "semantic_coherence_processor"):
            self.semantic_coherence_processor = SemanticCoherenceProcessor()
            # Initialize with proper configuration
            self.semantic_coherence_processor.initialize(
                {
                    "coherence_threshold": 0.2,
                    "enable_query_type_filtering": True,
                    "enable_pairwise_coherence": True,
                    "enable_clustering": False,
                }
            )
            logger.info(
                "Retriever.configure_semantic_coherence: Created new SemanticCoherenceProcessor"
            )

        # Add to post-processors if enabled and not already there
        if enable and self.semantic_coherence_processor not in self.post_processors:
            self.post_processors.append(self.semantic_coherence_processor)
            logger.info(
                "Retriever.configure_semantic_coherence: Added processor to post_processors"
            )
        # Remove from post-processors if disabled but present
        elif not enable and self.semantic_coherence_processor in self.post_processors:
            self.post_processors.remove(self.semantic_coherence_processor)
            logger.info(
                "Retriever.configure_semantic_coherence: Removed processor from post_processors"
            )

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
        """
        import logging
        import traceback

        logger = logging.getLogger(__name__)

        logger.debug(
            f"[Retriever.retrieve] Called on retriever id={id(self)} "
            f"with query='{query[:50]}...', strategy={strategy}, top_k={top_k}, "
            f"query_type_adaptation={self.query_type_adaptation}, use_two_stage={self.use_two_stage_retrieval}"
        )
        logger.debug("Call stack:\n" + "".join(traceback.format_stack(limit=5)))

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
            "in_evaluation": True,
            # Set feature flags to control component behavior
            "enable_query_type_adaptation": self.query_type_adaptation,
            "enable_semantic_coherence": (
                self.semantic_coherence_processor in self.post_processors
                if hasattr(self, "semantic_coherence_processor")
                else False
            ),
            "enable_two_stage_retrieval": self.use_two_stage_retrieval,
            # If user specified a strategy, that’s our config_name
            # Else use 'two_stage' if self.use_two_stage_retrieval else 'default'
            "config_name": strategy or ("two_stage" if self.use_two_stage_retrieval else "default"),
        }

        # Ensure components are initialized once
        if not self.query_analyzer:
            logger.debug(
                "[Retriever.retrieve] query_analyzer not found, calling initialize_components() now."
            )
            self.initialize_components()

        logger.debug(
            f"[Retriever.retrieve] Using retrieval strategy: {self.retrieval_strategy.__class__.__name__}"
        )
        logger.debug(f"[Retriever.retrieve] Using {len(self.post_processors)} post-processors")

        # Run query analyzer to get query type
        query_analysis = self.memory_manager.components.get("query_analyzer", {})
        if hasattr(query_analysis, "process_query"):
            analysis_result = query_analysis.process_query(query, query_context)
            # If query_analyzer returns e.g. {"enable_query_type_adaptation": False}, it might override
            # but typically it wouldn't. Let's see.
            query_context.update(analysis_result)

        #
        # *** Log the entire final query_context before pipeline ***
        #
        logger.debug(
            f"[Retriever.retrieve] Final query_context before execute_pipeline:\n{query_context}"
        )

        # If user explicitly gave a strategy, build a temporary pipeline
        if strategy:
            original_pipeline = self.memory_manager.pipeline
            modified_pipeline = list(original_pipeline)

            for i, step in enumerate(modified_pipeline):
                if isinstance(step["component"], RetrievalStrategy):
                    # Map your strategy to the correct component_name
                    if strategy == "similarity":
                        component_name = "similarity_retrieval"
                    elif strategy == "temporal":
                        component_name = "temporal_retrieval"
                    elif strategy == "two_stage":
                        component_name = "two_stage_retrieval"
                    elif strategy in ("bm25_hybrid", "hybrid_bm25"):
                        component_name = "hybrid_bm25_vector_retrieval"
                    elif strategy == "contextual_fabric":
                        component_name = "contextual_fabric_strategy"
                    else:
                        component_name = "hybrid_retrieval"

                    component = self.memory_manager.components.get(component_name)
                    if not component:
                        logger.warning(f"Strategy {component_name} not available, using default.")
                        continue

                    # If user said "two_stage" but the pipeline is not two_stage
                    if strategy == "two_stage":
                        if not isinstance(component, TwoStageRetrievalStrategy):
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

            self.memory_manager.pipeline = modified_pipeline

            pipeline_result = self.memory_manager.execute_pipeline(query, query_context)

            if not (results := pipeline_result.get("results", [])):
                logger.debug(
                    "[Retriever.retrieve] No results from pipeline, attempting fallback retrieval."
                )
                if hasattr(self.memory, "retrieve_memories") and query_embedding is not None:
                    benchmark_results = self.memory.retrieve_memories(
                        query_embedding,
                        top_k=top_k,
                        confidence_threshold=0.0,
                        activation_boost=False,
                    )
                    for idx, score, metadata in benchmark_results:
                        result_dict = {
                            "memory_id": idx,
                            "relevance_score": min(score, 0.1),
                            "below_threshold": True,
                            "benchmark_fallback": True,
                            "content": str(metadata.get("content", "Unknown")),
                            **metadata,
                        }
                        results.append(result_dict)
                elif (
                    hasattr(self.memory, "memory_metadata") and len(self.memory.memory_metadata) > 0
                ):
                    results = [
                        {
                            "memory_id": 0,
                            "relevance_score": 0.1,
                            "below_threshold": True,
                            "benchmark_fallback": True,
                            "content": str(
                                self.memory.memory_metadata[0].get("content", "Unknown")
                            ),
                            **self.memory.memory_metadata[0],
                        }
                    ]

            if self.dynamic_threshold_adjustment:
                self._adjust_thresholds(results)
            self._update_retrieval_metrics(results)

            # Restore pipeline
            self.memory_manager.pipeline = original_pipeline
            return results
        else:
            # Use the pipeline as-built
            if hasattr(self.memory_manager, "config_name"):
                query_context["config_name"] = self.memory_manager.config_name

            pipeline_result = self.memory_manager.execute_pipeline(query, query_context)
            results = pipeline_result.get("results", [])

            if not results:
                logger.debug(
                    "[Retriever.retrieve] No results from pipeline, attempting fallback retrieval (no explicit strategy)."
                )
                if hasattr(self.memory, "retrieve_memories") and query_embedding is not None:
                    benchmark_results = self.memory.retrieve_memories(
                        query_embedding,
                        top_k=top_k,
                        confidence_threshold=0.0,
                        activation_boost=False,
                    )
                    for idx, score, metadata in benchmark_results:
                        result_dict = {
                            "memory_id": idx,
                            "relevance_score": min(score, 0.1),
                            "below_threshold": True,
                            "benchmark_fallback": True,
                            "content": str(metadata.get("content", "Unknown")),
                            **metadata,
                        }
                        results.append(result_dict)
                elif (
                    hasattr(self.memory, "memory_metadata") and len(self.memory.memory_metadata) > 0
                ):
                    results = [
                        {
                            "memory_id": 0,
                            "relevance_score": 0.1,
                            "below_threshold": True,
                            "benchmark_fallback": True,
                            "content": str(
                                self.memory.memory_metadata[0].get("content", "Unknown")
                            ),
                            **self.memory.memory_metadata[0],
                        }
                    ]

            if self.dynamic_threshold_adjustment:
                self._adjust_thresholds(results)
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
        self.conversation_history.append(
            {
                "role": "user",
                "content": current_input,
                "timestamp": np.datetime64("now"),
            }
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
            self.two_stage_strategy.initialize(
                {
                    "confidence_threshold": self.minimum_relevance,
                    "first_stage_k": first_stage_k,
                    "first_stage_threshold_factor": first_stage_threshold_factor,
                }
            )

        # Rebuild pipeline with updated configuration
        self._build_default_pipeline()

    def configure_query_type_adaptation(
        self,
        enable: bool = True,
        adaptation_strength: float = 1.0,
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
            self.query_adapter.initialize(
                {
                    "adaptation_strength": adaptation_strength if enable else 0.0,
                    "confidence_threshold": self.minimum_relevance,
                    "first_stage_k": self.first_stage_k,
                    "first_stage_threshold_factor": self.first_stage_threshold_factor,
                }
            )

        # Rebuild pipeline with updated configuration
        self._build_default_pipeline()
