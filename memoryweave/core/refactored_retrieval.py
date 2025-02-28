# memoryweave/core/refactored_retrieval.py
from typing import Any, Optional

from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.personal_attributes import PersonalAttributeManager
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


class RefactoredRetriever:
    """
    Component-based implementation of contextual memory retrieval.
    """

    def __init__(
        self,
        memory,
        embedding_model,
        retrieval_strategy: str = "hybrid",
        confidence_threshold: float = 0.3,
        semantic_coherence_check: bool = False,
        adaptive_retrieval: bool = False,
        keyword_boost_weight: float = 0.5,
        adaptive_k_factor: float = 0.3,
        use_two_stage_retrieval: bool = False,
        query_type_adaptation: bool = False,
    ):
        # Store references to core components
        self.memory = memory
        self.embedding_model = embedding_model

        # Create memory manager
        self.memory_manager = MemoryManager()

        # set up components
        self._setup_components(
            retrieval_strategy=retrieval_strategy,
            confidence_threshold=confidence_threshold,
            semantic_coherence_check=semantic_coherence_check,
            adaptive_retrieval=adaptive_retrieval,
            keyword_boost_weight=keyword_boost_weight,
            adaptive_k_factor=adaptive_k_factor,
            use_two_stage_retrieval=use_two_stage_retrieval,
            query_type_adaptation=query_type_adaptation,
        )

    def _setup_components(self, **kwargs):
        """set up components and build retrieval pipeline."""
        # Register components
        self.query_analyzer = QueryAnalyzer()
        self.memory_manager.register_component("query_analyzer", self.query_analyzer)

        self.personal_attribute_manager = PersonalAttributeManager()
        self.memory_manager.register_component(
            "personal_attributes", self.personal_attribute_manager
        )

        # Retrieval strategies
        self.similarity_strategy = SimilarityRetrievalStrategy(self.memory)
        self.memory_manager.register_component("similarity_retrieval", self.similarity_strategy)

        self.temporal_strategy = TemporalRetrievalStrategy(self.memory)
        self.memory_manager.register_component("temporal_retrieval", self.temporal_strategy)

        self.hybrid_strategy = HybridRetrievalStrategy(self.memory)
        self.memory_manager.register_component("hybrid_retrieval", self.hybrid_strategy)

        # Post-processors
        self.keyword_boost = KeywordBoostProcessor()
        self.memory_manager.register_component("keyword_boost", self.keyword_boost)

        self.coherence_processor = SemanticCoherenceProcessor()
        self.memory_manager.register_component("coherence_check", self.coherence_processor)

        self.adaptive_k = AdaptiveKProcessor()
        self.memory_manager.register_component("adaptive_k", self.adaptive_k)

        # Initialize components with configuration
        self.similarity_strategy.initialize({
            "confidence_threshold": kwargs.get("confidence_threshold", 0.0)
        })
        self.hybrid_strategy.initialize({
            "relevance_weight": 0.7,
            "recency_weight": 0.3,
            "confidence_threshold": kwargs.get("confidence_threshold", 0.0),
        })
        self.keyword_boost.initialize({
            "keyword_boost_weight": kwargs.get("keyword_boost_weight", 0.5)
        })
        self.coherence_processor.initialize({"coherence_threshold": 0.2})
        self.adaptive_k.initialize({"adaptive_k_factor": kwargs.get("adaptive_k_factor", 0.3)})

        # Build retrieval pipeline based on configuration
        pipeline_config = [
            {"component": "query_analyzer"},
            {"component": "personal_attributes"},
        ]

        # Add retrieval strategy
        retrieval_strategy = kwargs.get("retrieval_strategy", "hybrid")
        if retrieval_strategy == "similarity":
            pipeline_config.append({"component": "similarity_retrieval"})
        elif retrieval_strategy == "temporal":
            pipeline_config.append({"component": "temporal_retrieval"})
        else:  # hybrid
            pipeline_config.append({"component": "hybrid_retrieval"})

        # Add post-processors
        pipeline_config.append({"component": "keyword_boost"})

        if kwargs.get("semantic_coherence_check", False):
            pipeline_config.append({"component": "coherence_check"})

        if kwargs.get("adaptive_retrieval", False):
            pipeline_config.append({"component": "adaptive_k"})

        # Build the pipeline
        self.memory_manager.build_pipeline(pipeline_config)

    def retrieve_for_context(
        self,
        current_input: str,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        top_k: int = 5,
        confidence_threshold: float = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories relevant to the current conversation context.

        Args:
            current_input: The current user input
            conversation_history: Recent conversation history
            top_k: Number of memories to retrieve
            confidence_threshold: Optional override for confidence threshold

        Returns:
            list of relevant memory entries with metadata
        """
        # Prepare context
        context = {
            "conversation_history": conversation_history or [],
            "top_k": top_k,
            "memory": self.memory,
            "confidence_threshold": confidence_threshold,
        }

        # Create query embedding
        query_embedding = self.embedding_model.encode(current_input)
        context["query_embedding"] = query_embedding

        # Execute retrieval pipeline
        result_context = self.memory_manager.execute_pipeline(current_input, context)

        # Extract and return results
        return result_context.get("results", [])
