"""
Component-based architecture for MemoryWeave.

This module provides modular components for memory retrieval, query analysis,
and post-processing that can be combined into flexible retrieval pipelines.
"""

from memoryweave.components.activation import ActivationManager
from memoryweave.components.associative_linking import (
    AssociativeMemoryLinker,
    AssociativeNetworkVisualizer,
)
from memoryweave.components.base import (
    Component,
    MemoryComponent,
    PostProcessor,
    RetrievalComponent,
    RetrievalStrategy,
)

# Import contextual fabric components
from memoryweave.components.context_enhancement import (
    ContextSignalExtractor,
    ContextualEmbeddingEnhancer,
)
from memoryweave.components.dynamic_threshold_adjuster import DynamicThresholdAdjuster
from memoryweave.components.keyword_expander import KeywordExpander
from memoryweave.components.memory_decay import MemoryDecayComponent
from memoryweave.components.memory_encoding import MemoryEncoder
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
from memoryweave.components.query_context_builder import QueryContextBuilder

# Import retrieval strategies
from memoryweave.components.retrieval_strategies import (
    CategoryRetrievalStrategy,
    ContextualFabricStrategy,
    HybridBM25VectorStrategy,
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
    TwoStageRetrievalStrategy,
)
from memoryweave.components.retriever import Retriever
from memoryweave.components.temporal_context import TemporalContextBuilder, TemporalDecayComponent

__all__ = [
    # Base classes
    "Component",
    "MemoryComponent",
    "RetrievalComponent",
    "RetrievalStrategy",
    "PostProcessor",
    # Components
    "KeywordExpander",
    "MemoryDecayComponent",
    "MemoryEncoder",
    "MemoryManager",
    "PersonalAttributeManager",
    "QueryAnalyzer",
    "QueryTypeAdapter",
    "QueryContextBuilder",
    "Retriever",
    "DynamicThresholdAdjuster",
    # Retrieval strategies
    "HybridRetrievalStrategy",
    "SimilarityRetrievalStrategy",
    "TemporalRetrievalStrategy",
    "TwoStageRetrievalStrategy",
    "CategoryRetrievalStrategy",
    "HybridBM25VectorStrategy",
    "ContextualFabricStrategy",
    # Post-processors
    "AdaptiveKProcessor",
    "KeywordBoostProcessor",
    "MinimumResultGuaranteeProcessor",
    "PersonalAttributeProcessor",
    "SemanticCoherenceProcessor",
    # Contextual Fabric Components
    "ContextualEmbeddingEnhancer",
    "ContextSignalExtractor",
    "AssociativeMemoryLinker",
    "AssociativeNetworkVisualizer",
    "TemporalContextBuilder",
    "TemporalDecayComponent",
    "ActivationManager",
]
