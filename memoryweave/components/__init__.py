"""
Component-based architecture for MemoryWeave.

This module provides modular components for memory retrieval, query analysis,
and post-processing that can be combined into flexible retrieval pipelines.
"""

from memoryweave.components.base import (
    Component,
    MemoryComponent,
    PostProcessor,
    RetrievalComponent,
    RetrievalStrategy,
)
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
from memoryweave.components.retrieval_strategies import (
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
    TwoStageRetrievalStrategy,
)
from memoryweave.components.retriever import Retriever

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
    "MemoryManager",
    "PersonalAttributeManager",
    "QueryAnalyzer",
    "QueryTypeAdapter",
    "Retriever",
    # Retrieval strategies
    "HybridRetrievalStrategy",
    "SimilarityRetrievalStrategy",
    "TemporalRetrievalStrategy",
    "TwoStageRetrievalStrategy",
    # Post-processors
    "AdaptiveKProcessor",
    "KeywordBoostProcessor",
    "MinimumResultGuaranteeProcessor",
    "PersonalAttributeProcessor",
    "SemanticCoherenceProcessor",
]
