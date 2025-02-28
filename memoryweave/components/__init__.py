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
from memoryweave.components.retriever import Retriever

__all__ = [
    # Base classes
    "Component",
    "MemoryComponent",
    "RetrievalComponent",
    "RetrievalStrategy",
    "PostProcessor",
    # Components
    "MemoryManager",
    "PersonalAttributeManager",
    "QueryAnalyzer",
    "Retriever",
    # Retrieval strategies
    "HybridRetrievalStrategy",
    "SimilarityRetrievalStrategy",
    "TemporalRetrievalStrategy",
    # Post-processors
    "AdaptiveKProcessor",
    "KeywordBoostProcessor",
    "SemanticCoherenceProcessor",
]
