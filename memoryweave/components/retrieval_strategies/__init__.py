"""
Retrieval strategies for MemoryWeave.

This module contains various retrieval strategies for retrieving memories based
on different approaches, such as similarity, temporal factors, or hybrid methods.
"""

# Import strategy implementations from the parent module
from memoryweave.components.retrieval_strategies.contextual_fabric_strategy import (
    ContextualFabricStrategy,
)
from memoryweave.components.retrieval_strategies_impl import (
    CategoryRetrievalStrategy,
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
    TwoStageRetrievalStrategy,
)

__all__ = [
    "SimilarityRetrievalStrategy",
    "TemporalRetrievalStrategy",
    "HybridRetrievalStrategy",
    "TwoStageRetrievalStrategy",
    "CategoryRetrievalStrategy",
    "ContextualFabricStrategy",
]

# Define a mapping of strategy names to classes
# This allows dynamic instantiation by name
available_strategies = {
    "similarity": SimilarityRetrievalStrategy,
    "temporal": TemporalRetrievalStrategy,
    "hybrid": HybridRetrievalStrategy,
    "two_stage": TwoStageRetrievalStrategy,
    "category": CategoryRetrievalStrategy,
    "contextual_fabric": ContextualFabricStrategy,
}
