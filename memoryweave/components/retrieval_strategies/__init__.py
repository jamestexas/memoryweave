"""
Retrieval strategies for MemoryWeave.

This module contains various retrieval strategies for retrieving memories based
on different approaches, such as similarity, temporal factors, or hybrid methods.
"""

# Import strategy implementations from the parent module
from memoryweave.components.retrieval_strategies_impl import (
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
    HybridRetrievalStrategy,
    TwoStageRetrievalStrategy,
    CategoryRetrievalStrategy,
)

# Import new strategies
from memoryweave.components.retrieval_strategies.hybrid_bm25_vector_strategy import (
    HybridBM25VectorStrategy,
)
from memoryweave.components.retrieval_strategies.contextual_fabric_strategy import (
    ContextualFabricStrategy,
)

__all__ = [
    "SimilarityRetrievalStrategy",
    "TemporalRetrievalStrategy",
    "HybridRetrievalStrategy",
    "TwoStageRetrievalStrategy",
    "CategoryRetrievalStrategy",
    "HybridBM25VectorStrategy",
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
    "hybrid_bm25_vector": HybridBM25VectorStrategy,
    "contextual_fabric": ContextualFabricStrategy,
}
