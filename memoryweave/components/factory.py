"""
Factory for creating memory components.
"""

from typing import Any

from memoryweave.components.adapters import CategoryAdapter
from memoryweave.components.base import Component
from memoryweave.components.category_manager import CategoryManager
from memoryweave.components.memory_adapter import MemoryAdapter
from memoryweave.components.retrieval_strategies_impl import (
    CategoryRetrievalStrategy,
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
    TwoStageRetrievalStrategy,
)


def create_memory_system(config: dict[str, Any]) -> dict[str, Component]:
    """
    Create a memory system from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of components
    """
    components: dict[str, Component] = {}

    # Create memory adapter
    memory_config = config.get("memory", {})
    memory_adapter = MemoryAdapter(**memory_config)
    components["memory"] = memory_adapter

    # Create category manager if enabled
    if config.get("use_categories", False):
        category_config = config.get("category", {})
        category_manager = CategoryManager()
        category_manager.initialize(category_config)
        components["category_manager"] = category_manager

        # Create category adapter
        category_adapter = CategoryAdapter(category_manager)
        components["category_adapter"] = category_adapter

    # Create retrieval strategy
    retrieval_type = config.get("retrieval_type", "similarity")
    retrieval_config = config.get("retrieval", {})

    if retrieval_type == "similarity":
        retrieval_strategy = SimilarityRetrievalStrategy(memory_adapter.memory)
        retrieval_strategy.initialize(retrieval_config)
        components["retrieval_strategy"] = retrieval_strategy
    elif retrieval_type == "hybrid":
        retrieval_strategy = HybridRetrievalStrategy(memory_adapter.memory)
        retrieval_strategy.initialize(retrieval_config)
        components["retrieval_strategy"] = retrieval_strategy
    elif retrieval_type == "temporal":
        retrieval_strategy = TemporalRetrievalStrategy(memory_adapter.memory)
        retrieval_strategy.initialize(retrieval_config)
        components["retrieval_strategy"] = retrieval_strategy
    elif retrieval_type == "two_stage":
        # Create base strategy
        base_type = retrieval_config.get("base_strategy", "similarity")
        base_config = retrieval_config.get("base_config", {})

        if base_type == "similarity":
            base_strategy = SimilarityRetrievalStrategy(memory_adapter.memory)
        elif base_type == "hybrid":
            base_strategy = HybridRetrievalStrategy(memory_adapter.memory)
        else:
            base_strategy = SimilarityRetrievalStrategy(memory_adapter.memory)

        base_strategy.initialize(base_config)

        # Create two-stage strategy
        retrieval_strategy = TwoStageRetrievalStrategy(
            memory_adapter.memory,
            base_strategy=base_strategy,
        )
        retrieval_strategy.initialize(retrieval_config)
        components["retrieval_strategy"] = retrieval_strategy
    elif retrieval_type == "category":
        if "category_manager" not in components:
            # Create category manager if not already created
            category_config = config.get("category", {})
            category_manager = CategoryManager()
            category_manager.initialize(category_config)
            components["category_manager"] = category_manager

        retrieval_strategy = CategoryRetrievalStrategy(memory_adapter.memory)
        retrieval_strategy.initialize(retrieval_config)
        components["retrieval_strategy"] = retrieval_strategy

    return components


def configure_memory_pipeline(config: dict[str, Any]) -> list[Component]:
    """
    Configure a memory pipeline from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of pipeline components
    """
    components = create_memory_system(config)

    # Create pipeline
    pipeline: list[Component] = []

    # Add components to pipeline in the correct order
    if "memory" in components:
        pipeline.append(components["memory"])

    if "category_adapter" in components:
        pipeline.append(components["category_adapter"])

    if "retrieval_strategy" in components:
        pipeline.append(components["retrieval_strategy"])

    return pipeline
