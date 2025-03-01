"""Retrieval component factory for MemoryWeave.

This module provides factories for creating retrieval-related components,
such as retrieval strategies and query processors.
"""

from typing import Any, Dict, List, Optional

from memoryweave.config.options import get_default_config
from memoryweave.config.validation import ConfigValidationError, validate_config
from memoryweave.interfaces.memory import IActivationManager, IMemoryStore, IVectorStore
from memoryweave.interfaces.query import IQueryAdapter, IQueryAnalyzer, IQueryExpander
from memoryweave.interfaces.retrieval import IRetrievalStrategy
from memoryweave.query.adaptation import QueryTypeAdapter
from memoryweave.query.analyzer import SimpleQueryAnalyzer
from memoryweave.query.keyword import KeywordExpander
from memoryweave.retrieval.hybrid import HybridRetrievalStrategy
from memoryweave.retrieval.similarity import SimilarityRetrievalStrategy
from memoryweave.retrieval.temporal import TemporalRetrievalStrategy
from memoryweave.retrieval.two_stage import TwoStageRetrievalStrategy


class RetrievalFactory:
    """Factory for creating retrieval-related components."""

    @staticmethod
    def create_retrieval_strategy(
        strategy_type: str,
        memory_store: IMemoryStore,
        vector_store: IVectorStore,
        activation_manager: Optional[IActivationManager] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> IRetrievalStrategy:
        """Create a retrieval strategy component.
        
        Args:
            strategy_type: Type of retrieval strategy to create
            memory_store: Memory store component
            vector_store: Vector store component
            activation_manager: Optional activation manager component
            config: Optional configuration for the retrieval strategy
            
        Returns:
            Configured retrieval strategy
            
        Raises:
            ValueError: If the strategy_type is invalid
            ConfigValidationError: If the configuration is invalid
        """
        # Use default config if none provided
        if config is None:
            config = {}

        # Merge with defaults
        defaults = get_default_config("retrieval_strategy")
        merged_config = {**defaults, **config}

        # Validate config
        is_valid, errors = validate_config(merged_config, "retrieval_strategy")
        if not is_valid:
            raise ConfigValidationError(errors, "retrieval_strategy")

        # Create strategy based on type
        if strategy_type == "similarity":
            strategy = SimilarityRetrievalStrategy(memory_store, vector_store)
        elif strategy_type == "temporal":
            if activation_manager is None:
                raise ValueError("Activation manager is required for temporal retrieval strategy")
            strategy = TemporalRetrievalStrategy(memory_store, activation_manager)
        elif strategy_type == "hybrid":
            if activation_manager is None:
                raise ValueError("Activation manager is required for hybrid retrieval strategy")
            strategy = HybridRetrievalStrategy(memory_store, vector_store, activation_manager)
        elif strategy_type == "two_stage":
            # Optionally use first and second stage strategies
            first_stage = None
            second_stage = None

            if "first_stage_config" in merged_config:
                first_stage_type = merged_config["first_stage_config"].get("type", "similarity")
                first_stage = RetrievalFactory.create_retrieval_strategy(
                    first_stage_type,
                    memory_store,
                    vector_store,
                    activation_manager,
                    merged_config["first_stage_config"]
                )

            if "second_stage_config" in merged_config:
                second_stage_type = merged_config["second_stage_config"].get("type", "similarity")
                second_stage = RetrievalFactory.create_retrieval_strategy(
                    second_stage_type,
                    memory_store,
                    vector_store,
                    activation_manager,
                    merged_config["second_stage_config"]
                )

            strategy = TwoStageRetrievalStrategy(
                memory_store,
                vector_store,
                first_stage,
                second_stage
            )
        else:
            raise ValueError(f"Unknown retrieval strategy type: {strategy_type}")

        # Configure the strategy
        strategy.configure(merged_config)

        return strategy

    @staticmethod
    def create_query_analyzer(config: Optional[Dict[str, Any]] = None) -> IQueryAnalyzer:
        """Create a query analyzer component.
        
        Args:
            config: Optional configuration for the query analyzer
            
        Returns:
            Configured query analyzer
            
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        # Use default config if none provided
        if config is None:
            config = {}

        # Merge with defaults
        defaults = get_default_config("query_analyzer")
        merged_config = {**defaults, **config}

        # Validate config
        is_valid, errors = validate_config(merged_config, "query_analyzer")
        if not is_valid:
            raise ConfigValidationError(errors, "query_analyzer")

        # Create analyzer
        analyzer = SimpleQueryAnalyzer()

        # Configure the analyzer
        analyzer.configure(merged_config)

        return analyzer

    @staticmethod
    def create_query_adapter(config: Optional[Dict[str, Any]] = None) -> IQueryAdapter:
        """Create a query adapter component.
        
        Args:
            config: Optional configuration for the query adapter
            
        Returns:
            Configured query adapter
            
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        # Use default config if none provided
        if config is None:
            config = {}

        # Merge with defaults
        defaults = get_default_config("query_adapter")
        merged_config = {**defaults, **config}

        # Validate config
        is_valid, errors = validate_config(merged_config, "query_adapter")
        if not is_valid:
            raise ConfigValidationError(errors, "query_adapter")

        # Create adapter
        adapter = QueryTypeAdapter()

        # Configure the adapter
        adapter.configure(merged_config)

        return adapter

    @staticmethod
    def create_query_expander(
        word_embeddings: Optional[Dict[str, List[float]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> IQueryExpander:
        """Create a query expander component.
        
        Args:
            word_embeddings: Optional word embeddings for expansion
            config: Optional configuration for the query expander
            
        Returns:
            Configured query expander
        """
        # Use default config if none provided
        if config is None:
            config = {}

        # Create expander
        expander = KeywordExpander(word_embeddings)

        # Configure the expander
        expander.configure(config)

        return expander
