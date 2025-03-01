"""Feature migration utilities for MemoryWeave.

This module provides utilities for migrating features from the old architecture
to the new component-based architecture, ensuring functional equivalence.
"""

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from memoryweave.adapters.memory_adapter import (
    LegacyActivationManagerAdapter,
    LegacyMemoryAdapter,
    LegacyVectorStoreAdapter,
)
from memoryweave.adapters.pipeline_adapter import PipelineToLegacyAdapter
from memoryweave.adapters.retrieval_adapter import LegacyRetrieverAdapter
from memoryweave.factory.memory import MemoryFactory
from memoryweave.factory.pipeline import PipelineFactory
from memoryweave.factory.retrieval import RetrievalFactory
from memoryweave.interfaces.pipeline import IComponent, IPipeline


class FeatureMigrator:
    """
    Utility for migrating features from old to new architecture.

    This class provides methods for mapping between old and new components,
    creating equivalent components, and ensuring feature parity during migration.
    """

    def __init__(self):
        """Initialize the feature migrator."""
        self._logger = logging.getLogger(__name__)
        self._component_mappings = {}  # Maps old component types to new component types

    def migrate_memory_system(self, legacy_memory) -> Dict[str, IComponent]:
        """
        Migrate a legacy ContextualMemory to the new architecture.

        Args:
            legacy_memory: A legacy ContextualMemory object

        Returns:
            Dictionary mapping component IDs to migrated components
        """
        self._logger.info("Migrating legacy memory system to new architecture")
        components = {}

        # Create adapters first
        memory_adapter = LegacyMemoryAdapter(legacy_memory)
        components[memory_adapter.get_id()] = memory_adapter

        vector_adapter = LegacyVectorStoreAdapter(legacy_memory, memory_adapter)
        components[vector_adapter.get_id()] = vector_adapter

        activation_adapter = LegacyActivationManagerAdapter(legacy_memory, memory_adapter)
        components[activation_adapter.get_id()] = activation_adapter

        # Check for and create native components based on legacy configuration
        # Memory Store
        try:
            memory_store = MemoryFactory.create_memory_store(
                {"max_memories": getattr(legacy_memory, "max_memories", 1000)}
            )
            components["native_memory_store"] = memory_store
        except Exception as e:
            self._logger.warning(f"Failed to create native memory store: {e}")

        # Retrieval Strategy - create native versions of strategies in legacy retriever
        if hasattr(legacy_memory, "memory_retriever"):
            retriever = legacy_memory.memory_retriever
            try:
                # Create similarity strategy
                similarity_strategy = RetrievalFactory.create_retrieval_strategy(
                    "similarity",
                    memory_store,
                    vector_adapter,
                    activation_adapter,
                    {
                        "similarity_threshold": getattr(
                            retriever, "default_confidence_threshold", 0.0
                        ),
                        "max_results": 10,
                    },
                )
                components["native_similarity_strategy"] = similarity_strategy

                # Create hybrid strategy if activation boost is used
                hybrid_strategy = RetrievalFactory.create_retrieval_strategy(
                    "hybrid",
                    memory_store,
                    vector_adapter,
                    activation_adapter,
                    {
                        "similarity_threshold": getattr(
                            retriever, "default_confidence_threshold", 0.0
                        ),
                        "max_results": 10,
                        "recency_bias": 0.3,
                        "activation_boost": 0.2,
                    },
                )
                components["native_hybrid_strategy"] = hybrid_strategy

                # Create two-stage strategy if semantic coherence is used
                if getattr(retriever, "semantic_coherence_check", False):
                    two_stage_strategy = RetrievalFactory.create_retrieval_strategy(
                        "two_stage",
                        memory_store,
                        vector_adapter,
                        activation_adapter,
                        {
                            "first_stage_threshold": 0.5,
                            "second_stage_threshold": getattr(
                                retriever, "default_confidence_threshold", 0.0
                            ),
                            "first_stage_max": 30,
                            "final_max_results": 10,
                        },
                    )
                    components["native_two_stage_strategy"] = two_stage_strategy
            except Exception as e:
                self._logger.warning(f"Failed to create native retrieval strategies: {e}")

        # Create a retriever adapter
        if hasattr(legacy_memory, "memory_retriever"):
            retriever_adapter = LegacyRetrieverAdapter(
                legacy_memory.memory_retriever, memory_adapter
            )
        else:
            retriever_adapter = LegacyRetrieverAdapter(legacy_memory, memory_adapter)
        components[retriever_adapter.get_id()] = retriever_adapter

        # Create a query analyzer
        try:
            query_analyzer = RetrievalFactory.create_query_analyzer()
            components["native_query_analyzer"] = query_analyzer
        except Exception as e:
            self._logger.warning(f"Failed to create native query analyzer: {e}")

        # Create a query adapter
        try:
            query_adapter = RetrievalFactory.create_query_adapter()
            components["native_query_adapter"] = query_adapter
        except Exception as e:
            self._logger.warning(f"Failed to create native query adapter: {e}")

        self._logger.info(f"Migrated {len(components)} components")
        return components

    def create_migration_pipeline(self, components: Dict[str, IComponent]) -> IPipeline:
        """
        Create a pipeline that combines the migrated components.

        Args:
            components: Dictionary of migrated components

        Returns:
            A configured pipeline using the migrated components
        """
        self._logger.info("Creating migration pipeline")

        # Create pipeline manager
        pipeline_manager = PipelineFactory.create_pipeline_manager()

        # Register all components
        for component_id, component in components.items():
            pipeline_manager.register_component(component)

        # Create a pipeline with appropriate stages
        # Structure: QueryAnalyzer -> QueryAdapter -> RetrievalStrategy
        stage_ids = []

        # Add query analyzer if available
        if "native_query_analyzer" in components:
            stage_ids.append("native_query_analyzer")

        # Add query adapter if available
        if "native_query_adapter" in components:
            stage_ids.append("native_query_adapter")

        # Add retrieval strategy - prefer native strategies if available
        if "native_two_stage_strategy" in components:
            stage_ids.append("native_two_stage_strategy")
        elif "native_hybrid_strategy" in components:
            stage_ids.append("native_hybrid_strategy")
        elif "native_similarity_strategy" in components:
            stage_ids.append("native_similarity_strategy")
        else:
            # Fall back to legacy adapter
            for component_id in components:
                if component_id.endswith("retriever_adapter"):
                    stage_ids.append(component_id)
                    break

        # Create the pipeline
        pipeline = pipeline_manager.create_pipeline(name="migration_pipeline", stage_ids=stage_ids)

        if pipeline is None:
            self._logger.error("Failed to create migration pipeline")
            raise ValueError("Failed to create migration pipeline")

        self._logger.info(f"Created migration pipeline with {len(stage_ids)} stages")
        return pipeline

    def validate_migration(
        self,
        legacy_retriever,
        migrated_pipeline: IPipeline,
        test_queries: List[Union[str, np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Validate that the migrated pipeline produces equivalent results to the legacy retriever.

        Args:
            legacy_retriever: The legacy retriever to compare against
            migrated_pipeline: The migrated pipeline
            test_queries: List of test queries (embeddings or text)

        Returns:
            Dictionary with validation results and metrics
        """
        self._logger.info("Validating migration")
        validation_results = {
            "total_queries": len(test_queries),
            "success_count": 0,
            "failure_count": 0,
            "recall_metrics": [],
            "precision_metrics": [],
            "result_count_diffs": [],
        }

        # Create pipeline adapter for legacy interface
        pipeline_adapter = PipelineToLegacyAdapter(migrated_pipeline)

        # Run test queries through both systems
        for i, query in enumerate(test_queries):
            self._logger.debug(f"Testing query {i + 1}/{len(test_queries)}")

            try:
                # Get results from legacy retriever
                if hasattr(legacy_retriever, "retrieve_for_context"):
                    legacy_results = legacy_retriever.retrieve_for_context(query)
                elif hasattr(legacy_retriever, "retrieve_memories"):
                    legacy_results = legacy_retriever.retrieve_memories(query)
                else:
                    self._logger.warning("Unknown legacy retriever interface")
                    continue

                # Get results from migrated pipeline
                migrated_results = pipeline_adapter.retrieve_for_context(query)

                # Compare results
                if self._compare_retrieval_results(legacy_results, migrated_results):
                    validation_results["success_count"] += 1
                else:
                    validation_results["failure_count"] += 1

                # Calculate metrics
                recall, precision = self._calculate_result_metrics(legacy_results, migrated_results)
                validation_results["recall_metrics"].append(recall)
                validation_results["precision_metrics"].append(precision)
                validation_results["result_count_diffs"].append(
                    len(migrated_results) - len(legacy_results)
                )
            except Exception as e:
                self._logger.error(f"Error validating query {i + 1}: {e}")
                validation_results["failure_count"] += 1

        # Calculate summary metrics
        if validation_results["recall_metrics"]:
            validation_results["avg_recall"] = sum(validation_results["recall_metrics"]) / len(
                validation_results["recall_metrics"]
            )
            validation_results["avg_precision"] = sum(
                validation_results["precision_metrics"]
            ) / len(validation_results["precision_metrics"])

        self._logger.info(
            f"Validation complete: {validation_results['success_count']} successes, "
            f"{validation_results['failure_count']} failures"
        )
        return validation_results

    def _compare_retrieval_results(
        self, legacy_results: List[Tuple], migrated_results: List[Tuple]
    ) -> bool:
        """
        Compare legacy and migrated retrieval results for equivalence.

        Args:
            legacy_results: Results from legacy retriever
            migrated_results: Results from migrated pipeline

        Returns:
            True if results are equivalent, False otherwise
        """
        # Check if both result sets are empty
        if not legacy_results and not migrated_results:
            return True

        # Check if both result sets have the same length
        if len(legacy_results) != len(migrated_results):
            return False

        # Check if both result sets contain the same memory IDs/indices
        legacy_ids = set(result[0] for result in legacy_results)
        migrated_ids = set(result[0] for result in migrated_results)

        return legacy_ids == migrated_ids

    def _calculate_result_metrics(
        self, legacy_results: List[Tuple], migrated_results: List[Tuple]
    ) -> Tuple[float, float]:
        """
        Calculate recall and precision metrics for migrated results.

        Args:
            legacy_results: Results from legacy retriever
            migrated_results: Results from migrated pipeline

        Returns:
            Tuple of (recall, precision)
        """
        if not legacy_results:
            return 1.0 if not migrated_results else 0.0

        if not migrated_results:
            return 0.0

        # Extract memory IDs
        legacy_ids = set(result[0] for result in legacy_results)
        migrated_ids = set(result[0] for result in migrated_results)

        # Calculate intersection
        intersection = legacy_ids.intersection(migrated_ids)

        # Calculate recall and precision
        recall = len(intersection) / len(legacy_ids) if legacy_ids else 0.0
        precision = len(intersection) / len(migrated_ids) if migrated_ids else 0.0

        return recall, precision
