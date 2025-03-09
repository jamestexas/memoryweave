# tests/components/retrieval_strategies/test_contextual_fabric_strategy.py
"""
Tests for the ContextualFabricStrategy.

This file contains tests for the main contextual fabric strategy that
leverages multiple context dimensions for memory retrieval.
"""

from unittest.mock import patch

import numpy as np

from memoryweave.components.retrieval_strategies.contextual_fabric_strategy import (
    ContextualFabricStrategy,
)


class TestContextualFabricStrategy:
    """Tests for the ContextualFabricStrategy class."""

    def test_initialization(
        self, memory_store, associative_linker, temporal_context, activation_manager
    ):
        """Test initialization with different configurations."""
        # Test default initialization
        strategy = ContextualFabricStrategy(
            memory_store=memory_store,
            associative_linker=associative_linker,
            temporal_context=temporal_context,
            activation_manager=activation_manager,
        )
        assert strategy.memory_store == memory_store
        assert strategy.associative_linker == associative_linker
        assert strategy.temporal_context == temporal_context
        assert strategy.activation_manager == activation_manager
        assert strategy.confidence_threshold == 0.1  # Default value

        # Test initialization with configuration
        config = {
            "confidence_threshold": 0.2,
            "similarity_weight": 0.6,
            "associative_weight": 0.2,
            "temporal_weight": 0.1,
            "activation_weight": 0.1,
            "max_associative_hops": 3,
            "debug": True,
        }
        strategy.initialize(config)
        assert strategy.confidence_threshold == 0.2
        assert strategy.similarity_weight == 0.6
        assert strategy.associative_weight == 0.2
        assert strategy.temporal_weight == 0.1
        assert strategy.activation_weight == 0.1
        assert strategy.max_associative_hops == 3
        assert strategy.debug is True

    def test_retrieve_basic(self, memory_store, query_embedding, base_context):
        """Test basic retrieval functionality with minimal dependencies."""
        strategy = ContextualFabricStrategy(memory_store=memory_store)
        strategy.initialize({"confidence_threshold": 0.0})  # No threshold for testing

        results = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # Check basic properties of results
        assert len(results) > 0
        assert "memory_id" in results[0]
        assert "relevance_score" in results[0]
        assert "similarity_score" in results[0]

        # Results should be sorted by relevance_score
        scores = [r["relevance_score"] for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_retrieve_with_full_context(
        self,
        memory_store,
        query_embedding,
        base_context,
        associative_linker,
        temporal_context,
        activation_manager,
    ):
        """Test retrieval with all context components available."""
        strategy = ContextualFabricStrategy(
            memory_store=memory_store,
            associative_linker=associative_linker,
            temporal_context=temporal_context,
            activation_manager=activation_manager,
        )
        strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "similarity_weight": 0.5,
                "associative_weight": 0.2,
                "temporal_weight": 0.1,
                "activation_weight": 0.2,
            }
        )

        # Set up temporal context to detect temporal references
        temporal_context.extract_time_references.return_value = {
            "has_temporal_reference": True,
            "time_type": "relative",
            "relative_time": 300,  # This should match memory 2's creation time
            "time_keywords": ["yesterday"],
        }

        # Create a context with query text for temporal extraction
        context = base_context.copy()
        context["query"] = "What happened yesterday?"

        results = strategy.retrieve(
            query_embedding, top_k=5, context=context, query=context["query"]
        )

        # Check that we have results
        assert len(results) > 0

        # Check that temporal context was used
        temporal_context.extract_time_references.assert_called_with("What happened yesterday?")

        # Check that associative linker was used
        associative_linker.traverse_associative_network.assert_called()

        # Check that activation manager was used
        activation_manager.get_activated_memories.assert_called()

        # Verify result structure contains all contribution scores
        assert "similarity_contribution" in results[0]
        assert "associative_contribution" in results[0]
        assert "temporal_contribution" in results[0]
        assert "activation_contribution" in results[0]

    def test_temporal_retrieval_behavior(self, memory_store, temporal_context, base_context):
        """Test that temporal references in queries affect memory retrieval."""
        # Create strategies with different temporal weights
        temporal_focused_strategy = ContextualFabricStrategy(
            memory_store=memory_store, temporal_context=temporal_context
        )
        temporal_focused_strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "similarity_weight": 0.2,
                "temporal_weight": 0.8,  # High temporal weight
            }
        )

        non_temporal_strategy = ContextualFabricStrategy(
            memory_store=memory_store, temporal_context=temporal_context
        )
        non_temporal_strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "similarity_weight": 1.0,
                "temporal_weight": 0.0,  # No temporal weight
            }
        )

        # Set up temporal context to match memory with creation_time=300
        temporal_context.extract_time_references.return_value = {
            "has_temporal_reference": True,
            "time_type": "relative",
            "relative_time": 300,  # Should match memory 2's creation time
            "time_keywords": ["yesterday"],
        }

        # Create a context with query containing temporal reference
        context = base_context.copy()
        context["query"] = "What happened yesterday?"

        # Get results from both strategies
        temporal_results = temporal_focused_strategy.retrieve(
            np.array([0.2, 0.3, 0.4]), top_k=5, context=context, query=context["query"]
        )
        non_temporal_results = non_temporal_strategy.retrieve(
            np.array([0.2, 0.3, 0.4]), top_k=5, context=context, query=context["query"]
        )

        # Memory 2 (with creation_time=300) should rank higher in temporal results
        memory_2_temporal_rank = next(
            (i for i, r in enumerate(temporal_results) if r["memory_id"] == 2), float("inf")
        )
        memory_2_non_temporal_rank = next(
            (i for i, r in enumerate(non_temporal_results) if r["memory_id"] == 2), float("inf")
        )

        # Memory 2 should rank better (lower index) in temporal results
        if memory_2_temporal_rank != float("inf") and memory_2_non_temporal_rank != float("inf"):
            assert memory_2_temporal_rank <= memory_2_non_temporal_rank, (
                "Memory with matching timestamp should rank higher with temporal weighting"
            )

    def test_retrieval_influenced_by_associations(
        self, memory_store, query_embedding, base_context, associative_linker
    ):
        """Test that memory retrieval is influenced by associative connections."""
        # Configure strategy to prioritize associative connections
        strategy = ContextualFabricStrategy(
            memory_store=memory_store, associative_linker=associative_linker
        )
        strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "similarity_weight": 0.3,  # Lower weight for similarity
                "associative_weight": 0.7,  # Higher weight for associations
                "max_associative_hops": 2,
            }
        )

        # Set up associative linker to strongly link to memory IDs 1 and 3
        associative_linker.traverse_associative_network.return_value = {
            1: 0.9,  # Strong link to memory 1
            3: 0.7,  # Moderate link to memory 3
        }

        # First retrieval - with associations
        results_with_associations = strategy.retrieve(
            query_embedding, top_k=5, context=base_context
        )

        # Second retrieval - without associations (by using a different strategy instance)
        strategy_no_associations = ContextualFabricStrategy(memory_store=memory_store)
        strategy_no_associations.initialize({"confidence_threshold": 0.0})
        results_without_associations = strategy_no_associations.retrieve(
            query_embedding, top_k=5, context=base_context
        )

        # Test behavior: Associative memories should rank higher with associations than without
        memory_1_with_associations = next(
            (r for r in results_with_associations if r["memory_id"] == 1), None
        )
        memory_1_without_associations = next(
            (r for r in results_without_associations if r["memory_id"] == 1), None
        )

        assert memory_1_with_associations is not None, (
            "Memory 1 should be in results with associations"
        )

        # Either memory 1 should be absent in results_without_associations or its rank should be worse
        if memory_1_without_associations is not None:
            with_rank = [r["memory_id"] for r in results_with_associations].index(1)
            without_rank = [r["memory_id"] for r in results_without_associations].index(1)
            assert with_rank < without_rank, "Memory 1 should rank higher with associations"

    def test_retrieval_influenced_by_activation(
        self,
        memory_store,
        query_embedding,
        base_context,
        activation_manager,
    ):
        """Test that memory retrieval is influenced by memory activation levels."""
        # Configure strategy to prioritize activated memories
        strategy = ContextualFabricStrategy(
            memory_store=memory_store,
            activation_manager=activation_manager,
        )
        strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "similarity_weight": 0.3,  # Lower weight for similarity
                "activation_weight": 0.7,  # Higher weight for activation
            }
        )

        # Set up activation manager to return specific activations
        activation_manager.get_activated_memories.return_value = [
            (2, 0.9),  # High activation for memory 2
            (4, 0.7),  # Moderate activation for memory 4
        ]

        # First retrieval - with activation
        results_with_activation = strategy.retrieve(query_embedding, top_k=5, context=base_context)

        # Second retrieval - without activation (by using a different strategy instance)
        strategy_no_activation = ContextualFabricStrategy(memory_store=memory_store)
        strategy_no_activation.initialize({"confidence_threshold": 0.0})
        results_without_activation = strategy_no_activation.retrieve(
            query_embedding, top_k=5, context=base_context
        )

        # Test behavior: Activated memories should rank higher with activation than without
        memory_2_with_activation = next(
            (r for r in results_with_activation if r["memory_id"] == 2), None
        )
        memory_2_without_activation = next(
            (r for r in results_without_activation if r["memory_id"] == 2), None
        )

        assert memory_2_with_activation is not None, "Memory 2 should be in results with activation"

        # Either memory 2 should be absent in results_without_activation or its rank should be worse
        if memory_2_without_activation is not None:
            with_rank = [r["memory_id"] for r in results_with_activation].index(2)
            without_rank = [r["memory_id"] for r in results_without_activation].index(2)
            assert with_rank < without_rank, "Memory 2 should rank higher with activation"

    def test_parameter_adaptation_affects_results(
        self, memory_store, query_embedding, base_context
    ):
        """Test that parameter adaptation via context affects retrieval results."""
        strategy = ContextualFabricStrategy(memory_store=memory_store)

        # Initialize with base configuration
        base_config = {
            "confidence_threshold": 0.1,
            "similarity_weight": 0.5,
            "associative_weight": 0.3,
            "temporal_weight": 0.1,
            "activation_weight": 0.1,
        }
        strategy.initialize(base_config)

        # Create a context with adapted parameters that emphasize similarity more
        context_adapted = base_context.copy()
        context_adapted["adapted_retrieval_params"] = {
            "confidence_threshold": 0.05,
            "similarity_weight": 0.8,  # Much higher similarity weight
            "associative_weight": 0.1,
            "temporal_weight": 0.05,
            "activation_weight": 0.05,
        }

        # Get results with adapted parameters
        results_adapted = strategy.retrieve(query_embedding, top_k=5, context=context_adapted)

        # Get results with original parameters
        results_original = strategy.retrieve(query_embedding, top_k=5, context=base_context)

        # Results should differ when using adapted parameters
        adapted_ids = [r["memory_id"] for r in results_adapted]
        original_ids = [r["memory_id"] for r in results_original]

        # Either the order or the IDs should be different
        assert (adapted_ids != original_ids) or any(
            results_adapted[i]["relevance_score"] != results_original[i]["relevance_score"]
            for i in range(min(len(results_adapted), len(results_original)))
        ), "Adapted parameters should change retrieval results"

    def test_confidence_threshold_filtering_behavior(
        self, memory_store, query_embedding, base_context
    ):
        """Test that confidence threshold filters low-confidence results."""
        # Strategy with high threshold
        high_threshold_strategy = ContextualFabricStrategy(memory_store=memory_store)
        high_threshold_strategy.initialize(
            {
                "confidence_threshold": 0.9,  # Very high threshold
                "min_results": 0,  # No minimum results
            }
        )

        # Strategy with lower threshold but same minimum
        low_threshold_strategy = ContextualFabricStrategy(memory_store=memory_store)
        low_threshold_strategy.initialize(
            {
                "confidence_threshold": 0.0,  # No threshold
                "min_results": 0,  # No minimum results
            }
        )

        # Strategy with high threshold but minimum results
        min_results_strategy = ContextualFabricStrategy(memory_store=memory_store)
        min_results_strategy.initialize(
            {
                "confidence_threshold": 0.9,  # Very high threshold
                "min_results": 2,  # Minimum of 2 results
            }
        )

        # Get results for each configuration
        results_high_threshold = high_threshold_strategy.retrieve(
            query_embedding, top_k=5, context=base_context
        )
        results_low_threshold = low_threshold_strategy.retrieve(
            query_embedding, top_k=5, context=base_context
        )
        results_with_minimum = min_results_strategy.retrieve(
            query_embedding, top_k=5, context=base_context
        )

        # High threshold should filter more results than low threshold
        assert len(results_high_threshold) <= len(results_low_threshold), (
            "Higher threshold should return fewer or equal results"
        )

        # Min results should guarantee at least that number of results
        assert len(results_with_minimum) >= 2, (
            "Strategy with min_results should return at least that many results"
        )

    def test_results_combination_behavior(self, memory_store, query_embedding):
        """Test that results from different dimensions are properly combined."""
        # Create two strategy instances
        balanced_strategy = ContextualFabricStrategy(memory_store=memory_store)
        balanced_strategy.initialize(
            {
                "similarity_weight": 0.25,
                "associative_weight": 0.25,
                "temporal_weight": 0.25,
                "activation_weight": 0.25,
            }
        )

        similarity_biased_strategy = ContextualFabricStrategy(memory_store=memory_store)
        similarity_biased_strategy.initialize(
            {
                "similarity_weight": 0.7,
                "associative_weight": 0.1,
                "temporal_weight": 0.1,
                "activation_weight": 0.1,
            }
        )

        # Mock the result components for testing
        test_data = {
            "similarity_results": [
                {"memory_id": 0, "similarity_score": 0.9},
                {"memory_id": 1, "similarity_score": 0.8},
                {"memory_id": 2, "similarity_score": 0.7},
            ],
            "associative_results": {1: 0.8, 3: 0.6},
            "temporal_results": {2: 0.9, 4: 0.7},
            "activation_results": {0: 0.9, 2: 0.7},
        }

        # Use patch to inject our test data
        with (
            patch.object(
                balanced_strategy,
                "_retrieve_by_similarity",
                return_value=test_data["similarity_results"],
            ),
            patch.object(
                balanced_strategy,
                "_retrieve_associative_results",
                return_value=test_data["associative_results"],
            ),
            patch.object(
                balanced_strategy,
                "_retrieve_temporal_results",
                return_value=test_data["temporal_results"],
            ),
            patch.object(
                balanced_strategy,
                "_get_activation_results",
                return_value=test_data["activation_results"],
            ),
            patch.object(
                similarity_biased_strategy,
                "_retrieve_by_similarity",
                return_value=test_data["similarity_results"],
            ),
            patch.object(
                similarity_biased_strategy,
                "_retrieve_associative_results",
                return_value=test_data["associative_results"],
            ),
            patch.object(
                similarity_biased_strategy,
                "_retrieve_temporal_results",
                return_value=test_data["temporal_results"],
            ),
            patch.object(
                similarity_biased_strategy,
                "_get_activation_results",
                return_value=test_data["activation_results"],
            ),
        ):
            # Get results from both strategies
            balanced_results = balanced_strategy.retrieve(
                query_embedding, top_k=5, context={"query": "test"}
            )
            similarity_biased_results = similarity_biased_strategy.retrieve(
                query_embedding, top_k=5, context={"query": "test"}
            )

            # Ranks should differ between strategies
            balanced_ranks = {r["memory_id"]: i for i, r in enumerate(balanced_results)}
            biased_ranks = {r["memory_id"]: i for i, r in enumerate(similarity_biased_results)}

            # At least one memory should have different rank
            assert any(
                balanced_ranks.get(mem_id) != biased_ranks.get(mem_id)
                for mem_id in set(balanced_ranks.keys()) | set(biased_ranks.keys())
            ), "Different weighting strategies should produce different result rankings"

    def test_empty_query(self, memory_store):
        """Test behavior with empty query embedding."""
        strategy = ContextualFabricStrategy(memory_store=memory_store)
        empty_query = np.zeros(3)
        results = strategy.retrieve(empty_query, top_k=3, context={"query": ""})

        # Allow either empty results or low relevance scores
        if len(results) == 0:
            return
        for r in results:
            assert r["relevance_score"] < 0.2, "Relevance score should be low for empty queries"

    def test_memory_metadata_inclusion(self, memory_store, query_embedding, base_context):
        """Test that memory metadata is properly included in results."""
        # Set up memory store with metadata
        memory_store.memory_embeddings = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        memory_store.memory_metadata = [
            {"content": "Memory content 0", "custom_field": "value0"},
            {"content": "Memory content 1", "custom_field": "value1"},
            {"content": "Memory content 2", "custom_field": "value2"},
        ]

        strategy = ContextualFabricStrategy(memory_store=memory_store)
        strategy.initialize({"confidence_threshold": 0.0})

        # Retrieve memories
        results = strategy.retrieve(query_embedding, top_k=3, context=base_context)
        print(f"RESULTS ARE: {results}\n\n")
        # Check that metadata is included
        assert len(results) > 0
        for result in results:
            memory_id = result["memory_id"]
            assert "metadata" in result
            assert result["metadata"]["content"] == f"Memory content {memory_id}"
            assert result["metadata"]["custom_field"] == f"value{memory_id}"

        # Check that specific fields are correctly carried over
        content_values = [r["metadata"]["content"] for r in results]
        assert all(isinstance(content, str) for content in content_values)
        assert all("Memory content" in content for content in content_values)
