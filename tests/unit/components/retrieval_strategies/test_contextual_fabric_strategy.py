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

    def test_retrieve_temporal_query(
        self, memory_store, query_embedding, base_context, temporal_context
    ):
        """Test retrieval with temporal queries."""
        strategy = ContextualFabricStrategy(
            memory_store=memory_store, temporal_context=temporal_context
        )
        strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "similarity_weight": 0.5,
                "temporal_weight": 0.5,
            }
        )

        # Set up temporal context to match memory with creation_time=300
        temporal_context.extract_time_references.return_value = {
            "has_temporal_reference": True,
            "time_type": "relative",
            "relative_time": 300,  # This should match memory 2's creation time
            "time_keywords": ["yesterday"],
        }

        # Create a mock for _retrieve_temporal_results that returns a specific memory
        with patch.object(strategy, "_retrieve_temporal_results") as mock_temporal:
            # Set up the mock to return a temporal match for memory 2
            mock_temporal.return_value = {2: 0.9}  # Memory 2 with high temporal relevance

            context = base_context.copy()
            context["query"] = "What happened yesterday?"

            results = strategy.retrieve(
                query_embedding, top_k=3, context=context, query=context["query"]
            )

            # Check that we have results
            assert len(results) > 0

            # Check that temporal context was used
            mock_temporal.assert_called_with("What happened yesterday?", context, memory_store)

            # Check if memory 2 is in the results with high temporal contribution
            memory_2_result = next((r for r in results if r["memory_id"] == 2), None)
            assert memory_2_result is not None
            assert memory_2_result["temporal_contribution"] > 0

    def test_retrieve_associative_query(
        self, memory_store, query_embedding, base_context, associative_linker
    ):
        """Test retrieval with associative network traversal."""
        strategy = ContextualFabricStrategy(
            memory_store=memory_store, associative_linker=associative_linker
        )
        strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "similarity_weight": 0.5,
                "associative_weight": 0.5,
                "max_associative_hops": 2,
            }
        )

        # Set up associative linker to return specific links
        associative_linker.traverse_associative_network.return_value = {
            1: 0.9,  # Strong link to memory 1
            3: 0.7,  # Moderate link to memory 3
        }

        results = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # Check that associative linker was used
        associative_linker.traverse_associative_network.assert_called()

        # Check that memory 1 is in the results with high associative contribution
        memory_1_result = next((r for r in results if r["memory_id"] == 1), None)
        assert memory_1_result is not None
        assert memory_1_result["associative_contribution"] > 0

    def test_retrieve_activation_boost(
        self, memory_store, query_embedding, base_context, activation_manager
    ):
        """Test retrieval with activation boosting."""
        # Set up memory store with embeddings
        memory_store.memory_embeddings = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.9, 0.8, 0.7],
                [0.6, 0.5, 0.4],
            ]
        )
        memory_store.memory_metadata = [
            {"content": "Memory content 0"},
            {"content": "Memory content 1"},
            {"content": "Memory content 2"},
            {"content": "Memory content 3"},
            {"content": "Memory content 4"},
        ]

        strategy = ContextualFabricStrategy(
            memory_store=memory_store, activation_manager=activation_manager
        )
        strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "similarity_weight": 0.5,
                "activation_weight": 0.5,
                "activation_boost_factor": 2.0,
            }
        )

        # Set up activation manager to return specific activations
        activation_manager.get_activated_memories.return_value = [
            (0, 0.9),  # High activation for memory 0
            (2, 0.7),  # Moderate activation for memory 2
        ]

        # Create a context with additional info
        context = base_context.copy()
        context["query"] = "Test query about activation"

        # Retrieve memories
        results = strategy.retrieve(query_embedding, top_k=3, context=context)

        # Check that activation manager was used
        activation_manager.get_activated_memories.assert_called()

        # Results should include activation contribution
        assert len(results) > 0

        # Check specific memories with activation contributions
        for result in results:
            if result["memory_id"] in [0, 2]:
                assert result["activation_contribution"] > 0

    def test_retrieve_with_parameter_adaptation(self, memory_store, query_embedding, base_context):
        """Test retrieval with parameter adaptation via context."""
        strategy = ContextualFabricStrategy(memory_store=memory_store)
        strategy.initialize(
            {
                "confidence_threshold": 0.1,
                "similarity_weight": 0.5,
                "associative_weight": 0.3,
                "temporal_weight": 0.1,
                "activation_weight": 0.1,
            }
        )

        # Create a context with adapted parameters
        context = base_context.copy()
        context["adapted_retrieval_params"] = {
            "confidence_threshold": 0.05,
            "similarity_weight": 0.6,
            "associative_weight": 0.2,
            "temporal_weight": 0.1,
            "activation_weight": 0.1,
        }

        # Mock the _apply_adapted_params method to check it's called correctly
        with patch.object(strategy, "_apply_adapted_params") as mock_apply:
            strategy.retrieve(query_embedding, top_k=3, context=context)

            # Check that adapted parameters were applied
            mock_apply.assert_called_with(context["adapted_retrieval_params"])

    def test_empty_query(self, memory_store):
        """Test behavior with empty query embedding."""
        strategy = ContextualFabricStrategy(memory_store=memory_store)

        # Create empty query embedding
        empty_query = np.zeros(3)

        # Should not raise error but return empty results
        results = strategy.retrieve(empty_query, top_k=3, context={"query": ""})

        # Results should be empty or at least have very low scores
        for r in results:
            assert r["relevance_score"] < 0.1  # Very low scores expected

    def test_confidence_threshold_filtering(self, memory_store, query_embedding, base_context):
        """Test that confidence threshold filtering works properly."""
        strategy = ContextualFabricStrategy(memory_store=memory_store)

        # Set a high threshold that should filter out most results
        strategy.initialize({"confidence_threshold": 0.9})

        # First without min_results to check filtering
        strategy.min_results = 0  # Set to 0 for this test
        results_filtered = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # Then with min_results to check minimum guarantee
        strategy.min_results = 2  # Set to 2 for this test
        results_with_min = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # Finally with lower threshold for comparison
        strategy.initialize({"confidence_threshold": 0.0})
        results_unfiltered = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # High threshold should filter more results
        assert len(results_filtered) <= len(results_unfiltered)

        # Min results should guarantee at least that many results
        assert len(results_with_min) >= 2  # Use literal number instead of strategy.min_results

    def test_combine_results(self, memory_store, query_embedding):
        """Test the _combine_results method directly."""
        strategy = ContextualFabricStrategy(memory_store=memory_store)
        strategy.initialize(
            {
                "similarity_weight": 0.5,
                "associative_weight": 0.2,
                "temporal_weight": 0.1,
                "activation_weight": 0.2,
            }
        )

        # Create test inputs
        similarity_results = [
            {"memory_id": 0, "similarity_score": 0.9},
            {"memory_id": 1, "similarity_score": 0.8},
            {"memory_id": 2, "similarity_score": 0.7},
        ]

        associative_results = {1: 0.8, 3: 0.6}
        temporal_results = {2: 0.9, 4: 0.7}
        activation_results = {0: 0.9, 2: 0.7}

        # Call the method directly
        combined = strategy._combine_results(
            similarity_results,
            associative_results,
            temporal_results,
            activation_results,
            memory_store,
        )

        # Check that results were combined properly
        assert len(combined) >= 5  # Should include all unique memory IDs

        # Check that all memory IDs are present
        memory_ids = {r["memory_id"] for r in combined}
        assert memory_ids.issuperset({0, 1, 2, 3, 4})

        # Check that memory 0 has both similarity and activation contributions
        memory_0 = next(r for r in combined if r["memory_id"] == 0)
        assert memory_0["similarity_contribution"] > 0
        assert memory_0["activation_contribution"] > 0

        # Check that memory 1 has both similarity and associative contributions
        memory_1 = next(r for r in combined if r["memory_id"] == 1)
        assert memory_1["similarity_contribution"] > 0
        assert memory_1["associative_contribution"] > 0

        # Check that memory 2 has similarity, temporal, and activation contributions
        memory_2 = next(r for r in combined if r["memory_id"] == 2)
        assert memory_2["similarity_contribution"] > 0
        assert memory_2["temporal_contribution"] > 0
        assert memory_2["activation_contribution"] > 0

    def test_retrieve_temporal_results(self, memory_store, temporal_context, base_context):
        """Test the _retrieve_temporal_results method directly."""
        strategy = ContextualFabricStrategy(
            memory_store=memory_store, temporal_context=temporal_context
        )

        # Set up temporal context to return specific time info
        temporal_context.extract_time_references.return_value = {
            "has_temporal_reference": True,
            "time_type": "relative",
            "relative_time": 300,  # This matches memory 2's creation time
            "time_keywords": ["yesterday"],
        }

        # Create a context with current_time
        context = base_context.copy()

        # Call the method directly
        temporal_results = strategy._retrieve_temporal_results(
            "What happened yesterday?", context, memory_store
        )

        # Should return a dictionary of memory_id -> score
        assert isinstance(temporal_results, dict)

        # Memory 2 should have a high temporal relevance
        assert 2 in temporal_results
        assert temporal_results[2] > 0.5
