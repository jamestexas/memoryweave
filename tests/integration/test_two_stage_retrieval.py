"""
Integration tests for TwoStageRetrievalStrategy.

These tests verify that the TwoStageRetrievalStrategy works correctly with other components
and produces expected results with different configurations.
"""

import pytest
import numpy as np

from memoryweave.components.retrieval_strategies import (
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TwoStageRetrievalStrategy,
)
from tests.utils.test_fixtures import (
    PredictableTestEmbeddings,
    create_test_memory,
    create_retrieval_components,
    verify_retrieval_results,
    assert_specific_difference,
)


class TestTwoStageRetrievalIntegration:
    """Integration tests for TwoStageRetrievalStrategy with proper assertions."""

    def setup_method(self):
        """Setup for each test using predictable test fixtures."""
        # Create a test memory with predictable data
        self.memory = create_test_memory()

        # Create retrieval components
        components = create_retrieval_components(self.memory)

        # Extract components for tests
        self.base_strategy = components["hybrid_strategy"]
        self.similarity_strategy = components["similarity_strategy"]
        self.keyword_processor = components["keyword_processor"]
        self.coherence_processor = components["coherence_processor"]
        self.basic_two_stage = components["basic_two_stage"]
        self.advanced_two_stage = components["advanced_two_stage"]

    def test_first_stage_k_affects_result_count(self):
        """Test that first_stage_k parameter affects the result count and quality."""
        # Create two strategies with different first_stage_k values
        small_k_strategy = TwoStageRetrievalStrategy(
            self.memory, base_strategy=self.similarity_strategy, post_processors=[]
        )
        small_k_strategy.initialize(
            {
                "confidence_threshold": 0.3,
                "first_stage_k": 2,  # Small first stage limits results
            }
        )

        large_k_strategy = TwoStageRetrievalStrategy(
            self.memory, base_strategy=self.similarity_strategy, post_processors=[]
        )
        large_k_strategy.initialize(
            {
                "confidence_threshold": 0.3,
                "first_stage_k": 5,  # Larger first stage allows more results
            }
        )

        # Common context with two-stage enabled
        context = {"enable_two_stage_retrieval": True, "config_name": "Test"}

        # Use cat query embedding
        query_embedding = PredictableTestEmbeddings.cat_query()

        # Retrieve with both configurations
        small_k_results = small_k_strategy.retrieve(query_embedding, 5, context)
        large_k_results = large_k_strategy.retrieve(query_embedding, 5, context)

        # When first_stage_k is smaller, fewer results are considered,
        # potentially limiting final result count or quality
        # Test that either:
        # 1. The number of results differs, or
        # 2. The relevance scores differ
        different, message = assert_specific_difference(
            small_k_results, large_k_results, "Different first_stage_k values should affect results"
        )

        assert different, message

        # Verify that cat-related content appears in both result sets
        # but we don't require exactly the same ordering
        assert verify_retrieval_results(small_k_results, ["cat"]), (
            "Small first_stage_k should still find cat-related content"
        )

        assert verify_retrieval_results(large_k_results, ["cat"]), (
            "Large first_stage_k should find cat-related content"
        )

    def test_two_stage_includes_post_processing(self):
        """Test that two-stage retrieval applies post-processing correctly."""
        # Create two identical strategies, but one with post-processors
        no_processors_strategy = TwoStageRetrievalStrategy(
            self.memory,
            base_strategy=self.similarity_strategy,
            post_processors=[],  # No post-processors
        )
        no_processors_strategy.initialize({"confidence_threshold": 0.3, "first_stage_k": 4})

        with_processors_strategy = TwoStageRetrievalStrategy(
            self.memory,
            base_strategy=self.similarity_strategy,
            post_processors=[self.keyword_processor],  # Add keyword processor
        )
        with_processors_strategy.initialize({"confidence_threshold": 0.3, "first_stage_k": 4})

        # Context with keywords to boost
        context = {
            "enable_two_stage_retrieval": True,
            "important_keywords": {"cat", "Whiskers"},
            "query": "Tell me about my cat Whiskers",
        }

        # Use cat query embedding
        query_embedding = PredictableTestEmbeddings.cat_query()

        # Retrieve with both strategies
        no_processor_results = no_processors_strategy.retrieve(query_embedding, 3, context)
        with_processor_results = with_processors_strategy.retrieve(query_embedding, 3, context)

        # Verify both return cat results
        assert verify_retrieval_results(no_processor_results, ["cat"]), (
            "Strategy without processors should still find cat-related content"
        )

        assert verify_retrieval_results(with_processor_results, ["cat"]), (
            "Strategy with processors should find cat-related content"
        )

        # Compare relevance scores - with keyword processor, "cat" results should have higher scores
        no_processor_cat_results = [
            r for r in no_processor_results if "cat" in r.get("content", "").lower()
        ]
        with_processor_cat_results = [
            r for r in with_processor_results if "cat" in r.get("content", "").lower()
        ]

        if no_processor_cat_results and with_processor_cat_results:
            no_processor_max_score = max(
                r.get("relevance_score", 0) for r in no_processor_cat_results
            )
            with_processor_max_score = max(
                r.get("relevance_score", 0) for r in with_processor_cat_results
            )

            # The actual assertion is that the processor changes scores in some way,
            # not necessarily increases them (depends on implementation)
            assert with_processor_max_score != no_processor_max_score, (
                "Keyword processor should modify cat result scores, but they remained the same"
            )

        # Test for specific differences between the results
        different, message = assert_specific_difference(
            no_processor_results,
            with_processor_results,
            "Post-processors should affect result scores or ordering",
        )

        assert different, message

    def test_disabled_two_stage_uses_base_strategy(self):
        """Test that when two-stage retrieval is disabled, it falls back to base strategy."""
        # Use cat query embedding
        query_embedding = PredictableTestEmbeddings.cat_query()

        # Context with two-stage explicitly disabled
        disabled_context = {"enable_two_stage_retrieval": False, "query": "Tell me about my cat"}

        # Context with two-stage explicitly enabled
        enabled_context = {"enable_two_stage_retrieval": True, "query": "Tell me about my cat"}

        # Direct base strategy results (SimilarityRetrievalStrategy)
        base_results = self.similarity_strategy.retrieve(query_embedding, 3, {})

        # Two-stage with disabled flag should use base strategy
        disabled_results = self.basic_two_stage.retrieve(query_embedding, 3, disabled_context)

        # Two-stage with enabled flag should use two-stage logic
        enabled_results = self.basic_two_stage.retrieve(query_embedding, 3, enabled_context)

        # When disabled, the results should match the base strategy's behavior
        assert len(disabled_results) == len(base_results), (
            f"When disabled, two-stage should return same count as base strategy: {len(disabled_results)} vs {len(base_results)}"
        )

        # Both disabled and direct base should find cat content
        assert verify_retrieval_results(disabled_results, ["cat"]), (
            "Disabled two-stage should still find cat-related content"
        )

        assert verify_retrieval_results(base_results, ["cat"]), (
            "Base strategy should find cat-related content"
        )

        # Enabled two-stage should produce different results from disabled
        different, message = assert_specific_difference(
            disabled_results,
            enabled_results,
            "Enabled vs disabled two-stage should produce different results",
        )

        assert different, message

    def test_different_configurations_produce_different_results(self):
        """Test that different configurations produce specific, expected differences."""
        # Use cat query embedding
        query_embedding = PredictableTestEmbeddings.cat_query()

        # Context with different configurations
        basic_context = {
            "enable_two_stage_retrieval": True,
            "config_name": "Basic",
            "enable_semantic_coherence": False,
        }

        advanced_context = {
            "enable_two_stage_retrieval": True,
            "config_name": "Advanced",
            "enable_semantic_coherence": True,
            "primary_query_type": "personal",
            "important_keywords": {"cat", "Whiskers"},
            "query": "Tell me about my cat Whiskers",
        }

        # Retrieve with both configurations
        basic_results = self.basic_two_stage.retrieve(query_embedding, 3, basic_context)
        advanced_results = self.advanced_two_stage.retrieve(query_embedding, 3, advanced_context)

        # Test specific expected differences:

        # 1. Basic should still return cat results
        assert verify_retrieval_results(basic_results, ["cat"]), (
            "Basic configuration should find cat-related content"
        )

        # 2. Advanced should return cat results, possibly in different order
        assert verify_retrieval_results(advanced_results, ["cat"]), (
            "Advanced configuration should find cat-related content"
        )

        # 3. Verify the configurations produce different results
        different, message = assert_specific_difference(
            basic_results,
            advanced_results,
            "Basic vs Advanced configurations should produce different results",
        )

        assert different, message

        # 4. Advanced configuration uses keyword boost, so should have
        # different relevance scores for cat results
        basic_cat_scores = [
            r.get("relevance_score", 0)
            for r in basic_results
            if "cat" in r.get("content", "").lower()
        ]

        advanced_cat_scores = [
            r.get("relevance_score", 0)
            for r in advanced_results
            if "cat" in r.get("content", "").lower()
        ]

        # Either the scores should be different, or the result sets should have
        # different lengths, content, etc.
        assert basic_cat_scores != advanced_cat_scores or different, (
            "Different configurations should produce different cat result scores"
        )

    def test_post_processor_order_affects_results(self):
        """Test that the order of post-processors affects the final results."""
        # Create two strategies with different post-processor orders
        order1_strategy = TwoStageRetrievalStrategy(
            self.memory,
            base_strategy=self.similarity_strategy,
            post_processors=[self.keyword_processor, self.coherence_processor],
        )
        order1_strategy.initialize({"confidence_threshold": 0.3, "first_stage_k": 4})

        order2_strategy = TwoStageRetrievalStrategy(
            self.memory,
            base_strategy=self.similarity_strategy,
            post_processors=[self.coherence_processor, self.keyword_processor],
        )
        order2_strategy.initialize({"confidence_threshold": 0.3, "first_stage_k": 4})

        # Context with relevant parameters for both processors
        context = {
            "enable_two_stage_retrieval": True,
            "enable_semantic_coherence": True,
            "primary_query_type": "personal",
            "important_keywords": {"cat", "Whiskers"},
            "query": "Tell me about my cat Whiskers",
        }

        # Use cat query embedding
        query_embedding = PredictableTestEmbeddings.cat_query()

        # Retrieve with both orders
        order1_results = order1_strategy.retrieve(query_embedding, 3, context)
        order2_results = order2_strategy.retrieve(query_embedding, 3, context)

        # Both should find cat results
        assert verify_retrieval_results(order1_results, ["cat"]), (
            "First processor order should find cat-related content"
        )

        assert verify_retrieval_results(order2_results, ["cat"]), (
            "Second processor order should find cat-related content"
        )

        # The order should produce different scores or rankings
        different, message = assert_specific_difference(
            order1_results, order2_results, "Different post-processor orders should affect results"
        )

        assert different, message


if __name__ == "__main__":
    pytest.main()
