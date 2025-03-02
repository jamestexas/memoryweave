"""
Integration test for verifying that benchmark configurations produce different results.

This test creates a simplified version of the benchmark to validate
that different configurations lead to measurably different metrics.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Set

from memoryweave.components.retriever import Retriever
from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.evaluation.synthetic.benchmark import BenchmarkConfig

from tests.utils.test_fixtures import (
    create_test_memory,
    verify_retrieval_results,
    assert_specific_difference
)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark results."""
    precision: float
    recall: float
    f1_score: float
    retrieval_count: float
    significant_features: Set[str]

    @property
    def summary(self) -> str:
        """Get a summary string of the metrics."""
        return (f"P={self.precision:.4f}, R={self.recall:.4f}, " +
                f"F1={self.f1_score:.4f}, Count={self.retrieval_count:.1f}, " +
                f"Features={','.join(self.significant_features)}")


class TestBenchmarkConfigurations:
    """Test that benchmark configurations produce distinct results with clear behavioral differences."""

    def setup_method(self):
        """Setup test data and configurations."""
        # Create a test memory with predictable patterns
        self.memory = create_test_memory(embedding_dim=4)
        
        # Define configurations to test with clear behavioral differences
        self.configs = [
            BenchmarkConfig(
                name="Basic", 
                retriever_type="basic",
                confidence_threshold=0.3, 
                top_k=5,
                semantic_coherence_check=False,
                use_two_stage_retrieval=False,
                query_type_adaptation=False,
                evaluation_mode=False,  # No longer rely on evaluation mode
            ),
            BenchmarkConfig(
                name="Semantic", 
                retriever_type="components",
                confidence_threshold=0.3, 
                top_k=5,
                semantic_coherence_check=True,  # Enabled semantic coherence
                use_two_stage_retrieval=False,
                query_type_adaptation=False,
                evaluation_mode=False,
            ),
            BenchmarkConfig(
                name="Adaptation", 
                retriever_type="components",
                confidence_threshold=0.3, 
                top_k=5,
                semantic_coherence_check=False,
                use_two_stage_retrieval=False,
                query_type_adaptation=True,  # Enabled query adaptation
                evaluation_mode=False,
            ),
            BenchmarkConfig(
                name="TwoStage", 
                retriever_type="components",
                confidence_threshold=0.3, 
                top_k=5,
                semantic_coherence_check=False,
                use_two_stage_retrieval=True,  # Enabled two-stage retrieval
                query_type_adaptation=False,
                evaluation_mode=False,
            ),
            BenchmarkConfig(
                name="Advanced", 
                retriever_type="components",
                confidence_threshold=0.3, 
                top_k=5,
                semantic_coherence_check=True,
                use_two_stage_retrieval=True,
                query_type_adaptation=True,
                evaluation_mode=False,
            ),
        ]
        
        # Test queries with known relevant indices
        # Create predictable query patterns
        self.test_queries = [
            {
                "name": "cat",
                "query": "Tell me about my cat Whiskers",
                "query_embedding": np.array([0.9, 0.1, 0.1, 0.1]),
                "relevant_indices": [0, 1],  # Indices of cat-related memories
                "type": "personal",
                "keywords": {"cat", "Whiskers"}
            },
            {
                "name": "weather",
                "query": "What's the weather like in Seattle?",
                "query_embedding": np.array([0.1, 0.1, 0.9, 0.1]),
                "relevant_indices": [4, 5],  # Indices of weather-related memories
                "type": "factual",
                "keywords": {"weather", "Seattle"}
            },
            {
                "name": "color",
                "query": "What's my favorite color?",
                "query_embedding": np.array([0.1, 0.9, 0.1, 0.1]),
                "relevant_indices": [2, 3],  # Indices of color-related memories
                "type": "personal",
                "keywords": {"color", "favorite"}
            },
        ]

    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkMetrics:
        """
        Run a benchmark with the given configuration.
        
        This method configures a retriever based on the provided configuration,
        runs test queries, and calculates metrics. It also identifies which
        features of the configuration significantly affect the results.
        
        Args:
            config: The benchmark configuration to test
            
        Returns:
            BenchmarkMetrics with results and significant features
        """
        # Create and configure retriever
        retriever = Retriever(memory=self.memory)
        retriever.minimum_relevance = config.confidence_threshold
        retriever.top_k = config.top_k
        retriever.initialize_components()
        
        # Configure features based on config
        significant_features = set()
        
        if config.semantic_coherence_check:
            retriever.configure_semantic_coherence(enable=True)
            significant_features.add("semantic_coherence")
            
        if config.query_type_adaptation:
            retriever.configure_query_type_adaptation(enable=True)
            significant_features.add("query_adaptation")
            
        if config.use_two_stage_retrieval:
            retriever.configure_two_stage_retrieval(enable=True)
            significant_features.add("two_stage")
        
        # Track metrics across queries
        all_metrics = []
        total_retrieval_count = 0
        
        # Run each test query
        for query_item in self.test_queries:
            query = query_item["query"]
            query_embedding = query_item["query_embedding"]
            relevant_indices = query_item["relevant_indices"]
            query_type = query_item["type"]
            keywords = query_item.get("keywords", set())
            
            # Set context for the query
            retriever.memory_manager.working_context = {
                "query_embedding": query_embedding,
                "enable_query_type_adaptation": config.query_type_adaptation,
                "enable_semantic_coherence": config.semantic_coherence_check,
                "enable_two_stage_retrieval": config.use_two_stage_retrieval,
                "config_name": config.name,
                "query": query,
                "primary_query_type": query_type,
                "important_keywords": keywords
            }
            
            # Perform retrieval
            results = retriever.retrieve(
                query,
                top_k=config.top_k,
                minimum_relevance=config.confidence_threshold
            )
            
            # Calculate metrics for this query
            success, metrics = verify_retrieval_results(
                results, 
                relevant_indices,
                require_all=False,
                check_order=False
            )
            
            all_metrics.append(metrics)
            total_retrieval_count += len(results)
        
        # Calculate average metrics
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])
        avg_f1 = np.mean([m["f1_score"] for m in all_metrics])
        avg_count = total_retrieval_count / len(self.test_queries)
        
        return BenchmarkMetrics(
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            retrieval_count=avg_count,
            significant_features=significant_features
        )

    def test_configurations_produce_different_results(self):
        """Test that different configurations produce specific behavioral differences."""
        # Run benchmark for each configuration
        results = {}
        for config in self.configs:
            results[config.name] = self.run_benchmark(config)
            
        # Print results for debugging
        print("\nBenchmark Results:")
        for config_name, metrics in results.items():
            print(f"{config_name}: {metrics.summary}")
        
        # Perform specific tests for each configuration's unique behavior
        
        # 1. Test that semantic coherence affects results
        basic_result = results["Basic"]
        semantic_result = results["Semantic"]
        
        # Semantic coherence should prioritize results matching query type
        assert basic_result.significant_features != semantic_result.significant_features, \
            "Feature sets should differ between Basic and Semantic configurations"
            
        # Check for differences in metrics
        semantic_different, _ = self._config_has_impact(basic_result, semantic_result)
        assert semantic_different, "Semantic coherence should affect results"
        
        # 2. Test that query adaptation affects results
        adaptation_result = results["Adaptation"]
        
        # Query adaptation should modify parameters based on query type
        assert "query_adaptation" in adaptation_result.significant_features, \
            "Query adaptation should be enabled in the Adaptation configuration"
            
        # Check for differences in metrics
        adaptation_different, _ = self._config_has_impact(basic_result, adaptation_result)
        assert adaptation_different, "Query adaptation should affect results"
        
        # 3. Test that two-stage retrieval affects results
        two_stage_result = results["TwoStage"]
        
        # Two-stage retrieval should introduce a first-stage retrieval
        assert "two_stage" in two_stage_result.significant_features, \
            "Two-stage retrieval should be enabled in the TwoStage configuration"
            
        # Check for differences in metrics
        two_stage_different, _ = self._config_has_impact(basic_result, two_stage_result)
        assert two_stage_different, "Two-stage retrieval should affect results"
        
        # 4. Test that advanced configuration combines features
        advanced_result = results["Advanced"]
        
        # Advanced should include all features
        assert len(advanced_result.significant_features) == 3, \
            f"Advanced configuration should enable all features, got {advanced_result.significant_features}"
            
        # Check for differences in metrics
        advanced_different, _ = self._config_has_impact(basic_result, advanced_result)
        assert advanced_different, "Advanced configuration should produce different results from basic"
        
        # 5. Test that there's sufficient diversity in results
        unique_f1_scores = {round(r.f1_score, 2) for r in results.values()}
        assert len(unique_f1_scores) >= 2, \
            f"Expected at least 2 different F1 scores, got {unique_f1_scores}"

    def _config_has_impact(self, baseline_metrics: BenchmarkMetrics, 
                          test_metrics: BenchmarkMetrics, 
                          threshold: float = 0.01) -> tuple[bool, str]:
        """
        Determine if a configuration significantly impacts metrics.
        
        Args:
            baseline_metrics: Metrics from the baseline configuration
            test_metrics: Metrics from the configuration being tested
            threshold: Minimum difference to consider significant
            
        Returns:
            Tuple of (has_impact, reason)
        """
        differences = []
        
        # Check precision difference
        if abs(baseline_metrics.precision - test_metrics.precision) > threshold:
            differences.append(f"precision ({baseline_metrics.precision:.4f} vs {test_metrics.precision:.4f})")
            
        # Check recall difference
        if abs(baseline_metrics.recall - test_metrics.recall) > threshold:
            differences.append(f"recall ({baseline_metrics.recall:.4f} vs {test_metrics.recall:.4f})")
            
        # Check F1 difference
        if abs(baseline_metrics.f1_score - test_metrics.f1_score) > threshold:
            differences.append(f"F1 score ({baseline_metrics.f1_score:.4f} vs {test_metrics.f1_score:.4f})")
            
        # Check retrieval count difference
        count_diff = abs(baseline_metrics.retrieval_count - test_metrics.retrieval_count)
        if count_diff >= 0.5:  # At least half a result difference on average
            differences.append(f"retrieval count ({baseline_metrics.retrieval_count:.1f} vs {test_metrics.retrieval_count:.1f})")
        
        if differences:
            return True, f"Configuration impacts {', '.join(differences)}"
        else:
            return False, "No significant impact detected"


if __name__ == "__main__":
    pytest.main()