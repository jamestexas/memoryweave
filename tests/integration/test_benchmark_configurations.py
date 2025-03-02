"""
Integration test for verifying that benchmark configurations produce different results.

This test creates a simplified version of the benchmark to validate
that different configurations lead to measurably different metrics.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List

from memoryweave.components.retriever import Retriever
from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.evaluation.synthetic.benchmark import BenchmarkConfig


@dataclass
class SimpleBenchmarkResult:
    """Simple container for benchmark results."""
    precision: float
    recall: float
    f1_score: float
    avg_retrieval_count: float
    config_name: str


class TestBenchmarkConfigurations:
    """Test that benchmark configurations produce distinct results."""

    def setup_method(self):
        """Setup test data and configurations."""
        # Create a small test memory
        self.memory = ContextualMemory(embedding_dim=4)
        
        # Create some test data with controlled relevance
        self._create_test_data()
        
        # Define configurations to test
        self.configs = [
            BenchmarkConfig(
                name="Basic",
                retriever_type="basic",
                confidence_threshold=0.3,
                top_k=5,
                semantic_coherence_check=False,
                use_two_stage_retrieval=False,
                query_type_adaptation=False,
                evaluation_mode=True,
            ),
            BenchmarkConfig(
                name="Semantic-Coherence",
                retriever_type="components",
                confidence_threshold=0.3,
                top_k=5,
                semantic_coherence_check=True,
                use_two_stage_retrieval=False,
                query_type_adaptation=False,
                evaluation_mode=True,
            ),
            BenchmarkConfig(
                name="Query-Adaptation",
                retriever_type="components",
                confidence_threshold=0.3,
                top_k=5,
                semantic_coherence_check=False,
                use_two_stage_retrieval=False,
                query_type_adaptation=True,
                evaluation_mode=True,
            ),
            BenchmarkConfig(
                name="Two-Stage",
                retriever_type="components",
                confidence_threshold=0.3,
                top_k=5,
                semantic_coherence_check=False,
                use_two_stage_retrieval=True,
                query_type_adaptation=False,
                evaluation_mode=True,
            ),
            BenchmarkConfig(
                name="Full-Advanced",
                retriever_type="components",
                confidence_threshold=0.3,
                top_k=5,
                semantic_coherence_check=True,
                use_two_stage_retrieval=True,
                query_type_adaptation=True,
                evaluation_mode=True,
            ),
        ]
        
        # Test queries with known relevant indices
        self.test_queries = [
            {
                "query": "Tell me about my cat",
                "query_embedding": np.array([0.9, 0.1, 0.1, 0.1]),
                "relevant_indices": [0, 1]  # Indices of cat-related memories
            },
            {
                "query": "What's the weather like?",
                "query_embedding": np.array([0.1, 0.1, 0.9, 0.1]),
                "relevant_indices": [4, 5]  # Indices of weather-related memories
            },
            {
                "query": "Tell me about my travels",
                "query_embedding": np.array([0.1, 0.1, 0.1, 0.9]),
                "relevant_indices": [8, 9]  # Indices of travel-related memories
            },
        ]

    def _create_test_data(self):
        """Create test data with controlled relevance."""
        # Create embeddings for different categories
        cat_embedding = np.array([0.9, 0.1, 0.1, 0.1])
        dog_embedding = np.array([0.1, 0.9, 0.1, 0.1])
        weather_embedding = np.array([0.1, 0.1, 0.9, 0.1])
        travel_embedding = np.array([0.1, 0.1, 0.1, 0.9])
        
        # Cat-related memories (for personal queries)
        self.memory.add_memory(
            cat_embedding,
            "My cat Whiskers loves to sleep on the couch.",
            {"type": "personal", "category": "pets", "index": 0}
        )
        self.memory.add_memory(
            cat_embedding * 0.95,
            "I feed my cat twice a day with premium food.",
            {"type": "personal", "category": "pets", "index": 1}
        )
        
        # Dog-related memories (should not be relevant for cat queries)
        self.memory.add_memory(
            dog_embedding,
            "My dog Rover likes to fetch balls.",
            {"type": "personal", "category": "pets", "index": 2}
        )
        self.memory.add_memory(
            dog_embedding * 0.95,
            "Dogs need regular walks and exercise.",
            {"type": "factual", "category": "pets", "index": 3}
        )
        
        # Weather-related memories (for factual queries)
        self.memory.add_memory(
            weather_embedding,
            "It was raining heavily in Seattle yesterday.",
            {"type": "factual", "category": "weather", "index": 4}
        )
        self.memory.add_memory(
            weather_embedding * 0.95,
            "The forecast predicts sunny weather tomorrow.",
            {"type": "factual", "category": "weather", "index": 5}
        )
        
        # Random memories (noise)
        self.memory.add_memory(
            np.array([0.5, 0.5, 0.1, 0.1]),
            "The library has many interesting books.",
            {"type": "factual", "category": "general", "index": 6}
        )
        self.memory.add_memory(
            np.array([0.4, 0.4, 0.4, 0.1]),
            "Coffee tastes best when freshly brewed.",
            {"type": "general", "category": "food", "index": 7}
        )
        
        # Travel-related memories
        self.memory.add_memory(
            travel_embedding,
            "I visited Paris last summer and saw the Eiffel Tower.",
            {"type": "personal", "category": "travel", "index": 8}
        )
        self.memory.add_memory(
            travel_embedding * 0.95,
            "Rome has amazing historical architecture.",
            {"type": "factual", "category": "travel", "index": 9}
        )

    def _run_mini_benchmark(self, config: BenchmarkConfig) -> SimpleBenchmarkResult:
        """Run a mini-benchmark with the given configuration."""
        # Create retriever
        retriever = Retriever(memory=self.memory)
        
        # Configure retriever according to configuration
        retriever.minimum_relevance = config.confidence_threshold
        retriever.top_k = config.top_k
        
        # Initialize components
        retriever.initialize_components()
        
        # Configure features
        if config.semantic_coherence_check:
            retriever.configure_semantic_coherence(enable=True)
            
        if config.query_type_adaptation:
            retriever.configure_query_type_adaptation(enable=True)
            
        if config.use_two_stage_retrieval:
            retriever.configure_two_stage_retrieval(enable=True)
            
        # Set config name
        retriever.memory_manager.config_name = config.name
            
        # Run queries
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_retrieval_counts = []
        
        for query_item in self.test_queries:
            query = query_item["query"]
            query_embedding = query_item["query_embedding"]
            relevant_indices = set(query_item["relevant_indices"])
            
            # Set query embedding directly in memory manager context
            retriever.memory_manager.working_context = {
                "query_embedding": query_embedding,
                "in_evaluation": config.evaluation_mode,
                "enable_query_type_adaptation": config.query_type_adaptation,
                "enable_semantic_coherence": config.semantic_coherence_check,
                "enable_two_stage_retrieval": config.use_two_stage_retrieval,
                "config_name": config.name,
                "query": query,
                "primary_query_type": "personal" if "my" in query.lower() else "factual"
            }
            
            # For personal queries, add important keywords
            if "my" in query.lower():
                if "cat" in query.lower():
                    retriever.memory_manager.working_context["important_keywords"] = {"cat", "Whiskers"}
                elif "travel" in query.lower():
                    retriever.memory_manager.working_context["important_keywords"] = {"travel", "visited", "Paris"}
            
            # Retrieve memories with the specified configuration
            results = retriever.retrieve(
                query,
                top_k=config.top_k,
                minimum_relevance=config.confidence_threshold
            )
            
            # Extract retrieved indices
            retrieved_indices = set()
            for r in results:
                # Get index from metadata
                if "index" in r:
                    retrieved_indices.add(r["index"])
                elif "memory_id" in r:
                    # Try to get from memory metadata
                    memory_id = r["memory_id"]
                    if isinstance(memory_id, int) and memory_id < len(self.memory.memory_metadata):
                        metadata = self.memory.memory_metadata[memory_id]
                        if "index" in metadata:
                            retrieved_indices.add(metadata["index"])
            
            # Calculate metrics
            if retrieved_indices:
                precision = len(relevant_indices.intersection(retrieved_indices)) / len(retrieved_indices)
            else:
                precision = 0.0
                
            if relevant_indices:
                recall = len(relevant_indices.intersection(retrieved_indices)) / len(relevant_indices)
            else:
                recall = 1.0
                
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
                
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
            all_retrieval_counts.append(len(results))
        
        # Calculate average metrics
        return SimpleBenchmarkResult(
            precision=np.mean(all_precisions),
            recall=np.mean(all_recalls),
            f1_score=np.mean(all_f1_scores),
            avg_retrieval_count=np.mean(all_retrieval_counts),
            config_name=config.name
        )

    def test_configurations_produce_different_results(self):
        """Test that different configurations produce different benchmark results."""
        # Run mini-benchmark for each configuration
        results = {}
        for config in self.configs:
            results[config.name] = self._run_mini_benchmark(config)
            
        # Verify that at least some configurations have different results
        unique_precisions = set(round(r.precision, 4) for r in results.values())
        unique_recalls = set(round(r.recall, 4) for r in results.values())
        unique_f1_scores = set(round(r.f1_score, 4) for r in results.values())
        unique_retrieval_counts = set(r.avg_retrieval_count for r in results.values())
        
        # Print out results for debugging
        print("\nMini-Benchmark Results:")
        for config_name, result in results.items():
            print(f"{config_name}: P={result.precision:.4f}, R={result.recall:.4f}, F1={result.f1_score:.4f}, Count={result.avg_retrieval_count:.1f}")
        
        # Some metrics should be different across configurations
        assert len(unique_f1_scores) > 1, "All configurations produced identical F1 scores"
        assert len(unique_precisions) > 1 or len(unique_recalls) > 1, "All configurations produced identical precision and recall"
        
        # Specific comparisons
        basic_result = results["Basic"]
        semantic_result = results["Semantic-Coherence"]
        adaptation_result = results["Query-Adaptation"]
        two_stage_result = results["Two-Stage"]
        advanced_result = results["Full-Advanced"]
        
        # Check that at least some configurations differ significantly in their results
        sig_diffs = 0
        
        # Check precision differences
        if abs(basic_result.precision - semantic_result.precision) > 0.01:
            sig_diffs += 1
        if abs(basic_result.precision - adaptation_result.precision) > 0.01:
            sig_diffs += 1
        if abs(basic_result.precision - two_stage_result.precision) > 0.01:
            sig_diffs += 1
        if abs(basic_result.precision - advanced_result.precision) > 0.01:
            sig_diffs += 1
            
        # Check recall differences
        if abs(basic_result.recall - semantic_result.recall) > 0.01:
            sig_diffs += 1
        if abs(basic_result.recall - adaptation_result.recall) > 0.01:
            sig_diffs += 1
        if abs(basic_result.recall - two_stage_result.recall) > 0.01:
            sig_diffs += 1
        if abs(basic_result.recall - advanced_result.recall) > 0.01:
            sig_diffs += 1
            
        # Ensure there are at least some significant differences
        assert sig_diffs >= 2, "Not enough significant differences between configurations"


if __name__ == "__main__":
    pytest.main()