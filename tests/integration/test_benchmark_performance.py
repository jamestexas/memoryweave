import pytest
import time
import json
import os
from pathlib import Path
import statistics

from memoryweave.evaluation.synthetic.benchmark import run_benchmark_with_config


@pytest.mark.integration
class TestBenchmarkPerformance:
    """Test performance metrics for benchmark runs."""
    
    @pytest.fixture
    def small_dataset(self):
        """Create a small test dataset for performance testing."""
        dataset_path = Path(__file__).parent.parent / "test_data" / "small_benchmark_dataset.json"
        
        # Check if the test dataset already exists
        if not dataset_path.exists():
            # Create the directory if it doesn't exist
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a small test dataset
            test_data = {
                "memories": [
                    {
                        "id": f"mem_{i}",
                        "content": f"This is test memory {i} about topic {i % 5}",
                        "embedding": [0.1] * 10,  # Small dummy embedding
                        "metadata": {"timestamp": time.time() - i * 3600, "source": "test"}
                    }
                    for i in range(50)  # 50 memories should be enough for testing
                ],
                "queries": [
                    {
                        "id": f"query_{i}",
                        "text": f"Tell me about topic {i % 5}",
                        "expected_ids": [f"mem_{j}" for j in range(50) if j % 5 == i % 5][:3],
                        "embedding": [0.2] * 10  # Small dummy embedding
                    }
                    for i in range(10)  # 10 queries is enough for testing
                ]
            }
            
            # Save the test dataset
            with open(dataset_path, 'w') as f:
                json.dump(test_data, f)
        
        return str(dataset_path)
    
    @pytest.fixture
    def basic_config(self):
        """Return a basic configuration for testing."""
        return {
            "name": "Performance-Test-Basic",
            "components": {
                "retriever": {
                    "class": "Retriever",
                    "params": {
                        "retrieval_strategy": "SimilarityRetrievalStrategy",
                        "top_k": 5,
                        "confidence_threshold": 0.1
                    }
                }
            }
        }
    
    @pytest.fixture
    def advanced_config(self):
        """Return an advanced configuration for testing."""
        return {
            "name": "Performance-Test-Advanced",
            "components": {
                "retriever": {
                    "class": "Retriever",
                    "params": {
                        "retrieval_strategy": "TwoStageRetrievalStrategy",
                        "top_k": 5,
                        "confidence_threshold": 0.1
                    }
                },
                "post_processors": [
                    {
                        "class": "KeywordBoostProcessor",
                        "params": {
                            "keyword_boost_weight": 0.5
                        }
                    },
                    {
                        "class": "SemanticCoherenceProcessor",
                        "params": {
                            "coherence_threshold": 0.2,
                            "max_penalty": 0.3,
                            "enable_query_type_filtering": True,
                            "enable_pairwise_coherence": True
                        }
                    }
                ]
            }
        }
    
    def test_basic_performance_tracking(self, small_dataset, basic_config):
        """Test basic performance tracking for benchmarks."""
        # Run benchmark multiple times and track execution time
        execution_times = []
        for _ in range(3):  # Run 3 times for reliable average
            start_time = time.time()
            result = run_benchmark_with_config(
                dataset_path=small_dataset,
                config=basic_config,
                metrics=["precision", "recall"],
                verbose=False,
                max_queries=5  # Limit for faster testing
            )
            end_time = time.time()
            execution_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = statistics.mean(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Log performance metrics
        print(f"Basic config performance metrics:")
        print(f"Average execution time: {avg_time:.3f}s")
        print(f"Min execution time: {min_time:.3f}s")
        print(f"Max execution time: {max_time:.3f}s")
        print(f"Standard deviation: {std_dev:.3f}s")
        
        # Store benchmark results for future comparison
        performance_data = {
            "config_name": basic_config["name"],
            "execution_times": execution_times,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "timestamp": time.time()
        }
        
        # Create performance tracking file
        performance_file = Path(__file__).parent.parent / "test_data" / "benchmark_performance.json"
        
        # Load existing data if file exists
        if performance_file.exists():
            with open(performance_file, 'r') as f:
                try:
                    all_performance_data = json.load(f)
                except json.JSONDecodeError:
                    all_performance_data = {"runs": []}
        else:
            # Create new data structure
            all_performance_data = {"runs": []}
        
        # Add current run
        all_performance_data["runs"].append(performance_data)
        
        # Save updated data
        performance_file.parent.mkdir(parents=True, exist_ok=True)
        with open(performance_file, 'w') as f:
            json.dump(all_performance_data, f, indent=2)
        
        # Performance assertions
        # Note: These thresholds are placeholders and should be adjusted based on actual performance
        assert avg_time < 5.0, f"Average execution time ({avg_time:.3f}s) exceeds threshold (5.0s)"
        assert std_dev < 1.0, f"Execution time variability ({std_dev:.3f}s) exceeds threshold (1.0s)"
    
    def test_performance_comparison(self, small_dataset, basic_config, advanced_config):
        """Compare performance between basic and advanced configurations."""
        # Run each configuration and track time
        config_times = {}
        
        for config_name, config in [("basic", basic_config), ("advanced", advanced_config)]:
            # Run multiple times for reliable average
            times = []
            for _ in range(3):
                start_time = time.time()
                result = run_benchmark_with_config(
                    dataset_path=small_dataset,
                    config=config,
                    metrics=["precision", "recall"],
                    verbose=False,
                    max_queries=5  # Limit for faster testing
                )
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Store average time
            config_times[config_name] = {
                "avg_time": statistics.mean(times),
                "times": times
            }
        
        # Log comparison results
        basic_avg = config_times["basic"]["avg_time"]
        advanced_avg = config_times["advanced"]["avg_time"]
        difference = advanced_avg - basic_avg
        percentage = (difference / basic_avg) * 100 if basic_avg > 0 else 0
        
        print(f"Performance comparison:")
        print(f"Basic config average time: {basic_avg:.3f}s")
        print(f"Advanced config average time: {advanced_avg:.3f}s")
        print(f"Difference: {difference:.3f}s ({percentage:.1f}%)")
        
        # Store comparison data
        comparison_data = {
            "timestamp": time.time(),
            "basic_config": {
                "name": basic_config["name"],
                "avg_time": basic_avg,
                "times": config_times["basic"]["times"]
            },
            "advanced_config": {
                "name": advanced_config["name"],
                "avg_time": advanced_avg,
                "times": config_times["advanced"]["times"]
            },
            "difference": difference,
            "percentage_difference": percentage
        }
        
        # Create comparison tracking file
        comparison_file = Path(__file__).parent.parent / "test_data" / "performance_comparisons.json"
        
        # Load existing data if file exists
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                try:
                    all_comparisons = json.load(f)
                except json.JSONDecodeError:
                    all_comparisons = {"comparisons": []}
        else:
            # Create new data structure
            all_comparisons = {"comparisons": []}
        
        # Add current comparison
        all_comparisons["comparisons"].append(comparison_data)
        
        # Save updated data
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_file, 'w') as f:
            json.dump(all_comparisons, f, indent=2)
        
        # Performance assertions
        # Check that the advanced config is not significantly slower than the basic config
        # This threshold can be adjusted based on your performance expectations
        assert percentage < 200, f"Advanced config is {percentage:.1f}% slower than basic config, exceeding 200% threshold"
    
    def test_query_performance_tracking(self, small_dataset, basic_config):
        """Track performance at the individual query level."""
        # Set up special parameter to enable per-query tracking
        config = basic_config.copy()
        config["track_query_performance"] = True
        
        # Run benchmark with query tracking
        start_time = time.time()
        result = run_benchmark_with_config(
            dataset_path=small_dataset,
            config=config,
            metrics=["precision", "recall"],
            verbose=False,
            max_queries=5  # Limit for faster testing
        )
        total_time = time.time() - start_time
        
        # Get per-query times if available in result
        # Note: This assumes the benchmark function has been modified to return timing info
        # If this isn't already implemented, you'll need to modify the benchmark code
        query_times = result.get("query_times", {})
        
        if query_times:
            # Calculate query time statistics
            times = list(query_times.values())
            avg_query_time = statistics.mean(times)
            max_query_time = max(times)
            min_query_time = min(times)
            
            # Log query performance
            print(f"Query performance metrics:")
            print(f"Average query time: {avg_query_time:.3f}s")
            print(f"Min query time: {min_query_time:.3f}s")
            print(f"Max query time: {max_query_time:.3f}s")
            print(f"Total benchmark time: {total_time:.3f}s")
            
            # Verify that the sum of query times is less than the total time
            # (there should be some overhead beyond just query processing)
            assert sum(times) <= total_time, "Sum of query times exceeds total benchmark time"
            
            # Check for outlier queries (significantly slower than average)
            for query_id, query_time in query_times.items():
                assert query_time < avg_query_time * 3, f"Query {query_id} took {query_time:.3f}s, which is more than 3x the average time"
        else:
            # Skip assertions if query timing isn't implemented
            pytest.skip("Query timing information not available in benchmark results")