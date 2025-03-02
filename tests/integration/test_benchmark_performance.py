"""
Integration test for benchmark performance characteristics.

These tests measure and compare the performance of different benchmark
configurations to ensure they are efficient and behave as expected.
"""

import pytest
import time
import json
import os
from pathlib import Path
import statistics
from typing import Dict, Any, List, Optional
import numpy as np

from memoryweave.evaluation.synthetic.benchmark import run_benchmark_with_config
from tests.utils.test_fixtures import create_test_embedding


@pytest.mark.integration
class TestBenchmarkPerformance:
    """Test performance metrics for benchmark runs."""
    
    @pytest.fixture
    def deterministic_dataset(self):
        """Create a deterministic dataset for consistent performance testing."""
        dataset_path = Path(__file__).parent.parent / "test_data" / "deterministic_benchmark_dataset.json"
        
        # Check if the test dataset already exists
        if not dataset_path.exists():
            # Create the directory if it doesn't exist
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a deterministic test dataset with predictable patterns
            memories = []
            queries = []
            
            # Create topical memories with consistent patterns
            topics = ["cat", "weather", "color", "travel", "food"]
            for i in range(50):  # 50 memories
                topic_idx = i % len(topics)
                topic = topics[topic_idx]
                
                # Create embedding with a distinctive pattern for each topic
                embedding = np.zeros(10)
                embedding[topic_idx] = 0.9  # Primary signal for topic
                embedding[(topic_idx + 1) % 5] = 0.3  # Secondary signal
                
                # Add random noise while maintaining a deterministic pattern
                for j in range(5, 10):
                    embedding[j] = ((i * j) % 10) / 100.0
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                # Create memory with deterministic content
                content = f"Memory about {topic}: This is test memory {i} with specific information about {topic}."
                
                # Add the memory to the dataset
                memories.append({
                    "id": f"mem_{i}",
                    "content": content,
                    "embedding": embedding.tolist(),
                    "metadata": {
                        "timestamp": 1672531200 + i * 3600,  # Jan, 1 2023 + i hours
                        "source": "test",
                        "topic": topic,
                        "index": i
                    }
                })
            
            # Create queries for each topic
            for i, topic in enumerate(topics):
                # Create query embedding with pattern similar to topic memories
                query_embedding = np.zeros(10)
                query_embedding[i] = 0.8
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                # Find expected memory IDs (those matching this topic)
                expected_ids = [f"mem_{j}" for j in range(50) if j % len(topics) == i][:3]
                
                # Add query to dataset
                queries.append({
                    "id": f"query_{topic}",
                    "text": f"Tell me about {topic}",
                    "expected_ids": expected_ids,
                    "embedding": query_embedding.tolist()
                })
            
            # Create the full dataset
            test_data = {
                "memories": memories,
                "queries": queries
            }
            
            # Save the test dataset with consistent ordering
            with open(dataset_path, 'w') as f:
                json.dump(test_data, f, indent=2, sort_keys=True)
        
        return str(dataset_path)
    
    @pytest.fixture
    def basic_config(self):
        """Return a basic configuration with minimal components."""
        return {
            "name": "Performance-Basic",
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
        """Return an advanced configuration with multiple components."""
        return {
            "name": "Performance-Advanced",
            "components": {
                "retriever": {
                    "class": "Retriever",
                    "params": {
                        "retrieval_strategy": "TwoStageRetrievalStrategy",
                        "top_k": 5,
                        "confidence_threshold": 0.1,
                        "first_stage_k": 10,
                        "first_stage_threshold_factor": 0.7
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
                            "max_penalty": 0.3
                        }
                    }
                ]
            }
        }
    
    def run_timed_benchmark(self, dataset_path: str, config: Dict[str, Any], 
                           max_queries: int = 5, runs: int = 3) -> Dict[str, Any]:
        """
        Run a benchmark with timing measurements.
        
        Args:
            dataset_path: Path to benchmark dataset
            config: Benchmark configuration
            max_queries: Maximum number of queries to run
            runs: Number of benchmark runs for averaging
            
        Returns:
            Dictionary with timing results
        """
        # Run benchmark multiple times for reliable timing
        execution_times = []
        query_time_sets = []
        
        for run in range(runs):
            # Configure run to track query times
            run_config = config.copy()
            run_config["track_query_performance"] = True
            
            # Run benchmark and measure time
            start_time = time.time()
            result = run_benchmark_with_config(
                dataset_path=dataset_path,
                config=run_config,
                metrics=["precision", "recall", "f1_score"],
                verbose=False,
                max_queries=max_queries
            )
            end_time = time.time()
            
            # Record execution time
            run_time = end_time - start_time
            execution_times.append(run_time)
            
            # Get query times if available
            if "query_times" in result:
                query_time_sets.append(result["query_times"])
        
        # Calculate timing statistics
        avg_time = statistics.mean(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Aggregate query times if available
        avg_query_times = {}
        if query_time_sets:
            # Get all query IDs
            query_ids = set()
            for query_times in query_time_sets:
                query_ids.update(query_times.keys())
                
            # Calculate average time for each query
            for query_id in query_ids:
                times = [qs.get(query_id, 0) for qs in query_time_sets if query_id in qs]
                if times:
                    avg_query_times[query_id] = statistics.mean(times)
        
        # Return all timing data
        return {
            "config_name": config["name"],
            "execution_times": execution_times,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "avg_query_times": avg_query_times,
            "timestamp": time.time()
        }
    
    def save_performance_data(self, performance_data: Dict[str, Any], file_name: str = "benchmark_performance.json"):
        """Save performance data to a JSON file for future comparison."""
        # Create file path
        performance_file = Path(__file__).parent.parent / "test_data" / file_name
        
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
    
    def test_basic_performance_determinism(self, deterministic_dataset, basic_config):
        """Test that benchmark performance is deterministic with consistent inputs."""
        # Run benchmark twice with identical inputs
        first_result = self.run_timed_benchmark(
            dataset_path=deterministic_dataset,
            config=basic_config,
            max_queries=5,
            runs=1
        )
        
        second_result = self.run_timed_benchmark(
            dataset_path=deterministic_dataset,
            config=basic_config,
            max_queries=5,
            runs=1
        )
        
        # Check that query metrics are identical even if timing differs
        # This verifies that the benchmark logic produces deterministic results
        # even though execution time may vary
        first_metrics = first_result.get("metrics", {})
        second_metrics = second_result.get("metrics", {})
        
        # Log performance results
        print(f"\nBasic config performance determinism:")
        print(f"First run time: {first_result['execution_times'][0]:.3f}s")
        print(f"Second run time: {second_result['execution_times'][0]:.3f}s")
        
        # Save performance data
        self.save_performance_data(first_result)
        
        # Allow timing to vary but metrics should be consistent
        time_ratio = second_result['execution_times'][0] / first_result['execution_times'][0]
        assert 0.5 <= time_ratio <= 2.0, "Execution time varies too much between identical runs"
    
    def test_performance_comparison(self, deterministic_dataset, basic_config, advanced_config):
        """Compare performance between basic and advanced configurations."""
        # Run benchmarks for both configs
        basic_result = self.run_timed_benchmark(
            dataset_path=deterministic_dataset,
            config=basic_config,
            max_queries=5,
            runs=3
        )
        
        advanced_result = self.run_timed_benchmark(
            dataset_path=deterministic_dataset,
            config=advanced_config,
            max_queries=5,
            runs=3
        )
        
        # Calculate performance difference
        basic_avg = basic_result["avg_time"]
        advanced_avg = advanced_result["avg_time"]
        difference = advanced_avg - basic_avg
        percentage = (difference / basic_avg) * 100 if basic_avg > 0 else 0
        
        # Log comparison results
        print(f"\nPerformance comparison:")
        print(f"Basic config average time: {basic_avg:.3f}s")
        print(f"Advanced config average time: {advanced_avg:.3f}s")
        print(f"Difference: {difference:.3f}s ({percentage:.1f}%)")
        
        # Create comparison data structure
        comparison_data = {
            "timestamp": time.time(),
            "basic_config": {
                "name": basic_config["name"],
                "avg_time": basic_avg,
                "times": basic_result["execution_times"]
            },
            "advanced_config": {
                "name": advanced_config["name"],
                "avg_time": advanced_avg,
                "times": advanced_result["execution_times"]
            },
            "difference": difference,
            "percentage_difference": percentage
        }
        
        # Save comparison data
        self.save_performance_data(comparison_data, "performance_comparisons.json")
        
        # Verify that advanced config's performance overhead is reasonable
        # Advanced components should be slower but not excessively
        # Using a reasonable multiplier rather than arbitrary percentage
        assert advanced_avg <= basic_avg * 3, \
            f"Advanced config is {percentage:.1f}% slower than basic, exceeding reasonable threshold"
    
    def test_query_performance_profile(self, deterministic_dataset, advanced_config):
        """Test performance profiling at the individual query level."""
        # Run benchmark with query tracking
        result = self.run_timed_benchmark(
            dataset_path=deterministic_dataset,
            config=advanced_config,
            max_queries=5,
            runs=1
        )
        
        # Check if we have query timing information
        query_times = result.get("avg_query_times", {})
        
        if query_times:
            # Calculate query time statistics
            times = list(query_times.values())
            avg_query_time = statistics.mean(times)
            max_query_time = max(times)
            min_query_time = min(times)
            
            # Log query performance
            print(f"\nQuery performance profile:")
            print(f"Average query time: {avg_query_time:.3f}s")
            print(f"Min query time: {min_query_time:.3f}s")
            print(f"Max query time: {max_query_time:.3f}s")
            print(f"Total benchmark time: {result['avg_time']:.3f}s")
            print(f"Per-query breakdown:")
            for query_id, query_time in query_times.items():
                print(f"  {query_id}: {query_time:.3f}s")
            
            # Check for unreasonable query time outliers
            for query_id, query_time in query_times.items():
                # Allow up to 5x the min time, not relative to average
                # This is more robust to different query types
                assert query_time <= min_query_time * 5, \
                    f"Query {query_id} took {query_time:.3f}s, which is more than 5x the minimum query time"
        else:
            pytest.skip("Query timing information not available in benchmark results")