import pytest
import time
import os
import json
import gc
import psutil
import sys
from pathlib import Path

from memoryweave.evaluation.synthetic.benchmark import run_benchmark_with_config
from memoryweave.components import factory


@pytest.mark.integration
class TestMemoryUsage:
    """Test memory usage during benchmark runs."""
    
    @pytest.fixture
    def small_dataset(self):
        """Create a small test dataset for memory testing."""
        dataset_path = Path(__file__).parent.parent / "test_data" / "small_benchmark_dataset.json"
        
        # Create the directory if it doesn't exist
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if the test dataset already exists
        if not dataset_path.exists():
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
    def simple_config(self):
        """Return a simple configuration for testing."""
        return {
            "name": "Memory-Test-Basic",
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
    def complex_config(self):
        """Return a more complex configuration for testing."""
        return {
            "name": "Memory-Test-Complex",
            "components": {
                "retriever": {
                    "class": "Retriever",
                    "params": {
                        "retrieval_strategy": "TwoStageRetrievalStrategy",
                        "top_k": 5,
                        "confidence_threshold": 0.1,
                        "keyword_expansion_factor": 1.5
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
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        gc.collect()  # Force garbage collection
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    
    def test_memory_usage_simple_config(self, small_dataset, simple_config):
        """Test memory usage with a simple configuration."""
        # Skip this test if psutil is not available
        pytest.importorskip("psutil")
        
        # Record initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Run benchmark with simple config
        result = run_benchmark_with_config(
            dataset_path=small_dataset,
            config=simple_config,
            metrics=["precision", "recall"],
            verbose=False,
            max_queries=5  # Limit to 5 queries for faster testing
        )
        
        # Record final memory usage
        final_memory = self.get_memory_usage()
        
        # Calculate memory increase
        memory_increase = final_memory - initial_memory
        
        # Log memory usage
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Assert that memory increase is within reasonable limits
        # This threshold might need adjustment based on your specific implementation
        assert memory_increase < 70, f"Memory increase ({memory_increase:.2f} MB) exceeds threshold (70 MB)"
    
    def test_memory_usage_complex_config(self, small_dataset, complex_config):
        """Test memory usage with a more complex configuration."""
        # Skip this test if psutil is not available
        pytest.importorskip("psutil")
        
        # Record initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Run benchmark with complex config
        result = run_benchmark_with_config(
            dataset_path=small_dataset,
            config=complex_config,
            metrics=["precision", "recall", "f1_score"],
            verbose=False,
            max_queries=5  # Limit to 5 queries for faster testing
        )
        
        # Record final memory usage
        final_memory = self.get_memory_usage()
        
        # Calculate memory increase
        memory_increase = final_memory - initial_memory
        
        # Log memory usage
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Assert that memory increase is within reasonable limits
        # Complex config might use more memory, so threshold is higher
        assert memory_increase < 150, f"Memory increase ({memory_increase:.2f} MB) exceeds threshold (150 MB)"
    
    def test_memory_leak_multiple_runs(self, small_dataset, simple_config):
        """Test for memory leaks by running multiple benchmarks."""
        # Skip this test if psutil is not available
        pytest.importorskip("psutil")
        
        # Record initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Run benchmark multiple times
        memory_after_runs = []
        for i in range(3):  # Run 3 times
            result = run_benchmark_with_config(
                dataset_path=small_dataset,
                config=simple_config,
                metrics=["precision"],
                verbose=False,
                max_queries=3  # Use fewer queries for faster testing
            )
            
            # Force garbage collection
            gc.collect()
            
            # Record memory after this run
            memory_after_runs.append(self.get_memory_usage())
            
            # Short sleep to allow any async cleanup
            time.sleep(0.5)
        
        # Calculate memory differences between runs
        memory_differences = [memory_after_runs[i] - memory_after_runs[i-1] for i in range(1, len(memory_after_runs))]
        
        # Log memory usage
        print(f"Initial memory: {initial_memory:.2f} MB")
        for i, mem in enumerate(memory_after_runs):
            print(f"Memory after run {i+1}: {mem:.2f} MB")
        
        print(f"Memory differences between runs: {[f'{diff:.2f} MB' for diff in memory_differences]}")
        
        # Check if memory usage is stable or growing
        # If memory is properly cleaned up, later runs should not significantly increase memory usage
        for i, diff in enumerate(memory_differences):
            assert diff < 20, f"Memory growth between runs {i+1} and {i+2} ({diff:.2f} MB) suggests a memory leak"