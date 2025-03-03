"""
Integration tests for baseline comparison.

These tests validate that the baseline comparison framework
works correctly with real retrievers and memory managers.
"""

import os
import tempfile
import json
import numpy as np
import pytest
from typing import List, Dict, Any

from memoryweave.baselines import BM25Retriever, VectorBaselineRetriever
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.retrieval_strategies import SimilarityRetrievalStrategy
from memoryweave.evaluation.baseline_comparison import (
    BaselineComparison, BaselineConfig, ComparisonResult
)
from memoryweave.interfaces.retrieval import Query
from memoryweave.storage.memory_store import Memory, MemoryStore


@pytest.fixture
def test_memories() -> List[Memory]:
    """Create a set of test memories with embeddings."""
    memories = [
        Memory(
            id="mem1",
            embedding=np.random.rand(128),
            content={"text": "Python is a high-level programming language known for its readability.", "metadata": {}},
            metadata={"category": "fact", "topic": "programming"}
        ),
        Memory(
            id="mem2",
            embedding=np.random.rand(128),
            content={"text": "JavaScript is a programming language commonly used for web development.", "metadata": {}},
            metadata={"category": "fact", "topic": "programming"}
        ),
        Memory(
            id="mem3",
            embedding=np.random.rand(128),
            content={"text": "Paris is the capital city of France and known for the Eiffel Tower.", "metadata": {}},
            metadata={"category": "fact", "topic": "geography"}
        ),
        Memory(
            id="mem4",
            embedding=np.random.rand(128),
            content={"text": "I prefer to eat pizza on Friday nights while watching movies.", "metadata": {}},
            metadata={"category": "preference", "topic": "food"}
        ),
        Memory(
            id="mem5",
            embedding=np.random.rand(128),
            content={"text": "My favorite programming language is Python because of its simplicity.", "metadata": {}},
            metadata={"category": "preference", "topic": "programming"}
        ),
        Memory(
            id="mem6",
            embedding=np.random.rand(128),
            content={"text": "The Great Wall of China is one of the seven wonders of the world.", "metadata": {}},
            metadata={"category": "fact", "topic": "geography"}
        ),
        Memory(
            id="mem7",
            embedding=np.random.rand(128),
            content={"text": "Machine learning is a subset of artificial intelligence focused on data-based learning.", "metadata": {}},
            metadata={"category": "fact", "topic": "technology"}
        ),
        Memory(
            id="mem8",
            embedding=np.random.rand(128),
            content={"text": "I enjoy hiking in the mountains during summer vacations.", "metadata": {}},
            metadata={"category": "preference", "topic": "activity"}
        ),
    ]
    return memories


@pytest.fixture
def memory_manager(test_memories) -> MemoryManager:
    """Create a memory manager with test memories."""
    memory_store = MemoryStore()
    memory_store.add_multiple(test_memories)
    
    return MemoryManager(memory_store=memory_store)


@pytest.fixture
def test_queries() -> List[Query]:
    """Create test queries for evaluation."""
    return [
        Query(
            text="What programming language is known for readability?",
            embedding=np.random.rand(128)
        ),
        Query(
            text="What is the capital of France?",
            embedding=np.random.rand(128)
        ),
        Query(
            text="What food do I like to eat?",
            embedding=np.random.rand(128)
        ),
    ]


@pytest.fixture
def relevant_memory_ids() -> List[List[str]]:
    """Define relevant memory IDs for each test query."""
    return [
        ["mem1", "mem5"],  # Relevant to programming language query
        ["mem3"],          # Relevant to France query
        ["mem4"],          # Relevant to food preference query
    ]


@pytest.mark.integration
def test_baseline_comparison_integration(
    memory_manager, test_queries, relevant_memory_ids
):
    """Test the baseline comparison framework with actual retrievers."""
    # Create a MemoryWeave similarity retriever
    memoryweave_retriever = SimilarityRetrievalStrategy()
    
    # Define baseline configurations
    baseline_configs = [
        BaselineConfig(
            name="bm25",
            retriever_class=BM25Retriever,
            parameters={}
        ),
        BaselineConfig(
            name="vector_baseline",
            retriever_class=VectorBaselineRetriever,
            parameters={}
        )
    ]
    
    # Create comparison framework
    comparison = BaselineComparison(
        memory_manager=memory_manager,
        memoryweave_retriever=memoryweave_retriever,
        baseline_configs=baseline_configs,
        metrics=["precision", "recall", "f1", "mrr"]
    )
    
    # Run comparison
    result = comparison.run_comparison(
        queries=test_queries,
        relevant_memory_ids=relevant_memory_ids,
        max_results=5,
        threshold=0.0
    )
    
    # Verify structure of results
    assert isinstance(result, ComparisonResult)
    assert "memoryweave" in result.runtime_stats
    assert "bm25" in result.runtime_stats
    assert "vector_baseline" in result.runtime_stats
    
    assert "bm25" in result.baseline_metrics
    assert "vector_baseline" in result.baseline_metrics
    
    # Check that metrics are computed
    for system_metrics in [result.memoryweave_metrics] + list(result.baseline_metrics.values()):
        assert "average" in system_metrics
        assert "precision" in system_metrics["average"]
        assert "recall" in system_metrics["average"]
        assert "f1" in system_metrics["average"]
        assert "mrr" in system_metrics["average"]
    
    # Test saving results to file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        temp_path = temp.name
    
    try:
        comparison.save_results(result, temp_path)
        
        # Verify file content
        with open(temp_path, 'r') as f:
            saved_data = json.load(f)
        
        assert "memoryweave_metrics" in saved_data
        assert "baseline_metrics" in saved_data
        assert "runtime_stats" in saved_data
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Test visualization
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        viz_path = temp.name
    
    try:
        comparison.visualize_results(result, viz_path)
        assert os.path.exists(viz_path)
        assert os.path.getsize(viz_path) > 0
    finally:
        if os.path.exists(viz_path):
            os.remove(viz_path)
    
    # Test HTML report generation
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp:
        html_path = temp.name
    
    try:
        comparison.generate_html_report(result, html_path)
        assert os.path.exists(html_path)
        assert os.path.getsize(html_path) > 0
    finally:
        if os.path.exists(html_path):
            os.remove(html_path)
        # Also remove the chart image
        chart_path = html_path.replace(".html", "_chart.png")
        if os.path.exists(chart_path):
            os.remove(chart_path)
