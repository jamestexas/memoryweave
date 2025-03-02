"""
Tests for baseline comparison framework.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
import uuid
import tempfile
import os

from memoryweave.baselines import BM25Retriever, VectorBaselineRetriever
from memoryweave.evaluation.baseline_comparison import (
    BaselineComparison, BaselineConfig, ComparisonResult
)
from memoryweave.interfaces.memory import IMemoryManager
from memoryweave.interfaces.retrieval import (
    IRetrievalStrategy, Query, RetrievalResult, RetrievalParameters
)
from memoryweave.storage.memory_store import Memory


class MockMemoryManager:
    """Mock memory manager for testing."""
    
    def __init__(self, memories: List[Memory]):
        """Initialize with pre-defined memories."""
        self.memories = memories
    
    def get_all_memories(self) -> List[Memory]:
        """Return all memories."""
        return self.memories


class MockRetriever:
    """Mock retrieval strategy for testing."""
    
    def __init__(self, memories: List[Memory], name: str = "mock_retriever"):
        """Initialize with pre-defined memories and name."""
        self.memories = memories
        self.name = name
    
    def retrieve(
        self, 
        query: Query, 
        top_k: int = 10, 
        threshold: float = 0.0,
        **kwargs
    ) -> RetrievalResult:
        """Mock retrieval that matches memories containing query terms."""
        query_terms = set(query.text.lower().split())
        
        matches = []
        scores = []
        
        for memory in self.memories:
            memory_terms = set(memory.text.lower().split())
            
            # Simple term overlap scoring
            overlap = query_terms.intersection(memory_terms)
            if overlap:
                score = len(overlap) / len(query_terms)
                if score >= threshold:
                    matches.append(memory)
                    scores.append(score)
        
        # Sort by score and take top_k
        sorted_pairs = sorted(zip(matches, scores), key=lambda pair: pair[1], reverse=True)
        sorted_pairs = sorted_pairs[:top_k]
        
        if sorted_pairs:
            matches, scores = zip(*sorted_pairs)
        else:
            matches, scores = [], []
        
        return RetrievalResult(
            memories=list(matches),
            scores=list(scores),
            strategy=self.name,
            parameters=RetrievalParameters(
                max_results=top_k,
                threshold=threshold
            ),
            metadata={"query_time": 0.001}
        )


@pytest.fixture
def sample_memories() -> List[Memory]:
    """Create a set of sample memories for testing."""
    memories = [
        Memory(
            id="1",
            text="Python is a programming language",
            embedding=np.array([0.1, 0.2, 0.3]),
            metadata={"category": "fact"}
        ),
        Memory(
            id="2",
            text="I like to eat pizza on Fridays",
            embedding=np.array([0.2, 0.3, 0.4]),
            metadata={"category": "preference"}
        ),
        Memory(
            id="3",
            text="The capital of France is Paris",
            embedding=np.array([0.3, 0.4, 0.5]),
            metadata={"category": "fact"}
        ),
        Memory(
            id="4",
            text="My favorite color is blue",
            embedding=np.array([0.4, 0.5, 0.6]),
            metadata={"category": "preference"}
        ),
        Memory(
            id="5",
            text="Python has simple and easy to learn syntax",
            embedding=np.array([0.12, 0.22, 0.32]),
            metadata={"category": "fact"}
        ),
    ]
    return memories


@pytest.fixture
def sample_queries() -> List[Query]:
    """Create a set of sample queries for testing."""
    return [
        Query(
            text="What programming language is easy to learn?",
            embedding=np.array([0.11, 0.21, 0.31])
        ),
        Query(
            text="What is the capital of France?",
            embedding=np.array([0.31, 0.41, 0.51])
        ),
        Query(
            text="What food do I like?",
            embedding=np.array([0.21, 0.31, 0.41])
        )
    ]


@pytest.fixture
def relevant_memory_ids() -> List[List[str]]:
    """Create a set of relevant memory IDs for each query."""
    return [
        ["1", "5"],  # Relevant to "What programming language is easy to learn?"
        ["3"],       # Relevant to "What is the capital of France?"
        ["2"]        # Relevant to "What food do I like?"
    ]


class TestBaselineComparison:
    """Tests for baseline comparison framework."""
    
    def test_initialization(self, sample_memories):
        """Test initialization of comparison framework."""
        memory_manager = MockMemoryManager(sample_memories)
        memoryweave_retriever = MockRetriever(sample_memories, name="memoryweave")
        
        baseline_configs = [
            BaselineConfig(
                name="bm25",
                retriever_class=BM25Retriever,
                parameters={"b": 0.75, "k1": 1.2}
            ),
            BaselineConfig(
                name="vector",
                retriever_class=VectorBaselineRetriever,
                parameters={"use_exact_search": True}
            )
        ]
        
        comparison = BaselineComparison(
            memory_manager=memory_manager,
            memoryweave_retriever=memoryweave_retriever,
            baseline_configs=baseline_configs
        )
        
        assert comparison.memory_manager == memory_manager
        assert comparison.memoryweave_retriever == memoryweave_retriever
        assert len(comparison.baseline_retrievers) == 2
        assert "bm25" in comparison.baseline_retrievers
        assert "vector" in comparison.baseline_retrievers
    
    def test_run_comparison(self, sample_memories, sample_queries, relevant_memory_ids):
        """Test running a comparison between MemoryWeave and baselines."""
        memory_manager = MockMemoryManager(sample_memories)
        memoryweave_retriever = MockRetriever(sample_memories, name="memoryweave")
        
        baseline_configs = [
            BaselineConfig(
                name="mock_baseline",
                retriever_class=MockRetriever,
                parameters={"memories": sample_memories, "name": "mock_baseline"}
            )
        ]
        
        comparison = BaselineComparison(
            memory_manager=memory_manager,
            memoryweave_retriever=memoryweave_retriever,
            baseline_configs=baseline_configs
        )
        
        result = comparison.run_comparison(
            queries=sample_queries,
            relevant_memory_ids=relevant_memory_ids,
            max_results=3,
            threshold=0.0
        )
        
        # Check that the result has the expected structure
        assert isinstance(result, ComparisonResult)
        assert "memoryweave" in result.runtime_stats
        assert "mock_baseline" in result.runtime_stats
        assert "mock_baseline" in result.baseline_metrics
        assert "average" in result.memoryweave_metrics
        assert "by_query" in result.memoryweave_metrics
        assert len(result.query_details) == len(sample_queries)
    
    def test_calculate_metrics(self, sample_memories, sample_queries, relevant_memory_ids):
        """Test calculation of evaluation metrics."""
        memory_manager = MockMemoryManager(sample_memories)
        retriever = MockRetriever(sample_memories)
        
        # Run retrieval for each query
        results = []
        for query in sample_queries:
            results.append(retriever.retrieve(query, top_k=3))
        
        # Initialize comparison
        comparison = BaselineComparison(
            memory_manager=memory_manager,
            memoryweave_retriever=retriever,
            baseline_configs=[]
        )
        
        # Calculate metrics
        metrics = comparison._compute_metrics(results, sample_queries, relevant_memory_ids)
        
        # Check metric structure
        assert "average" in metrics
        assert "by_query" in metrics
        assert "precision" in metrics["average"]
        assert "recall" in metrics["average"]
        assert "f1" in metrics["average"]
        
        # Check individual query metrics
        for i in range(len(sample_queries)):
            query_id = f"query_{i}"
            assert query_id in metrics["by_query"]
            assert "precision" in metrics["by_query"][query_id]
            assert "recall" in metrics["by_query"][query_id]
            assert "f1" in metrics["by_query"][query_id]
    
    def test_save_results(self, sample_memories, sample_queries, relevant_memory_ids):
        """Test saving comparison results to a file."""
        memory_manager = MockMemoryManager(sample_memories)
        memoryweave_retriever = MockRetriever(sample_memories, name="memoryweave")
        
        baseline_configs = [
            BaselineConfig(
                name="mock_baseline",
                retriever_class=MockRetriever,
                parameters={"memories": sample_memories, "name": "mock_baseline"}
            )
        ]
        
        comparison = BaselineComparison(
            memory_manager=memory_manager,
            memoryweave_retriever=memoryweave_retriever,
            baseline_configs=baseline_configs
        )
        
        result = comparison.run_comparison(
            queries=sample_queries,
            relevant_memory_ids=relevant_memory_ids
        )
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            temp_path = temp.name
        
        try:
            # Save results
            comparison.save_results(result, temp_path)
            
            # Check that file exists and has content
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_visualize_results(self, sample_memories, sample_queries, relevant_memory_ids):
        """Test visualizing comparison results."""
        memory_manager = MockMemoryManager(sample_memories)
        memoryweave_retriever = MockRetriever(sample_memories, name="memoryweave")
        
        baseline_configs = [
            BaselineConfig(
                name="mock_baseline",
                retriever_class=MockRetriever,
                parameters={"memories": sample_memories, "name": "mock_baseline"}
            )
        ]
        
        comparison = BaselineComparison(
            memory_manager=memory_manager,
            memoryweave_retriever=memoryweave_retriever,
            baseline_configs=baseline_configs
        )
        
        result = comparison.run_comparison(
            queries=sample_queries,
            relevant_memory_ids=relevant_memory_ids
        )
        
        # Create a temporary file for the visualization
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            temp_path = temp.name
        
        try:
            # Generate visualization
            comparison.visualize_results(result, temp_path)
            
            # Check that file exists and has content
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)