"""
Tests for baseline retrieval implementations.
"""

import uuid
from typing import List

import numpy as np
import pytest

from memoryweave.baselines import BM25Retriever, VectorBaselineRetriever
from memoryweave.interfaces.retrieval import Query, QueryType
from memoryweave.interfaces.memory import Memory


@pytest.fixture
def sample_memories() -> List[Memory]:
    """Create a set of sample memories for testing."""
    memories = [
        Memory(
            id=str(uuid.uuid4()),
            embedding=np.array([0.1, 0.2, 0.3]),
            content={"text": "Python is a programming language", "metadata": {}},
            metadata={"category": "fact"},
        ),
        Memory(
            id=str(uuid.uuid4()),
            embedding=np.array([0.2, 0.3, 0.4]),
            content={"text": "I like to eat pizza on Fridays", "metadata": {}},
            metadata={"category": "preference"},
        ),
        Memory(
            id=str(uuid.uuid4()),
            embedding=np.array([0.3, 0.4, 0.5]),
            content={"text": "The capital of France is Paris", "metadata": {}},
            metadata={"category": "fact"},
        ),
        Memory(
            id=str(uuid.uuid4()),
            embedding=np.array([0.4, 0.5, 0.6]),
            content={"text": "My favorite color is blue", "metadata": {}},
            metadata={"category": "preference"},
        ),
        Memory(
            id=str(uuid.uuid4()),
            embedding=np.array([0.12, 0.22, 0.32]),
            content={"text": "Python has simple and easy to learn syntax", "metadata": {}},
            metadata={"category": "fact"},
        ),
    ]
    return memories


@pytest.fixture
def sample_query() -> Query:
    """Create a sample query for testing."""
    return Query(
        text="What programming language has easy syntax?",
        embedding=np.array([0.11, 0.21, 0.31]),
        query_type=QueryType.FACTUAL,
        extracted_keywords=["programming", "language", "syntax"],
        extracted_entities=[],
    )


class TestBM25Retriever:
    """Tests for BM25 retriever implementation."""

    def test_index_memories(self, sample_memories):
        """Test indexing memories with BM25."""
        retriever = BM25Retriever()
        retriever.index_memories(sample_memories)

        assert retriever.stats["index_size"] == len(sample_memories)
        assert "indexing_time" in retriever.stats

    def test_retrieve(self, sample_memories, sample_query):
        """Test retrieving memories with BM25."""
        retriever = BM25Retriever()
        retriever.index_memories(sample_memories)

        result = retriever.retrieve(sample_query, top_k=2)

        assert len(result["memories"]) <= 2
        assert len(result["scores"]) == len(result["memories"])
        assert result["strategy"] == "bm25"
        assert "query_time" in result["metadata"]
        assert "bm25_params" in result["metadata"]

    def test_threshold_filtering(self, sample_memories):
        """Test that threshold parameter correctly filters results."""
        retriever = BM25Retriever()
        retriever.index_memories(sample_memories)

        # Query that should match with high score
        query = Query(
            text="Python programming syntax",
            embedding=np.array([0.1, 0.2, 0.3]),
            query_type=QueryType.FACTUAL,
            extracted_keywords=["python", "programming", "syntax"],
            extracted_entities=[],
        )

        # Get all results
        result_all = retriever.retrieve(query, top_k=5, threshold=0.0)

        # Get only high-scoring results
        result_threshold = retriever.retrieve(query, top_k=5, threshold=0.7)

        assert len(result_all["memories"]) >= len(result_threshold["memories"])

    def test_get_statistics(self, sample_memories, sample_query):
        """Test getting statistics from BM25 retriever."""
        retriever = BM25Retriever()
        retriever.index_memories(sample_memories)

        # Run a query to generate stats
        retriever.retrieve(sample_query)

        stats = retriever.get_statistics()
        assert "index_size" in stats
        assert "query_times" in stats
        assert "avg_query_time" in stats


class TestVectorBaselineRetriever:
    """Tests for vector baseline retriever implementation."""

    def test_index_memories(self, sample_memories):
        """Test indexing memories with vector search."""
        retriever = VectorBaselineRetriever()
        retriever.index_memories(sample_memories)

        assert retriever.stats["index_size"] == len(sample_memories)
        assert retriever.stats["dimensions"] == sample_memories[0].embedding.shape[0]
        assert "indexing_time" in retriever.stats

    def test_retrieve(self, sample_memories, sample_query):
        """Test retrieving memories with vector search."""
        retriever = VectorBaselineRetriever()
        retriever.index_memories(sample_memories)

        result = retriever.retrieve(sample_query, top_k=2)

        assert len(result["memories"]) <= 2
        assert len(result["scores"]) == len(result["memories"])
        assert result["strategy"] == "vector_baseline"
        assert "query_time" in result["metadata"]
        assert "search_type" in result["metadata"]

    def test_threshold_filtering(self, sample_memories):
        """Test that threshold parameter correctly filters results."""
        retriever = VectorBaselineRetriever()
        retriever.index_memories(sample_memories)

        # Create query with embedding similar to first memory
        query = Query(
            text="What is Python?",
            embedding=np.array([0.11, 0.21, 0.31]),
            query_type=QueryType.FACTUAL,
            extracted_keywords=["python"],
            extracted_entities=[],
        )

        # Get all results
        result_all = retriever.retrieve(query, top_k=5, threshold=0.0)

        # Get only high-scoring results
        result_threshold = retriever.retrieve(query, top_k=5, threshold=0.9)

        assert len(result_all["memories"]) >= len(result_threshold["memories"])

    def test_get_statistics(self, sample_memories, sample_query):
        """Test getting statistics from vector retriever."""
        retriever = VectorBaselineRetriever()
        retriever.index_memories(sample_memories)

        # Run a query to generate stats
        retriever.retrieve(sample_query)

        stats = retriever.get_statistics()
        assert "index_size" in stats
        assert "dimensions" in stats
        assert "query_times" in stats
        assert "avg_query_time" in stats

    def test_empty_query_embedding(self, sample_memories):
        """Test handling of query without embedding."""
        retriever = VectorBaselineRetriever()
        retriever.index_memories(sample_memories)

        # Query without embedding
        query = Query(
            text="What is Python?",
            embedding=None,
            query_type=QueryType.FACTUAL,
            extracted_keywords=["python"],
            extracted_entities=[],
        )

        result = retriever.retrieve(query)

        # Should return empty result
        assert len(result["memories"]) == 0
        assert len(result["scores"]) == 0
        assert result["strategy"] == "vector_baseline"
