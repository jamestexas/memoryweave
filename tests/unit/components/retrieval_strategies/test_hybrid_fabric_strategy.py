# tests/components/retrieval_strategies/test_hybrid_fabric_strategy.py
"""
Tests for the HybridFabricStrategy.

This file contains tests for the memory-efficient hybrid fabric retrieval
strategy that optimizes resource usage while maintaining retrieval quality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memoryweave.components.retrieval_strategies.hybrid_fabric_strategy import HybridFabricStrategy


@pytest.fixture
def query_embedding():
    """Create a sample query embedding for testing."""
    vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    return vector / np.linalg.norm(vector)


@pytest.fixture
def memory_store():
    """Create a properly configured mock memory store."""
    mock_store = MagicMock()
    # Configure essential attributes
    mock_store.memory_embeddings = np.random.randn(10, 5)  # 10 memories with dim=5
    mock_store.memory_embeddings = mock_store.memory_embeddings / np.linalg.norm(
        mock_store.memory_embeddings, axis=1, keepdims=True
    )
    mock_store.memory_metadata = [
        {"content": f"Test memory {i}", "source": "test"} for i in range(10)
    ]

    # Add search capabilities
    mock_store.search_by_vector = MagicMock()
    mock_store.search_by_vector.return_value = [
        {"memory_id": 0, "relevance_score": 0.9, "content": "Memory 0"},
        {"memory_id": 1, "relevance_score": 0.8, "content": "Memory 1"},
        {"memory_id": 2, "relevance_score": 0.7, "content": "Memory 2"},
    ]

    # Define search_hybrid - this will signal supports_hybrid=True
    mock_store.search_hybrid = MagicMock()
    mock_store.search_hybrid.return_value = [
        {"memory_id": 0, "relevance_score": 0.9, "content": "Memory 0"},
        {"memory_id": 1, "relevance_score": 0.8, "content": "Memory 1"},
    ]

    return mock_store


@pytest.fixture
def associative_linker():
    """Create a mock associative linker."""
    mock_linker = MagicMock()

    def get_associative_links(memory_id):
        if memory_id == 0:
            return [(1, 0.8), (2, 0.6)]
        elif memory_id == 1:
            return [(0, 0.8), (3, 0.7)]
        return []

    mock_linker.get_associative_links = MagicMock(side_effect=get_associative_links)
    mock_linker.traverse_associative_network = MagicMock(return_value={1: 0.8, 2: 0.6, 3: 0.7})

    return mock_linker


@pytest.fixture
def base_context(memory_store):
    """Create a base context dict for testing."""
    return {
        "memory_store": memory_store,
        "top_k": 3,
        "query": "Test query",
    }


class TestHybridFabricStrategy:
    """Tests for the HybridFabricStrategy class."""

    def test_initialization(
        self, memory_store, associative_linker, temporal_context, activation_manager
    ):
        """Test initialization with different configurations."""
        # Test default initialization
        strategy = HybridFabricStrategy(
            memory_store=memory_store,
            associative_linker=associative_linker,
            temporal_context=temporal_context,
            activation_manager=activation_manager,
        )

        # Initialize to ensure proper setup
        strategy.initialize({})

        # Verify attributes are set correctly
        assert strategy.memory_store == memory_store
        assert strategy.associative_linker == associative_linker
        assert strategy.temporal_context == temporal_context
        assert strategy.activation_manager == activation_manager

        # Default hybrid parameters
        assert strategy.use_keyword_filtering is True
        assert strategy.keyword_boost_factor == 0.3
        assert strategy.max_chunks_per_memory == 3
        assert strategy.prioritize_full_embeddings is True

        # Verify supports_hybrid is detected correctly based on memory_store
        assert strategy.supports_hybrid is True, "Should detect hybrid support"

        # Test initialization with configuration
        config = {
            "confidence_threshold": 0.2,
            "similarity_weight": 0.6,
            "associative_weight": 0.2,
            "temporal_weight": 0.1,
            "activation_weight": 0.1,
            "use_keyword_filtering": False,
            "keyword_boost_factor": 0.4,
            "max_chunks_per_memory": 5,
            "prioritize_full_embeddings": False,
            "use_two_stage": True,
            "first_stage_k": 40,
            "first_stage_threshold_factor": 0.6,
        }
        strategy.initialize(config)
        assert strategy.confidence_threshold == 0.2
        assert strategy.similarity_weight == 0.6
        assert strategy.associative_weight == 0.2
        assert strategy.temporal_weight == 0.1
        assert strategy.activation_weight == 0.1
        assert strategy.use_keyword_filtering is False
        assert strategy.keyword_boost_factor == 0.4
        assert strategy.max_chunks_per_memory == 5
        assert strategy.prioritize_full_embeddings is False
        assert strategy.use_two_stage_by_default is True
        assert strategy.first_stage_k == 40
        assert strategy.first_stage_threshold_factor == 0.6

    def test_hybrid_support_detection(self):
        """Test detection of hybrid support in memory store."""
        # Create memory stores with different capabilities
        memory_with_hybrid = MagicMock()
        memory_with_hybrid.search_hybrid = MagicMock()

        nested_memory = MagicMock()
        nested_memory.search_hybrid = MagicMock()
        memory_with_nested = MagicMock()
        memory_with_nested.memory_store = nested_memory

        memory_with_chunks = MagicMock()
        memory_with_chunks.search_chunks = MagicMock()

        memory_without_hybrid = MagicMock()

        # Verify direct search_hybrid support
        strategy1 = HybridFabricStrategy(memory_store=memory_with_hybrid)
        strategy1.initialize({})  # Let initialize() detect capabilities
        assert strategy1.supports_hybrid is True, "Direct search_hybrid should be detected"

        # Verify nested search_hybrid support
        strategy2 = HybridFabricStrategy(memory_store=memory_with_nested)
        strategy2.initialize({})  # Let initialize() detect capabilities
        assert strategy2.supports_hybrid is True, "Nested search_hybrid should be detected"

        # Verify search_chunks support
        strategy3 = HybridFabricStrategy(memory_store=memory_with_chunks)
        strategy3.initialize({})  # Let initialize() detect capabilities
        assert strategy3.supports_hybrid is True, (
            "search_chunks should be detected as hybrid-capable"
        )

        # Verify lack of hybrid support
        strategy4 = HybridFabricStrategy(memory_store=memory_without_hybrid)
        strategy4.initialize({})  # Let initialize() detect capabilities
        assert strategy4.supports_hybrid is False, "Should not detect hybrid support"

    def test_retrieve_basic(self, memory_store, query_embedding, base_context):
        """Test basic retrieval functionality for benchmarking."""
        # Initialize strategy and let it detect capabilities naturally
        strategy = HybridFabricStrategy(memory_store=memory_store)
        strategy.initialize({"confidence_threshold": 0.0})

        # Verify supports_hybrid is set from memory_store (has search_hybrid)
        assert strategy.supports_hybrid is True

        # Retrieve memories
        results = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # Check that search_by_vector was called with expected parameters
        memory_store.search_by_vector.assert_called_once()
        call_args = memory_store.search_by_vector.call_args
        assert call_args[1]["query_vector"] is query_embedding
        assert call_args[1]["limit"] == 6  # top_k * 2

        # Verify results
        assert len(results) == 3, "Should return 3 results"
        assert results[0]["memory_id"] == 0, "First result should be memory_id 0"
        assert results[0]["relevance_score"] == 0.9, "First result score should be 0.9"

    def test_combine_results_with_rank_fusion(self):
        """Test the _combine_results_with_rank_fusion method directly."""
        strategy = HybridFabricStrategy()
        strategy.initialize({})  # Initialize with defaults

        # Create test results
        keyword_results = [
            {"memory_id": 1, "relevance_score": 0.9, "content": "Memory 1"},
            {"memory_id": 3, "relevance_score": 0.8, "content": "Memory 3"},
            {"memory_id": 5, "relevance_score": 0.7, "content": "Memory 5"},
        ]

        vector_results = [
            {"memory_id": 2, "relevance_score": 0.95, "content": "Memory 2"},
            {"memory_id": 1, "relevance_score": 0.85, "content": "Memory 1"},
            {"memory_id": 4, "relevance_score": 0.75, "content": "Memory 4"},
        ]

        # Call the method
        combined = strategy._combine_results_with_rank_fusion(
            results1=keyword_results, results2=vector_results, k1=60.0, k2=40.0, top_k=4
        )

        # Should combine the results with rank-based fusion
        assert len(combined) == 4, "Should return top_k=4 results"

        # Memory 1 should be ranked highly as it appears in both results
        memory_1 = next(r for r in combined if r["memory_id"] == 1)
        assert memory_1["rrf_score"] > 0
        assert "retrieval_sources" in memory_1
        assert set(memory_1["retrieval_sources"]) == {"keyword", "vector"}

        # All results should have rrf_score and retrieval_sources
        for result in combined:
            assert "rrf_score" in result
            assert "retrieval_sources" in result

    def test_combined_search(self, query_embedding):
        """Test the _combined_search method directly."""
        strategy = HybridFabricStrategy()
        strategy.initialize(
            {
                "keyword_boost_factor": 0.4,
            }
        )

        # Create memory store with search methods
        memory_store = MagicMock()
        memory_store.search_by_vector = MagicMock()
        memory_store.search_by_vector.return_value = [
            {"memory_id": 1, "relevance_score": 0.9, "content": "Memory 1"},
            {"memory_id": 2, "relevance_score": 0.8, "content": "Memory 2"},
            {"memory_id": 3, "relevance_score": 0.7, "content": "Memory 3"},
        ]

        memory_store.search_chunks = MagicMock()
        memory_store.search_chunks.return_value = [
            {
                "memory_id": 4,
                "chunk_index": 0,
                "chunk_similarity": 0.85,
                "content": "Chunk of memory 4",
                "metadata": {"memory_id": 4},
            },
            {
                "memory_id": 1,
                "chunk_index": 0,
                "chunk_similarity": 0.75,
                "content": "Chunk of memory 1",
                "metadata": {"memory_id": 1},
            },
        ]

        # Call the method with keywords
        keywords = ["memory", "test"]
        results = strategy._combined_search(
            query_embedding=query_embedding,
            max_results=4,
            threshold=0.0,
            keywords=keywords,
            memory_store=memory_store,
        )

        # Should call both search methods
        memory_store.search_by_vector.assert_called_once()
        memory_store.search_chunks.assert_called_once()

        # Should include results from both sources
        memory_ids = {r["memory_id"] for r in results}
        assert 1 in memory_ids, "Memory 1 should be included (from both sources)"
        assert 2 in memory_ids, "Memory 2 should be included (from vector)"
        assert 3 in memory_ids, "Memory 3 should be included (from vector)"
        assert 4 in memory_ids, "Memory 4 should be included (from chunks)"

        # Results should be sorted by relevance_score
        scores = [r.get("relevance_score", 0) for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), (
            "Results should be sorted by relevance_score"
        )

    def test_extract_simple_keywords(self):
        """Test the keyword extraction functionality."""
        # Create strategy with mock for nltk check
        strategy = HybridFabricStrategy()
        strategy.initialize({})

        # Mock _load_module to return a value causing nltk fallback
        with patch(
            "memoryweave.components.retrieval_strategies.hybrid_fabric_strategy._load_module",
            return_value=True,
        ):
            text = "This is a comprehensive test with some technical keywords like extraction, parsing, and analysis of semantic content."
            keywords = strategy._extract_simple_keywords(text)

            # Verify reasonable keywords are extracted
            expected_keywords = [
                "comprehensive",
                "technical",
                "keywords",
                "extraction",
                "parsing",
                "analysis",
                "semantic",
                "content",
            ]
            assert any(kw in keywords for kw in expected_keywords), (
                "Should extract meaningful keywords"
            )

            # Verify stopwords are filtered
            stopwords = ["this", "is", "a", "with", "some", "like", "and", "of"]
            assert all(sw not in keywords for sw in stopwords), "Should filter out stopwords"

    def test_retrieve_two_stage(self, memory_store, query_embedding, base_context):
        """Test the retrieve_two_stage method directly."""
        # Initialize strategy
        strategy = HybridFabricStrategy(memory_store=memory_store)
        strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "use_two_stage": True,
                "first_stage_k": 10,
                "first_stage_threshold_factor": 0.7,
            }
        )

        # Set up memory store search method returns
        memory_store.search_by_keywords = MagicMock()
        memory_store.search_by_keywords.return_value = [
            {"memory_id": 3, "relevance_score": 0.75, "content": "Memory 3"},
            {"memory_id": 1, "relevance_score": 0.65, "content": "Memory 1"},
        ]

        # Context with keywords
        context = base_context.copy()
        context["important_keywords"] = ["memory", "test"]

        # Use context manager for patches to ensure cleanup
        with patch.object(
            strategy,
            "_apply_keyword_boosting",
            return_value=[
                {"memory_id": 0, "relevance_score": 0.92, "content": "Memory 0 with keyword"},
                {"memory_id": 1, "relevance_score": 0.88, "content": "Memory 1 with keyword"},
                {"memory_id": 2, "relevance_score": 0.7, "content": "Memory 2"},
                {"memory_id": 3, "relevance_score": 0.85, "content": "Memory 3 with keyword"},
            ],
        ) as mock_keyword_boost:
            with patch.object(
                strategy,
                "_check_semantic_coherence",
                lambda results, _: sorted(
                    results, key=lambda x: x.get("relevance_score", 0), reverse=True
                ),
            ) as _mock_coherence:
                with patch.object(
                    strategy, "_enhance_with_associative_context", lambda results: results
                ) as _mock_assoc:
                    # Call the method
                    results = strategy.retrieve_two_stage(
                        query_embedding=query_embedding,
                        top_k=3,
                        context=context,
                        query="test query",
                    )

                    # Check that search methods were called
                    memory_store.search_by_vector.assert_called_once()
                    memory_store.search_by_keywords.assert_called_once()

                    # Check that post-processing was applied
                    mock_keyword_boost.assert_called_once()

                    # Verify results
                    assert len(results) == 3, "Should return top_k=3 results"
                    assert results[0]["relevance_score"] >= results[1]["relevance_score"], (
                        "Results should be sorted"
                    )
                    assert results[1]["relevance_score"] >= results[2]["relevance_score"], (
                        "Results should be sorted"
                    )

    def test_apply_keyword_boosting(self):
        """Test the _apply_keyword_boosting method directly."""
        strategy = HybridFabricStrategy()
        strategy.initialize(
            {
                "keyword_boost_factor": 0.4,
            }
        )

        # Create test results
        results = [
            {
                "memory_id": 1,
                "relevance_score": 0.8,
                "content": "This contains keyword1 and keyword2",
            },
            {"memory_id": 2, "relevance_score": 0.7, "content": "This contains only keyword1"},
            {"memory_id": 3, "relevance_score": 0.6, "content": "This has no keywords"},
        ]

        # Call the method with keywords
        keywords = ["keyword1", "keyword2"]
        boosted = strategy._apply_keyword_boosting(results, keywords)

        # Memory 1 has both keywords, should get highest boost
        assert boosted[0]["relevance_score"] > 0.8, "Memory 1 should be boosted"
        assert "keyword_boost" in boosted[0], "Should include boost factor"
        assert boosted[0]["keyword_matches"] == 2, "Should count 2 keyword matches"

        # Memory 2 has one keyword, should get medium boost
        assert boosted[1]["relevance_score"] > 0.7, "Memory 2 should be boosted"
        assert boosted[1]["relevance_score"] < boosted[0]["relevance_score"], (
            "Less boost than Memory 1"
        )
        assert "keyword_boost" in boosted[1], "Should include boost factor"
        assert boosted[1]["keyword_matches"] == 1, "Should count 1 keyword match"

        # Memory 3 has no keywords, should not be boosted
        assert boosted[2]["relevance_score"] == 0.6, "Memory 3 should not be boosted"
        assert "keyword_boost" not in boosted[2], "Should not have boost factor"

    def test_check_semantic_coherence(self, query_embedding):
        """Test the _check_semantic_coherence method directly."""
        strategy = HybridFabricStrategy()
        strategy.initialize({})

        # Create test results with embeddings
        results = [
            {
                "memory_id": 1,
                "relevance_score": 0.8,
                "content": "Content 1",
                "embedding": np.array([0.9, 0.1, 0.1]),  # Similar to 2, different from 3
            },
            {
                "memory_id": 2,
                "relevance_score": 0.7,
                "content": "Content 2",
                "embedding": np.array([0.8, 0.2, 0.2]),  # Similar to 1, different from 3
            },
            {
                "memory_id": 3,
                "relevance_score": 0.6,
                "content": "Content 3",
                "embedding": np.array([0.2, 0.8, 0.8]),  # Different from 1 and 2
            },
        ]

        # Normalize embeddings
        for r in results:
            r["embedding"] = r["embedding"] / np.linalg.norm(r["embedding"])

        # Call the method
        checked = strategy._check_semantic_coherence(results, query_embedding)

        # Each result should have a coherence score
        for result in checked:
            assert "coherence_score" in result, "Should add coherence_score to each result"

        # Memory 3 should have lower coherence with others
        assert checked[2]["coherence_score"] < checked[0]["coherence_score"], (
            "Memory 3 should have lower coherence"
        )

    def test_enhance_with_associative_context(self, memory_store, associative_linker):
        """Test the _enhance_with_associative_context method directly."""
        # Initialize strategy with associative linker
        strategy = HybridFabricStrategy(
            memory_store=memory_store, associative_linker=associative_linker
        )
        strategy.initialize({})

        # Create test results representing memories 0 and 1
        results = [
            {"memory_id": 0, "relevance_score": 0.9, "content": "Memory 0"},
            {"memory_id": 1, "relevance_score": 0.8, "content": "Memory 1"},
        ]

        # Call the method
        enhanced = strategy._enhance_with_associative_context(results)

        # Should include original memories
        assert any(r["memory_id"] == 0 for r in enhanced), "Should include Memory 0"
        assert any(r["memory_id"] == 1 for r in enhanced), "Should include Memory 1"

        # Should include associated memories
        assert any(r["memory_id"] == 2 for r in enhanced), (
            "Should include Memory 2 (associated with 0)"
        )
        assert any(r["memory_id"] == 3 for r in enhanced), (
            "Should include Memory 3 (associated with 1)"
        )

        # Associated memories should have metadata from associative link
        memory_3 = next((r for r in enhanced if r["memory_id"] == 3), None)
        assert memory_3 is not None, "Memory 3 should be included"
        assert "associative_link" in memory_3, "Should be marked as associative link"
        assert memory_3["link_source"] == 1, "Should identify source memory"
        assert "link_strength" in memory_3, "Should include link strength"

    def test_configure_two_stage(self):
        """Test the configure_two_stage method."""
        strategy = HybridFabricStrategy()
        strategy.initialize({})  # Initialize with defaults

        # Configure with new values
        strategy.configure_two_stage(
            enable=False, first_stage_k=40, first_stage_threshold_factor=0.5
        )

        # Verify values were updated
        assert strategy.use_two_stage_by_default is False, "Two-stage retrieval should be disabled"
        assert strategy.first_stage_k == 40, "first_stage_k should be updated"
        assert strategy.first_stage_threshold_factor == 0.5, (
            "first_stage_threshold_factor should be updated"
        )
