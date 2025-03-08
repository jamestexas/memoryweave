# tests/components/retrieval_strategies/test_hybrid_fabric_strategy.py
"""
Tests for the HybridFabricStrategy.

This file contains tests for the memory-efficient hybrid fabric retrieval
strategy that optimizes resource usage while maintaining retrieval quality.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from memoryweave.components.retrieval_strategies.hybrid_fabric_strategy import HybridFabricStrategy


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
        assert strategy.memory_store == memory_store
        assert strategy.associative_linker == associative_linker
        assert strategy.temporal_context == temporal_context
        assert strategy.activation_manager == activation_manager

        # Default hybrid parameters
        assert strategy.use_keyword_filtering is True
        assert strategy.keyword_boost_factor == 0.3
        assert strategy.max_chunks_per_memory == 3
        assert strategy.prioritize_full_embeddings is True

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
        # Create memory store with search_hybrid method
        memory_with_hybrid = MagicMock()
        memory_with_hybrid.search_hybrid = MagicMock()

        # Create memory store with nested hybrid support
        nested_memory = MagicMock()
        memory_store = MagicMock()
        memory_store.memory_store = nested_memory
        memory_store.memory_store.search_hybrid = MagicMock()

        # Create memory store with chunk support
        memory_with_chunks = MagicMock()
        memory_with_chunks.search_chunks = MagicMock()

        # Create memory store without hybrid support
        memory_without_hybrid = MagicMock()

        # Test detection with direct search_hybrid
        strategy = HybridFabricStrategy(memory_store=memory_with_hybrid)
        strategy.initialize({})
        assert strategy.supports_hybrid is True

        # Test detection with nested memory store
        strategy = HybridFabricStrategy(memory_store=memory_store)
        strategy.initialize({})
        assert strategy.supports_hybrid is True

        # Test detection with search_chunks (also works for hybrid)
        strategy = HybridFabricStrategy(memory_store=memory_with_chunks)
        strategy.initialize({})
        assert strategy.supports_hybrid is True

        # Test without hybrid support
        strategy = HybridFabricStrategy(memory_store=memory_without_hybrid)
        strategy.initialize({})
        assert strategy.supports_hybrid is False

    def test_retrieve_basic(self, memory_store, query_embedding, base_context):
        """Test basic retrieval functionality for benchmarking."""
        # This case is for benchmarking where we return direct vector results
        strategy = HybridFabricStrategy(memory_store=memory_store)
        strategy.initialize({"confidence_threshold": 0.0})

        # Make memory store searchable
        memory_store.search_by_vector = MagicMock()
        memory_store.search_by_vector.return_value = [
            {"memory_id": 0, "relevance_score": 0.9, "content": "Memory 0"},
            {"memory_id": 1, "relevance_score": 0.8, "content": "Memory 1"},
            {"memory_id": 2, "relevance_score": 0.7, "content": "Memory 2"},
        ]

        # Retrieve memories
        results = strategy.retrieve(query_embedding, top_k=3, context=base_context)

        # Check that search_by_vector was called
        memory_store.search_by_vector.assert_called_with(
            query_vector=query_embedding,
            limit=6,  # top_k * 2
            threshold=0.05,  # min(confidence_threshold, 0.05)
        )

        # Should return results directly for benchmark
        assert len(results) == 3
        assert results[0]["memory_id"] == 0
        assert results[0]["relevance_score"] == 0.9

    def test_combine_results_with_rank_fusion(self):
        """Test the _combine_results_with_rank_fusion method directly."""
        strategy = HybridFabricStrategy()

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
        assert len(combined) == 4

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
        assert 1 in memory_ids  # From both
        assert 2 in memory_ids  # From vector
        assert 3 in memory_ids  # From vector
        assert 4 in memory_ids  # From chunks

        # Results should be sorted by relevance_score
        scores = [r.get("relevance_score", 0) for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_extract_simple_keywords(self):
        """Test the _extract_simple_keywords method directly."""
        strategy = HybridFabricStrategy()

        # Test with simple text
        text = "This is a test with some keywords like extraction and analysis."
        keywords = strategy._extract_simple_keywords(text)

        # Should extract words longer than 3 chars, excluding stopwords
        assert "test" in keywords
        assert "keywords" in keywords
        assert "extraction" in keywords
        assert "analysis" in keywords

        # Should not include stopwords and short words
        assert "this" not in keywords
        assert "is" not in keywords
        assert "a" not in keywords
        assert "with" not in keywords
        assert "and" not in keywords

        # Test with empty text
        assert strategy._extract_simple_keywords("") == []

    def test_retrieve_two_stage(self, memory_store, query_embedding, base_context):
        """Test the retrieve_two_stage method directly."""
        strategy = HybridFabricStrategy(memory_store=memory_store)
        strategy.initialize(
            {
                "confidence_threshold": 0.0,
                "use_two_stage": True,
                "first_stage_k": 10,
                "first_stage_threshold_factor": 0.7,
            }
        )

        # Make memory store searchable
        memory_store.search_by_vector = MagicMock()
        memory_store.search_by_vector.return_value = [
            {"memory_id": 0, "relevance_score": 0.9, "content": "Memory 0"},
            {"memory_id": 1, "relevance_score": 0.8, "content": "Memory 1"},
            {"memory_id": 2, "relevance_score": 0.7, "content": "Memory 2"},
        ]

        memory_store.search_by_keywords = MagicMock()
        memory_store.search_by_keywords.return_value = [
            {"memory_id": 3, "relevance_score": 0.75, "content": "Memory 3"},
            {"memory_id": 1, "relevance_score": 0.65, "content": "Memory 1"},
        ]

        # Add important keywords to context
        context = base_context.copy()
        context["important_keywords"] = ["memory", "test"]

        # Mock post-processing methods
        with patch.object(strategy, "_apply_keyword_boosting") as mock_keyword:
            mock_keyword.side_effect = lambda results, _: results

            with patch.object(strategy, "_check_semantic_coherence") as mock_coherence:
                mock_coherence.side_effect = lambda results, _: results

                with patch.object(strategy, "_enhance_with_associative_context") as mock_assoc:
                    mock_assoc.side_effect = lambda results: results

                    # Call the method
                    results = strategy.retrieve_two_stage(
                        query_embedding=query_embedding,
                        top_k=3,
                        context=context,
                        query="test query",
                    )

                    # Check that both search methods were called
                    memory_store.search_by_vector.assert_called_once()
                    memory_store.search_by_keywords.assert_called_once()

                    # Check that post-processing methods were called
                    mock_keyword.assert_called_once()
                    mock_coherence.assert_called_once()
                    mock_assoc.assert_called_once()

                    # Should return candidate results
                    assert len(results) == 3  # top_k

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

        # Call the method
        keywords = ["keyword1", "keyword2"]
        boosted = strategy._apply_keyword_boosting(results, keywords)

        # Results should be boosted based on keyword matches
        # Result 1 has both keywords, should get highest boost
        assert boosted[0]["relevance_score"] > 0.8
        assert "keyword_boost" in boosted[0]
        assert boosted[0]["keyword_matches"] == 2

        # Result 2 has one keyword, should get medium boost
        assert boosted[1]["relevance_score"] > 0.7
        assert "keyword_boost" in boosted[1]
        assert boosted[1]["keyword_matches"] == 1

        # Result 3 has no keywords, should not be boosted
        assert boosted[2]["relevance_score"] == 0.6
        assert "keyword_boost" not in boosted[2]

    def test_check_semantic_coherence(self, query_embedding):
        """Test the _check_semantic_coherence method directly."""
        strategy = HybridFabricStrategy()

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

        # Should add coherence scores
        for result in checked:
            assert "coherence_score" in result

        # Result 3 should have lower coherence score than others
        assert checked[2]["coherence_score"] < checked[0]["coherence_score"]
        assert checked[2]["coherence_score"] < checked[1]["coherence_score"]

        # If coherence is very low, should have coherence_penalty
        if checked[2]["coherence_score"] < 0.3:
            assert "coherence_penalty" in checked[2]

    def test_enhance_with_associative_context(self):
        """Test the _enhance_with_associative_context method directly."""
        strategy = HybridFabricStrategy()

        # Create memory store
        memory_store = MagicMock()
        memory_store.get = MagicMock()
        memory_store.get.side_effect = lambda id: type(
            "Memory",
            (),
            {
                "id": id,
                "content": f"Memory content {id}",
                "metadata": {"content": f"Memory content {id}"},
            },
        )
        strategy.memory_store = memory_store

        # Create associative linker
        associative_linker = MagicMock()
        associative_linker.get_associative_links = MagicMock()
        associative_linker.get_associative_links.side_effect = lambda id: [
            (id + 1, 0.8),  # Link to memory id+1 with strength 0.8
            (id + 2, 0.6),  # Link to memory id+2 with strength 0.6
        ]
        strategy.associative_linker = associative_linker

        # Create test results
        results = [
            {"memory_id": 1, "relevance_score": 0.8, "content": "Content 1"},
            {"memory_id": 4, "relevance_score": 0.7, "content": "Content 4"},
        ]

        # Call the method
        enhanced = strategy._enhance_with_associative_context(results)

        # Should add associative memories
        assert len(enhanced) > len(results)

        # Should include the original memories
        assert any(r["memory_id"] == 1 for r in enhanced)
        assert any(r["memory_id"] == 4 for r in enhanced)

        # Should include linked memories
        assert any(r["memory_id"] == 2 for r in enhanced)  # Linked to 1
        assert any(r["memory_id"] == 3 for r in enhanced)  # Linked to 1
        assert any(r["memory_id"] == 5 for r in enhanced)  # Linked to 4
        assert any(r["memory_id"] == 6 for r in enhanced)  # Linked to 4

        # Linked memories should have associative_link flag
        linked_memories = [r for r in enhanced if r.get("associative_link")]
        assert len(linked_memories) > 0

        # Linked memories should have link_source and link_strength
        for linked in linked_memories:
            assert "link_source" in linked
            assert "link_strength" in linked

    def test_configure_two_stage(self):
        """Test the configure_two_stage method."""
        strategy = HybridFabricStrategy()

        # Initial values
        assert strategy.use_two_stage_by_default is True  # Default from class
        assert strategy.first_stage_k == 30  # Default from class
        assert strategy.first_stage_threshold_factor == 0.7  # Default from class

        # Configure with different values
        strategy.configure_two_stage(
            enable=False, first_stage_k=40, first_stage_threshold_factor=0.5
        )

        # Check that values were updated
        assert strategy.use_two_stage_by_default is False
        assert strategy.first_stage_k == 40
        assert strategy.first_stage_threshold_factor == 0.5
