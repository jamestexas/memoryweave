"""
Integration tests for category-based retrieval in the pipeline.
"""

import numpy as np
import pytest

from memoryweave.components.factory import create_memory_system
from tests.utils.test_fixtures import create_test_embedding


class TestCategoryRetrieval:
    """Integration tests for category-based retrieval."""

    def test_category_manager_integration(self):
        """Test CategoryManager integration with core memory system."""
        # Create memory system with default config
        memory_system = create_memory_system()
        memory = memory_system["memory"]
        category_manager = memory_system["category_manager"]
        category_adapter = memory_system["category_adapter"]

        # Verify components were created
        assert memory is not None
        assert category_manager is not None
        assert category_adapter is not None

        # Verify that memory has a category manager
        assert hasattr(memory, "category_manager")
        assert memory.category_manager is not None

        # Add some test memories
        embeddings = [
            create_test_embedding("cat memory", 768),
            create_test_embedding("dog memory", 768),
            create_test_embedding("weather memory", 768),
        ]

        for i, embedding in enumerate(embeddings):
            memory.add_memory(
                embedding, f"Test memory {i}", {"content": f"Memory content {i}", "index": i}
            )

            # Test assigning to category via the adapter
            category_idx = category_adapter.assign_to_category(embedding)
            category_adapter.add_memory_category_mapping(i, category_idx)

        # Verify we can retrieve memory-category mappings
        category0 = category_adapter.get_category_for_memory(0)
        assert isinstance(category0, int)

        memories_in_cat = category_adapter.get_memories_for_category(category0)
        assert len(memories_in_cat) > 0

        # Test category similarity calculation
        test_query = create_test_embedding("cat query", 768)
        similarities = category_adapter.get_category_similarities(test_query)
        assert similarities is not None
        assert len(similarities) > 0

    def test_category_retrieval_in_pipeline(self):
        """Test category retrieval in a complete pipeline."""
        # This test requires the QueryAnalyzer component which may not be available
        # in all environments, so we'll skip it
        try:
            from memoryweave.components.query_analysis import QueryAnalyzer
        except ImportError:
            # Skip test if required dependencies are not available
            pytest.skip("QueryAnalyzer not available - required for this test")

        # Create memory system
        memory_system = create_memory_system(
            {
                "memory": {"embedding_dim": 128},
                "category": {"vigilance_threshold": 0.8, "dynamic_vigilance": True},
                "confidence_threshold": 0.3,
                "max_categories": 3,
                "category_selection_threshold": 0.5,
            }
        )

        memory = memory_system["memory"]
        manager = memory_system["manager"]
        category_retrieval = memory_system["category_retrieval"]

        # Add required components for pipeline
        query_analyzer = QueryAnalyzer()
        manager.register_component("query_analyzer", query_analyzer)

        # Configure a simple pipeline with just the category retrieval
        manager.build_pipeline(
            [
                {
                    "component": "category_retrieval",
                    "config": {
                        "confidence_threshold": 0.3,
                        "max_categories": 3,
                        "category_selection_threshold": 0.5,
                        "activation_boost": True,
                    },
                }
            ]
        )

        # Add test memories with clear category patterns
        categories = {
            "cat": np.array([1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 123),
            "dog": np.array([0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 123),
            "weather": np.array([0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 123),
            "food": np.array([0.0, 0.0, 0.0, 1.0, 0.0] + [0.0] * 123),
        }

        # Normalize embeddings
        for key in categories:
            categories[key] = categories[key] / np.linalg.norm(categories[key])

        # Add memories for each category
        memory_ids = {}
        for category, pattern in categories.items():
            ids = []
            for i in range(3):  # 3 memories per category
                # Add some noise to make it slightly different
                noise = np.random.randn(128) * 0.05
                embedding = pattern + noise
                embedding = embedding / np.linalg.norm(embedding)

                # Add to memory
                idx = memory.add_memory(
                    embedding,
                    f"{category} memory {i}",
                    {
                        "category": category,
                        "content": f"{category} content {i}",
                        "index": len(memory.memory_embeddings) - 1,
                    },
                )
                ids.append(idx)
            memory_ids[category] = ids

        # Define test query for each category
        queries = {
            "cat": {
                "embedding": categories["cat"] * 0.9 + np.random.randn(128) * 0.1,
                "text": "Tell me about cats",
            },
            "dog": {
                "embedding": categories["dog"] * 0.9 + np.random.randn(128) * 0.1,
                "text": "Information about dogs",
            },
            "weather": {
                "embedding": categories["weather"] * 0.9 + np.random.randn(128) * 0.1,
                "text": "What's the weather like?",
            },
            "food": {
                "embedding": categories["food"] * 0.9 + np.random.randn(128) * 0.1,
                "text": "What food do you like?",
            },
        }

        # Normalize query embeddings
        for category in queries:
            queries[category]["embedding"] = queries[category]["embedding"] / np.linalg.norm(
                queries[category]["embedding"]
            )

        # Test retrieval for each category
        for category, query_info in queries.items():
            query_text = query_info["text"]
            query_embedding = query_info["embedding"]

            # Execute the pipeline
            results = manager.execute_pipeline(
                {
                    "query": query_text,
                    "query_embedding": query_embedding,
                    "top_k": 3,
                }
            )

            # Check that results include expected memories
            assert "results" in results
            retrieved_results = results["results"]
            assert len(retrieved_results) > 0

            # Verify at least one memory from the target category is retrieved
            memory_indices = [result["memory_id"] for result in retrieved_results]
            target_indices = memory_ids[category]

            assert any(idx in target_indices for idx in memory_indices), (
                f"Expected at least one memory from category {category} to be retrieved"
            )

            # Check that category information is included in results
            assert "category_id" in retrieved_results[0]
            assert "category_similarity" in retrieved_results[0]

    def test_hybrid_category_pipeline(self):
        """Test the hybrid category pipeline which combines strategies."""
        # This test requires several components that may not be available
        # in all environments, so we'll skip it
        try:
            from memoryweave.components.post_processors import (
                KeywordBoostProcessor,
                SemanticCoherenceProcessor,
            )
            from memoryweave.components.query_adapter import QueryTypeAdapter
            from memoryweave.components.query_analysis import QueryAnalyzer
        except ImportError:
            # Skip test if required dependencies are not available
            pytest.skip("Required components not available for this test")

        # Create memory system
        memory_system = create_memory_system(
            {
                "memory": {"embedding_dim": 128},
                "category": {"vigilance_threshold": 0.75},
                "confidence_threshold": 0.2,
            }
        )

        memory = memory_system["memory"]
        manager = memory_system["manager"]
        category_retrieval = memory_system["category_retrieval"]

        # Register required components
        query_analyzer = QueryAnalyzer()
        query_adapter = QueryTypeAdapter()
        keyword_boost = KeywordBoostProcessor()
        coherence = SemanticCoherenceProcessor()

        manager.register_component("query_analyzer", query_analyzer)
        manager.register_component("query_adapter", query_adapter)
        manager.register_component("keyword_boost", keyword_boost)
        manager.register_component("coherence", coherence)
        manager.register_component("two_stage_retrieval", category_retrieval)

        # Configure a simple pipeline with just category retrieval
        manager.build_pipeline(
            [
                {
                    "component": "category_retrieval",
                    "config": {
                        "confidence_threshold": 0.25,
                        "max_categories": 3,
                        "category_selection_threshold": 0.5,
                        "activation_boost": True,
                    },
                },
                {
                    "component": "keyword_boost",
                    "config": {
                        "keyword_boost_weight": 0.5,
                    },
                },
            ]
        )

        # Add some test memories with clear patterns
        embeddings = [
            (
                create_test_embedding("cat content", 128),
                "All about cats",
                {"type": "animal", "category": "cat"},
            ),
            (
                create_test_embedding("more cat stuff", 128),
                "Cat behaviors",
                {"type": "animal", "category": "cat"},
            ),
            (
                create_test_embedding("dog content", 128),
                "Dogs are loyal",
                {"type": "animal", "category": "dog"},
            ),
            (
                create_test_embedding("weather today", 128),
                "Weather forecast",
                {"type": "weather", "category": "weather"},
            ),
            (
                create_test_embedding("food recipes", 128),
                "Cooking recipes",
                {"type": "food", "category": "food"},
            ),
        ]

        for i, (embedding, text, metadata) in enumerate(embeddings):
            metadata["index"] = i
            memory.add_memory(embedding, text, metadata)

        # Create a cat-related query
        cat_query = create_test_embedding("Tell me about cats", 128)

        # Execute the pipeline
        results = manager.execute_pipeline(
            {
                "query": "Tell me about cats",
                "query_embedding": cat_query,
                "top_k": 3,
                "important_keywords": {"cat", "cats", "tell"},
                "enable_semantic_coherence": True,
            }
        )

        # Check that results are returned
        assert "results" in results
        retrieved_results = results["results"]
        assert len(retrieved_results) > 0

        # Verify results contain cat-related memories
        cat_indices = [0, 1]  # Indices of cat memories
        retrieved_indices = [result["memory_id"] for result in retrieved_results]

        assert any(idx in cat_indices for idx in retrieved_indices), (
            f"Expected cat memories (indices {cat_indices}) in results, got {retrieved_indices}"
        )

    def test_category_consolidation(self):
        """Test category consolidation in a real pipeline."""
        # Create memory system with consolidation enabled
        memory_system = create_memory_system(
            {
                "memory": {"embedding_dim": 64},
                "category": {
                    "vigilance_threshold": 0.99,  # Start high to create many categories
                    "enable_category_consolidation": True,
                    "consolidation_threshold": 0.6,
                    "min_category_size": 1,
                    "consolidation_frequency": 5,
                },
            }
        )

        memory = memory_system["memory"]
        category_adapter = memory_system["category_adapter"]

        # Add memories with very similar patterns to force separate categories initially
        patterns = {
            "cat1": np.array([0.95, 0.05, 0.0, 0.0] + [0.0] * 60),
            "cat2": np.array([0.9, 0.1, 0.0, 0.0] + [0.0] * 60),
            "cat3": np.array([0.85, 0.15, 0.0, 0.0] + [0.0] * 60),
            "dog1": np.array([0.0, 0.0, 0.95, 0.05] + [0.0] * 60),
            "dog2": np.array([0.0, 0.0, 0.9, 0.1] + [0.0] * 60),
            "dog3": np.array([0.0, 0.0, 0.85, 0.15] + [0.0] * 60),
        }

        # Normalize patterns
        for key in patterns:
            patterns[key] = patterns[key] / np.linalg.norm(patterns[key])

        # Add memories to force creation of multiple initial categories
        for i, (name, pattern) in enumerate(patterns.items()):
            memory.add_memory(pattern, f"Memory {name}", {"name": name, "index": i})

            # Assign to category
            category_idx = category_adapter.assign_to_category(pattern)
            category_adapter.add_memory_category_mapping(i, category_idx)

        # Get statistics before consolidation
        stats_before = category_adapter.get_category_statistics()

        # Now lower vigilance and force consolidation
        # This simulates what would happen after many memories are added and
        # consolidation is triggered automatically
        num_categories = category_adapter.consolidate_categories(threshold=0.5)

        # Get statistics after consolidation
        stats_after = category_adapter.get_category_statistics()

        # Verify that consolidation reduced the number of categories
        assert stats_after["num_categories"] < stats_before["num_categories"]

        # The cat memories should be in one category and dog memories in another
        cat_indices = [0, 1, 2]  # Indices of cat memories
        dog_indices = [3, 4, 5]  # Indices of dog memories

        # Get category for first cat and dog memory
        cat_category = category_adapter.get_category_for_memory(0)
        dog_category = category_adapter.get_category_for_memory(3)

        # They should be in different categories
        assert cat_category != dog_category

        # Other cat memories should be in same category as first cat
        for idx in cat_indices[1:]:
            assert category_adapter.get_category_for_memory(idx) == cat_category

        # Other dog memories should be in same category as first dog
        for idx in dog_indices[1:]:
            assert category_adapter.get_category_for_memory(idx) == dog_category
