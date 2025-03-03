"""
Integration tests for the migrated MemoryWeave pipeline.

These tests verify that the new component architecture can be integrated
into a complete, working pipeline that provides the same functionality
as the legacy system with well-defined, consistent behavior.
"""

import pytest
import numpy as np
from typing import List, Tuple, Dict, Any

from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.interfaces.retrieval import QueryType, Query, RetrievalParameters
from memoryweave.factory.memory import MemoryFactory
from memoryweave.factory.retrieval import RetrievalFactory
from memoryweave.factory.pipeline import PipelineFactory, PipelineManager
from memoryweave.adapters.component_migration import FeatureMigrator

from tests.utils.test_fixtures import (
    create_test_embedding,
    verify_retrieval_results,
    assert_specific_difference,
)


class TestMigratedPipeline:
    """Test suite for verifying the integrated migrated pipeline with consistent behavior."""

    def setup_predictable_test_data(self, embedding_dim: int = 768) -> Dict[str, Any]:
        """
        Create predictable test data for consistent migration tests.

        Uses deterministic embedding patterns for different memory categories
        and query types, ensuring consistent test results.

        Args:
            embedding_dim: Dimension of embeddings to create

        Returns:
            Dictionary containing embeddings, texts, metadata, and queries
        """
        # Create memory embeddings with predictable patterns
        embeddings = []
        texts = []
        metadata = []

        # Define category patterns
        categories = ["personal", "factual", "event", "concept", "opinion"]

        # Generate test memories with predictable patterns
        for i in range(15):
            # Determine category based on index
            cat_idx = i % len(categories)
            category = categories[cat_idx]

            # Create embedding with distinctive pattern for this category
            embedding = np.zeros(embedding_dim)
            embedding[cat_idx] = 0.9  # Primary category signal

            # Add secondary signals to create specific patterns
            embedding[(cat_idx + 1) % len(categories)] = 0.3
            embedding[(cat_idx + 2) % len(categories)] = 0.1

            # Add position information for memory ordering
            pos_idx = 100 + i
            embedding[pos_idx % embedding_dim] = 0.05 * (i + 1)

            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)

            # Create text with identifiable content
            if category == "personal":
                text = f"Memory {i}: My favorite color is {['blue', 'red', 'green'][i % 3]} and I enjoy it."
            elif category == "factual":
                text = f"Memory {i}: The capital of {['France', 'Japan', 'Brazil'][i % 3]} is {['Paris', 'Tokyo', 'Brasília'][i % 3]}."
            elif category == "event":
                text = f"Memory {i}: Yesterday I {['went to the park', 'had dinner with friends', 'watched a movie'][i % 3]}."
            elif category == "concept":
                text = f"Memory {i}: {['Democracy', 'Philosophy', 'Psychology'][i % 3]} is an important field of study."
            else:  # opinion
                text = f"Memory {i}: I think {['artificial intelligence', 'renewable energy', 'space exploration'][i % 3]} will change the future."

            # Create metadata with consistent structure
            meta = {
                "source": "test",
                "importance": 0.5 + (0.05 * i),
                "created_at": 1672531200 + (i * 3600),  # Jan 1, 2023 + i hours
                "category": category,
                "index": i,
            }

            # Add to collections
            embeddings.append(embedding)
            texts.append(text)
            metadata.append(meta)

        # Create test queries for multiple query types
        queries = []

        # Personal query about favorite color
        personal_query_text = "What's my favorite color?"
        personal_query_embedding = np.zeros(embedding_dim)
        personal_query_embedding[0] = 0.9  # Match personal category
        personal_query_embedding = personal_query_embedding / np.linalg.norm(
            personal_query_embedding
        )
        queries.append((personal_query_text, personal_query_embedding, QueryType.PERSONAL))

        # Factual query about capitals
        factual_query_text = "What is the capital of France?"
        factual_query_embedding = np.zeros(embedding_dim)
        factual_query_embedding[1] = 0.9  # Match factual category
        factual_query_embedding = factual_query_embedding / np.linalg.norm(factual_query_embedding)
        queries.append((factual_query_text, factual_query_embedding, QueryType.FACTUAL))

        # Conceptual query
        concept_query_text = "Tell me about philosophy."
        concept_query_embedding = np.zeros(embedding_dim)
        concept_query_embedding[3] = 0.9  # Match concept category
        concept_query_embedding = concept_query_embedding / np.linalg.norm(concept_query_embedding)
        queries.append((concept_query_text, concept_query_embedding, QueryType.CONCEPTUAL))

        return {"embeddings": embeddings, "texts": texts, "metadata": metadata, "queries": queries}

    @pytest.fixture
    def test_data(self):
        """Create predictable test data for integration tests."""
        return self.setup_predictable_test_data(embedding_dim=768)

    @pytest.fixture
    def legacy_memory(self, test_data):
        """Create and populate a legacy ContextualMemory with deterministic data."""
        memory = ContextualMemory(embedding_dim=768, max_memories=100, use_art_clustering=True)

        # Add test data with explicit metadata to ensure consistency
        for i in range(len(test_data["embeddings"])):
            memory.add_memory(
                test_data["embeddings"][i], test_data["texts"][i], test_data["metadata"][i]
            )

        return memory

    @pytest.fixture
    def migrated_pipeline(self, test_data):
        """Create and populate a migrated pipeline with consistent component structure."""
        # Create memory components with explicit parameters
        memory_store = MemoryFactory.create_memory_store({"max_memories": 100})
        vector_store = MemoryFactory.create_vector_store(
            {}
        )  # No need to specify embedding_dim for vector_store
        activation_manager = MemoryFactory.create_activation_manager({"decay_rate": 0.01})

        # Create retrieval components with explicit parameters
        similarity_strategy = RetrievalFactory.create_retrieval_strategy(
            "similarity",
            memory_store,
            vector_store,
            {"confidence_threshold": 0.3, "activation_boost": True},
        )

        hybrid_strategy = RetrievalFactory.create_retrieval_strategy(
            "hybrid",
            memory_store,
            vector_store,
            activation_manager,
            {"relevance_weight": 0.7, "recency_weight": 0.3},
        )

        two_stage_strategy = RetrievalFactory.create_retrieval_strategy(
            "two_stage",
            memory_store,
            vector_store,
            activation_manager,
            {"first_stage_k": 10, "first_stage_threshold_factor": 0.7},
        )

        # Create analysis components with explicit parameters
        query_analyzer = RetrievalFactory.create_query_analyzer(
            {"personal_keywords": ["my", "me", "i", "favorite"]}
        )

        query_adapter = RetrievalFactory.create_query_adapter(
            {"personal_threshold": 0.6, "factual_threshold": 0.7}
        )

        # Create pipeline manager and register all components
        pipeline_manager = PipelineFactory.create_pipeline_manager()
        for component in [
            memory_store,
            vector_store,
            activation_manager,
            similarity_strategy,
            hybrid_strategy,
            two_stage_strategy,
            query_analyzer,
            query_adapter,
        ]:
            pipeline_manager.register_component(component)

        # Create specialized pipelines for different query types
        factual_pipeline = pipeline_manager.create_pipeline(
            "factual_pipeline", [query_adapter.get_id(), similarity_strategy.get_id()]
        )

        personal_pipeline = pipeline_manager.create_pipeline(
            "personal_pipeline", [query_adapter.get_id(), hybrid_strategy.get_id()]
        )

        advanced_pipeline = pipeline_manager.create_pipeline(
            "advanced_pipeline", [query_adapter.get_id(), two_stage_strategy.get_id()]
        )

        # Add test data with explicit metadata to ensure consistency
        for i in range(len(test_data["embeddings"])):
            memory_store.add(
                test_data["embeddings"][i], test_data["texts"][i], test_data["metadata"][i]
            )

        # Return all components for testing
        return {
            "memory_store": memory_store,
            "vector_store": vector_store,
            "activation_manager": activation_manager,
            "query_analyzer": query_analyzer,
            "query_adapter": query_adapter,
            "factual_pipeline": factual_pipeline,
            "personal_pipeline": personal_pipeline,
            "advanced_pipeline": advanced_pipeline,
            "pipeline_manager": pipeline_manager,
        }

    def test_basic_retrieval_functionality(self, legacy_memory, migrated_pipeline, test_data):
        """Test that both legacy and migrated systems can retrieve relevant memories."""
        # Create a simple query with predictable results
        query_idx = 0  # Personal query about favorite color
        query_text, query_embedding, expected_type = test_data["queries"][query_idx]

        # Get results from legacy system
        legacy_results = legacy_memory.retrieve_memories(
            query_embedding=query_embedding, top_k=3, confidence_threshold=0.3
        )

        # Create query for new system with expected structure
        query = Query(
            text=query_text,
            embedding=query_embedding,
            query_type=expected_type,
            extracted_keywords=set(["favorite", "color"]),
            extracted_entities=[],
        )

        # Get results from the personal pipeline
        migrated_results = migrated_pipeline["personal_pipeline"].execute(query)

        # Verify both systems returned results
        assert len(legacy_results) > 0, "Legacy system should return results"
        assert len(migrated_results) > 0, "Migrated system should return results"

        # Extract content for comparison
        legacy_content = [result[2].get("text", "") for result in legacy_results]
        migrated_content = [result.get("content", "") for result in migrated_results]

        # Check that expected content is present in results
        # For personal query about color, we expect color-related memories
        color_found_legacy = any("color" in content.lower() for content in legacy_content)
        color_found_migrated = any("color" in content.lower() for content in migrated_content)

        assert color_found_legacy, "Legacy system should return color-related memories"
        assert color_found_migrated, "Migrated system should return color-related memories"

        # Log results for debugging
        print("\nLegacy results:")
        for i, result in enumerate(legacy_results):
            print(f"  {i + 1}. {result[2].get('text', '')} (score: {result[1]:.4f})")

        print("\nMigrated results:")
        for i, result in enumerate(migrated_results):
            print(
                f"  {i + 1}. {result.get('content', '')} (score: {result.get('relevance_score', 0):.4f})"
            )

    def test_query_type_specific_pipelines(self, migrated_pipeline, test_data):
        """Test that different pipelines are optimized for different query types."""
        # Get pipeline components
        query_analyzer = migrated_pipeline["query_analyzer"]
        personal_pipeline = migrated_pipeline["personal_pipeline"]
        factual_pipeline = migrated_pipeline["factual_pipeline"]

        # Test with multiple query types
        results_by_type = {}

        for query_text, query_embedding, expected_type in test_data["queries"]:
            # Create query object
            query = Query(
                text=query_text,
                embedding=query_embedding,
                query_type=expected_type,
                extracted_keywords=query_analyzer.extract_keywords(query_text),
                extracted_entities=[],
            )

            # Get results from type-specific pipeline
            if expected_type == QueryType.PERSONAL:
                results = personal_pipeline.execute(query)
                pipeline_name = "personal"
            else:
                results = factual_pipeline.execute(query)
                pipeline_name = "factual"

            # Store results by query type
            results_by_type[expected_type.name] = {
                "query": query_text,
                "results": results,
                "pipeline": pipeline_name,
            }

        # Verify each pipeline returns appropriate results for its query type
        for query_type, result_info in results_by_type.items():
            # Check result count
            assert len(result_info["results"]) > 0, (
                f"{result_info['pipeline']} pipeline should return results for {query_type} query"
            )

            # Check result relevance - should have at least one high relevance result
            high_relevance = any(r.get("relevance_score", 0) > 0.5 for r in result_info["results"])
            assert high_relevance, (
                f"{result_info['pipeline']} pipeline should return high relevance results for {query_type} query"
            )

            # Print results for debugging
            print(f"\n{query_type} query results from {result_info['pipeline']} pipeline:")
            print(f"Query: {result_info['query']}")
            for i, r in enumerate(result_info["results"][:3]):
                print(
                    f"  {i + 1}. {r.get('content', '')} (score: {r.get('relevance_score', 0):.4f})"
                )

    def test_query_adaptation_parameters(self, migrated_pipeline, test_data):
        """Test that query adaptation produces appropriate parameter adjustments."""
        # Get adapter component
        query_adapter = migrated_pipeline["query_adapter"]

        # Test adaptation for different query types
        adaptation_results = {}

        for query_text, query_embedding, expected_type in test_data["queries"]:
            # Create query object
            query = Query(
                text=query_text,
                embedding=query_embedding,
                query_type=expected_type,
                extracted_keywords=set(query_text.lower().split()),
                extracted_entities=[],
            )

            # Get adapted parameters
            params = query_adapter.adapt_parameters(query)

            # Store parameters by query type
            adaptation_results[expected_type.name] = {"query": query_text, "parameters": params}

        # Verify appropriate parameter adjustments for each query type

        # Personal queries should prioritize recency and have lower thresholds
        personal_params = adaptation_results.get("PERSONAL", {}).get("parameters", {})
        assert "similarity_threshold" in personal_params, (
            "Missing similarity_threshold for personal query"
        )
        assert "recency_bias" in personal_params, "Missing recency_bias for personal query"

        # Factual queries should prioritize similarity and have higher thresholds
        factual_params = adaptation_results.get("FACTUAL", {}).get("parameters", {})
        assert "similarity_threshold" in factual_params, (
            "Missing similarity_threshold for factual query"
        )

        # Compare parameters between query types
        if "PERSONAL" in adaptation_results and "FACTUAL" in adaptation_results:
            personal_threshold = personal_params.get("similarity_threshold", 1.0)
            factual_threshold = factual_params.get("similarity_threshold", 0.0)

            # Personal should have lower threshold than factual
            assert personal_threshold <= factual_threshold, (
                f"Personal threshold ({personal_threshold}) should be <= factual threshold ({factual_threshold})"
            )

            # Recency bias should be higher for personal queries
            personal_recency = personal_params.get("recency_bias", 0.0)
            factual_recency = factual_params.get("recency_bias", 0.0)
            assert personal_recency >= factual_recency, (
                f"Personal recency bias ({personal_recency}) should be >= factual recency bias ({factual_recency})"
            )

        # Print adaptation results for debugging
        print("\nQuery adaptation results:")
        for query_type, result in adaptation_results.items():
            print(f"\n{query_type} query: {result['query']}")
            print(f"Adapted parameters: {result['parameters']}")

    def test_migration_consistency(self, legacy_memory, test_data):
        """Test that the FeatureMigrator produces consistent migration results."""
        # Create migrator
        migrator = FeatureMigrator()

        # Run migration twice with same inputs
        # First migration
        components1 = migrator.migrate_memory_system(legacy_memory)
        pipeline1 = migrator.create_migration_pipeline(components1)

        # Second migration
        components2 = migrator.migrate_memory_system(legacy_memory)
        pipeline2 = migrator.create_migration_pipeline(components2)

        # Prepare test queries
        test_queries = [embedding for _, embedding, _ in test_data["queries"]]

        # Get results from both pipelines for same query
        query = test_queries[0]  # Use first query (personal)

        results1 = pipeline1.execute({"query_embedding": query})
        results2 = pipeline2.execute({"query_embedding": query})

        # Compare result structures - should be identical
        assert len(results1) == len(results2), (
            f"Migration results should have same length: {len(results1)} vs {len(results2)}"
        )

        # Check for identical memory IDs in same order
        ids1 = [r.get("memory_id") for r in results1]
        ids2 = [r.get("memory_id") for r in results2]

        assert ids1 == ids2, "Migration should produce identical results with same inputs"

        # Get validation metrics
        metrics1 = migrator.validate_migration(
            legacy_memory.memory_retriever, pipeline1, test_queries
        )

        # Print metrics for debugging
        print("\nMigration validation metrics:")
        for key, value in metrics1.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Verify reasonable recall and precision
        assert metrics1["avg_recall"] >= 0.5, (
            f"Migration recall is too low: {metrics1['avg_recall']:.4f}"
        )
        assert metrics1["avg_precision"] >= 0.5, (
            f"Migration precision is too low: {metrics1['avg_precision']:.4f}"
        )
