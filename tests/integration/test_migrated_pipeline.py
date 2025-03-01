"""
Integration tests for the migrated MemoryWeave pipeline.

These tests verify that the new component architecture can be integrated
into a complete, working pipeline that provides the same functionality
as the legacy system.
"""

import pytest
import numpy as np
from typing import List, Tuple

from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.interfaces.retrieval import QueryType, Query, RetrievalParameters
from memoryweave.factory.memory import MemoryFactory
from memoryweave.factory.retrieval import RetrievalFactory
from memoryweave.factory.pipeline import PipelineFactory, PipelineManager
from memoryweave.adapters.component_migration import FeatureMigrator


class TestMigratedPipeline:
    """Test suite for verifying the integrated migrated pipeline."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data for the integrations tests."""
        # Create test embeddings
        embeddings = []
        texts = []
        metadata = []
        
        # Generate some test data
        for i in range(10):
            # Create embedding vector with controlled similarity properties
            embedding = np.zeros(768)
            embedding[0] = 0.1 * i  # Makes similarity predictable
            embedding[1] = 0.2 * i
            embedding[2] = 0.3 * i
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            text = f"This is test memory {i} with some content for testing retrieval"
            if i % 2 == 0:
                text += " and it mentions personal information about favorite colors."
            else:
                text += " and it contains factual information about history."
                
            meta = {
                "source": "test",
                "importance": 0.5 + 0.05 * i,
                "created_at": 1672531200 + i * 3600  # Jan 1, 2023 + i hours
            }
            
            embeddings.append(embedding)
            texts.append(text)
            metadata.append(meta)
        
        # Create test queries
        queries = []
        # Personal query
        personal_query = np.zeros(768)
        personal_query[0] = 0.1  # Similar to memory 1
        personal_query = personal_query / np.linalg.norm(personal_query)
        queries.append(("What's my favorite color?", personal_query, QueryType.PERSONAL))
        
        # Factual query
        factual_query = np.zeros(768)
        factual_query[0] = 0.9  # Similar to memory 9
        factual_query = factual_query / np.linalg.norm(factual_query)
        queries.append(("Tell me about history.", factual_query, QueryType.FACTUAL))
        
        return {
            "embeddings": embeddings,
            "texts": texts,
            "metadata": metadata,
            "queries": queries
        }
    
    @pytest.fixture
    def legacy_memory(self, test_data):
        """Create and populate a legacy ContextualMemory."""
        memory = ContextualMemory(
            embedding_dim=768,
            max_memories=100,
            use_art_clustering=True
        )
        
        # Add test data
        for i in range(len(test_data["embeddings"])):
            memory.add_memory(
                test_data["embeddings"][i],
                test_data["texts"][i],
                test_data["metadata"][i]
            )
            
        return memory
    
    @pytest.fixture
    def migrated_pipeline(self, test_data):
        """Create and populate a migrated pipeline."""
        # Create memory components
        memory_store = MemoryFactory.create_memory_store({"max_memories": 100})
        vector_store = MemoryFactory.create_vector_store()
        activation_manager = MemoryFactory.create_activation_manager()
        
        # Create retrieval components
        similarity_strategy = RetrievalFactory.create_retrieval_strategy(
            'similarity', memory_store, vector_store
        )
        hybrid_strategy = RetrievalFactory.create_retrieval_strategy(
            'hybrid', memory_store, vector_store, activation_manager
        )
        two_stage_strategy = RetrievalFactory.create_retrieval_strategy(
            'two_stage', memory_store, vector_store, activation_manager
        )
        query_analyzer = RetrievalFactory.create_query_analyzer()
        query_adapter = RetrievalFactory.create_query_adapter()
        
        # Create pipeline manager and register components
        pipeline_manager = PipelineFactory.create_pipeline_manager()
        pipeline_manager.register_component(memory_store)
        pipeline_manager.register_component(vector_store)
        pipeline_manager.register_component(activation_manager)
        pipeline_manager.register_component(similarity_strategy)
        pipeline_manager.register_component(hybrid_strategy)
        pipeline_manager.register_component(two_stage_strategy)
        pipeline_manager.register_component(query_analyzer)
        pipeline_manager.register_component(query_adapter)
        
        # Create pipelines for different query types
        factual_pipeline = pipeline_manager.create_pipeline(
            "factual_pipeline", 
            [query_adapter.get_id(), similarity_strategy.get_id()]
        )
        
        personal_pipeline = pipeline_manager.create_pipeline(
            "personal_pipeline", 
            [query_adapter.get_id(), hybrid_strategy.get_id()]
        )
        
        advanced_pipeline = pipeline_manager.create_pipeline(
            "advanced_pipeline", 
            [query_adapter.get_id(), two_stage_strategy.get_id()]
        )
        
        # Add test data
        for i in range(len(test_data["embeddings"])):
            memory_store.add(
                test_data["embeddings"][i],
                test_data["texts"][i],
                test_data["metadata"][i]
            )
        
        return {
            "memory_store": memory_store,
            "vector_store": vector_store,
            "activation_manager": activation_manager,
            "query_analyzer": query_analyzer,
            "query_adapter": query_adapter,
            "factual_pipeline": factual_pipeline,
            "personal_pipeline": personal_pipeline,
            "advanced_pipeline": advanced_pipeline,
            "pipeline_manager": pipeline_manager
        }
    
    def test_basic_retrieval(self, legacy_memory, migrated_pipeline, test_data):
        """Test basic retrieval functionality of both systems."""
        # Get components from migrated pipeline
        memory_store = migrated_pipeline["memory_store"]
        vector_store = migrated_pipeline["vector_store"]
        query_analyzer = migrated_pipeline["query_analyzer"]
        
        # Create a simple query
        query_idx = 0
        query_text, query_embedding, _ = test_data["queries"][query_idx]
        
        # Get results from legacy system
        legacy_results = legacy_memory.retrieve_memories(
            query_embedding=query_embedding,
            top_k=3
        )
        
        # Create query for new system
        query_type = query_analyzer.analyze(query_text)
        query = Query(
            text=query_text,
            embedding=query_embedding,
            query_type=query_type,
            extracted_keywords=query_analyzer.extract_keywords(query_text),
            extracted_entities=[]
        )
        
        # Get results from the appropriate pipeline
        if query_type == QueryType.PERSONAL:
            migrated_results = migrated_pipeline["personal_pipeline"].execute(query)
        elif query_type == QueryType.FACTUAL:
            migrated_results = migrated_pipeline["factual_pipeline"].execute(query)
        else:
            migrated_results = migrated_pipeline["advanced_pipeline"].execute(query)
        
        # Compare result count
        assert len(migrated_results) == len(legacy_results), \
            "Migrated pipeline should return the same number of results as legacy system"
        
        # Check that results contain similar content (not necessarily identical order)
        legacy_content = {result[2]["text"] for result in legacy_results}
        migrated_content = {result["content"] for result in migrated_results}
        
        # Allow for some differences (since both systems might have slightly different
        # sorting for very similar items), but ensure substantial overlap
        overlap = legacy_content.intersection(migrated_content)
        similarity_ratio = len(overlap) / len(legacy_content)
        
        assert similarity_ratio >= 0.5, \
            "Results from migrated pipeline should substantially overlap with legacy results"
    
    def test_query_type_adaptation(self, legacy_memory, migrated_pipeline, test_data):
        """Test query type adaptation in both systems."""
        # Get components
        query_analyzer = migrated_pipeline["query_analyzer"]
        query_adapter = migrated_pipeline["query_adapter"]
        
        # Test with different query types
        for query_text, query_embedding, expected_type in test_data["queries"]:
            # Analyze query
            query_type = query_analyzer.analyze(query_text)
            
            # Check query type is correct
            assert query_type == expected_type, \
                f"Query '{query_text}' should be analyzed as {expected_type}"
            
            # Create query
            query = Query(
                text=query_text,
                embedding=query_embedding,
                query_type=query_type,
                extracted_keywords=query_analyzer.extract_keywords(query_text),
                extracted_entities=[]
            )
            
            # Get adapted parameters
            params = query_adapter.adapt_parameters(query)
            
            # Verify parameters are adapted based on query type
            if query_type == QueryType.PERSONAL:
                # Personal queries should have lower similarity threshold
                assert params["similarity_threshold"] <= 0.7
                # And higher recency bias
                assert params.get("recency_bias", 0) >= 0.3
            elif query_type == QueryType.FACTUAL:
                # Factual queries should have higher similarity threshold
                assert params["similarity_threshold"] >= 0.7
                # And lower recency bias
                assert params.get("recency_bias", 1.0) <= 0.2
    
    def test_migrator_utility(self, legacy_memory, test_data):
        """Test the FeatureMigrator utility."""
        # Create migrator
        migrator = FeatureMigrator()
        
        # Migrate legacy system
        components = migrator.migrate_memory_system(legacy_memory)
        
        # Create migration pipeline
        pipeline = migrator.create_migration_pipeline(components)
        
        # Prepare test queries for validation
        test_queries = []
        for _, embedding, _ in test_data["queries"]:
            test_queries.append(embedding)
        
        # Validate migration
        validation_results = migrator.validate_migration(
            legacy_memory.memory_retriever,
            pipeline,
            test_queries
        )
        
        # Check validation metrics
        assert validation_results["total_queries"] == len(test_queries)
        
        # We don't expect perfect recall/precision due to different sorting algorithms
        # and implementation details, but should be reasonably close
        if "avg_recall" in validation_results:
            assert validation_results["avg_recall"] >= 0.5, \
                "Migration should have reasonable recall"
            assert validation_results["avg_precision"] >= 0.5, \
                "Migration should have reasonable precision"