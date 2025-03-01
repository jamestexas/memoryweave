"""
Example demonstrating migration from the legacy MemoryWeave architecture
to the new component-based architecture.

This example shows three migration approaches:
1. Using adapters to bridge legacy and new components
2. Creating equivalent components from scratch
3. Using the FeatureMigrator utility for automatic migration
"""

import numpy as np
from typing import List, Dict, Any

# Legacy imports
from memoryweave.core.contextual_memory import ContextualMemory

# New architecture imports
from memoryweave.interfaces.retrieval import QueryType, Query
from memoryweave.factory.memory import MemoryFactory
from memoryweave.factory.retrieval import RetrievalFactory
from memoryweave.factory.pipeline import PipelineFactory

# Adapter imports
from memoryweave.adapters.memory_adapter import LegacyMemoryAdapter, LegacyVectorStoreAdapter
from memoryweave.adapters.retrieval_adapter import LegacyRetrieverAdapter, NewToLegacyRetrieverAdapter
from memoryweave.adapters.pipeline_adapter import LegacyToPipelineAdapter, PipelineToLegacyAdapter
from memoryweave.adapters.component_migration import FeatureMigrator


def create_test_data(num_memories: int = 10) -> Dict[str, Any]:
    """Create test data for the example."""
    embeddings = []
    texts = []
    metadata = []
    
    # Generate test data
    for i in range(num_memories):
        # Create embedding vector
        embedding = np.random.rand(768)
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
    
    # Create a test query
    query_embedding = np.random.rand(768)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_text = "What's my favorite color?"
    
    return {
        "embeddings": embeddings,
        "texts": texts,
        "metadata": metadata,
        "query_embedding": query_embedding,
        "query_text": query_text
    }


def using_legacy_system(test_data: Dict[str, Any]) -> List[Any]:
    """Demonstrate using the legacy MemoryWeave system."""
    print("\n=== Using Legacy System ===")
    
    # Create legacy memory system
    memory = ContextualMemory(
        embedding_dim=768,
        max_memories=100,
        default_confidence_threshold=0.6,
        adaptive_retrieval=True
    )
    
    # Add test data
    for i in range(len(test_data["embeddings"])):
        memory.add_memory(
            test_data["embeddings"][i],
            test_data["texts"][i],
            test_data["metadata"][i]
        )
    
    print(f"Added {len(test_data['embeddings'])} memories to legacy system")
    
    # Retrieve memories
    results = memory.retrieve_memories(
        query_embedding=test_data["query_embedding"],
        top_k=3,
        activation_boost=True
    )
    
    print(f"Retrieved {len(results)} memories from legacy system")
    for i, (memory_idx, similarity, metadata) in enumerate(results):
        print(f"  {i+1}. Similarity: {similarity:.4f}, Text: {metadata['text'][:50]}...")
    
    return results


def migration_approach_1_adapters(legacy_memory, test_data: Dict[str, Any]) -> List[Any]:
    """Demonstrate migration approach 1: Using adapters."""
    print("\n=== Migration Approach 1: Using Adapters ===")
    
    # Create adapters for legacy memory
    memory_adapter = LegacyMemoryAdapter(legacy_memory)
    vector_adapter = LegacyVectorStoreAdapter(legacy_memory, memory_adapter)
    retriever_adapter = LegacyRetrieverAdapter(legacy_memory.memory_retriever, memory_adapter)
    
    print("Created adapters for legacy memory components")
    
    # Create new query components
    query_analyzer = RetrievalFactory.create_query_analyzer()
    query_adapter = RetrievalFactory.create_query_adapter()
    
    print("Created new query components")
    
    # Analyze query
    query_text = test_data["query_text"]
    query_type = query_analyzer.analyze(query_text)
    print(f"Query '{query_text}' analyzed as {query_type}")
    
    # Extract keywords
    keywords = query_analyzer.extract_keywords(query_text)
    print(f"Extracted keywords: {keywords}")
    
    # Create query
    query = Query(
        text=query_text,
        embedding=test_data["query_embedding"],
        query_type=query_type,
        extracted_keywords=keywords,
        extracted_entities=[]
    )
    
    # Get adapted parameters
    params = query_adapter.adapt_parameters(query)
    print(f"Adapted parameters: {params}")
    
    # Retrieve using adapter
    results = retriever_adapter.retrieve(test_data["query_embedding"], params)
    
    print(f"Retrieved {len(results)} memories using adapter")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['relevance_score']:.4f}, Text: {result['content'][:50]}...")
    
    return results


def migration_approach_2_new_components(test_data: Dict[str, Any]) -> List[Any]:
    """Demonstrate migration approach 2: Creating new components."""
    print("\n=== Migration Approach 2: Creating New Components ===")
    
    # Create memory components
    memory_store = MemoryFactory.create_memory_store({"max_memories": 100})
    vector_store = MemoryFactory.create_vector_store()
    activation_manager = MemoryFactory.create_activation_manager()
    
    print("Created new memory components")
    
    # Create retrieval components
    retrieval_strategy = RetrievalFactory.create_retrieval_strategy(
        'hybrid', memory_store, vector_store, activation_manager
    )
    query_analyzer = RetrievalFactory.create_query_analyzer()
    query_adapter = RetrievalFactory.create_query_adapter()
    
    print("Created new retrieval components")
    
    # Create pipeline
    pipeline_manager = PipelineFactory.create_pipeline_manager()
    pipeline_manager.register_component(memory_store)
    pipeline_manager.register_component(vector_store)
    pipeline_manager.register_component(activation_manager)
    pipeline_manager.register_component(retrieval_strategy)
    pipeline_manager.register_component(query_analyzer)
    pipeline_manager.register_component(query_adapter)
    
    retrieval_pipeline = pipeline_manager.create_pipeline(
        "retrieval_pipeline", 
        [query_analyzer.get_id(), query_adapter.get_id(), retrieval_strategy.get_id()]
    )
    
    print("Created retrieval pipeline")
    
    # Add test data
    for i in range(len(test_data["embeddings"])):
        memory_store.add(
            test_data["embeddings"][i],
            test_data["texts"][i],
            test_data["metadata"][i]
        )
    
    print(f"Added {len(test_data['embeddings'])} memories to new system")
    
    # Create query
    query = Query(
        text=test_data["query_text"],
        embedding=test_data["query_embedding"],
        query_type=QueryType.UNKNOWN,  # Will be set by analyzer in pipeline
        extracted_keywords=[],  # Will be set by analyzer in pipeline
        extracted_entities=[]
    )
    
    # Execute pipeline
    results = retrieval_pipeline.execute(query)
    
    print(f"Retrieved {len(results)} memories using new pipeline")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['relevance_score']:.4f}, Text: {result['content'][:50]}...")
    
    return results


def migration_approach_3_migrator(legacy_memory, test_data: Dict[str, Any]) -> List[Any]:
    """Demonstrate migration approach 3: Using the FeatureMigrator utility."""
    print("\n=== Migration Approach 3: Using FeatureMigrator ===")
    
    # Create migrator
    migrator = FeatureMigrator()
    
    # Migrate legacy system
    components = migrator.migrate_memory_system(legacy_memory)
    
    print(f"Migrated {len(components)} components from legacy system")
    
    # Create migration pipeline
    pipeline = migrator.create_migration_pipeline(components)
    
    print("Created migration pipeline")
    
    # Create query
    query = Query(
        text=test_data["query_text"],
        embedding=test_data["query_embedding"],
        query_type=QueryType.UNKNOWN,
        extracted_keywords=[],
        extracted_entities=[]
    )
    
    # Execute pipeline
    results = pipeline.execute(query)
    
    print(f"Retrieved {len(results)} memories using migration pipeline")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['relevance_score']:.4f}, Text: {result['content'][:50]}...")
    
    # Validate migration
    validation_results = migrator.validate_migration(
        legacy_memory.memory_retriever,
        pipeline,
        [test_data["query_embedding"]]
    )
    
    print("\nMigration Validation Results:")
    print(f"  Success count: {validation_results['success_count']}")
    print(f"  Failure count: {validation_results['failure_count']}")
    if 'avg_recall' in validation_results:
        print(f"  Average recall: {validation_results['avg_recall']:.4f}")
        print(f"  Average precision: {validation_results['avg_precision']:.4f}")
    
    return results


def main():
    """Run the example."""
    # Create test data
    test_data = create_test_data(num_memories=10)
    
    # Run legacy system
    legacy_memory = ContextualMemory(
        embedding_dim=768,
        max_memories=100,
        default_confidence_threshold=0.6,
        adaptive_retrieval=True
    )
    
    # Add test data to legacy memory
    for i in range(len(test_data["embeddings"])):
        legacy_memory.add_memory(
            test_data["embeddings"][i],
            test_data["texts"][i],
            test_data["metadata"][i]
        )
    
    legacy_results = using_legacy_system(test_data)
    
    # Run migration approaches
    adapter_results = migration_approach_1_adapters(legacy_memory, test_data)
    new_component_results = migration_approach_2_new_components(test_data)
    migrator_results = migration_approach_3_migrator(legacy_memory, test_data)
    
    print("\n=== Summary ===")
    print(f"Legacy system:                     {len(legacy_results)} results")
    print(f"Migration approach 1 (adapters):   {len(adapter_results)} results")
    print(f"Migration approach 2 (new):        {len(new_component_results)} results")
    print(f"Migration approach 3 (migrator):   {len(migrator_results)} results")


if __name__ == "__main__":
    main()