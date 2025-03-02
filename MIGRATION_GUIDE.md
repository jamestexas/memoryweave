# MemoryWeave Migration Guide

This guide provides instructions for migrating from the original monolithic MemoryWeave architecture to the new modular, component-based architecture.

## Migration Overview

The MemoryWeave system has been refactored to use a modular, component-based architecture. This migration guide will help you transition from the old monolithic architecture to the new component-based system.

### Benefits of Migration

- **Improved maintainability**: Smaller, focused components are easier to understand and maintain
- **Better testability**: Components can be tested in isolation
- **Increased extensibility**: New components can be added without modifying existing code
- **Clearer dependencies**: More explicit component relationships
- **Performance optimization**: Components can be optimized independently

## Key Changes

| Legacy Architecture | Component Architecture | Description |
|---------------------|------------------------|-------------|
| `ContextualMemory` | `MemoryManager` | Memory management functionality |
| `ContextualRetriever` | `Retriever` | Memory retrieval functionality |
| - | `SimilarityRetrievalStrategy` | Vector similarity retrieval |
| - | `TemporalRetrievalStrategy` | Time-based retrieval |
| - | `HybridRetrievalStrategy` | Combined similarity and temporal retrieval |
| - | `TwoStageRetrievalStrategy` | More sophisticated multi-stage retrieval |
| - | `QueryAnalyzer` | Query analysis and classification |
| - | `PersonalAttributeManager` | Personal attribute extraction and management |

## Migration Options

There are three migration paths available depending on your needs:

1. **Adapter-based migration**: Continue using the old interface with new components underneath
2. **Gradual component migration**: Selectively migrate components over time
3. **Full migration**: Fully adopt the new component architecture

## Option 1: Adapter-Based Migration

The adapter-based approach allows using the old interface with new components underneath. This is the simplest migration path with minimal code changes.

### Steps

1. **Replace imports**:

```python
# Old imports
from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.core.memory_retriever import MemoryRetriever

# New imports
from memoryweave.adapters.pipeline_adapter import PipelineToLegacyAdapter
from memoryweave.factory.memory import MemoryFactory
from memoryweave.factory.retrieval import RetrievalFactory
from memoryweave.factory.pipeline import PipelineFactory
```

2. **Create adapter**:

```python
# Create native components
memory_store = MemoryFactory.create_memory_store()
vector_store = MemoryFactory.create_vector_store()
activation_manager = MemoryFactory.create_activation_manager()
retrieval_strategy = RetrievalFactory.create_retrieval_strategy(
    'hybrid', memory_store, vector_store, activation_manager
)

# Create pipeline
pipeline_manager = PipelineFactory.create_pipeline_manager()
pipeline_manager.register_component(memory_store)
pipeline_manager.register_component(vector_store)
pipeline_manager.register_component(activation_manager)
pipeline_manager.register_component(retrieval_strategy)
pipeline = pipeline_manager.create_pipeline(
    "retrieval_pipeline", [retrieval_strategy.get_id()]
)

# Create adapter that provides the legacy interface
adapter = PipelineToLegacyAdapter(pipeline)

# Use with legacy interface
memory_idx = adapter.add_memory(embedding, text, metadata)
results = adapter.retrieve_for_context(query_embedding, top_k=5)
```

### Example

```python
# Replace this:
memory = ContextualMemory(embedding_dim=768, max_memories=1000)
memory.add_memory(embedding, text, metadata)
results = memory.retrieve_memories(query_embedding, top_k=5)

# With this:
memory_store = MemoryFactory.create_memory_store({'max_memories': 1000})
vector_store = MemoryFactory.create_vector_store()
activation_manager = MemoryFactory.create_activation_manager()
retrieval_strategy = RetrievalFactory.create_retrieval_strategy(
    'hybrid', memory_store, vector_store, activation_manager
)

pipeline_manager = PipelineFactory.create_pipeline_manager()
pipeline_manager.register_component(memory_store)
pipeline_manager.register_component(vector_store)
pipeline_manager.register_component(activation_manager)
pipeline_manager.register_component(retrieval_strategy)
pipeline = pipeline_manager.create_pipeline(
    "retrieval_pipeline", [retrieval_strategy.get_id()]
)

adapter = PipelineToLegacyAdapter(pipeline)
adapter.add_memory(embedding, text, metadata)
results = adapter.retrieve_for_context(query_embedding, top_k=5)
```

## Option 2: Gradual Component Migration

The gradual component migration approach allows you to selectively migrate components over time, combining old and new components.

### Steps

1. **Start with legacy memory**:

```python
from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.adapters.memory_adapter import LegacyMemoryAdapter
from memoryweave.adapters.retrieval_adapter import LegacyRetrieverAdapter

# Create legacy memory
legacy_memory = ContextualMemory(embedding_dim=768, max_memories=1000)

# Create adapters for legacy components
memory_adapter = LegacyMemoryAdapter(legacy_memory)
retriever_adapter = LegacyRetrieverAdapter(legacy_memory.memory_retriever, memory_adapter)
```

2. **Gradually replace components**:

```python
from memoryweave.factory.retrieval import RetrievalFactory

# Create new query components
query_analyzer = RetrievalFactory.create_query_analyzer()
query_adapter = RetrievalFactory.create_query_adapter()

# Use new query components with legacy retriever
def process_query(query_text, query_embedding):
    query_type = query_analyzer.analyze(query_text)
    parameters = query_adapter.adapt_parameters({
        'text': query_text,
        'embedding': query_embedding,
        'query_type': query_type,
        'extracted_keywords': query_analyzer.extract_keywords(query_text),
        'extracted_entities': query_analyzer.extract_entities(query_text)
    })
    
    # Use legacy retriever with new parameters
    return retriever_adapter.retrieve(query_embedding, parameters)
```

### Example

```python
# Create mix of old and new components
legacy_memory = ContextualMemory(embedding_dim=768, max_memories=1000)
memory_adapter = LegacyMemoryAdapter(legacy_memory)
vector_adapter = LegacyVectorStoreAdapter(legacy_memory, memory_adapter)

# Create new retrieval strategy that uses legacy memory
new_strategy = RetrievalFactory.create_retrieval_strategy(
    'two_stage', 
    memory_adapter,  # Use legacy memory via adapter 
    vector_adapter,  # Use legacy vector store via adapter
    None  # No activation manager
)

# Create a pipeline with the new strategy
pipeline_manager = PipelineFactory.create_pipeline_manager()
pipeline_manager.register_component(memory_adapter)
pipeline_manager.register_component(vector_adapter)
pipeline_manager.register_component(new_strategy)
pipeline = pipeline_manager.create_pipeline(
    "mixed_pipeline", [new_strategy.get_id()]
)

# Use the pipeline
query = Query(
    text="What is my favorite color?",
    embedding=query_embedding,
    query_type=QueryType.PERSONAL,
    extracted_keywords=["favorite", "color"],
    extracted_entities=[]
)
results = pipeline.execute(query)
```

## Option 3: Full Migration

The full migration approach involves completely adopting the new component architecture.

### Basic Migration Steps

1. **Update Imports**

```python
# Legacy imports
from memoryweave.core import ContextualMemory, ContextualRetriever

# New imports
from memoryweave.components import Retriever
from memoryweave.components.memory_manager import MemoryManager
```

2. **Replace Memory Creation**

```python
# Legacy code
memory = ContextualMemory(
    embedding_dim=768,
    max_memories=1000
)

# New code
memory_manager = MemoryManager(
    embedding_dim=768,
    max_capacity=1000
)
```

3. **Replace Retriever Creation**

```python
# Legacy code
retriever = ContextualRetriever(
    memory=memory,
    embedding_model=embedding_model,
    confidence_threshold=0.7,
    semantic_coherence_check=True
)

# New code
retriever = Retriever(
    memory=memory_manager,
    embedding_model=embedding_model
)
retriever.minimum_relevance = 0.7
retriever.configure_semantic_coherence(enable=True)
retriever.initialize_components()
```

4. **Replace Retrieval Calls**

```python
# Legacy code
memories = retriever.retrieve_for_context(
    query, 
    top_k=5
)

# New code
memories = retriever.retrieve(
    query, 
    top_k=5
)
```

### Advanced Configuration

The new architecture provides more fine-grained control over retrieval behavior:

```python
# Configure two-stage retrieval
retriever.configure_two_stage_retrieval(
    enable=True,
    first_stage_k=20,
    first_stage_threshold_factor=0.7
)

# Configure query adaptation
retriever.configure_query_type_adaptation(
    enable=True,
    adaptation_strength=1.0
)

# Configure dynamic thresholds
retriever.enable_dynamic_threshold_adjustment(
    enable=True,
    window_size=5
)
```

### Full Implementation Example

```python
from memoryweave.interfaces.retrieval import Query, QueryType
from memoryweave.factory.memory import MemoryFactory
from memoryweave.factory.retrieval import RetrievalFactory
from memoryweave.factory.pipeline import PipelineFactory

# Create memory components
memory_store = MemoryFactory.create_memory_store({'max_memories': 1000})
vector_store = MemoryFactory.create_vector_store()
activation_manager = MemoryFactory.create_activation_manager({'use_temporal_decay': True})

# Create different retrieval strategies
similarity_strategy = RetrievalFactory.create_retrieval_strategy(
    'similarity', memory_store, vector_store
)
temporal_strategy = RetrievalFactory.create_retrieval_strategy(
    'temporal', memory_store, None, activation_manager
)
hybrid_strategy = RetrievalFactory.create_retrieval_strategy(
    'hybrid', memory_store, vector_store, activation_manager
)
two_stage_strategy = RetrievalFactory.create_retrieval_strategy(
    'two_stage', memory_store, vector_store, activation_manager
)

# Create query processors
query_analyzer = RetrievalFactory.create_query_analyzer()
query_adapter = RetrievalFactory.create_query_adapter()

# Create pipeline manager
pipeline_manager = PipelineFactory.create_pipeline_manager()
pipeline_manager.register_component(memory_store)
pipeline_manager.register_component(vector_store)
pipeline_manager.register_component(activation_manager)
pipeline_manager.register_component(similarity_strategy)
pipeline_manager.register_component(temporal_strategy)
pipeline_manager.register_component(hybrid_strategy)
pipeline_manager.register_component(two_stage_strategy)
pipeline_manager.register_component(query_analyzer)
pipeline_manager.register_component(query_adapter)

# Create pipelines for different query types
factual_pipeline = pipeline_manager.create_pipeline(
    "factual_pipeline", 
    [query_analyzer.get_id(), query_adapter.get_id(), similarity_strategy.get_id()]
)

personal_pipeline = pipeline_manager.create_pipeline(
    "personal_pipeline", 
    [query_analyzer.get_id(), query_adapter.get_id(), hybrid_strategy.get_id()]
)

temporal_pipeline = pipeline_manager.create_pipeline(
    "temporal_pipeline", 
    [query_analyzer.get_id(), query_adapter.get_id(), temporal_strategy.get_id()]
)

complex_pipeline = pipeline_manager.create_pipeline(
    "complex_pipeline", 
    [query_analyzer.get_id(), query_adapter.get_id(), two_stage_strategy.get_id()]
)

# Use the appropriate pipeline based on query type
def process_query(query_text, query_embedding):
    # Analyze query type
    query_type = query_analyzer.analyze(query_text)
    
    # Create query object
    query = Query(
        text=query_text,
        embedding=query_embedding,
        query_type=query_type,
        extracted_keywords=query_analyzer.extract_keywords(query_text),
        extracted_entities=query_analyzer.extract_entities(query_text)
    )
    
    # Select pipeline based on query type
    if query_type == QueryType.FACTUAL:
        return factual_pipeline.execute(query)
    elif query_type == QueryType.PERSONAL:
        return personal_pipeline.execute(query)
    elif query_type == QueryType.TEMPORAL:
        return temporal_pipeline.execute(query)
    else:
        return complex_pipeline.execute(query)
```

## Using the Migration Utility

To simplify migration, MemoryWeave includes a `FeatureMigrator` utility that can automatically create new components equivalent to your legacy components.

```python
from memoryweave.adapters.component_migration import FeatureMigrator

# Create legacy memory
legacy_memory = ContextualMemory(embedding_dim=768, max_memories=1000)

# Use migrator to create equivalent components
migrator = FeatureMigrator()
components = migrator.migrate_memory_system(legacy_memory)

# Create a pipeline with the migrated components
pipeline = migrator.create_migration_pipeline(components)

# Validate that the migration was successful
test_queries = [...]  # List of test queries
validation_results = migrator.validate_migration(
    legacy_memory.memory_retriever, 
    pipeline,
    test_queries
)

print(f"Migration success rate: {validation_results['success_count']/validation_results['total_queries']:.2f}")
```

## Common Issues and Solutions

1. **Different Result Ordering**: The new architecture may return results in a slightly different order due to the more sophisticated ranking algorithms. This is generally an improvement but may require adjustments to downstream processing.

2. **Configuration Differences**: The new architecture requires explicit initialization after configuration changes. Always call `retriever.initialize_components()` after configuring.

3. **Return Format**: The new architecture uses a consistent return format for all retrieval strategies. Check the documentation for the exact structure.

4. **Advanced Features**: Some advanced features (memory decay, ART clustering integration) may require additional configuration in the new architecture.

## Frequently Asked Questions

### Do I need to migrate all at once?

No, the adapter-based approach allows you to migrate gradually, moving one component at a time.

### Will my existing code break after migration?

If you use the adapter-based approach, your existing code should continue to work. The adapters provide the same interface as the legacy components.

### How can I tell if my migration was successful?

Use the `validate_migration` method of the `FeatureMigrator` utility to compare results between the old and new systems.

### What if I'm using custom components?

You can create adapters for your custom components following the same pattern as the provided adapters. Implement the appropriate interfaces from the `memoryweave.interfaces` package.

### Will there be performance differences after migration?

The new architecture is designed to be more efficient, but there may be slight performance differences due to the added abstraction. In most cases, the benefits of the new architecture outweigh any minor performance impact.

## Need Help?

If you need assistance with migration, please open an issue on the MemoryWeave GitHub repository with the label "migration". The maintainers will provide guidance and support.