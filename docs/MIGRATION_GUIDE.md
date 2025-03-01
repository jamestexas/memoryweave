# MemoryWeave Migration Guide

This document provides guidance for migrating from the legacy `core` architecture to the new component-based architecture.

## Overview

MemoryWeave has undergone a major architectural refactoring, moving from a monolithic design to a modular, component-based architecture. This refactoring improves maintainability, testability, and extensibility while providing more fine-grained control over memory management and retrieval.

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

## Basic Migration Steps

### 1. Update Imports

Replace legacy imports with component-based imports:

```python
# Legacy imports
from memoryweave.core import ContextualMemory, ContextualRetriever

# New imports
from memoryweave.components import Retriever
from memoryweave.components.memory_manager import MemoryManager
```

### 2. Replace Memory Creation

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

### 3. Replace Retriever Creation

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

### 4. Replace Retrieval Calls

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

## Advanced Configuration

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

## Using Migration Utilities

For complex use cases, you can use the provided migration utilities to help with transition:

```python
from memoryweave.adapters.component_migration import FeatureMigrator

# Create a migrator
migrator = FeatureMigrator()

# Migrate a legacy memory system to the new architecture
components = migrator.migrate_memory_system(legacy_memory)

# Create a pipeline using migrated components
pipeline = migrator.create_migration_pipeline(components)

# Test migration quality
validation_results = migrator.validate_migration(
    legacy_retriever=legacy_retriever,
    migrated_pipeline=pipeline,
    test_queries=["What is my favorite color?", "Tell me about quantum physics"]
)
```

## Common Issues and Solutions

1. **Different Result Ordering**: The new architecture may return results in a slightly different order due to the more sophisticated ranking algorithms. This is generally an improvement but may require adjustments to downstream processing.

2. **Configuration Differences**: The new architecture requires explicit initialization after configuration changes. Always call `retriever.initialize_components()` after configuring.

3. **Return Format**: The new architecture uses a consistent return format for all retrieval strategies. Check the documentation for the exact structure.

4. **Advanced Features**: Some advanced features (memory decay, ART clustering integration) may require additional configuration in the new architecture.

## Support

If you encounter issues during migration, please:

1. Check the feature matrix documentation for implementation status
2. Review examples in the tests directory
3. File an issue on the GitHub repository with a minimal reproducible example