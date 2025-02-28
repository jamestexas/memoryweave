# Core Module Knowledge

## Module Purpose
The core module contains the fundamental building blocks of the MemoryWeave memory management system, implementing the contextual fabric approach.

## Key Components
- `ContextualMemory`: Main memory system that stores and manages memory traces
- `MemoryEncoder`: Encodes different content types into context-rich memory representations
- `ContextualRetriever`: Retrieves memories based on context using various strategies

## Architecture
The core components work together in the following way:
1. `MemoryEncoder` converts raw content into embeddings with rich contextual metadata
2. `ContextualMemory` stores these embeddings and manages their activation levels
3. `ContextualRetriever` uses sophisticated retrieval strategies to find relevant memories

## Implementation Details
- Memory is stored as embeddings with associated metadata
- Activation levels track recency and relevance of memories
- Retrieval uses a combination of similarity, recency, and keyword matching
- Personal attributes are extracted and stored for enhanced retrieval
- ART-inspired clustering organizes memories into dynamic categories

## ART-Inspired Clustering
The `ContextualMemory` class includes an optional Adaptive Resonance Theory (ART) inspired clustering mechanism:

- **Dynamic Category Formation**: Memories self-organize into categories based on similarity
- **Vigilance Parameter**: Controls the threshold for creating new categories vs. modifying existing ones
- **Resonance-Based Matching**: Categories are matched based on similarity to the input
- **Prototype Learning**: Category prototypes adapt over time as new memories are added

To enable ART clustering:
```python
memory = ContextualMemory(
    embedding_dim=768,
    use_art_clustering=True,
    vigilance_threshold=0.85,  # Higher = more categories
    learning_rate=0.2  # Higher = faster adaptation
)
```

## Dynamic Vigilance
The ART-inspired clustering supports dynamic vigilance adjustment strategies:

- **Decreasing Vigilance**: Starts high and gradually decreases, encouraging more merging over time
- **Increasing Vigilance**: Starts low and gradually increases, creating broader categories first
- **Category-Based**: Adjusts vigilance based on the number of categories formed
- **Density-Based**: Adjusts vigilance based on the density of memories in embedding space

To enable dynamic vigilance:
```python
memory = ContextualMemory(
    embedding_dim=768,
    use_art_clustering=True,
    vigilance_threshold=0.7,  # Initial threshold
    dynamic_vigilance=True,
    vigilance_strategy="decreasing",  # "decreasing", "increasing", "category_based", or "density_based"
    min_vigilance=0.5,
    max_vigilance=0.9,
    target_categories=5  # Target number for category_based strategy
)
```

## Category Consolidation
To address category fragmentation, the system supports hierarchical clustering-based category consolidation:

- **Periodic Consolidation**: Automatically merges similar categories after a specified number of memories
- **Similarity Threshold**: Controls how aggressively categories are merged
- **Hierarchical Clustering**: Uses various linkage methods to identify category relationships
- **Manual Consolidation**: Can be triggered on demand with custom thresholds

To enable category consolidation:
```python
memory = ContextualMemory(
    embedding_dim=768,
    use_art_clustering=True,
    vigilance_threshold=0.8,  # High vigilance creates more initial categories
    enable_category_consolidation=True,
    consolidation_threshold=0.7,  # Higher = less aggressive merging
    min_category_size=3,  # Categories smaller than this are prioritized for merging
    consolidation_frequency=50,  # Consolidate every 50 memories
    hierarchical_method="average"  # "single", "complete", "average", or "weighted"
)
```

## Confidence Thresholding
The system supports filtering out low-confidence retrievals to improve precision:

- **Confidence Threshold**: Minimum similarity score for memory inclusion
- **Semantic Coherence Check**: Ensures retrieved memories form a coherent set
- **Adaptive K Selection**: Dynamically determines how many memories to retrieve

To enable confidence thresholding:
```python
memory = ContextualMemory(
    embedding_dim=768,
    default_confidence_threshold=0.4,  # Minimum similarity score
    semantic_coherence_check=True,  # Check coherence between memories
    coherence_threshold=0.2,  # Minimum average similarity between memories
    adaptive_retrieval=True  # Dynamically determine how many memories to retrieve
)

# When retrieving, you can override the default threshold
memories = memory.retrieve_memories(
    query_embedding,
    top_k=5,
    confidence_threshold=0.5  # Override default threshold
)
```

## Advanced Retrieval Strategies
The system supports several advanced retrieval strategies to balance precision and recall:

- **Two-Stage Retrieval**: First retrieves a larger candidate set with lower threshold, then re-ranks
- **Query Type Adaptation**: Adjusts retrieval parameters based on query type (factual vs. personal)
- **Minimum Memory Guarantees**: Ensures a minimum number of memories are returned even with strict filtering

To enable advanced retrieval:
```python
retriever = ContextualRetriever(
    memory=memory,
    embedding_model=embedding_model,
    retrieval_strategy="hybrid",
    confidence_threshold=0.3,
    semantic_coherence_check=True,
    adaptive_retrieval=True,
    adaptive_k_factor=0.15,  # Lower = more results (less conservative)
    use_two_stage_retrieval=True,  # Enable two-stage retrieval
    first_stage_k=20,  # Number of candidates in first stage
    query_type_adaptation=True  # Adapt to query type
)
```

## Design Principles
- Biologically-inspired memory management
- Rich contextual signatures rather than isolated facts
- Dynamic activation patterns for memory retrieval
- Temporal relationships and episodic anchoring
- Self-organizing memory categories
- Balance between precision and recall through adaptive strategies
