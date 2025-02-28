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
- **Dynamic Threshold Adjustment**: Automatically adjusts thresholds based on retrieval performance
- **Memory Decay**: Applies exponential decay to memory activations over time

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
    query_type_adaptation=True,  # Adapt to query type
    dynamic_threshold_adjustment=True,  # Dynamically adjust thresholds
    threshold_adjustment_window=5,  # Window size for adjustment
    memory_decay_enabled=True,  # Enable memory decay
    memory_decay_rate=0.99,  # Rate of decay
    memory_decay_interval=10  # Apply decay every N interactions
)
```

### Two-Stage Retrieval
The two-stage retrieval process works as follows:

1. **First Stage**: Retrieve a larger set of candidate memories using a lower confidence threshold
   - This improves recall by considering more potential matches
   - Typically retrieves 20-30 candidates instead of just 5-10

2. **Second Stage**: Re-rank and filter the candidates
   - Apply semantic coherence check to ensure retrieved memories form a coherent set
   - Use adaptive K selection to dynamically determine how many memories to return
   - Apply keyword boosting to prioritize memories that match important keywords

This approach significantly improves recall for factual queries while maintaining precision for personal queries.

### Query Type Adaptation
The system can automatically adapt retrieval parameters based on the type of query:

- **Factual Queries**: Use lower thresholds and less conservative adaptive K selection
  - Examples: "What is the capital of France?", "Who wrote Hamlet?"
  - Detection: Pattern matching for question words, absence of personal pronouns

- **Personal Queries**: Maintain higher thresholds for better precision
  - Examples: "What's my name?", "Where do I live?"
  - Detection: Presence of personal pronouns and possessives

This adaptation helps balance precision and recall based on the query context.

### Dynamic Threshold Adjustment
The system can automatically adjust confidence thresholds based on retrieval performance:

- Monitors retrieval metrics over a sliding window of recent interactions
- Lowers thresholds if too few memories are being retrieved
- Raises thresholds if too many or low-quality memories are being retrieved

This self-tuning mechanism helps the system adapt to different conversation contexts without manual intervention.

### Memory Decay
To focus on more recent and contextually relevant memories:

- Applies exponential decay to memory activations over time
- Older memories naturally become less dominant in retrieval
- Helps the system focus on recent context without manual consolidation

## Design Principles
- Biologically-inspired memory management
- Rich contextual signatures rather than isolated facts
- Dynamic activation patterns for memory retrieval
- Temporal relationships and episodic anchoring
- Self-organizing memory categories
- Balance between precision and recall through adaptive strategies

## Testability and Modularity
The component-based architecture is designed with testability and modularity in mind:

- **Separation of Concerns**: Each component has a clear, single responsibility
- **Interface-Based Design**: Components interact through well-defined interfaces
- **Dependency Injection**: Components can be easily replaced with mock versions for testing
- **Configuration-Driven**: Components are configured at runtime rather than hardcoded
- **Pipeline Architecture**: Retrieval process is built as a configurable pipeline of components
- **Small, Focused Classes**: Each class is small and focused on a specific task
- **Unit Testability**: Components can be tested in isolation from the rest of the system
- **Integration Testing**: Pipeline can be tested as a whole with different component combinations
- **Strategy Pattern**: Different retrieval strategies can be swapped out without changing the rest of the system
- **Dynamic Configuration**: System can be reconfigured at runtime to adapt to different workloads

This approach ensures that the system is maintainable and testable, with strong modularity that allows components to be enhanced or replaced independently.

## Diagnostic Analysis
The Diagnostic Analysis Phase has identified several key issues with memory retrieval:

1. **Query Type Differences**: Personal queries and factual queries require different retrieval strategies and thresholds
   - Personal queries benefit from higher thresholds (0.4-0.6) for better precision
   - Factual queries need lower thresholds (0.2-0.3) for better recall

2. **Embedding Quality**: The quality of embeddings affects retrieval performance
   - Related memories should have high intra-set similarity (>0.6)
   - Unrelated memories should have low inter-set similarity (<0.3)
   - Current embedding models may not provide sufficient separation

3. **Threshold Optimization**: Different thresholds are optimal for different query types
   - Using a single threshold for all queries leads to poor overall performance
   - Dynamic threshold adjustment based on query type can significantly improve F1 scores

4. **Retrieval Pipeline**: A two-stage retrieval pipeline can improve performance
   - First stage: Retrieve a larger candidate set with lower threshold
   - Second stage: Re-rank and filter candidates based on relevance and coherence

These findings inform the next phase of development: Retrieval Strategy Overhaul.
