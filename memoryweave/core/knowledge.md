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

## Design Principles
- Biologically-inspired memory management
- Rich contextual signatures rather than isolated facts
- Dynamic activation patterns for memory retrieval
- Temporal relationships and episodic anchoring
- Self-organizing memory categories
