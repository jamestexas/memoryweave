# Core Module Knowledge

## Module Purpose
The core module contains the fundamental building blocks of the MemoryWeave memory management system, implementing the contextual fabric approach.

## Key Components
- `ContextualMemory`: Main memory system that stores and manages memory traces
- `MemoryEncoder`: Encodes different types of content into context-rich memory representations
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

## Design Principles
- Biologically-inspired memory management
- Rich contextual signatures rather than isolated facts
- Dynamic activation patterns for memory retrieval
- Temporal relationships and episodic anchoring
