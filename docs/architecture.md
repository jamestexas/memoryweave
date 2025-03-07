# MemoryWeave Architecture

## Overview

MemoryWeave is a memory management system for language models that uses a "contextual fabric" approach inspired by biological memory systems. Rather than traditional knowledge graph approaches with discrete nodes and edges, MemoryWeave focuses on capturing rich contextual signatures of information for improved long-context coherence in LLM conversations.

## Component Architecture

MemoryWeave uses a modular component-based architecture organized into several layers:

### Memory Storage Layer

- **StandardMemoryStore**: Basic storage for memory embeddings and content
- **HybridMemoryStore**: Combines vector search with keyword-based retrieval
- **ChunkedMemoryStore**: Manages large contexts by breaking them into chunks
- **VectorStore**: Handles vector similarity search with different implementations:
  - `SimpleVectorStore`: Basic vector similarity search
  - `ActivationVectorStore`: Incorporates activation levels into search
  - `ANNVectorStore`: Approximate nearest neighbor search for large datasets

### Retrieval Layer

- **Retriever**: Main component coordinating the retrieval process
- **Retrieval Strategies**:
  - `SimilarityRetrievalStrategy`: Pure vector similarity-based retrieval
  - `HybridRetrievalStrategy`: Combines similarity with other signals
  - `ContextualFabricStrategy`: Advanced strategy using the contextual fabric approach
  - `HybridBM25VectorStrategy`: Combines BM25 keyword search with vector similarity
  - `TwoStageRetrievalStrategy`: Multi-stage retrieval with candidates and re-ranking

### Query Processing Layer

- **QueryAnalyzer**: Analyzes and classifies query types (factual, personal, etc.)
- **QueryAdapter**: Adapts retrieval parameters based on query type
- **KeywordExpander**: Expands queries with related keywords
- **QueryContextBuilder**: Builds context information for queries
- **DynamicContextAdapter**: Dynamically adapts context based on query

### Post-Processing Layer

- **SemanticCoherenceProcessor**: Ensures retrieved memories form a coherent set
- **KeywordBoostProcessor**: Boosts memories matching important keywords
- **AdaptiveKProcessor**: Dynamically adjusts number of results
- **PersonalAttributeProcessor**: Enhances retrieval with personal attributes

### Pipeline System

- **PipelineBuilder**: Constructs configurable processing pipelines
- **PipelineManager**: Manages pipeline execution
- **ComponentRegistry**: Registers and provides access to components

## Data Flow

### Memory Addition Flow

1. Application calls `add_memory` on the MemoryManager
1. Memory is encoded (if needed) using MemoryEncoder
1. Memory is stored in the appropriate MemoryStore
1. Vector representation is added to VectorStore
1. Optional: Memory is categorized (if CategoryManager is used)
1. Optional: Activation levels are initialized

### Memory Retrieval Flow

1. Application calls `retrieve` on the Retriever
1. Query is encoded into a vector representation
1. Query is analyzed to determine query type and extract keywords
1. Query parameters are adapted based on query type
1. Pipeline is executed with appropriate retrieval strategy
1. Post-processors refine and re-rank results
1. Results are returned to the application

## Key Features

### Contextual Fabric Approach

MemoryWeave's contextual fabric approach is inspired by biological memory systems:

- **Rich Context Encoding**: Memories include surrounding context and metadata
- **Activation Dynamics**: Recently or frequently accessed memories have higher activation levels
- **Temporal Organization**: Memories maintain their relationship to other events in time
- **Associative Retrieval**: Memories are retrieved through multiple pathways, not just similarity

### Advanced Retrieval Strategies

- **Two-Stage Retrieval**: First retrieves a larger set of candidates, then re-ranks
- **Query Type Adaptation**: Adjusts retrieval parameters based on query type
- **Dynamic Threshold Adjustment**: Automatically adjusts thresholds based on retrieval performance
- **Semantic Coherence Check**: Ensures retrieved memories form a coherent set
- **Hybrid Retrieval**: Combines vector similarity, BM25, and other signals

## Directory Structure

```
memoryweave/
├── api/               # API layer for application integration
├── components/        # Core components and implementations
│   ├── retrieval_strategies/  # Retrieval strategy implementations
├── factory/           # Factory methods for creating components
├── interfaces/        # Interface definitions
├── nlp/               # NLP utilities
├── pipeline/          # Pipeline orchestration components
├── query/             # Query processing components
├── retrieval/         # Retrieval components
├── storage/           # Memory and vector storage
│   ├── vector_search/  # Vector search implementations
└── utils/             # Utility functions
```

## Current Implementation Status

The current focus is on:

- Completing the component-based architecture
- Optimizing retrieval precision and recall
- Enhancing query analysis and adaptation
- Expanding support for different retrieval strategies
- Improving performance with large memory sets

## Future Directions

- Memory persistence for long-term storage
- Enhanced category management with dynamic category formation
- Hierarchical memory organization
- Integration with more LLM frameworks
- Advanced temporal and episodic memory features
