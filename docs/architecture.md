# MemoryWeave Architecture

## Overview

MemoryWeave is a memory management system for language models that uses a "contextual fabric" approach inspired by biological memory systems. Rather than traditional knowledge graph approaches with discrete nodes and edges, MemoryWeave focuses on capturing rich contextual signatures of information for improved long-context coherence in LLM conversations.

This document describes the architecture of MemoryWeave, including its components, their relationships, and the flow of data through the system.

## Component Architecture

MemoryWeave is organized into several key components that work together to provide memory management capabilities. The architecture has evolved from a monolithic design to a more modular, component-based approach.

### Core Components (Original System)

The core components form the foundation of MemoryWeave's memory management capabilities:

1. **CoreMemory**: Provides basic storage for memory embeddings, metadata, and activation levels.
   - Handles memory addition, retrieval, and activation updates
   - Manages memory capacity through consolidation

2. **CategoryManager**: Implements ART-inspired clustering for organizing memories.
   - Dynamically forms categories based on memory similarity
   - Updates category prototypes as new memories are added
   - Supports category consolidation to prevent fragmentation

3. **MemoryRetriever**: Implements memory retrieval strategies.
   - Supports similarity-based, category-based, and hybrid retrieval
   - Applies confidence thresholding and semantic coherence checks
   - Implements adaptive k selection for optimal result counts

4. **ContextualMemory**: Provides a unified interface to the core components.
   - Combines CoreMemory, CategoryManager, and MemoryRetriever
   - Exposes a simple API for memory operations
   - Maintains backward compatibility with the original API

5. **MemoryEncoder**: Converts different content types into memory representations.
   - Encodes text, interactions, and concepts into embeddings
   - Adds contextual information to enhance retrieval

6. **ContextualRetriever (Deprecated)**: Original retriever implementation.
   - Being phased out in favor of the component-based approach
   - Maintained for backward compatibility

### Component-Based System (New Architecture)

The new component-based architecture provides more modularity and testability:

1. **MemoryManager**: Orchestrates memory components and pipelines.
   - Registers components and builds retrieval pipelines
   - Executes pipelines to process queries
   - Manages component configuration

2. **MemoryAdapter**: Adapts core memory to the component interface.
   - Wraps ContextualMemory for use in the pipeline
   - Translates between different data formats

3. **CoreRetrieverAdapter**: Adapts core retriever to the component interface.
   - Wraps MemoryRetriever for use in the pipeline
   - Handles parameter adaptation based on query type

4. **Retriever**: New component-based retriever implementation.
   - Uses the MemoryManager to orchestrate retrieval
   - Configures and executes retrieval pipelines
   - Supports advanced features like two-stage retrieval and query type adaptation

### Bridge Components

Bridge components facilitate the transition between the original and new architectures:

1. **RefactoredRetriever**: Provides compatibility between old and new systems.
   - Implements the same interface as ContextualRetriever
   - Uses both the original and new components internally
   - Allows gradual migration to the new architecture

## Data Flow

### Memory Addition Flow

1. Application calls `add_memory` on ContextualMemory
2. CoreMemory stores the embedding and metadata
3. If ART clustering is enabled, CategoryManager assigns the memory to a category
4. Activation levels and temporal markers are updated

### Memory Retrieval Flow

#### Legacy Flow
1. Application calls `retrieve_for_context` on ContextualRetriever
2. Query is encoded and analyzed
3. Retrieval strategy is selected based on query type
4. Memories are retrieved using similarity, recency, or hybrid approach
5. Post-processing is applied (keyword boosting, coherence check, etc.)
6. Results are returned to the application

#### New Component-Based Flow
1. Application calls `retrieve` on Retriever
2. MemoryManager executes the retrieval pipeline:
   - Query analyzer identifies query type and extracts keywords
   - Query adapter adjusts parameters based on query type
   - Retrieval strategy retrieves candidate memories
   - Post-processors refine and re-rank results
3. Results are returned to the application

#### Transition Flow (Using RefactoredRetriever)
1. Application calls `retrieve_for_context` on RefactoredRetriever
2. RefactoredRetriever delegates to either:
   - The new Retriever for component-based retrieval
   - The original MemoryRetriever for direct retrieval
3. Results are formatted consistently and returned to the application

## Key Features

### ART-Inspired Clustering

MemoryWeave uses an Adaptive Resonance Theory (ART) inspired approach to organize memories into categories:

- **Dynamic Category Formation**: Memories self-organize into categories based on similarity
- **Vigilance Parameter**: Controls the threshold for creating new categories vs. modifying existing ones
- **Prototype Learning**: Category prototypes adapt over time as new memories are added
- **Category Consolidation**: Similar categories can be merged to prevent fragmentation

### Advanced Retrieval Strategies

MemoryWeave supports several advanced retrieval strategies:

- **Two-Stage Retrieval**: First retrieves a larger set of candidates with lower threshold, then re-ranks
- **Query Type Adaptation**: Adjusts retrieval parameters based on query type (factual vs. personal)
- **Dynamic Threshold Adjustment**: Automatically adjusts thresholds based on retrieval performance
- **Semantic Coherence Check**: Ensures retrieved memories form a coherent set
- **Adaptive K Selection**: Dynamically determines how many memories to retrieve

## Migration Path

MemoryWeave is transitioning from the original monolithic architecture to the new component-based architecture:

1. **Current State**: Both architectures coexist, with RefactoredRetriever providing compatibility
2. **Transition**: Applications gradually migrate from ContextualRetriever to Retriever
3. **Future State**: The original ContextualRetriever will be fully deprecated, and all applications will use the new component-based architecture

## Conclusion

MemoryWeave's architecture provides a flexible and powerful approach to memory management for language models. The transition to a component-based architecture improves modularity, testability, and extensibility while maintaining backward compatibility through bridge components.

The system's biologically-inspired approach to memory management, with features like ART-inspired clustering and advanced retrieval strategies, enables more natural and coherent conversations with language models.
