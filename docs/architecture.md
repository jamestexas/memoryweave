# MemoryWeave Architecture

## Overview

MemoryWeave is a memory management system for language models that uses a "contextual fabric" approach inspired by biological memory systems. Rather than traditional knowledge graph approaches with discrete nodes and edges, MemoryWeave focuses on capturing rich contextual signatures of information for improved long-context coherence in LLM conversations.

This document describes the architecture of MemoryWeave, including its components, their relationships, and the flow of data through the system.

> **UPDATED March 2025**: This architecture document reflects both the current state and the ongoing refactoring efforts. We're transitioning from the original monolithic design to a more modular, component-based architecture as described in the [Architecture Decision Record](#architecture-decision-record) below.

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

## Architecture Decision Record

### Title: Modular Retrieval Architecture Refactoring

#### Status
Accepted (March 2025)

#### Context
The MemoryWeave system has grown organically, leading to several architectural challenges:

1. **Monolithic components**: Core retrieval functionality in large, tightly-coupled modules
2. **Code duplication**: Similar functionality duplicated across files
3. **Poor separation of concerns**: Memory storage, retrieval logic, and query processing mixed
4. **Testing difficulties**: Complex components are difficult to test in isolation
5. **Limited extensibility**: Adding new features requires modifying core classes

The current architecture centers around a monolithic `ContextualRetriever` with many responsibilities and a similarly large `nlp_extraction.py` utility. As we add features like advanced retrieval strategies, personal attribute tracking, and pipeline configurations, the codebase has become increasingly difficult to maintain.

#### Decision
We will refactor the architecture to a modular, component-based system with clear interfaces and separation of concerns. The new architecture will follow these principles:

1. **Interface-first design**: Define clear interfaces for all components
2. **Single responsibility**: Each component should do one thing well
3. **Composition over inheritance**: Build complex behavior through composition
4. **Progressive enhancement**: Start with simple implementations and add features incrementally
5. **Backward compatibility**: Maintain adapter layers for smooth transition

#### Implementation Plan

##### Phase 1: Interface Definition & Core Boundaries (1-2 days)
1. Create `/interfaces/` module with core protocol definitions
   - `memory.py`: Memory storage and retrieval interfaces
   - `retrieval.py`: Retrieval strategy interfaces
   - `query.py`: Query processing interfaces
   - `pipeline.py`: Pipeline configuration interfaces

2. Define clear data models for all components
   - Memory representation
   - Query representation
   - Retrieval results
   - Configuration options

##### Phase 2: Core Components Implementation (2-3 days)
1. Create `/storage/` module for memory management
   - `memory_store.py`: Basic memory storage
   - `vector_store.py`: Vector search optimizations
   - `activation.py`: Memory activation tracking
   - `category.py`: Category management

2. Create `/retrieval/` module for retrieval strategies
   - `similarity.py`: Pure similarity-based retrieval
   - `temporal.py`: Recency-based retrieval
   - `hybrid.py`: Combined similarity and temporal
   - `two_stage.py`: Two-stage retrieval process

3. Create `/query/` module for query processing
   - `analyzer.py`: Query type analysis
   - `keyword.py`: Keyword extraction
   - `adaptation.py`: Parameter adaptation

4. Create `/nlp/` module for NLP utilities
   - `extraction.py`: Core extraction logic
   - `matchers.py`: Pattern matching
   - `patterns.py`: Regular expression patterns
   - `keywords.py`: Keyword utilities

##### Phase 3: Pipeline Architecture (1-2 days)
1. Create `/pipeline/` module for orchestration
   - `manager.py`: Pipeline orchestration
   - `registry.py`: Component registry
   - `builder.py`: Pipeline construction
   - `executor.py`: Pipeline execution

2. Create `/config/` module for configuration
   - `options.py`: Configuration options
   - `validation.py`: Config validation
   - `loaders.py`: Config loading utilities

3. Create `/factory/` module for component creation
   - `memory.py`: Memory component factories
   - `retrieval.py`: Retrieval component factories
   - `pipeline.py`: Pipeline component factories

##### Phase 4: Implementation Migration (2-3 days)
1. Create adapter implementations for all components
2. Incrementally migrate features from the old to new architecture
3. Update tests to work with new architecture

##### Phase 5: Cleanup & Documentation (1 day)
1. Remove deprecated code
2. Create comprehensive documentation
3. Update examples and tests

#### Advantages
1. **Improved maintainability**: Smaller, focused components are easier to understand and maintain
2. **Better testability**: Isolated components can be tested independently
3. **Increased extensibility**: New components can be added without modifying existing code
4. **Clearer dependencies**: More explicit component dependencies
5. **Easier collaboration**: Team members can work on different components simultaneously
6. **Performance optimization**: Components can be optimized independently

#### Implementation Priorities

##### Highest Priority
1. Break up monolithic retrieval.py into modular components
2. Create clear interfaces for all components
3. Separate memory storage from retrieval logic

##### Secondary Priority
1. Refactor NLP extraction into focused components
2. Create pipeline architecture for flexible component arrangement
3. Implement adapter layer for backward compatibility

##### Third Priority
1. Improve test organization and coverage
2. Create comprehensive documentation
3. Optimize performance-critical paths

#### Migration Strategy
1. **Adapter pattern**: Create adapters implementing new interfaces that delegate to existing code
2. **Parallel implementations**: Maintain both implementations during transition
3. **Feature parity tracking**: Use feature matrix to track migration progress
4. **Test-driven migration**: Ensure all features have tests before migration

#### Metrics for Success
1. **Code quality**: Reduced complexity metrics for individual components
2. **Test coverage**: Maintain or improve test coverage
3. **Performance**: No regression in benchmark performance
4. **Feature parity**: All features from original implementation available
5. **Documentation**: Comprehensive documentation for all components

#### Implementation Timeline
- Phase 1 (Interface Definition): Days 1-2
- Phase 2 (Core Components): Days 3-5
- Phase 3 (Pipeline Architecture): Days 6-7
- Phase 4 (Implementation Migration): Days 8-10
- Phase 5 (Cleanup & Documentation): Day 11

## Conclusion

MemoryWeave's architecture provides a flexible and powerful approach to memory management for language models. The transition to a component-based architecture improves modularity, testability, and extensibility while maintaining backward compatibility through bridge components.

The system's biologically-inspired approach to memory management, with features like ART-inspired clustering and advanced retrieval strategies, enables more natural and coherent conversations with language models.

The new modular architecture will allow us to more easily implement the features described in our roadmap while maintaining the quality and performance of the system.
