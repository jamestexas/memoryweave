# MemoryWeave Components Feature Matrix

## Executive Summary

The `memoryweave/components` directory implements a sophisticated component-based architecture for memory management in language models. This directory contains modules responsible for query analysis, memory manipulation, retrieval adaptation, and contextual enhancement. The architecture follows a pipeline pattern where components can be combined flexibly for different retrieval strategies.

Several files show direct dependencies on the deprecated "core" module, particularly in adapter classes and integration points. These should be prioritized for refactoring to complete the architectural transition.

## Foundation Components

<details>

| File | Purpose | Key Features | Core Dependencies | Code Quality Notes |
|------|---------|--------------|-------------------|-------------------|
| `base.py` | Defines base component interfaces | • Abstract `Component` base class<br>• `MemoryComponent` for memory operations<br>• `RetrievalComponent` for queries<br>• `RetrievalStrategy` for memory retrieval | None | Clean implementation with clear abstractions |
| `component_names.py` | Defines component type enumeration | • `ComponentName` enum for all components<br>• Used for component registration and pipeline config | None | Concise, well-organized enumeration |
| `pipeline_config.py` | Defines configuration models | • `PipelineStep` for single steps<br>• `PipelineConfig` for complete pipeline<br>• Pydantic validation | None | Good use of Pydantic for validation |

</details>

## Query Processing

<details>

| File | Purpose | Key Features | Core Dependencies | Code Quality Notes |
|------|---------|--------------|-------------------|-------------------|
| `query_analysis.py` | Analyzes query type and content | • Query type classification (personal, factual, etc.)<br>• Keyword extraction<br>• Parameter recommendations by query type | None | Has hardcoded patterns for test cases; could benefit from refactoring |
| `query_adapter.py` | Adapts retrieval parameters | • Parameter adjustment by query type<br>• Configurable adaptation strength<br>• Optimizes for different query patterns | None | Complex adaptation logic with some redundant logging |
| `query_context_builder.py` | Enriches queries with context | • Extracts temporal markers and entities<br>• Builds context from conversation history<br>• Enhances query embeddings | None | Good separation of concerns; could use more documentation |
| `keyword_expander.py` | Expands keywords for better recall | • Handles singular/plural forms including irregulars<br>• Extensive synonym dictionary<br>• Configurable expansion parameters | None | Well-structured but large hardcoded dictionaries could be externalized |

</details>

## Memory Organization

<details>

| File | Purpose | Key Features | Core Dependencies | Code Quality Notes |
|------|---------|--------------|-------------------|-------------------|
| `memory_manager.py` | Orchestrates memory components | • Component registration<br>• Pipeline building and execution<br>• Centralized memory access | Indirect through `BaseMemoryStore` | Clear responsibilities and good error handling |
| `category_manager.py` | Organizes memories into categories | • ART-inspired clustering<br>• Dynamic category management<br>• Category consolidation<br>• Category-based retrieval | Depends on `CoreCategoryManager` | Mostly clean implementation; adapter pattern to core |
| `personal_attributes.py` | Manages user attributes | • Extracts attributes from text<br>• Categorizes into preferences, demographics, etc.<br>• Retrieves attributes relevant to queries | None | Complex methods with multiple responsibilities; some hardcoded patterns |
| `text_chunker.py` | Breaks down large texts | • Chunking by paragraphs, sentences, or size<br>• Maintains context between chunks<br>• Special handling for conversations | None | Good implementation with appropriate regex usage |

</details>

## Contextual Enhancement

<details>

| File | Purpose | Key Features | Core Dependencies | Code Quality Notes |
|------|---------|--------------|-------------------|-------------------|
| `context_enhancement.py` | Enhances memory embeddings | • `ContextualEmbeddingEnhancer` for richer embeddings<br>• Integrates conversation, temporal, topical context<br>• `ContextSignalExtractor` for context signals | None | Good separation of concerns; could use more documentation |
| `temporal_context.py` | Manages temporal aspects | • `TemporalContextBuilder` for time context<br>• `TemporalDecayComponent` for activation decay<br>• Groups memories into temporal episodes<br>• Extracts time references from queries | Indirect through `BaseMemoryStore` | Complex implementation with multiple responsibilities |
| `associative_linking.py` | Creates memory connections | • Creates bidirectional links between related memories<br>• Calculates link strength (similarity + temporal)<br>• Implements spreading activation<br>• Includes network visualization | Uses `BaseMemoryStore` and `MemoryID` | Good implementation with clear methods |
| `activation.py` | Manages memory activation levels | • Tracks activation for all memories<br>• Implements spreading activation<br>• Applies decay over time<br>• Boosts retrieval based on activation | Interacts with `AssociativeMemoryLinker` | Well-structured with good separation of concerns |

</details>

## Dynamic Adaptation

<details>

| File | Purpose | Key Features | Core Dependencies | Code Quality Notes |
|------|---------|--------------|-------------------|-------------------|
| `dynamic_threshold_adjuster.py` | Adjusts confidence thresholds | • Analyzes retrieval metrics<br>• Adapts thresholds by query type<br>• Distribution-based adjustment<br>• Minimum results guarantee | None | Well-structured with good Pydantic usage |
| `dynamic_context_adapter.py` | Context-aware adaptation | • Adapts for memory size, query type, complexity<br>• Implements adaptive weights<br>• Handles conversation context<br>• Supports extensive logging | None | Comprehensive adaptation logic |
| `memory_decay.py` | Applies decay to activations | • Configurable decay rate and interval<br>• Exponential decay implementation<br>• Handles both component and legacy memory | Has compatibility code for core activation levels | Clean implementation with good error handling |

</details>

## Post-Processing

<details>

| File | Purpose | Key Features | Core Dependencies | Code Quality Notes |
|------|---------|--------------|-------------------|-------------------|
| `post_processors.py` | Refines retrieval results | • `KeywordBoostProcessor`: Boosts keyword matches<br>• `SemanticCoherenceProcessor`: Ensures coherence<br>• `AdaptiveKProcessor`: Adjusts result count<br>• `MinimumResultGuaranteeProcessor`: Ensures min results<br>• `PersonalAttributeProcessor`: Applies attributes | None | Good separation of concerns; complex coherence processor could be further split |

</details>

## Integration and Adapters

<details>

| File | Purpose | Key Features | Core Dependencies | Code Quality Notes |
|------|---------|--------------|-------------------|-------------------|
| `adapters.py` | Connects core with components | • `CoreRetrieverAdapter`: Adapts core retriever<br>• `CategoryAdapter`: Adapts category managers<br>• Bidirectional compatibility | Directly uses `ContextualMemory` and `CoreCategoryManager` | Clean adapters with clear responsibilities |
| `memory_adapter.py` | Adapts core memory | • Wraps `ContextualMemory`<br>• Methods for adding/retrieving memories<br>• Implements hybrid search | Directly uses `ContextualMemory` | Clear adapter implementation |
| `retrieval_strategies_impl.py` | Implements retrieval strategies | • Multiple strategy implementations<br>• Each with unique retrieval approach<br>• Extensive error handling and logging | Directly uses `ContextualMemory` | Complex strategies with some redundancy |
| `retriever.py` | Main entry point for retrieval | • Initializes and configures components<br>• Builds retrieval pipelines<br>• High-level retrieval interface<br>• Conversation state tracking | Uses `StandardMemoryStore` | Complex with many responsibilities |
| `factory.py` | Creates memory components | • `create_memory_system` for complete system<br>• `configure_memory_pipeline` for different pipelines<br>• Standard configs for different use cases | Directly uses `CoreCategoryManager` and `ContextualMemory` | Clean factory implementation |

</details>

## Documentation Reorganization Recommendations

<details>

1. **Consolidate Planning Documents**:

   - Merge `next_steps.md` with `plan_for_improvement.md` to create a single document for improvement planning
   - Align `development_priorities.md` with `roadmap.md` to ensure consistency in priorities

1. **Consolidate Refactoring Documentation**:

   - Merge `refactoring_summary.md` (root) with `docs/refactoring_progress.md` to create a comprehensive refactoring history

1. **Update Key Documents**:

   - Replace `docs/feature_matrix.md` with this new feature matrix
   - Update `readme.md` to reflect current architecture and capabilities
   - Update `architecture.md` with latest component interactions

1. **Create New Documentation**:

   - Create a component-specific README for the `memoryweave/components` directory
   - Consider creating visual diagrams showing component interactions

</details>

## Code Quality Assessment Summary

<details>

1. **Strengths**:

   - Clear component boundaries and interfaces
   - Good use of modern Python features (Pydantic, type hints)
   - Well-structured class hierarchy
   - Good error handling throughout

1. **Areas for Improvement**:

   - Several files have dependencies on deprecated "core" module
   - Some complex methods could be refactored for clarity
   - Hardcoded test patterns in several components
   - Some redundancy between different implementations

1. **Priority Refactoring Targets**:

   - `adapters.py` and `memory_adapter.py` (direct core dependencies)
   - `factory.py` (direct core dependencies)
   - `retrieval_strategies_impl.py` (direct core dependencies)
   - `query_analysis.py` (hardcoded patterns)
   - `personal_attributes.py` (complex methods)

</details>

## Deprecated Core Usage Summary

<details>

| File | Core Dependencies | Impact | Refactoring Priority |
|------|-------------------|--------|---------------------|
| `adapters.py` | `ContextualMemory`, `CoreCategoryManager` | High - Direct dependency | 1 - Critical |
| `memory_adapter.py` | `ContextualMemory` | High - Direct dependency | 1 - Critical |
| `factory.py` | `ContextualMemory`, `CoreCategoryManager` | High - Direct dependency | 1 - Critical |
| `retrieval_strategies_impl.py` | `ContextualMemory` | High - Direct dependency | 1 - Critical |
| `category_manager.py` | `CoreCategoryManager` | Medium - Adapter pattern | 2 - Important |
| `temporal_context.py` | Indirect through `BaseMemoryStore` | Low - Indirect | 3 - Optional |
| `activation.py` | None, but interacts with deprecated pattern | Low - Pattern | 3 - Optional |

</details>
