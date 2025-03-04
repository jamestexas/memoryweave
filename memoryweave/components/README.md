# MemoryWeave Components

This directory contains the component-based architecture for MemoryWeave's memory retrieval system. The components are designed to be modular, reusable, and easily testable, implementing a biologically-inspired memory system.

## Architecture Overview

The components system follows a pipeline architecture where each component performs a specific task in the memory retrieval process. Components can be combined in different ways to create flexible retrieval pipelines, mimicking the interconnected processes in human memory.

### Key Concepts

- **Component**: Base class for all components
- **MemoryComponent**: Components that operate on memory data
- **RetrievalComponent**: Components involved in memory retrieval
- **RetrievalStrategy**: Strategies for retrieving memories
- **PostProcessor**: Components that process retrieved results
- **MemoryManager**: Coordinates components and orchestrates the pipeline

## Component Types

### Query Analysis

- **QueryAnalyzer**: Analyzes queries to determine type and extract important information
- **QueryTypeAdapter**: Adapts retrieval parameters based on query type (factual, personal, hypothetical)
- **QueryContextBuilder**: Builds context information from the query for improved retrieval

### Retrieval Strategies

- **SimilarityRetrievalStrategy**: Retrieves memories based on similarity to query embedding
- **TemporalRetrievalStrategy**: Retrieves memories based on recency and activation
- **HybridRetrievalStrategy**: Combines similarity, recency, and keyword matching
- **TwoStageRetrievalStrategy**: Two-stage retrieval with candidate generation and re-ranking
- **HybridBM25VectorStrategy**: Combines BM25 keyword matching with vector similarity
- **ContextualFabricStrategy**: Advanced strategy that integrates direct similarity, associative links, temporal context, and activation patterns

### Post-Processors

- **KeywordBoostProcessor**: Boosts relevance scores of results containing important keywords
- **SemanticCoherenceProcessor**: Adjusts relevance scores based on semantic coherence
- **AdaptiveKProcessor**: Adjusts the number of results based on query characteristics
- **DynamicThresholdAdjuster**: Updates dynamic threshold metrics and ensures minimum results
- **MinimumResultGuaranteeProcessor**: Ensures a minimum number of results are returned
- **PersonalAttributeProcessor**: Adjusts relevance of results related to personal attributes

### Memory Organization

- **CategoryManager**: Implements ART (Adaptive Resonance Theory) clustering for memory categorization
- **AssociativeMemoryLinker**: Creates and traverses associative links between related memories
- **TemporalContextBuilder**: Maintains temporal relationships between memories
- **MemoryDecayComponent**: Implements memory decay over time to simulate forgetting
- **ActivationManager**: Tracks memory activation levels based on usage

### Context Enhancement

- **DynamicContextAdapter**: Adapts retrieval based on conversation context
- **DynamicThresholdAdjuster**: Dynamically adjusts thresholds based on retrieval performance
- **KeywordExpander**: Expands important keywords with related terms
- **ContextEnhancement**: Enriches context with conversation history and temporal information

### Other Components

- **PersonalAttributeManager**: Manages extraction and storage of personal attributes
- **PipelineConfig**: Configuration model for retrieval pipelines
- **MemoryAdapter**: Adapts between different memory storage implementations
- **ComponentFactory**: Creates components based on configuration

## Usage

The components are typically used through the `Retriever` class, which provides a high-level interface for memory retrieval operations:

```python
from memoryweave.components import Retriever

# Initialize retriever with memory and embedding model
retriever = Retriever(memory=memory, embedding_model=embedding_model)

# Retrieve memories
results = retriever.retrieve(
    query="What is my favorite color?",
    top_k=5,
    strategy="contextual_fabric",  # Use the advanced contextual fabric strategy
    minimum_relevance=0.3,
    conversation_history=conversation_history  # Provide conversation context
)
```

For more advanced usage, you can configure the pipeline directly:

```python
# Configure two-stage retrieval
retriever.configure_two_stage_retrieval(
    enable=True,
    first_stage_k=20,
    first_stage_threshold_factor=0.7
)

# Configure query type adaptation
retriever.configure_query_type_adaptation(
    enable=True,
    adaptation_strength=0.8
)

# Configure semantic coherence
retriever.configure_semantic_coherence(
    enable=True
)

# Enable dynamic threshold adjustment
retriever.enable_dynamic_threshold_adjustment(
    enable=True,
    window_size=5
)
```

## Creating Custom Components

To create a custom component, inherit from the appropriate base class and implement the required methods:

```python
from memoryweave.components.base import PostProcessor

class CustomPostProcessor(PostProcessor):
    def initialize(self, config: dict[str, Any]) -> None:
        # Initialize with configuration
        self.custom_param = config.get("custom_param", 0.5)
        
    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # Process retrieved results
        for result in results:
            # Custom processing logic
            pass
        return results
```

Then register your component with the memory manager:

```python
from memoryweave.components import MemoryManager

memory_manager = MemoryManager()
custom_processor = CustomPostProcessor()
memory_manager.register_component("custom_processor", custom_processor)
```

## Pipeline Configuration

Pipelines are configured as a list of steps, each with a component name and configuration:

```python
pipeline_config = [
    {
        "component": "query_analyzer",
        "config": {}
    },
    {
        "component": "query_adapter",
        "config": {
            "adaptation_strength": 0.8
        }
    },
    {
        "component": "contextual_fabric_strategy",
        "config": {
            "similarity_weight": 0.5,
            "associative_weight": 0.3,
            "temporal_weight": 0.1,
            "activation_weight": 0.1,
            "confidence_threshold": 0.3
        }
    },
    {
        "component": "semantic_coherence",
        "config": {
            "coherence_threshold": 0.2
        }
    }
]

memory_manager.build_pipeline(pipeline_config)
```

## Advanced Features

### Contextual Fabric Strategy

The contextual fabric strategy is inspired by human memory retrieval and combines multiple sources of information:

1. **Direct Similarity**: Vector similarity between query and memories
2. **Associative Links**: Traversal of associative connections between memories
3. **Temporal Context**: Time-based relevance of memories
4. **Activation Patterns**: Recency and frequency of memory access

This strategy provides more human-like retrieval that considers not just direct matches but also related concepts and contextual factors.

### ART Clustering

The system implements Adaptive Resonance Theory (ART) inspired clustering to organize memories into categories:

1. **Dynamic Categories**: Creates and updates categories based on similarity
2. **Vigilance Parameter**: Controls the specificity of categories
3. **Prototype Learning**: Updates category prototypes as new memories are added
4. **Category Consolidation**: Merges similar categories to maintain organization

This allows for efficient retrieval by category and improves the organization of the memory store.

### Two-Stage Retrieval

The two-stage retrieval process improves recall while maintaining precision:

1. **First Stage**: Retrieve a larger set of candidate memories using a lower confidence threshold
2. **Second Stage**: Re-rank and filter candidates using post-processors

### Query Type Adaptation

The system can adapt retrieval parameters based on query type:

- **Factual Queries**: Use lower thresholds for better recall
- **Personal Queries**: Use higher thresholds for better precision
- **Hypothetical Queries**: Balance recall and precision

### Dynamic Threshold Adjustment

The system can automatically adjust confidence thresholds based on retrieval performance:

- Monitors retrieval metrics over a sliding window
- Lowers thresholds if too few memories are being retrieved
- Raises thresholds if too many low-quality memories are being retrieved

### Progressive Filtering

For large memory stores, the system can use progressive filtering to improve performance:

1. **First Pass**: Quick approximation to find candidate memories
2. **Second Pass**: Detailed computation only on the candidates

### Batched Computation

For very large memory sets, the system can process embeddings in batches to reduce memory pressure and improve performance.

## Usage

The components are typically used through the `Retriever` class, which provides a high-level interface for memory retrieval operations:

```python
from memoryweave.components import Retriever

# Initialize retriever with memory and embedding model
retriever = Retriever(memory=memory, embedding_model=embedding_model)

# Retrieve memories
results = retriever.retrieve(
    query="What is my favorite color?",
    top_k=5,
    strategy="hybrid",
    minimum_relevance=0.3
)
```

For more advanced usage, you can configure the pipeline directly:

```python
# Configure two-stage retrieval
retriever.configure_two_stage_retrieval(
    enable=True,
    first_stage_k=20,
    first_stage_threshold_factor=0.7
)

# Configure query type adaptation
retriever.configure_query_type_adaptation(
    enable=True,
    adaptation_strength=0.8
)
```

## Creating Custom Components

To create a custom component, inherit from the appropriate base class and implement the required methods:

```python
from memoryweave.components.base import PostProcessor

class CustomPostProcessor(PostProcessor):
    def initialize(self, config: dict[str, Any]) -> None:
        # Initialize with configuration
        self.custom_param = config.get("custom_param", 0.5)
        
    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # Process retrieved results
        for result in results:
            # Custom processing logic
            pass
        return results
```

Then register your component with the memory manager:

```python
from memoryweave.components import MemoryManager

memory_manager = MemoryManager()
custom_processor = CustomPostProcessor()
memory_manager.register_component("custom_processor", custom_processor)
```

## Pipeline Configuration

Pipelines are configured as a list of steps, each with a component name and configuration:

```python
pipeline_config = [
    {
        "component": "query_analyzer",
        "config": {}
    },
    {
        "component": "query_adapter",
        "config": {
            "adaptation_strength": 0.8
        }
    },
    {
        "component": "hybrid_retrieval",
        "config": {
            "relevance_weight": 0.7,
            "recency_weight": 0.3,
            "confidence_threshold": 0.3
        }
    },
    {
        "component": "custom_processor",
        "config": {
            "custom_param": 0.6
        }
    }
]

memory_manager.build_pipeline(pipeline_config)
```

## Advanced Features

### Two-Stage Retrieval

The two-stage retrieval process improves recall while maintaining precision:

1. **First Stage**: Retrieve a larger set of candidate memories using a lower confidence threshold
2. **Second Stage**: Re-rank and filter candidates using post-processors

### Query Type Adaptation

The system can adapt retrieval parameters based on query type:

- **Factual Queries**: Use lower thresholds for better recall
- **Personal Queries**: Use higher thresholds for better precision

### Dynamic Threshold Adjustment

The system can automatically adjust confidence thresholds based on retrieval performance:

- Monitors retrieval metrics over a sliding window
- Lowers thresholds if too few memories are being retrieved
- Raises thresholds if too many low-quality memories are being retrieved
