# MemoryWeave Components

This directory contains the component-based architecture for MemoryWeave's memory retrieval system. The components are designed to be modular, reusable, and easily testable.

## Architecture Overview

The components system follows a pipeline architecture where each component performs a specific task in the memory retrieval process. Components can be combined in different ways to create flexible retrieval pipelines.

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
- **QueryTypeAdapter**: Adapts retrieval parameters based on query type

### Retrieval Strategies

- **SimilarityRetrievalStrategy**: Retrieves memories based on similarity to query embedding
- **TemporalRetrievalStrategy**: Retrieves memories based on recency and activation
- **HybridRetrievalStrategy**: Combines similarity, recency, and keyword matching
- **TwoStageRetrievalStrategy**: Two-stage retrieval with candidate generation and re-ranking

### Post-Processors

- **KeywordBoostProcessor**: Boosts relevance scores of results containing important keywords
- **SemanticCoherenceProcessor**: Adjusts relevance scores based on semantic coherence
- **AdaptiveKProcessor**: Adjusts the number of results based on query characteristics
- **DynamicThresholdAdjuster**: Updates dynamic threshold metrics and ensures minimum results

### Other Components

- **PersonalAttributeManager**: Manages extraction and storage of personal attributes
- **PipelineConfig**: Configuration model for retrieval pipelines

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
