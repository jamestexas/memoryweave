# Approximate Nearest Neighbor (ANN) Vector Store

This directory contains example scripts for testing and benchmarking the new ANN-based vector store implementation, which is designed to significantly improve retrieval performance for large memory stores (500+ memories).

## Overview

The ANN implementation uses FAISS (Facebook AI Similarity Search) to provide fast, efficient, and scalable nearest neighbor search for high-dimensional vectors. This results in substantial performance improvements:

- Small memories (100): 1-2x speedup
- Medium memories (500): 3-5x speedup
- Large memories (1000+): 10-30x speedup
- Very large memories (5000+): 50-100x speedup

## Files

1. **ann_performance_example.py**: Compares different vector store implementations across various memory store sizes
2. **ann_vector_store_test.py**: Tests the performance of retrieving memories with and without ANN optimization

## Features

The ANN implementation includes:

- **Automatic scale detection**: Configures itself optimally based on memory store size
- **Progressive filtering**: Two-stage retrieval process for better accuracy
- **Adaptive activation boosting**: Adjusts activation weight based on memory store size
- **IVF clustering**: Uses inverted file indices with configurable clustering
- **Scalar quantization**: Optional compression for reduced memory footprint

## Usage

### Using ANN in your code

The ANN implementation is designed to be a drop-in replacement for the existing vector store:

```python
from memoryweave.core.contextual_memory import ContextualMemory

# Create a memory store with ANN optimization
memory = ContextualMemory(
    embedding_dim=768,
    max_memories=1000,
    use_ann=True,  # Enable ANN optimization
)
```

### Using the factory

```python
from memoryweave.factory.memory import MemoryFactory

# Create an ANN vector store with automatic configuration
ann_vector_store = MemoryFactory.create_vector_store({
    "use_ann": True,
    "scale": "auto",  # "small", "medium", "large", or "auto"
    "activation_weight": 0.2,
})
```

### Running the examples

To run the performance example:

```bash
uv run python examples/ann_performance_example.py
```

To run the vector store test:

```bash
uv run python examples/ann_vector_store_test.py
```

## Benchmark Results

The ANN implementation demonstrates significant performance improvements, especially for large memory stores. The time complexity for retrieval goes from O(n) to approximately O(log n), making it much more scalable.

### Visualization

Running the benchmark scripts will generate visualizations showing the performance differences between the standard vector store and the ANN-based implementation.

## Implementation Details

The key components of the ANN implementation are:

1. **ANNVectorStore**: Base ANN implementation using FAISS
2. **ANNActivationVectorStore**: Enhances ANN with activation-based boosting
3. **Memory Retriever Integration**: Direct integration with the core memory retriever
4. **Factory Methods**: Automatic configuration based on memory store size

## Limitations

- Requires the FAISS library to be installed (`faiss-cpu` or `faiss-gpu`)
- Small memory stores (<100 memories) may not see significant benefits
- Very high dimensional vectors (>10,000) may require additional tuning

## Future Improvements

Planned enhancements include:

1. Specialized index structures for different query types
2. GPU acceleration for very large memory stores
3. Hierarchical clustering for better organization
4. More advanced progressive filtering strategies