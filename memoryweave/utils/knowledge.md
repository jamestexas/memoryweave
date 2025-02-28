# Utils Module Knowledge

## Module Purpose
The utils module provides utility functions used throughout the MemoryWeave system.

## Key Components
- `similarity.py`: Functions for calculating similarity between embeddings and texts
- `analysis.py`: Tools for analyzing memory retrieval performance and distributions

## Key Functions

### Similarity Functions
- `cosine_similarity_batched`: Efficient batch calculation of cosine similarity
- `embed_text_batch`: Batch embedding of text
- `fuzzy_string_match`: Fuzzy matching for text comparison

### Analysis Functions
- `analyze_query_similarities`: Analyze similarity distributions for specific queries
- `visualize_memory_categories`: Visualize memory categories and their relationships
- `analyze_retrieval_performance`: Analyze retrieval performance across parameter settings

## Implementation Details
- Optimized for performance with batch operations
- Uses numpy for efficient vector operations
- Provides both exact and fuzzy matching capabilities
- Includes visualization tools for memory analysis

## Usage
These utilities are used internally by the core components but can also be used directly for:
1. Custom similarity calculations
2. Text embedding operations
3. String matching with tolerance for minor differences
4. Analyzing memory retrieval performance
5. Visualizing memory categories and distributions

### Example: Analyzing Query Similarities
```python
from memoryweave.utils.analysis import analyze_query_similarities

# Analyze why a specific query might be failing
results = analyze_query_similarities(
    memory_system=memory_system,
    query="What is the capital of France?",
    expected_relevant_indices=[5, 10],  # Indices of memories that should be relevant
    plot=True,
    save_path="query_analysis.png"
)

# Check if relevant memories are below the threshold
for idx, sim, boosted_sim in results["below_threshold"]:
    print(f"Memory {idx} has similarity {sim:.3f} (boosted: {boosted_sim:.3f})")
```

### Example: Analyzing Retrieval Performance
```python
from memoryweave.utils.analysis import analyze_retrieval_performance

# Define parameter variations to test
parameter_variations = [
    {"confidence_threshold": 0.3, "adaptive_k_factor": 0.3},
    {"confidence_threshold": 0.2, "adaptive_k_factor": 0.15},
    {"confidence_threshold": 0.15, "adaptive_k_factor": 0.1},
]

# Analyze performance across parameter variations
performance = analyze_retrieval_performance(
    memory_system=memory_system,
    test_queries=test_queries,  # List of (query, expected_indices) tuples
    parameter_variations=parameter_variations,
    save_path="retrieval_performance.png"
)

# Get the best configuration
best_config = performance["best_f1"]
print(f"Best configuration: {best_config['parameters']}")
print(f"F1 score: {best_config['avg_f1']:.3f}")
```
