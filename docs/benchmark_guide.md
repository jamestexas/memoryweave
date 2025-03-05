# MemoryWeave Benchmarking Guide

This guide explains how to run benchmarks to evaluate and compare MemoryWeave's retrieval capabilities, with a special focus on comparing the contextual fabric approach with traditional retrieval methods.

## Unified Benchmark System

MemoryWeave now uses a unified benchmark system that can be configured through YAML files. This makes it easier to run consistent, reproducible benchmarks.

```bash
# Run a benchmark with its configuration file
uv run python run_benchmark.py --config configs/contextual_fabric_benchmark.yaml

# Override number of memories
uv run python run_benchmark.py --config configs/memory_retrieval_benchmark.yaml --memories 200

# Specify custom output file
uv run python run_benchmark.py --config configs/baseline_comparison.yaml --output my_results.json

# Enable debug output
uv run python run_benchmark.py --config configs/contextual_fabric_benchmark.yaml --debug
```

## Available Benchmarks

MemoryWeave includes several benchmarking tools:

1. **Contextual Fabric Benchmark**: Compares contextual fabric against hybrid BM25+vector retrieval
1. **Memory Retrieval Benchmark**: Evaluates different retrieval configurations
1. **Baseline Comparison**: Compares against industry standard BM25 and vector search
1. **Synthetic Benchmark**: Tests performance on generated datasets with controlled properties

## Running the Contextual Fabric Benchmark

The contextual fabric benchmark is the most comprehensive evaluation, specifically designed to showcase the advantages of the contextual fabric architecture over traditional retrieval methods.

### Quick Run

```bash
# Run using the unified benchmark system
uv run python run_benchmark.py --config configs/contextual_fabric_benchmark.yaml
```

### Legacy Script (For Backward Compatibility)

To run the contextual fabric benchmark with all memory sizes (20, 100, and 500) and generate visualizations using the legacy script:

```bash
# Run the benchmark script (handles everything)
./run_contextual_fabric_benchmark.sh
```

This script will:

1. Run benchmarks with 20, 100, and 500 memories
1. Save results to `benchmark_results/contextual_fabric_*.json`
1. Generate visualizations in `evaluation_charts/contextual_fabric_*/`

### Understanding the Results

The contextual fabric benchmark evaluates several retrieval capabilities:

1. **Conversation Context**: Can the system use conversation history to improve retrieval?
1. **Temporal Context**: Can the system find memories based on temporal references?
1. **Associative Links**: Can the system find related memories through associative links?
1. **Activation Patterns**: Does the system boost recently used or important memories?
1. **Episodic Memory**: Can the system retrieve memories from the same episode?

For each test case, the benchmark compares:

- **Contextual Fabric Strategy**: Our advanced approach using the contextual fabric architecture
- **Hybrid BM25+Vector Strategy**: A baseline representing current industry-standard approaches

The main metrics reported are:

- **Precision**: Percentage of retrieved results that are relevant
- **Recall**: Percentage of relevant results that were retrieved
- **F1 Score**: Harmonic mean of precision and recall

## Running Other Benchmarks

### Memory Retrieval Benchmark

This benchmark evaluates different retrieval configurations:

```bash
# Run the memory retrieval benchmark with the unified system
uv run python run_benchmark.py --config configs/memory_retrieval_benchmark.yaml

# Legacy method (deprecated)
uv run python -m benchmarks.memory_retrieval_benchmark
```

### Baseline Comparison

Compare against industry-standard BM25 and vector search:

```bash
# Run with the unified benchmark system
uv run python run_benchmark.py --config configs/baseline_comparison.yaml

# Legacy method (deprecated)
uv run python run_baseline_comparison.py --dataset sample_baseline_dataset.json --config baselines_config.yaml
```

### Synthetic Benchmark

Test with generated datasets with controlled properties:

```bash
# Run with the unified benchmark system
uv run python run_benchmark.py --config configs/synthetic_benchmark.yaml

# Legacy method (deprecated)
uv run python run_synthetic_benchmark.py
```

## Visualizing Benchmark Results

For all benchmarks, you can create visualizations:

```bash
# Visualizations are automatically generated when running benchmarks
# To only visualize existing results:
python examples/visualize_results.py results.json

# Create comparison charts
python examples/visualize_results.py --compare result1.json result2.json --output comparison.png
```

## Customizing Benchmarks

You can customize benchmark parameters by creating or modifying configuration files:

### Example: Customizing the Contextual Fabric Benchmark

Create a new configuration file `configs/contextual_fabric_custom.yaml`:

```yaml
name: "Contextual Fabric Custom"
type: "contextual_fabric"
description: "Custom contextual fabric evaluation with tuned parameters"
memories: 200
embedding_dim: 384
output_file: "benchmark_results/contextual_fabric_custom.json"
visualize: true
parameters:
  contextual_fabric_strategy:
    confidence_threshold: 0.1
    similarity_weight: 0.7
    associative_weight: 0.1
    temporal_weight: 0.1
    activation_weight: 0.1
    max_associative_hops: 2
  baseline_strategy:
    confidence_threshold: 0.1
    vector_weight: 0.4
    bm25_weight: 0.6
    bm25_b: 0.5
    bm25_k1: 1.5
```

Then run it:

```bash
uv run python run_benchmark.py --config configs/contextual_fabric_custom.yaml
```

### Example: Customizing Memory Retrieval Configurations

Create a new configuration file `configs/memory_retrieval_custom.yaml`:

```yaml
name: "Memory Retrieval Custom Configurations"
type: "memory_retrieval"
description: "Tests optimized configurations"
memories: 300
queries: 30
output_file: "benchmark_results/memory_retrieval_custom.json"
visualize: true
configurations:
  - name: "Precision-Focused"
    retriever_type: "components"
    confidence_threshold: 0.5
    semantic_coherence_check: true
    adaptive_retrieval: true
    use_two_stage_retrieval: true
    query_type_adaptation: true
  
  - name: "Recall-Focused"
    retriever_type: "components"
    confidence_threshold: 0.1
    semantic_coherence_check: false
    adaptive_retrieval: true
    use_two_stage_retrieval: true
    query_type_adaptation: true
```

## Using the DynamicContextAdapter in Benchmarks

If you want to evaluate the newly implemented DynamicContextAdapter:

```python
from memoryweave.components.dynamic_context_adapter import DynamicContextAdapter

# Create and configure the adapter
dynamic_adapter = DynamicContextAdapter()
dynamic_adapter.initialize({"adaptation_strength": 1.0, "enable_memory_size_adaptation": True})

# In your benchmark loop, add this step before retrieval:
adaptation_context = dynamic_adapter.process_query(
    query, {"memory_store": memory_store, "primary_query_type": query_type}
)

# Then use the adapted parameters in retrieval
retrieval_context = {"query": query, "query_embedding": query_embedding, **adaptation_context}
results = retrieval_strategy.retrieve(query_embedding, top_k=5, context=retrieval_context)
```

## Troubleshooting

### BM25 Warnings

When running benchmarks with synthetic data, you may see warnings like:

```
WARNING:root:BM25 retrieval failed: minimal could not be calculated, returning default
```

These are expected with synthetic test data and don't indicate a problem. The hybrid strategy will fall back to vector retrieval when BM25 indexing fails.

### Memory Usage

Benchmarks with large memory sizes (500+) may require significant memory. If you encounter memory issues:

1. Reduce the number of memories:

```bash
uv run python run_benchmark.py --config configs/contextual_fabric_benchmark.yaml --memories 200
```

2. Run with a smaller embedding dimension (if configurable in your benchmark):

```yaml
# In your config file
embedding_dim: 128  # Reduced from default 384
```
