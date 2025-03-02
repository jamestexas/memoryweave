# MemoryWeave Benchmarking Guide

This guide explains how to run benchmarks to evaluate and compare MemoryWeave's retrieval capabilities, with a special focus on comparing the contextual fabric approach with traditional retrieval methods.

## Available Benchmarks

MemoryWeave includes several benchmarking tools:

1. **Contextual Fabric Benchmark**: Compares contextual fabric against hybrid BM25+vector retrieval
2. **Memory Retrieval Benchmark**: Evaluates different retrieval configurations
3. **Baseline Comparison**: Compares against industry standard BM25 and vector search
4. **Synthetic Benchmark**: Tests performance on generated datasets with controlled properties

## Running the Contextual Fabric Benchmark

The contextual fabric benchmark is the most comprehensive evaluation, specifically designed to showcase the advantages of the contextual fabric architecture over traditional retrieval methods.

### Quick Run

To run the contextual fabric benchmark with all memory sizes (20, 100, and 500) and generate visualizations:

```bash
# Run the benchmark script (handles everything)
./run_contextual_fabric_benchmark.sh
```

This script will:
1. Run benchmarks with 20, 100, and 500 memories
2. Save results to `benchmark_results/contextual_fabric_*.json`
3. Generate visualizations in `evaluation_charts/contextual_fabric_*/`

### Manual Execution

If you prefer to run specific tests manually:

```bash
# Run with a specific memory size
uv run python -m benchmarks.contextual_fabric_benchmark --memories 100 --output my_results.json

# Visualize the results
uv run python benchmarks/visualize_contextual_fabric.py my_results.json output_folder/
```

### Understanding the Results

The contextual fabric benchmark evaluates several retrieval capabilities:

1. **Conversation Context**: Can the system use conversation history to improve retrieval?
2. **Temporal Context**: Can the system find memories based on temporal references?
3. **Associative Links**: Can the system find related memories through associative links?
4. **Activation Patterns**: Does the system boost recently used or important memories?
5. **Episodic Memory**: Can the system retrieve memories from the same episode?

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
# Run the memory retrieval benchmark
uv run python -m benchmarks.memory_retrieval_benchmark
```

### Baseline Comparison

Compare against industry-standard BM25 and vector search:

```bash
# Run baseline comparison with default settings
uv run python run_baseline_comparison.py

# Run with custom settings
uv run python run_baseline_comparison.py --dataset sample_baseline_dataset.json --config baselines_config.yaml --output results.json --html-report report.html
```

### Synthetic Benchmark

Test with generated datasets with controlled properties:

```bash
# Run synthetic benchmark with default settings
uv run python run_synthetic_benchmark.py

# Run with custom configuration
uv run python run_synthetic_benchmark.py --config configs/benchmark_advanced.json
```

## Visualizing Benchmark Results

For all benchmarks, you can create visualizations:

```bash
# Visualize any benchmark results
python examples/visualize_results.py results.json

# Create comparison charts
python examples/visualize_results.py --compare result1.json result2.json --output comparison.png
```

## Customizing Benchmarks

You can customize benchmark parameters by modifying the benchmark scripts or using configuration files:

### For Contextual Fabric Benchmark:

```python
# In your own script:
from benchmarks.contextual_fabric_benchmark import ContextualFabricBenchmark

benchmark = ContextualFabricBenchmark(embedding_dim=768)

# Customize components
benchmark.contextual_fabric_strategy.initialize({
    "confidence_threshold": 0.1,
    "similarity_weight": 0.6,  # Adjust weights
    "associative_weight": 0.2,
    "temporal_weight": 0.1,
    "activation_weight": 0.1,
})

# Run benchmark
benchmark.run_benchmark(num_memories=200, output_file="custom_results.json")
```

## Adding the DynamicContextAdapter to Benchmarks

If you want to evaluate the newly implemented DynamicContextAdapter:

```python
from memoryweave.components.dynamic_context_adapter import DynamicContextAdapter

# Create and configure the adapter
dynamic_adapter = DynamicContextAdapter()
dynamic_adapter.initialize({
    "adaptation_strength": 1.0,
    "enable_memory_size_adaptation": True
})

# In your benchmark loop, add this step before retrieval:
adaptation_context = dynamic_adapter.process_query(
    query, 
    {
        "memory_store": memory_store,
        "primary_query_type": query_type
    }
)

# Then use the adapted parameters in retrieval
retrieval_context = {
    "query": query,
    "query_embedding": query_embedding,
    **adaptation_context
}
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
uv run python -m benchmarks.contextual_fabric_benchmark --memories 200
```

2. Run with a smaller embedding dimension:
```bash
uv run python -m benchmarks.contextual_fabric_benchmark --embedding-dim 384
```