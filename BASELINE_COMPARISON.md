# MemoryWeave Baseline Comparison

This document explains how to run and interpret the baseline comparison in MemoryWeave.

## Overview

The baseline comparison framework allows you to objectively evaluate MemoryWeave's retrieval capabilities against industry-standard methods like BM25 and vector search. This helps identify scenarios where the contextual fabric approach excels or needs improvement.

## Running the Comparison

### Quick Start with the Unified Benchmark System

The recommended way to run baseline comparisons is to use the unified benchmark system:

```bash
# Run with the sample dataset and configuration
uv run python run_benchmark.py --config configs/baseline_comparison.yaml
```

This will:
1. Load the sample dataset with memories and queries
2. Compare MemoryWeave against BM25 and vector search baselines
3. Generate results in JSON, visualization, and HTML report formats

### Legacy Method (Deprecated)

For backward compatibility, you can still use the direct script:

```bash
# Run with the sample dataset and configuration
uv run python run_baseline_comparison.py --dataset sample_baseline_dataset.json --config baselines_config.yaml
```

### Using Your Own Dataset

To compare on your own dataset, create a JSON file with the following structure:

```json
{
  "memories": [
    {
      "id": "mem1",
      "content": {
        "text": "Your memory text here",
        "metadata": {}
      },
      "embedding": [0.1, 0.2, 0.3, ...],
      "metadata": {
        "category": "fact",
        "topic": "example"
      }
    },
    ...
  ],
  "queries": [
    {
      "text": "Your query text here",
      "embedding": [0.1, 0.2, 0.3, ...],
      "keywords": ["keyword1", "keyword2"],
      "entities": ["entity1"]
    },
    ...
  ],
  "relevant_ids": [
    ["mem1", "mem3"],
    ["mem2"],
    ...
  ]
}
```

Then update your configuration file to point to your dataset:

```yaml
# In configs/baseline_comparison.yaml
name: "Baseline Comparison"
type: "baseline"
description: "Compares MemoryWeave against standard baseline methods"
output_file: "benchmark_results/baseline_comparison_results.json"
visualize: true
parameters:
  dataset: "path/to/your_dataset.json"  # Update this line
  config: "baselines_config.yaml"
  retriever: "hybrid_bm25"
  max_results: 10
  threshold: 0.0
```

And run the benchmark:

```bash
uv run python run_benchmark.py --config configs/baseline_comparison.yaml
```

### Customizing Baseline Configurations

You can customize the baseline configurations by modifying the `baselines_config.yaml` file:

```yaml
# BM25 baseline with default parameters
- name: bm25
  type: bm25
  parameters:
    b: 0.75  # Length normalization parameter
    k1: 1.2  # Term frequency scaling parameter

# Vector search baseline
- name: vector_search
  type: vector
  parameters:
    use_exact_search: true  # Use exact search instead of approximate

# Add more baseline configurations as needed
```

### Additional Options

The unified benchmark system supports several options:

```bash
# Override the output path
uv python run_benchmark.py --config configs/baseline_comparison.yaml --output my_results.json

# Enable debug logging
uv python run_benchmark.py --config configs/baseline_comparison.yaml --debug

# Disable visualization generation
uv python run_benchmark.py --config configs/baseline_comparison.yaml --no-viz
```

## Example Visualization

For a more direct example, you can run the provided example script:

```bash
uv run python examples/baseline_comparison_example.py
```

This will generate:
- `baseline_comparison_example.json` - Full comparison results
- `baseline_comparison_chart.png` - Visualization of metrics
- `baseline_comparison_report.html` - HTML report with interactive elements

## Interpreting Results

The comparison generates several metrics for each retrieval system:

- **Precision**: Percentage of retrieved items that are relevant
- **Recall**: Percentage of relevant items that were retrieved
- **F1 Score**: Harmonic mean of precision and recall
- **MRR (Mean Reciprocal Rank)**: Measures how early the first relevant item appears

The HTML report and visualization provide a side-by-side comparison of these metrics, helping you identify the strengths and weaknesses of each approach.

Performance metrics are also recorded, allowing you to compare query time and efficiency across different methods.

## Using in Custom Benchmarks

You can integrate the baseline comparison into your own benchmarking code:

```python
from memoryweave.baselines import BM25Retriever, VectorBaselineRetriever
from memoryweave.evaluation.baseline_comparison import BaselineComparison, BaselineConfig

# Initialize components
memory_manager = get_your_memory_manager()
memoryweave_retriever = get_your_retriever()

# Define baseline configurations
baseline_configs = [
    BaselineConfig(
        name="bm25",
        retriever_class=BM25Retriever,
        parameters={"b": 0.75, "k1": 1.2}
    ),
    BaselineConfig(
        name="vector_search",
        retriever_class=VectorBaselineRetriever,
        parameters={"use_exact_search": True}
    )
]

# Create comparison framework
comparison = BaselineComparison(
    memory_manager=memory_manager,
    memoryweave_retriever=memoryweave_retriever,
    baseline_configs=baseline_configs,
    metrics=["precision", "recall", "f1", "mrr"]
)

# Run comparison with your data
result = comparison.run_comparison(
    queries=your_queries,
    relevant_memory_ids=your_relevant_ids,
    max_results=10,
    threshold=0.0
)

# Visualize and save results
comparison.visualize_results(result, "visualization.png")
comparison.generate_html_report(result, "report.html")
comparison.save_results(result, "results.json")
```

## Creating Custom Benchmark Configurations

You can create custom baseline comparison configurations by creating a new YAML file:

```yaml
name: "Custom Baseline Comparison"
type: "baseline"
description: "Your custom comparison description"
output_file: "custom_results.json"
visualize: true
parameters:
  dataset: "your_dataset.json"
  config: "your_baselines_config.yaml"
  retriever: "hybrid_bm25"  # Options: "similarity", "hybrid", "hybrid_bm25"
  max_results: 10
  threshold: 0.0
```

Save this to a file (e.g., `configs/custom_baseline.yaml`) and run:

```bash
uv python run_benchmark.py --config configs/custom_baseline.yaml
```

This approach allows you to maintain different comparison configurations for different use cases.