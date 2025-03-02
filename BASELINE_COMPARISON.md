# MemoryWeave Baseline Comparison

This document explains how to run and interpret the baseline comparison in MemoryWeave.

## Overview

The baseline comparison framework allows you to objectively evaluate MemoryWeave's retrieval capabilities against industry-standard methods like BM25 and vector search. This helps identify scenarios where the contextual fabric approach excels or needs improvement.

## Running the Comparison

### Quick Start with the Sample Dataset

The simplest way to run the baseline comparison is to use the provided sample dataset:

```bash
# Run with the sample dataset and configuration
uv run python run_baseline_comparison.py --dataset sample_baseline_dataset.json --config baselines_config.yaml
```

This will:
1. Load the sample dataset with 8 memories and 3 queries
2. Compare MemoryWeave against BM25 and vector search baselines
3. Generate results in JSON, visualization, and HTML report formats

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

Then run the comparison with your dataset:

```bash
uv run python run_baseline_comparison.py --dataset your_dataset.json --config baselines_config.yaml
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

The comparison tool supports several options:

```bash
# Specify a different retriever type
uv run python run_baseline_comparison.py --dataset sample_baseline_dataset.json --config baselines_config.yaml --retriever hybrid

# Change the output paths
uv run python run_baseline_comparison.py --dataset sample_baseline_dataset.json --config baselines_config.yaml --output my_results.json --html-report my_report.html --visualization my_viz.png

# Adjust retrieval parameters
uv run python run_baseline_comparison.py --dataset sample_baseline_dataset.json --config baselines_config.yaml --max-results 20 --threshold 0.3
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

You can also integrate the baseline comparison into your own benchmarking code:

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

This allows you to incorporate baseline comparison into any custom evaluation pipeline.