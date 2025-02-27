# MemoryWeave Notebooks and Tests

This directory contains notebooks and test scripts for exploring and evaluating the MemoryWeave memory management system.

## About MemoryWeave

[MemoryWeave](https://github.com/jamestexas/memoryweave) is an experimental approach to memory management for language models using a "contextual fabric" approach inspired by biological memory systems. Rather than traditional knowledge graph approaches with discrete nodes and edges, MemoryWeave focuses on capturing rich contextual signatures of information for improved long-context coherence in LLM conversations.

## Available Notebooks and Tests

| File | Description |
|------|-------------|
| [`test_memory.py`](test_memory.py) | Basic demonstration of MemoryWeave's core memory functionality, showing storage and retrieval of simple memories. |
| [`test_confidence_thresholding.py`](test_confidence_thresholding.py) | Demonstrates how confidence thresholding improves retrieval precision by filtering out low-relevance memories. |
| [`test_category_consolidation.py`](test_category_consolidation.py) | Tests the category consolidation feature which reduces category fragmentation through hierarchical clustering. |
| [`test_dynamic_vigilance.py`](test_dynamic_vigilance.py) | Demonstrates different dynamic vigilance strategies for ART-inspired clustering. |
| [`test_large_memory_clustering.py`](test_large_memory_clustering.py) | Tests clustering performance with larger memory sets. |
| [`benchmark_memory.py`](benchmark_memory.py) | Comprehensive benchmark of different MemoryWeave configurations, measuring precision, recall, and performance. |

## Running the Tests

All test scripts are designed to be run from the project root with:

```bash
python notebooks/test_name.py
```

Or using `uv`:

```bash
uv run python notebooks/test_name.py
```

## Output Files

Test results, charts, and visualizations are saved to the `output/` directory in the project root.

## Key Features Demonstrated

These notebooks demonstrate several key features of MemoryWeave:

1. **Contextual Fabric Approach**: Memories store rich contextual signatures rather than isolated facts
2. **ART-Inspired Clustering**: Self-organizing memory categorization based on Adaptive Resonance Theory
3. **Category Consolidation**: Hierarchical clustering to reduce category fragmentation
4. **Confidence Thresholding**: Filtering out low-confidence retrievals for improved precision
5. **Semantic Coherence Check**: Ensuring retrieved memories form a coherent set
6. **Adaptive K Selection**: Dynamically selecting how many memories to retrieve

## Additional Resources

- [Main MemoryWeave Repository](https://github.com/jamestexas/memoryweave)
- [ROADMAP.md](../ROADMAP.md): Future development plans for MemoryWeave
