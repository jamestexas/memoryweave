#!/usr/bin/env python3
"""
Example showing how to use the baseline comparison framework.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

from memoryweave.baselines import BM25Retriever, VectorBaselineRetriever
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.retrieval_strategies import SimilarityRetriever
from memoryweave.evaluation.baseline_comparison import (
    BaselineComparison, BaselineConfig
)
from memoryweave.interfaces.retrieval import Query, QueryType
from memoryweave.storage.memory_store import Memory, MemoryStore


def main():
    """Run a simple baseline comparison example."""
    # Load the sample dataset
    dataset_path = Path(__file__).parent.parent / "sample_baseline_dataset.json"
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    # Create memory objects
    memories = []
    for mem_data in data.get("memories", []):
        embedding = np.array(mem_data.get("embedding", [])) if "embedding" in mem_data else None
        
        memory = Memory(
            id=mem_data.get("id", str(len(memories))),
            embedding=embedding,
            content=mem_data.get("content", {"text": "", "metadata": {}}),
            metadata=mem_data.get("metadata", {})
        )
        memories.append(memory)
    
    # Create query objects
    queries = []
    for q_data in data.get("queries", []):
        embedding = np.array(q_data.get("embedding", [])) if "embedding" in q_data else None
        
        query = Query(
            text=q_data.get("text", ""),
            embedding=embedding,
            query_type=QueryType.UNKNOWN,
            extracted_keywords=q_data.get("keywords", []),
            extracted_entities=q_data.get("entities", [])
        )
        queries.append(query)
    
    # Get relevant memory IDs
    relevant_ids = data.get("relevant_ids", [])
    
    # Initialize memory manager
    memory_store = MemoryStore()
    memory_store.add_multiple(memories)
    memory_manager = MemoryManager(memory_store=memory_store)
    
    # Initialize MemoryWeave retriever
    memoryweave_retriever = SimilarityRetriever()
    
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
    
    # Run comparison
    print("Running baseline comparison...")
    result = comparison.run_comparison(
        queries=queries,
        relevant_memory_ids=relevant_ids,
        max_results=5,
        threshold=0.0
    )
    
    # Save results
    output_path = "baseline_comparison_example.json"
    comparison.save_results(result, output_path)
    print(f"Saved comparison results to {output_path}")
    
    # Generate visualization
    visualization_path = "baseline_comparison_chart.png"
    comparison.visualize_results(result, visualization_path)
    print(f"Saved visualization to {visualization_path}")
    
    # Generate HTML report
    html_report_path = "baseline_comparison_report.html"
    comparison.generate_html_report(result, html_report_path, title="MemoryWeave Baseline Comparison Example")
    print(f"Saved HTML report to {html_report_path}")
    
    # Print summary
    print("\nComparison Summary:")
    print("-------------------")
    
    # Print MemoryWeave metrics
    mw_metrics = result.memoryweave_metrics["average"]
    print(f"MemoryWeave (SimilarityRetriever):")
    print(f"  Precision: {mw_metrics.get('precision', 0):.4f}")
    print(f"  Recall: {mw_metrics.get('recall', 0):.4f}")
    print(f"  F1: {mw_metrics.get('f1', 0):.4f}")
    print(f"  MRR: {mw_metrics.get('mrr', 0):.4f}")
    print(f"  Avg Query Time: {result.runtime_stats['memoryweave'].get('avg_query_time', 0):.5f}s")
    
    # Print baseline metrics
    for name, metrics in result.baseline_metrics.items():
        print(f"\n{name}:")
        avg_metrics = metrics["average"]
        print(f"  Precision: {avg_metrics.get('precision', 0):.4f}")
        print(f"  Recall: {avg_metrics.get('recall', 0):.4f}")
        print(f"  F1: {avg_metrics.get('f1', 0):.4f}")
        print(f"  MRR: {avg_metrics.get('mrr', 0):.4f}")
        print(f"  Avg Query Time: {result.runtime_stats[name].get('avg_query_time', 0):.5f}s")


if __name__ == "__main__":
    main()