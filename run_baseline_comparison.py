#!/usr/bin/env python3
"""
CLI tool for running baseline comparisons against MemoryWeave.

This script allows comparing MemoryWeave's retrieval capabilities
with standard baseline methods like BM25 and vector search.
"""

import argparse
import json
import os
import sys
import yaml
from typing import List, Dict, Any, Type

import numpy as np

from memoryweave.baselines import BaselineRetriever, BM25Retriever, VectorBaselineRetriever
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.retrieval_strategies import SimilarityRetriever, HybridRetriever
from memoryweave.evaluation.baseline_comparison import (
    BaselineComparison, BaselineConfig, ComparisonResult
)
from memoryweave.interfaces.retrieval import Query, QueryType, IRetrievalStrategy
from memoryweave.storage.memory_store import Memory, MemoryStore


def load_dataset(path: str) -> Dict[str, Any]:
    """Load a dataset from a JSON file.
    
    Args:
        path: Path to the dataset file
        
    Returns:
        Dictionary containing memories, queries and relevant IDs
    """
    with open(path, 'r') as f:
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
            extracted_entities=q_data.get("entities", []),
            metadata=q_data.get("metadata", {})
        )
        queries.append(query)
    
    # Get relevant memory IDs
    relevant_ids = data.get("relevant_ids", [])
    
    return {
        "memories": memories,
        "queries": queries,
        "relevant_ids": relevant_ids
    }


def get_retriever(retriever_type: str, **kwargs) -> IRetrievalStrategy:
    """Get a retriever of the specified type.
    
    Args:
        retriever_type: Type of retriever to create
        **kwargs: Additional parameters for the retriever
        
    Returns:
        An initialized retriever
    """
    if retriever_type == "similarity":
        return SimilarityRetriever(**kwargs)
    elif retriever_type == "hybrid":
        return HybridRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def load_baseline_configs(config_path: str) -> List[BaselineConfig]:
    """Load baseline configurations from a YAML file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        List of baseline configurations
    """
    retriever_classes: Dict[str, Type[BaselineRetriever]] = {
        "bm25": BM25Retriever,
        "vector": VectorBaselineRetriever
    }
    
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    baseline_configs = []
    for config in configs:
        retriever_type = config.get("type")
        if retriever_type not in retriever_classes:
            print(f"Warning: Unknown baseline type '{retriever_type}', skipping")
            continue
        
        baseline_configs.append(
            BaselineConfig(
                name=config.get("name", retriever_type),
                retriever_class=retriever_classes[retriever_type],
                parameters=config.get("parameters", {})
            )
        )
    
    return baseline_configs


def main():
    """Run baseline comparison based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare MemoryWeave against baseline retrieval methods"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Path to dataset JSON file containing memories, queries, and relevant IDs"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to baseline configuration YAML file"
    )
    
    parser.add_argument(
        "--retriever", 
        type=str, 
        default="similarity",
        choices=["similarity", "hybrid"],
        help="MemoryWeave retriever type to use"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="baseline_comparison_results.json",
        help="Path to save comparison results"
    )
    
    parser.add_argument(
        "--html-report", 
        type=str, 
        default=None,
        help="Path to save HTML report (optional)"
    )
    
    parser.add_argument(
        "--visualization", 
        type=str, 
        default=None,
        help="Path to save visualization (optional)"
    )
    
    parser.add_argument(
        "--max-results", 
        type=int, 
        default=10,
        help="Maximum number of results to retrieve"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.0,
        help="Minimum similarity score threshold"
    )
    
    args = parser.parse_args()
    
    # Check that input files exist
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' not found")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)
    
    # Load dataset
    try:
        dataset = load_dataset(args.dataset)
        print(f"Loaded dataset with {len(dataset['memories'])} memories and {len(dataset['queries'])} queries")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Load baseline configurations
    try:
        baseline_configs = load_baseline_configs(args.config)
        print(f"Loaded {len(baseline_configs)} baseline configurations")
    except Exception as e:
        print(f"Error loading baseline configurations: {e}")
        sys.exit(1)
    
    # Initialize memory manager
    memory_store = MemoryStore()
    memory_store.add_multiple(dataset["memories"])
    memory_manager = MemoryManager(memory_store=memory_store)
    
    # Initialize MemoryWeave retriever
    try:
        memoryweave_retriever = get_retriever(args.retriever)
        print(f"Using MemoryWeave retriever: {args.retriever}")
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        sys.exit(1)
    
    # Create comparison framework
    comparison = BaselineComparison(
        memory_manager=memory_manager,
        memoryweave_retriever=memoryweave_retriever,
        baseline_configs=baseline_configs,
        metrics=["precision", "recall", "f1", "mrr"]
    )
    
    # Run comparison
    print(f"Running baseline comparison...")
    result = comparison.run_comparison(
        queries=dataset["queries"],
        relevant_memory_ids=dataset["relevant_ids"],
        max_results=args.max_results,
        threshold=args.threshold
    )
    
    # Save results
    comparison.save_results(result, args.output)
    print(f"Saved comparison results to {args.output}")
    
    # Generate visualization if requested
    if args.visualization:
        comparison.visualize_results(result, args.visualization)
        print(f"Saved visualization to {args.visualization}")
    
    # Generate HTML report if requested
    if args.html_report:
        comparison.generate_html_report(result, args.html_report)
        print(f"Saved HTML report to {args.html_report}")
    
    # Print summary
    print("\nComparison Summary:")
    print("-------------------")
    
    # Print MemoryWeave metrics
    mw_metrics = result.memoryweave_metrics["average"]
    print(f"MemoryWeave ({args.retriever}):")
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