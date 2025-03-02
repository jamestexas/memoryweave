#!/usr/bin/env python
"""
Run the semantic benchmark for MemoryWeave retrieval.

This script tests memory retrieval using semantically rich synthetic data
with relationships, contradictions, and temporal structures to provide
more realistic and challenging evaluation scenarios.
"""

import argparse
import logging
import os
import sys

from memoryweave.evaluation.synthetic.benchmark import BenchmarkConfig, SyntheticBenchmark
from memoryweave.evaluation.synthetic.semantic_generator import generate_semantic_dataset


def main():
    """Run the semantic benchmark with enhanced test generation."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("semantic_benchmark.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("semantic_benchmark")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Semantic Memory Retrieval Benchmark")
    parser.add_argument(
        "--num-memories", type=int, default=100, help="Number of memories to generate"
    )
    parser.add_argument("--num-queries", type=int, default=20, help="Number of test queries to run")
    parser.add_argument(
        "--num-series", type=int, default=5, help="Number of memory series to generate"
    )
    parser.add_argument(
        "--contradictions", type=float, default=0.1, help="Ratio of contradictory memories"
    )
    parser.add_argument(
        "--complexity",
        type=str,
        default="medium",
        choices=["simple", "medium", "complex"],
        help="Complexity level for generated data",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Path to existing dataset file")
    parser.add_argument(
        "--save-dataset",
        type=str,
        default="datasets/semantic_dataset.json",
        help="Path to save generated dataset",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="semantic_benchmark_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    logger.info("Starting semantic benchmark")

    # Define configurations to benchmark
    configs = [
        BenchmarkConfig(
            name="Basic",
            retriever_type="basic",
            confidence_threshold=0.3,
            top_k=5,
            evaluation_mode=True,  # Ensure special case handling is disabled
        ),
        BenchmarkConfig(
            name="Semantic-Coherence",
            retriever_type="components",
            confidence_threshold=0.3,
            top_k=5,
            semantic_coherence_check=True,
            evaluation_mode=True,
        ),
        BenchmarkConfig(
            name="Query-Adaptation",
            retriever_type="components",
            confidence_threshold=0.3,
            top_k=5,
            query_type_adaptation=True,
            evaluation_mode=True,
        ),
        BenchmarkConfig(
            name="Two-Stage",
            retriever_type="components",
            confidence_threshold=0.3,
            top_k=5,
            use_two_stage_retrieval=True,
            evaluation_mode=True,
        ),
        BenchmarkConfig(
            name="Full-Advanced",
            retriever_type="components",
            confidence_threshold=0.3,
            top_k=5,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            dynamic_threshold_adjustment=True,
            evaluation_mode=True,
        ),
        BenchmarkConfig(
            name="Precision-Focused",
            retriever_type="components",
            confidence_threshold=0.5,  # Higher threshold for better precision
            top_k=3,  # Fewer results
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            evaluation_mode=True,
        ),
        BenchmarkConfig(
            name="Recall-Focused",
            retriever_type="components",
            confidence_threshold=0.2,  # Lower threshold for better recall
            top_k=10,  # More results
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            evaluation_mode=True,
        ),
    ]

    # Generate or load semantic dataset
    if args.dataset and os.path.exists(args.dataset):
        logger.info(f"Using existing dataset from {args.dataset}")
        dataset_path = args.dataset
    else:
        logger.info(
            f"Generating semantic dataset with {args.num_memories} memories and {args.num_queries} queries"
        )
        dataset_path = args.save_dataset

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)

        # Generate dataset
        generate_semantic_dataset(
            output_path=dataset_path,
            num_memories=args.num_memories,
            num_queries=args.num_queries,
            num_series=args.num_series,
            contradiction_rate=args.contradictions,
            complexity=args.complexity,
            random_seed=args.random_seed,
        )
        logger.info(f"Dataset saved to {dataset_path}")

    # Initialize benchmark
    benchmark = SyntheticBenchmark(
        configs=configs,
        dataset_path=dataset_path,
        random_seed=args.random_seed,
    )

    # Run benchmark
    logger.info("Running benchmark with all configurations")
    results = benchmark.run_benchmark(save_path=args.save_results)

    # Log results summary
    logger.info("\nBenchmark Results Summary:")
    # Check the structure of results - it's a dict with 'results' and 'summary' keys
    if isinstance(results, dict) and 'results' in results:
        for result in results['results']:
            logger.info(f"Configuration: {result.config.name}")
            logger.info(f"  Precision: {result.precision:.4f}")
            logger.info(f"  Recall: {result.recall:.4f}")
            logger.info(f"  F1 Score: {result.f1_score:.4f}")
            logger.info(f"  Avg Query Time: {result.avg_query_time:.4f}s")
            logger.info(f"  Avg Results: {result.avg_retrieval_count:.2f}")

        # Find best configurations
        best_precision = max(results['results'], key=lambda r: r.precision)
        best_recall = max(results['results'], key=lambda r: r.recall)
        best_f1 = max(results['results'], key=lambda r: r.f1_score)

        logger.info("\nBest Configurations:")
        logger.info(f"Best Precision: {best_precision.config.name} ({best_precision.precision:.4f})")
        logger.info(f"Best Recall: {best_recall.config.name} ({best_recall.recall:.4f})")
        logger.info(f"Best F1 Score: {best_f1.config.name} ({best_f1.f1_score:.4f})")
    else:
        # Handle string result case
        logger.info(f"Result: {results}")

    logger.info(f"\nFull results saved to {args.save_results}")
    logger.info("Visualization saved to semantic_benchmark_results.png")


if __name__ == "__main__":
    main()
