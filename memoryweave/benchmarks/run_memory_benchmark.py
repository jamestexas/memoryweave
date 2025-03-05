#!/usr/bin/env python3
"""
run_memory_benchmark.py - A streamlined benchmark runner for MemoryWeave

This script runs performance benchmarks for different memory retrieval configurations
and produces standardized visualizations and metrics.

Usage:
    python run_memory_benchmark.py --config benchmark_config.yaml --output results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import yaml

from memoryweave.benchmarks.performance.memory_retrieval_benchmark import (
    MemoryRetrievalBenchmark,
    MemoryRetrievalConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("memory_benchmark")


def load_config(config_path: str) -> Dict:
    """Load benchmark configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        if path.suffix.lower() in [".yaml", ".yml"]:
            with open(path) as f:
                return yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            with open(path) as f:
                return json.load(f)
        else:
            logger.error(f"Unsupported configuration file format: {path.suffix}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def main():
    """Run the memory benchmark."""
    parser = argparse.ArgumentParser(description="MemoryWeave Memory Retrieval Benchmark")
    parser.add_argument(
        "--config",
        "-c",
        default="memoryweave/benchmarks/configs/memory_retrieval_benchmark.yaml",
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to save benchmark results (overrides config file)",
    )
    parser.add_argument(
        "--memories",
        "-m",
        type=int,
        default=None,
        help="Number of memories to test (overrides config file)",
    )
    parser.add_argument(
        "--queries",
        "-q",
        type=int,
        default=None,
        help="Number of queries to run (overrides config file)",
    )
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualizations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load configuration
    config_data = load_config(args.config)

    # Extract benchmark configurations
    benchmark_configs = []
    for cfg in config_data.get("configurations", []):
        # Create a MemoryRetrievalConfig from each entry
        config = MemoryRetrievalConfig(
            name=cfg.get("name", "Unnamed"),
            description=cfg.get("description", ""),
            retriever_type=cfg.get("retriever_type", "legacy"),
            embedding_dim=cfg.get("embedding_dim", config_data.get("embedding_dim", 384)),
            memories_to_test=args.memories or cfg.get("memories", config_data.get("memories", 500)),
            queries_to_test=args.queries or cfg.get("queries", config_data.get("queries", 50)),
            output_file=args.output
            or cfg.get("output_file", config_data.get("output_file", "benchmark_results.json")),
            confidence_threshold=cfg.get("confidence_threshold", 0.3),
            semantic_coherence_check=cfg.get("semantic_coherence_check", False),
            adaptive_retrieval=cfg.get("adaptive_retrieval", False),
            use_two_stage_retrieval=cfg.get("use_two_stage_retrieval", False),
            query_type_adaptation=cfg.get("query_type_adaptation", False),
            dynamic_threshold_adjustment=cfg.get("dynamic_threshold_adjustment", False),
            memory_decay_enabled=cfg.get("memory_decay_enabled", False),
            use_ann=cfg.get("use_ann", False),
        )
        benchmark_configs.append(config)

    # Create and run benchmark
    benchmark = MemoryRetrievalBenchmark(benchmark_configs)
    results = benchmark.run_all()

    # Save results
    output_file = args.output or config_data.get("output_file", "benchmark_results.json")
    benchmark.save_results(output_file)
    logger.info(f"Results saved to {output_file}")

    # Generate visualizations if requested
    if args.visualize or config_data.get("visualize", False):
        benchmark.visualize_results()


if __name__ == "__main__":
    main()
