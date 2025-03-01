#!/usr/bin/env python
"""
Run the synthetic benchmark for MemoryWeave retrieval.

This script generates synthetic test data and runs benchmarks on various
retrieval configurations to evaluate performance in a controlled environment.
"""

import argparse
import logging
import sys

from memoryweave.evaluation.synthetic.benchmark import main as benchmark_main

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("synthetic_benchmark.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Pass arguments to benchmark main function
    benchmark_main()