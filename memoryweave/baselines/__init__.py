"""
Baseline retrieval implementations for MemoryWeave.

This package provides standard information retrieval baselines
for benchmarking MemoryWeave against established methods.
"""

from memoryweave.baselines.base import BaselineRetriever
from memoryweave.baselines.bm25 import BM25Retriever
from memoryweave.baselines.vector import VectorBaselineRetriever

__all__ = ["BaselineRetriever", "BM25Retriever", "VectorBaselineRetriever"]