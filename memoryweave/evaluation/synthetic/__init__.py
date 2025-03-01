# memoryweave/evaluation/synthetic/__init__.py
"""
Synthetic test generation tools for MemoryWeave evaluation.

This package provides utilities for generating synthetic memory datasets
and evaluation queries to reduce bias in testing and enable more
comprehensive evaluation.
"""

from memoryweave.evaluation.synthetic.generators import (
    SyntheticMemoryGenerator,
    SyntheticQueryGenerator,
)

from memoryweave.evaluation.synthetic.semantic_generator import (
    SemanticDataGenerator,
    generate_semantic_dataset,
    MemoryRelationship,
    MemorySeries,
    SemanticMemory,
)

__all__ = [
    "SyntheticMemoryGenerator",
    "SyntheticQueryGenerator",
    "SemanticDataGenerator",
    "generate_semantic_dataset",
    "MemoryRelationship",
    "MemorySeries",
    "SemanticMemory",
]