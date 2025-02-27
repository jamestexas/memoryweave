"""
Evaluation tools for measuring MemoryWeave performance.

This module provides metrics and methods for evaluating the effectiveness
of the contextual fabric approach to memory management.
"""

from memoryweave.evaluation.coherence_metrics import (
    coherence_score,
    context_relevance,
    evaluate_conversation,
    response_consistency,
)

__all__ = ["coherence_score", "context_relevance", "response_consistency", "evaluate_conversation"]
