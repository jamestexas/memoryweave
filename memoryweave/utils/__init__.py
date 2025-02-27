"""
Utility functions for MemoryWeave.

This module provides helper functions and utilities for working with
the contextual fabric memory system.
"""

from memoryweave.utils.similarity import (
    cosine_similarity_batched,
    embed_text_batch,
    fuzzy_string_match,
)

__all__ = ["cosine_similarity_batched", "embed_text_batch", "fuzzy_string_match"]
