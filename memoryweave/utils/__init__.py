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


def _get_device(self, device: str = "auto") -> str:
    import torch

    """Choose appropriate device."""
    if device != "auto":
        return device
    if torch.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


__all__ = ["cosine_similarity_batched", "embed_text_batch", "fuzzy_string_match", "_get_device"]
