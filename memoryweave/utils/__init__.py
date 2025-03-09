"""
Utility functions for MemoryWeave.

This module provides helper functions and utilities for working with
the contextual fabric memory system.
"""

import importlib
import logging
import sys

from memoryweave.utils.similarity import (
    cosine_similarity_batched,
    embed_text_batch,
    fuzzy_string_match,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("memoryweave")


def _load_module(module_name: str) -> bool:
    """Checks if a module is loaded, if not, checks if it can be, and loads it if possible."""
    result = False
    if module_name in sys.modules:
        logger.debug(f"{module_name!r} already in sys.modules")
        result = True
    elif (spec := importlib.util.find_spec(module_name)) is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        result = True
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        logger.debug(f"{module_name!r} has been imported")

    else:
        logger.debug(f"can't find the {module_name!r} module")
    return result


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
