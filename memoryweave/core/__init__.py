"""
Core module for MemoryWeave.

This module contains the core functionality for the MemoryWeave memory management system.
"""

from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.core.memory_encoding import MemoryEncoder
from memoryweave.core.retrieval import ContextualRetriever  # This now imports from deprecated

__all__ = ["ContextualMemory", "MemoryEncoder", "ContextualRetriever"]
