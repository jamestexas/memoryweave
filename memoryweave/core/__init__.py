"""
Core components of the MemoryWeave memory management system.

This module contains the fundamental building blocks of the contextual fabric
approach to memory management, including memory encoding, storage, and retrieval.
"""

from memoryweave.core.contextual_fabric import ContextualMemory
from memoryweave.core.memory_encoding import MemoryEncoder
from memoryweave.core.retrieval import ContextualRetriever

__all__ = ["ContextualMemory", "MemoryEncoder", "ContextualRetriever"]
