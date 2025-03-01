"""
Core module for MemoryWeave.

This module contains the core functionality for the MemoryWeave memory management system.

DEPRECATION WARNING:
The core module is being deprecated in favor of the component-based architecture.
Please use the components module instead:
- Use memoryweave.components.memory_manager.MemoryManager instead of ContextualMemory
- Use memoryweave.components.retriever.Retriever instead of ContextualRetriever
"""

import warnings

from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.core.memory_encoding import MemoryEncoder

# Emit deprecation warning
warnings.warn(
    "The memoryweave.core module is deprecated. "
    "Use memoryweave.components module instead for the new architecture.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ContextualMemory", "MemoryEncoder"]
