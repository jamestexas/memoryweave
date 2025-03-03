"""
DEPRECATED: Core module for MemoryWeave.

This module is completely deprecated. Please use the component-based architecture instead.

Migration guide:
- Use memoryweave.components.memory_manager.MemoryManager instead of ContextualMemory
- Use memoryweave.components.retriever.Retriever instead of ContextualRetriever
- Use memoryweave.adapters for bridge components during migration

See MIGRATION_GUIDE.md for detailed migration instructions.
"""

import warnings

# Re-export the core classes for backward compatibility
# These imports will trigger their own deprecation warnings
from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.core.memory_encoding import MemoryEncoder

# Emit deprecation warning
warnings.warn(
    "The memoryweave.core module is deprecated. "
    "Use memoryweave.components module instead for the new architecture. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ContextualMemory", "MemoryEncoder"]
