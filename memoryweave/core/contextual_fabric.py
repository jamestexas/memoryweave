"""
DEPRECATED: This module has been refactored into components.

The functionality has been moved to contextual_memory.py and should be
imported from there instead.
"""

import warnings

warnings.warn(
    "memoryweave.core.contextual_fabric is deprecated. "
    "Please use memoryweave.core.contextual_memory instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export ContextualMemory for backward compatibility
from memoryweave.core.contextual_memory import ContextualMemory
