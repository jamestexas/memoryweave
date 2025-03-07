"""
The core module has been deprecated and replaced with the component-based architecture.

Please use the following modules instead:
- memoryweave.components: Base components and implementations
- memoryweave.storage: Memory storage implementations
- memoryweave.factory: Factory methods for creating components

This directory will be removed in a future release.
"""

# Raise an ImportError when someone tries to import from core
raise ImportError(
    "The memoryweave.core module has been replaced with the component-based architecture. "
    "Please use memoryweave.components, memoryweave.storage, etc. instead."
)
