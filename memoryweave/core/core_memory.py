"""
DEPRECATED: Core memory module.

This module is deprecated. Please use the component-based architecture instead.
"""

import warnings

warnings.warn(
    "memoryweave.core.core_memory is deprecated. "
    "Use memoryweave.storage.refactored.memory_store or memoryweave.components.memory_manager instead.",
    DeprecationWarning,
    stacklevel=2,
)


class Memory:
    """
    DEPRECATED: Core memory class.

    This class is deprecated and will be removed in a future version.
    Please use memoryweave.storage.refactored.memory_store.StandardMemoryStore instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning."""
        warnings.warn(
            "Core Memory class is deprecated and will be removed in a future version. "
            "Use memoryweave.storage.refactored.memory_store.StandardMemoryStore instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Import the new memory store for delegation
        from memoryweave.storage.refactored.memory_store import StandardMemoryStore

        # Create the delegate
        self.delegate = StandardMemoryStore()

    def __getattr__(self, name):
        """Delegate all attribute access to the new implementation."""
        if hasattr(self.delegate, name):
            attr = getattr(self.delegate, name)

            # If the attribute is a method, wrap it with a deprecation warning
            if callable(attr):

                def wrapped_method(*args, **kwargs):
                    warnings.warn(
                        f"Method {name} of core.Memory is deprecated. "
                        f"Use memoryweave.storage.refactored.memory_store.StandardMemoryStore.{name} instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return attr(*args, **kwargs)

                return wrapped_method
            return attr

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
