"""
DEPRECATED: Core contextual memory module.

This module is deprecated. Please use the component-based architecture instead.
"""

import warnings

warnings.warn(
    "memoryweave.core.contextual_memory is deprecated. "
    "Use memoryweave.components.memory_manager instead.",
    DeprecationWarning,
    stacklevel=2,
)


class ContextualMemory:
    """
    DEPRECATED: Core contextual memory class.

    This class is deprecated and will be removed in a future version.
    Please use memoryweave.components.memory_manager.MemoryManager instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning."""
        warnings.warn(
            "ContextualMemory is deprecated and will be removed in a future version. "
            "Use memoryweave.components.memory_manager.MemoryManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Import the new MemoryManager for delegation
        from memoryweave.components.memory_manager import MemoryManager
        from memoryweave.storage.refactored.memory_store import StandardMemoryStore

        # Create a memory store and manager
        store = StandardMemoryStore()
        self.delegate = MemoryManager(store)
        self.memory_store = store

    def __getattr__(self, name):
        """Delegate attribute access appropriately."""
        # Some methods might need to be delegated to the memory_store instead
        store_methods = ["add", "get", "update", "remove", "add_memory", "retrieve_memories"]

        if name in store_methods and hasattr(self.memory_store, name):
            attr = getattr(self.memory_store, name)

            def wrapped_method(*args, **kwargs):
                warnings.warn(
                    f"Method {name} of ContextualMemory is deprecated. "
                    f"Use StandardMemoryStore.{name} instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return attr(*args, **kwargs)

            return wrapped_method

        elif hasattr(self.delegate, name):
            attr = getattr(self.delegate, name)

            if callable(attr):

                def wrapped_method(*args, **kwargs):
                    warnings.warn(
                        f"Method {name} of ContextualMemory is deprecated. "
                        f"Use memoryweave.components.memory_manager.MemoryManager.{name} instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return attr(*args, **kwargs)

                return wrapped_method
            return attr

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
