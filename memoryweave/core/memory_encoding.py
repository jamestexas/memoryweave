"""
DEPRECATED: Core memory encoding module.

This module is deprecated. Please use the component-based architecture instead.
"""

import warnings

warnings.warn(
    "memoryweave.core.memory_encoding is deprecated. "
    "Use memoryweave.components.memory_encoding instead.",
    DeprecationWarning,
    stacklevel=2,
)


class MemoryEncoder:
    """
    DEPRECATED: Core memory encoding class.

    This class is deprecated and will be removed in a future version.
    Please use memoryweave.components.memory_encoding.MemoryEncoder instead.
    """

    def __init__(self, embedding_model=None):
        """Initialize with deprecation warning."""
        warnings.warn(
            "MemoryEncoder from core module is deprecated and will be removed in a future version. "
            "Use memoryweave.components.memory_encoding.MemoryEncoder instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Import the new MemoryEncoder for delegation
        from memoryweave.components.memory_encoding import MemoryEncoder
        from memoryweave.factory.memory_factory import create_memory_encoder

        # Create the delegate - either use provided model or create a new one
        if embedding_model is not None:
            self.delegate = MemoryEncoder(embedding_model)
        else:
            self.delegate = create_memory_encoder()

        self.delegate.initialize({})

    def __getattr__(self, name):
        """Delegate all attribute access to the new implementation."""
        if hasattr(self.delegate, name):
            attr = getattr(self.delegate, name)

            # If the attribute is a method, wrap it with a deprecation warning
            if callable(attr):

                def wrapped_method(*args, **kwargs):
                    warnings.warn(
                        f"Method {name} of core.MemoryEncoder is deprecated. "
                        f"Use memoryweave.components.memory_encoding.MemoryEncoder.{name} instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return attr(*args, **kwargs)

                return wrapped_method
            return attr

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
