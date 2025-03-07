"""
DEPRECATED: Core category management system.

This module is deprecated. Please use the component-based architecture instead.
"""

import warnings

warnings.warn(
    "memoryweave.core.category_manager is deprecated. "
    "Use memoryweave.components.category_manager instead.",
    DeprecationWarning,
    stacklevel=2,
)


class CoreCategoryManager:
    """
    DEPRECATED: Core category management system.

    This class is deprecated and will be removed in a future version.
    Please use memoryweave.components.category_manager.CategoryManager instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning."""
        warnings.warn(
            "CoreCategoryManager is deprecated and will be removed in a future version. "
            "Use memoryweave.components.category_manager.CategoryManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Import the new CategoryManager for delegation
        from memoryweave.components.category_manager import CategoryManager

        # Create the delegate
        self.delegate = CategoryManager(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate all attribute access to the new implementation."""
        if hasattr(self.delegate, name):
            attr = getattr(self.delegate, name)

            # If the attribute is a method, wrap it with a deprecation warning
            if callable(attr):

                def wrapped_method(*args, **kwargs):
                    warnings.warn(
                        f"Method {name} of CoreCategoryManager is deprecated. "
                        f"Use memoryweave.components.category_manager.CategoryManager.{name} instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return attr(*args, **kwargs)

                return wrapped_method
            return attr

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
