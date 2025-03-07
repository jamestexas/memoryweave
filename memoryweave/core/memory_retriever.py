"""
DEPRECATED: Core memory retriever module.

This module is deprecated. Please use the component-based architecture instead.
"""

import warnings

warnings.warn(
    "memoryweave.core.memory_retriever is deprecated. "
    "Use memoryweave.components.retriever instead.",
    DeprecationWarning,
    stacklevel=2,
)


class ContextualRetriever:
    """
    DEPRECATED: Core contextual retriever class.

    This class is deprecated and will be removed in a future version.
    Please use memoryweave.components.retriever.Retriever instead.
    """

    def __init__(self, memory=None, *args, **kwargs):
        """Initialize with deprecation warning."""
        warnings.warn(
            "ContextualRetriever is deprecated and will be removed in a future version. "
            "Use memoryweave.components.retriever.Retriever instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Import the new Retriever for delegation
        from sentence_transformers import SentenceTransformer

        from memoryweave.components.retriever import Retriever

        # Create embedding model
        embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

        # Create the delegate
        self.delegate = Retriever(memory=memory, embedding_model=embedding_model)
        self.delegate.initialize_components()

    def __getattr__(self, name):
        """Delegate all attribute access to the new implementation."""
        if hasattr(self.delegate, name):
            attr = getattr(self.delegate, name)

            # If the attribute is a method, wrap it with a deprecation warning
            if callable(attr):

                def wrapped_method(*args, **kwargs):
                    warnings.warn(
                        f"Method {name} of ContextualRetriever is deprecated. "
                        f"Use memoryweave.components.retriever.Retriever.{name} instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return attr(*args, **kwargs)

                return wrapped_method
            return attr

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
