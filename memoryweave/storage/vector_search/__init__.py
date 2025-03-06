"""Vector search implementations for MemoryWeave."""

from importlib.util import find_spec
from typing import Any, Optional

from memoryweave.storage.vector_search.base import IVectorSearchProvider

# Lazy imports to avoid unnecessary dependencies
_NUMPY_SEARCH = None
_FAISS_SEARCH = None
_HYBRID_SEARCH = None


def get_numpy_search():
    """Get the NumPy vector search implementation."""
    global _NUMPY_SEARCH
    if _NUMPY_SEARCH is None:
        from memoryweave.storage.vector_search.numpy_search import NumpyVectorSearch

        _NUMPY_SEARCH = NumpyVectorSearch
    return _NUMPY_SEARCH


def get_faiss_search():
    """Get the FAISS vector search implementation."""
    global _FAISS_SEARCH
    if _FAISS_SEARCH is None:
        if find_spec("faiss") is None:
            raise ImportError(
                "FAISS is not installed. Install it with: pip install faiss-cpu or faiss-gpu"
            )
        from memoryweave.storage.vector_search.faiss_search import FaissVectorSearch

        _FAISS_SEARCH = FaissVectorSearch
    return _FAISS_SEARCH


def get_hybrid_search():
    """Get the hybrid BM25+vector search implementation."""
    global _HYBRID_SEARCH
    if _HYBRID_SEARCH is None:
        from memoryweave.storage.vector_search.hybrid_search import HybridBM25VectorSearch

        _HYBRID_SEARCH = HybridBM25VectorSearch
    return _HYBRID_SEARCH


def create_vector_search_provider(provider_type: str, **kwargs) -> IVectorSearchProvider:
    """
    Factory function for creating vector search providers.

    Args:
        provider_type: Type of provider ("numpy", "faiss", or "hybrid_bm25")
        **kwargs: Provider-specific arguments

    Returns:
        Vector search provider instance

    Raises:
        ValueError: If provider_type is unknown
    """
    if provider_type == "numpy":
        return get_numpy_search()(**kwargs)
    elif provider_type == "faiss":
        return get_faiss_search()(**kwargs)
    elif provider_type == "hybrid_bm25":
        return get_hybrid_search()(**kwargs)
    else:
        raise ValueError(f"Unknown vector search provider type: {provider_type}")


# Also expose the interface for direct imports
__all__ = ["IVectorSearchProvider", "create_vector_search_provider"]
