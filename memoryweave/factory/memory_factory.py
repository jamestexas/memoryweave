"""Factory for creating memory stores and adapters with appropriate configurations."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from memoryweave.storage.refactored.adapter import MemoryAdapter
from memoryweave.storage.refactored.chunked_adapter import ChunkedMemoryAdapter
from memoryweave.storage.refactored.chunked_store import ChunkedMemoryStore
from memoryweave.storage.refactored.hybrid_adapter import HybridMemoryAdapter
from memoryweave.storage.refactored.hybrid_store import HybridMemoryStore
from memoryweave.storage.refactored.memory_store import StandardMemoryStore
from memoryweave.storage.vector_search import create_vector_search_provider


class VectorSearchConfig(BaseModel):
    """Configuration for vector search providers."""

    type: Literal["numpy", "faiss", "hybrid_bm25"] = "numpy"

    # Common settings
    dimension: int = 768
    use_quantization: bool = False

    # FAISS-specific settings
    faiss_index_type: Optional[str] = None
    faiss_metric: Optional[str] = "cosine"
    faiss_nprobe: Optional[int] = None

    # BM25-specific settings
    bm25_k1: Optional[float] = 1.5
    bm25_b: Optional[float] = 0.75
    bm25_weight: Optional[float] = 0.5  # Weight for BM25 vs vector score


class MemoryStoreConfig(BaseModel):
    """Configuration for memory stores and adapters."""

    type: Literal["standard", "chunked", "hybrid"] = "standard"
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)

    # Common settings
    max_memories: int = 1000

    # Chunking settings (for chunked and hybrid stores)
    chunk_size: Optional[int] = 200
    chunk_overlap: Optional[int] = 50
    min_chunk_size: Optional[int] = 30

    # Hybrid-specific settings
    adaptive_threshold: Optional[int] = 800
    max_chunks_per_memory: Optional[int] = 3
    importance_threshold: Optional[float] = 0.6


def create_memory_store_and_adapter(config: MemoryStoreConfig):
    """
    Create a memory store and adapter based on the provided configuration.

    Args:
        config: Memory store configuration

    Returns:
        Appropriate memory adapter (MemoryAdapter, ChunkedMemoryAdapter, or HybridMemoryAdapter)
    """
    # Create vector search provider with appropriate settings
    vector_search_kwargs = {
        "dimension": config.vector_search.dimension,
        "use_quantization": config.vector_search.use_quantization,
    }

    # Add type-specific settings
    if config.vector_search.type == "faiss":
        vector_search_kwargs.update({
            "index_type": config.vector_search.faiss_index_type,
            "metric": config.vector_search.faiss_metric,
            "nprobe": config.vector_search.faiss_nprobe,
        })
    elif config.vector_search.type == "hybrid_bm25":
        vector_search_kwargs.update({
            "k1": config.vector_search.bm25_k1,
            "b": config.vector_search.bm25_b,
            "weight": config.vector_search.bm25_weight,
        })

    vector_search = create_vector_search_provider(
        config.vector_search.type,
        **{k: v for k, v in vector_search_kwargs.items() if v is not None},
    )

    # Create appropriate memory store and adapter
    if config.type == "standard":
        store = StandardMemoryStore()
        adapter = MemoryAdapter(store)

    elif config.type == "chunked":
        store = ChunkedMemoryStore()
        # Configure chunking parameters if provided
        if hasattr(store, "chunk_size") and config.chunk_size is not None:
            store.chunk_size = config.chunk_size
        if hasattr(store, "chunk_overlap") and config.chunk_overlap is not None:
            store.chunk_overlap = config.chunk_overlap
        if hasattr(store, "min_chunk_size") and config.min_chunk_size is not None:
            store.min_chunk_size = config.min_chunk_size

        adapter = ChunkedMemoryAdapter(store)

    elif config.type == "hybrid":
        store = HybridMemoryStore()
        # Configure hybrid parameters if provided
        if hasattr(store, "adaptive_threshold") and config.adaptive_threshold is not None:
            store.adaptive_threshold = config.adaptive_threshold
        if hasattr(store, "max_chunks_per_memory") and config.max_chunks_per_memory is not None:
            store.max_chunks_per_memory = config.max_chunks_per_memory
        if hasattr(store, "importance_threshold") and config.importance_threshold is not None:
            store.importance_threshold = config.importance_threshold

        adapter = HybridMemoryAdapter(store)

    else:
        raise ValueError(f"Unknown memory store type: {config.type}")

    # Attach vector search to adapter if supported
    if hasattr(adapter, "set_vector_search"):
        adapter.set_vector_search(vector_search)

    return adapter


# Usage example:
# config = MemoryStoreConfig(
#     type="hybrid",
#     vector_search=VectorSearchConfig(
#         type="faiss",
#         faiss_index_type="IVF100,Flat"
#     ),
#     adaptive_threshold=800,
#     max_chunks_per_memory=3
# )
# memory_adapter = create_memory_store_and_adapter(config)
