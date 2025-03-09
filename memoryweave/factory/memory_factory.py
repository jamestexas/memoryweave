"""
Factory for creating memory stores and adapters.
"""

import logging
from dataclasses import dataclass
from typing import Any

from memoryweave.components.memory_encoding import MemoryEncoder
from memoryweave.interfaces.memory import IMemoryStore
from memoryweave.storage.adapter import MemoryAdapter
from memoryweave.storage.chunked_adapter import ChunkedMemoryAdapter
from memoryweave.storage.chunked_store import ChunkedMemoryStore
from memoryweave.storage.hybrid_adapter import HybridMemoryAdapter
from memoryweave.storage.hybrid_store import HybridMemoryStore
from memoryweave.storage.memory_store import StandardMemoryStore
from memoryweave.storage.vector_search import create_vector_search_provider

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchConfig:
    """Configuration for vector search."""

    provider: str = "numpy"
    dimension: int = 768
    metric: str = "cosine"
    use_quantization: bool = False
    index_type: str | None = None
    nprobe: int = 10
    type: str = "faiss"


@dataclass
class MemoryStoreConfig:
    """Configuration for memory store."""

    store_type: str = "standard"
    vector_search: VectorSearchConfig | None = None  # Vector search configuration
    max_memories: int = 1000  # Maximum number of memories to store
    embedding_dim: int = 768  # Dimension of the memory embeddings
    type: str = "memory_store"  # hybrid / chunked / standard etc.

    # NOTE: Below is only for Chunked memory store configurations
    chunk_size: int = 1000  # Chunk size for chunked memory store
    chunk_overlap: int = 100  # Overlap size for chunked memory store
    min_chunk_size: int = 100  # Minimum chunk size for chunked memory store
    max_chunks_per_memory: int = 10  # Maximum number of chunks per memory

    # NOTE: Adaptive chunking configurations
    adaptive_threshold: float = 0.5  # Threshold for adaptive chunking
    adaptive_chunk_size: int = 1000  # Chunk size for adaptive chunking
    importance_threshold: float = 0.5  # Threshold for importance-based chunking


def create_memory_store_and_adapter(
    config: MemoryStoreConfig,
) -> tuple[IMemoryStore, MemoryAdapter | HybridMemoryAdapter | ChunkedMemoryAdapter]:
    """
    Create a memory store and adapter from configuration.

    Args:
        config: Configuration for the memory store

    Returns:
        Tuple of (memory_store, memory_adapter)
    """
    # Create memory store
    store_type = config.store_type

    # Create vector search provider
    if config.vector_search:
        vector_search = create_vector_search_provider(
            provider=config.vector_search.provider,
            dimension=config.vector_search.dimension,
            metric=config.vector_search.metric,
            use_quantization=config.vector_search.use_quantization,
            index_type=config.vector_search.index_type,
            nprobe=config.vector_search.nprobe,
            provider_type=config.vector_search.type,  # numpy / faiss / hybrid_bm25
        )

    print(f"Creating memory store of type: {store_type}")
    if config.store_type == "hybrid":
        memory_store = HybridMemoryStore()
        return memory_store, HybridMemoryAdapter(
            memory_store=memory_store,
            vector_search=vector_search,
        )

    elif config.store_type == "chunked":
        return ChunkedMemoryAdapter(memory_store=ChunkedMemoryStore())

    memory_store = StandardMemoryStore()
    return memory_store, MemoryAdapter(
        memory_store=memory_store,
        vector_search=vector_search,
    )


def create_memory_encoder(
    embedding_model_name: str = None, embedding_model: Any = None, **kwargs
) -> MemoryEncoder:
    """
    Create a memory encoder component with the specified embedding model.

    Args:
        embedding_model_name: Name of the embedding model to use (if model not provided directly)
        embedding_model: Direct embedding model instance (higher priority than model_name)
        **kwargs: Additional arguments for configuration or embedding model

    Returns:
        Configured MemoryEncoder component
    """
    from sentence_transformers import SentenceTransformer

    from memoryweave.utils import _get_device

    # Get device configuration
    device = _get_device(kwargs.pop("device", "auto"))

    # Initialize the embedding model if not provided
    if embedding_model is None:
        if embedding_model_name is None:
            embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
        embedding_model = SentenceTransformer(embedding_model_name, device=device, **kwargs)

    # Create and initialize the memory encoder
    encoder = MemoryEncoder(embedding_model)
    encoder.initialize(
        {
            "context_window_size": kwargs.get("context_window_size", 3),
            "use_episodic_markers": kwargs.get("use_episodic_markers", True),
            "context_enhancer": kwargs.get("context_enhancer_config", {}),
        }
    )

    return encoder
