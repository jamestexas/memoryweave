"""
Factory for creating memory stores and adapters.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from memoryweave.api.config import VectorSearchConfig
from memoryweave.components.memory_encoding import MemoryEncoder
from memoryweave.interfaces.memory import IMemoryStore
from memoryweave.storage.adapter import MemoryAdapter
from memoryweave.storage.chunked_adapter import ChunkedMemoryAdapter
from memoryweave.storage.chunked_store import ChunkedMemoryStore
from memoryweave.storage.hybrid_adapter import HybridMemoryAdapter
from memoryweave.storage.hybrid_store import HybridMemoryStore
from memoryweave.storage.memory_store import StandardMemoryStore
from memoryweave.storage.vector_search import create_vector_search_provider
from memoryweave.utils import _get_device

logger = logging.getLogger(__name__)


class MemoryStoreConfig(BaseModel):
    """Configuration for memory store."""

    store_type: str = Field(
        default="standard",
        description="Type of the memory store (standard, chunked, hybrid).",
    )
    vector_search: VectorSearchConfig | None = Field(
        default=None,
        description="Configuration for vector search.",
    )
    max_memories: int = Field(
        1000,
        description="Maximum number of memories to store.",
    )
    embedding_dim: int = Field(
        default=768,
        description="Embedding dimension for the memory store.",
    )
    type: str = Field(
        default="memory_store",
        description="High-level memory store type.",
    )
    chunk_size: int = Field(
        default=1000,
        description="Chunk size for chunked memory store.",
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap size between chunks in chunked memory store.",
    )
    min_chunk_size: int = Field(
        100,
        description="Minimum chunk size for chunked memory store.",
    )
    max_chunks_per_memory: int = Field(
        10,
        description="Maximum number of chunks per memory.",
    )
    adaptive_threshold: float = Field(
        0.5,
        description="Threshold for adaptive chunking.",
    )
    adaptive_chunk_size: int = Field(
        1000,
        description="Adaptive chunk size for chunking.",
    )
    importance_threshold: float = Field(
        0.5,
        description="Threshold for importance-based chunking.",
    )


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

    # Create vector search provider if provided, else default to None (from the model default)
    if (vector_search := config.vector_search) is not None:
        vector_search = create_vector_search_provider(
            provider=config.vector_search.provider,
            dimension=config.vector_search.dimension,
            metric=config.vector_search.metric,
            use_quantization=config.vector_search.use_quantization,
            index_type=config.vector_search.index_type,
            nprobe=config.vector_search.nprobe,
            provider_type=config.vector_search.type,  # numpy / faiss / hybrid_bm25
        )

    logger.debug(f"Creating memory store of type: {store_type}")
    # Return the appropriate memory store and adapter based on the store type
    if config.store_type == "hybrid":
        memory_store = HybridMemoryStore()
        return memory_store, HybridMemoryAdapter(
            memory_store=memory_store,
            vector_search=vector_search,
        )

    elif config.store_type == "chunked":
        chunked_store = ChunkedMemoryStore()
        return chunked_store, ChunkedMemoryAdapter(
            memory_store=chunked_store,
        )
    # Fall back to standard memory store
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

    # Get device configuration
    device = _get_device(kwargs.pop("device", "auto"))

    # Initialize the embedding model if not provided
    if embedding_model is None:
        if embedding_model_name is None:
            embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
        embedding_model = SentenceTransformer(embedding_model_name, device=device, **kwargs)
    context_enhancer_config = kwargs.get("context_enhancer_config", {})

    # Create and initialize the memory encoder
    encoder = MemoryEncoder(embedding_model=embedding_model)
    encoder.initialize(
        {
            "context_window_size": kwargs.get("context_window_size", 3),
            "use_episodic_markers": kwargs.get("use_episodic_markers", True),
            "context_enhancer": context_enhancer_config,
        }
    )

    return encoder
