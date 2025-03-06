"""
Factory for creating memory stores and adapters.
"""

from dataclasses import dataclass
from typing import Any, Optional

from memoryweave.components.memory_encoding import MemoryEncoder
from memoryweave.interfaces.memory import IMemoryEncoder, IMemoryStore
from memoryweave.storage.refactored.adapter import MemoryAdapter
from memoryweave.storage.refactored.chunked_store import ChunkedMemoryStore
from memoryweave.storage.refactored.hybrid_store import HybridMemoryStore
from memoryweave.storage.refactored.memory_store import StandardMemoryStore
from memoryweave.storage.vector_search import create_vector_search_provider


@dataclass
class VectorSearchConfig:
    """Configuration for vector search."""

    provider: str = "numpy"
    dimension: int = 768
    metric: str = "cosine"
    use_quantization: bool = False
    index_type: Optional[str] = None
    nprobe: int = 10


@dataclass
class MemoryStoreConfig:
    """Configuration for memory store."""

    store_type: str = "standard"
    vector_search: Optional[VectorSearchConfig] = None
    max_memories: int = 1000
    embedding_dim: int = 768


def create_memory_store_and_adapter(
    config: MemoryStoreConfig,
) -> tuple[IMemoryStore, MemoryAdapter]:
    """
    Create a memory store and adapter from configuration.

    Args:
        config: Configuration for the memory store

    Returns:
        Tuple of (memory_store, memory_adapter)
    """
    # Create memory store
    store_type = config.store_type
    memory_store: IMemoryStore

    if store_type == "standard":
        memory_store = StandardMemoryStore()
    elif store_type == "hybrid":
        memory_store = HybridMemoryStore()
    elif store_type == "chunked":
        memory_store = ChunkedMemoryStore()
    else:
        raise ValueError(f"Unknown memory store type: {store_type}")

    # Create vector search provider
    vector_search = None
    if config.vector_search:
        vector_search = create_vector_search_provider(
            provider=config.vector_search.provider,
            dimension=config.vector_search.dimension,
            metric=config.vector_search.metric,
            use_quantization=config.vector_search.use_quantization,
            index_type=config.vector_search.index_type,
            nprobe=config.vector_search.nprobe,
        )

    # Create memory adapter
    memory_adapter = MemoryAdapter(memory_store, vector_search)

    return memory_store, memory_adapter


def create_memory_encoder(embedding_model: Any, config: dict[str, Any] = None) -> IMemoryEncoder:
    """
    Create a memory encoder from configuration.

    Args:
        embedding_model: Model to use for creating embeddings
        config: Configuration for the memory encoder

    Returns:
        Memory encoder
    """
    if config is None:
        config = {}

    # Create memory encoder
    memory_encoder = MemoryEncoder(embedding_model)
    memory_encoder.initialize(config)

    return memory_encoder
