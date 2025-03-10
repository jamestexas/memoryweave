from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from memoryweave.api.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL


class VectorSearchConfig(BaseModel):
    """Configuration for vector search."""

    provider: str = Field(
        default="numpy", description="Vector search provider (e.g., numpy, faiss)."
    )
    dimension: int = Field(
        default=768, description="Dimension of the embeddings for vector search."
    )
    metric: str = Field(
        default="cosine", description="Distance metric for the vector search (e.g., cosine)."
    )
    use_quantization: bool = Field(default=False, description="Enable quantization for indexing.")
    index_type: Optional[str] = Field(
        default=None, description="Optional index type (e.g., IVF, HNSW)."
    )
    nprobe: int = Field(default=10, description="Number of probes for the vector index.")
    type: str = Field(
        default="faiss", description="Library or approach for the vector search (e.g., faiss)."
    )

    # Define model configuration directly
    model_config = ConfigDict(extra="allow")


class MemoryStoreConfig(BaseModel):
    """Configuration for memory store."""

    store_type: str = Field(
        default="standard", description="Type of memory store (standard, hybrid, chunked)."
    )
    vector_search: Optional[VectorSearchConfig] = Field(
        default=None, description="Vector search configuration."
    )
    max_memories: int = Field(default=1000, description="Maximum number of memories to store.")
    embedding_dim: int = Field(default=768, description="Dimension of the memory embeddings.")
    type: str = Field(
        default="memory_store", description="Type of memory store (hybrid/chunked/standard)."
    )

    # Chunked memory store configurations
    chunk_size: int = Field(default=1000, description="Chunk size for chunked memory store.")
    chunk_overlap: int = Field(default=100, description="Overlap size for chunked memory store.")
    min_chunk_size: int = Field(
        default=100, description="Minimum chunk size for chunked memory store."
    )
    max_chunks_per_memory: int = Field(
        default=10, description="Maximum number of chunks per memory."
    )

    # Adaptive chunking configurations
    adaptive_threshold: float = Field(default=0.5, description="Threshold for adaptive chunking.")
    adaptive_chunk_size: int = Field(default=1000, description="Chunk size for adaptive chunking.")
    importance_threshold: float = Field(
        default=0.5, description="Threshold for importance-based chunking."
    )

    # Define model configuration
    model_config = ConfigDict(extra="allow")


class StrategyConfig(BaseModel):
    """Configuration for retrieval strategies."""

    confidence_threshold: float = Field(0.1, description="Minimum confidence for results")
    similarity_weight: float = Field(0.4, description="Weight for similarity")
    associative_weight: float = Field(0.3, description="Weight for associative signals")
    temporal_weight: float = Field(0.2, description="Weight for temporal signals")
    activation_weight: float = Field(0.1, description="Weight for activation signals")
    max_associative_hops: int = Field(2, description="Maximum hops for associative traversal")
    debug: bool = Field(False, description="Enable debug logging")

    # Hybrid strategy parameters
    use_keyword_filtering: bool = Field(True, description="Enable keyword filtering")
    keyword_boost_factor: float = Field(0.3, description="Boost factor for keywords")
    prioritize_full_embeddings: bool = Field(True, description="Prioritize full embeddings")
    min_results: int = Field(3, description="Minimum number of results")
    max_candidates: int = Field(50, description="Maximum candidates to consider")

    # Two-stage parameters
    use_two_stage: bool = Field(True, description="Enable two-stage retrieval")
    first_stage_k: int = Field(30, description="Candidates in first stage")
    first_stage_threshold_factor: float = Field(0.7, description="Threshold factor")

    # Advanced parameters
    use_batched_computation: bool = Field(True, description="Use batched computation")
    batch_size: int = Field(200, description="Batch size for computation")


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = Field(300, description="Target chunk size")
    chunk_overlap: int = Field(30, description="Overlap between chunks")
    min_chunk_size: int = Field(50, description="Minimum chunk size")
    respect_paragraphs: bool = Field(True, description="Respect paragraph boundaries")
    respect_sentences: bool = Field(True, description="Respect sentence boundaries")


class MemoryWeaveConfig(BaseModel):
    """Main configuration for MemoryWeave."""

    model_name: str = Field(DEFAULT_MODEL, description="LLM model name")
    embedding_model_name: str = Field(DEFAULT_EMBEDDING_MODEL, description="Embedding model")
    device: str = Field("auto", description="Device (auto, cpu, cuda, mps)")
    max_memories: int = Field(1000, description="Maximum memories")
    consolidation_interval: int = Field(100, description="Consolidation interval")
    show_progress_bar: bool = Field(False, description="Show progress bars")
    debug: bool = Field(False, description="Enable debug logging")

    # Feature flags
    enable_category_management: bool = Field(True, description="Enable category management")
    enable_personal_attributes: bool = Field(True, description="Enable personal attributes")
    enable_semantic_coherence: bool = Field(True, description="Enable semantic coherence")
    enable_dynamic_thresholds: bool = Field(True, description="Enable dynamic thresholds")

    # Specific configurations
    memory_store: MemoryStoreConfig = Field(default_factory=MemoryStoreConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    # HybridMemoryWeaveAPI specific
    two_stage_retrieval: bool = Field(True, description="Enable two-stage retrieval")

    # Additional parameters
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ChunkingConfig",
    "MemoryStoreConfig",
    "MemoryWeaveConfig",
    "StrategyConfig",
    "VectorSearchConfig",
]
