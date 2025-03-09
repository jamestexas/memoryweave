from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from memoryweave.api.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL


class VectorSearchConfig(BaseModel):
    """Configuration for vector search."""

    type: str = Field("numpy", description="Vector search provider type")
    dimension: int = Field(384, description="Dimension of embedding vectors")
    provider: Optional[str] = Field(None, description="Custom provider name")
    metric: str = Field("cosine", description="Distance metric")
    use_quantization: bool = Field(False, description="Use vector quantization")
    index_type: Optional[str] = Field(None, description="Index type for specific providers")
    nprobe: int = Field(4, description="Number of probes for approximate search")


class MemoryStoreConfig(BaseModel):
    """Configuration for memory store."""

    store_type: Literal["standard", "hybrid", "chunked"] = Field(
        "standard", description="Memory store type"
    )
    vector_search: Optional[VectorSearchConfig] = None

    # Chunking parameters
    chunk_size: int = Field(300, description="Target chunk size")
    chunk_overlap: int = Field(30, description="Overlap between chunks")
    min_chunk_size: int = Field(50, description="Minimum chunk size")
    adaptive_threshold: int = Field(800, description="Character count that triggers chunking")
    max_chunks_per_memory: int = Field(3, description="Maximum chunks per memory")
    importance_threshold: float = Field(0.6, description="Threshold for keeping chunks")


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
