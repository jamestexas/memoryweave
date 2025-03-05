# memoryweave/components/component_names.py
from enum import Enum


class ComponentName(str, Enum):
    QUERY_ANALYZER = "query_analyzer"
    QUERY_ADAPTER = "query_adapter"
    QUERY_CONTEXT_BUILDER = "query_context_builder"
    TWO_STAGE_RETRIEVAL = "two_stage_retrieval"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    HYBRID_BM25_VECTOR_RETRIEVAL = "hybrid_bm25_vector_retrieval"
    SIMILARITY_RETRIEVAL = "similarity_retrieval"
    TEMPORAL_RETRIEVAL = "temporal_retrieval"
    CATEGORY_RETRIEVAL = "category_retrieval"
    KEYWORD_BOOST = "keyword_boost"
    COHERENCE = "coherence"
    ADAPTIVE_K = "adaptive_k"
    DYNAMIC_THRESHOLD = "dynamic_threshold"
    MINIMUM_RESULT_GUARANTEE = "minimum_result_guarantee"
    PERSONAL_ATTRIBUTE = "personal_attribute"
    MEMORY_DECAY = "memory_decay"
    KEYWORD_EXPANDER = "keyword_expander"
    # New contextual fabric components
    CONTEXTUAL_EMBEDDING_ENHANCER = "contextual_embedding_enhancer"
    ASSOCIATIVE_MEMORY_LINKER = "associative_memory_linker"
    TEMPORAL_CONTEXT_BUILDER = "temporal_context_builder"
    ACTIVATION_MANAGER = "activation_manager"
    CONTEXTUAL_FABRIC_STRATEGY = "contextual_fabric_strategy"
    # Dynamic context adaptation
    DYNAMIC_CONTEXT_ADAPTER = "dynamic_context_adapter"
    # Used for Text Chunking
    TEXT_CHUNKER = "text_chunker"
    CHUNKED_FABRIC_STRATEGY = "chunked_fabric_strategy"
    HYBRID_FABRIC_STRAETGY = "hybrid_fabric_strategy"
