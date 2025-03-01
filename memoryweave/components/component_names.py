# memoryweave/components/component_names.py
from enum import Enum


class ComponentName(str, Enum):
    QUERY_ANALYZER = "query_analyzer"
    QUERY_ADAPTER = "query_adapter"
    QUERY_CONTEXT_BUILDER = "query_context_builder"
    TWO_STAGE_RETRIEVAL = "two_stage_retrieval"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
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
