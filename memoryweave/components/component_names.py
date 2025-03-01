# memoryweave/components/component_names.py
from enum import Enum


class ComponentName(str, Enum):
    QUERY_ANALYZER = "query_analyzer"
    QUERY_ADAPTER = "query_adapter"
    TWO_STAGE_RETRIEVAL = "two_stage_retrieval"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    SIMILARITY_RETRIEVAL = "similarity_retrieval"
    TEMPORAL_RETRIEVAL = "temporal_retrieval"
    KEYWORD_BOOST = "keyword_boost"
    COHERENCE = "coherence"
    ADAPTIVE_K = "adaptive_k"
    DYNAMIC_THRESHOLD = "dynamic_threshold"
