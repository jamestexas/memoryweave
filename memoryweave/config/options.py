"""Configuration options for MemoryWeave.

This module defines the configuration options for MemoryWeave components,
including defaults, validation rules, and documentation.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


class ConfigValueType(Enum):
    """Types of configuration values."""

    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    LIST = auto()
    DICT = auto()
    ENUM = auto()


@dataclass
class ConfigOption:
    """Definition of a configuration option."""

    name: str
    value_type: ConfigValueType
    default_value: Any
    description: str
    required: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    enum_type: Optional[type] = None
    nested_options: Optional[List["ConfigOption"]] = None


@dataclass
class ComponentConfig:
    """Configuration for a component."""

    component_type: str
    options: List[ConfigOption]
    description: str = ""


# Memory Store Configuration
MEMORY_STORE_CONFIG = ComponentConfig(
    component_type="memory_store",
    description="Configuration for memory storage components",
    options=[
        ConfigOption(
            name="max_memories",
            value_type=ConfigValueType.INTEGER,
            default_value=1000,
            description="Maximum number of memories to store",
            min_value=10,
            max_value=100000,
        ),
        ConfigOption(
            name="initial_activation",
            value_type=ConfigValueType.FLOAT,
            default_value=0.0,
            description="Initial activation level for new memories",
            min_value=-10.0,
            max_value=10.0,
        ),
        ConfigOption(
            name="consolidation_strategy",
            value_type=ConfigValueType.STRING,
            default_value="lru",
            description="Strategy for memory consolidation",
            allowed_values=["lru", "activation", "random"],
        ),
    ],
)

# Vector Store Configuration
VECTOR_STORE_CONFIG = ComponentConfig(
    component_type="vector_store",
    description="Configuration for vector storage components",
    options=[
        ConfigOption(
            name="similarity_metric",
            value_type=ConfigValueType.STRING,
            default_value="cosine",
            description="Metric for measuring vector similarity",
            allowed_values=["cosine", "dot", "euclidean"],
        ),
        ConfigOption(
            name="activation_weight",
            value_type=ConfigValueType.FLOAT,
            default_value=0.2,
            description="Weight of activation in similarity scoring",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="use_ann",
            value_type=ConfigValueType.BOOLEAN,
            default_value=True,
            description="Whether to use approximate nearest neighbor search for better performance at scale",
        ),
        ConfigOption(
            name="scale",
            value_type=ConfigValueType.STRING,
            default_value="auto",
            description="Scale of memory store for optimizing ANN parameters",
            allowed_values=["auto", "small", "medium", "large"],
        ),
        ConfigOption(
            name="dimension",
            value_type=ConfigValueType.INTEGER,
            default_value=768,
            description="Dimensionality of embedding vectors",
            min_value=32,
            max_value=4096,
        ),
        ConfigOption(
            name="memory_count",
            value_type=ConfigValueType.INTEGER,
            default_value=0,
            description="Current memory count (0 means auto-detect)",
            min_value=0,
        ),
        ConfigOption(
            name="metric",
            value_type=ConfigValueType.STRING,
            default_value="cosine",
            description="Distance metric for ANN search",
            allowed_values=["cosine", "l2", "ip"],
        ),
        ConfigOption(
            name="index_type",
            value_type=ConfigValueType.STRING,
            default_value="auto",
            description="FAISS index type (auto means choose based on scale)",
            allowed_values=["auto", "Flat", "IVF100,Flat", "IVF256,Flat", "IVF100,PQ32"],
        ),
        ConfigOption(
            name="nprobe",
            value_type=ConfigValueType.INTEGER,
            default_value=0,
            description="Number of clusters to probe during search (0 means auto-configure)",
            min_value=0,
            max_value=1000,
        ),
        ConfigOption(
            name="build_threshold",
            value_type=ConfigValueType.INTEGER,
            default_value=50,
            description="Minimum number of vectors before building ANN index",
            min_value=10,
            max_value=1000,
        ),
        ConfigOption(
            name="quantize",
            value_type=ConfigValueType.BOOLEAN,
            default_value=False,
            description="Whether to use scalar quantization to reduce memory usage",
        ),
    ],
)

# Retrieval Strategy Configuration
RETRIEVAL_STRATEGY_CONFIG = ComponentConfig(
    component_type="retrieval_strategy",
    description="Configuration for retrieval strategy components",
    options=[
        ConfigOption(
            name="similarity_threshold",
            value_type=ConfigValueType.FLOAT,
            default_value=0.7,
            description="Minimum similarity score for retrieval",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="max_results",
            value_type=ConfigValueType.INTEGER,
            default_value=10,
            description="Maximum number of results to retrieve",
            min_value=1,
            max_value=100,
        ),
        ConfigOption(
            name="recency_bias",
            value_type=ConfigValueType.FLOAT,
            default_value=0.3,
            description="Weight of recency in retrieval scoring",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="activation_boost",
            value_type=ConfigValueType.FLOAT,
            default_value=0.2,
            description="Weight of activation in retrieval scoring",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="keyword_weight",
            value_type=ConfigValueType.FLOAT,
            default_value=0.3,
            description="Weight of keyword matches in retrieval scoring",
            min_value=0.0,
            max_value=1.0,
        ),
        # For HybridRetrievalStrategy
        ConfigOption(
            name="relevance_weight",
            value_type=ConfigValueType.FLOAT,
            default_value=0.7,
            description="Weight of semantic relevance in hybrid retrieval",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="recency_weight",
            value_type=ConfigValueType.FLOAT,
            default_value=0.3,
            description="Weight of recency in hybrid retrieval",
            min_value=0.0,
            max_value=1.0,
        ),
        # For TemporalRetrievalStrategy
        ConfigOption(
            name="recency_window_days",
            value_type=ConfigValueType.FLOAT,
            default_value=7.0,
            description="Window in days for considering memory recency",
            min_value=0.1,
            max_value=365.0,
        ),
        # For TwoStageRetrievalStrategy
        ConfigOption(
            name="first_stage_threshold",
            value_type=ConfigValueType.FLOAT,
            default_value=0.5,
            description="Threshold for first stage retrieval",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="second_stage_threshold",
            value_type=ConfigValueType.FLOAT,
            default_value=0.7,
            description="Threshold for second stage retrieval",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="first_stage_max",
            value_type=ConfigValueType.INTEGER,
            default_value=30,
            description="Maximum results from first stage",
            min_value=1,
            max_value=100,
        ),
        ConfigOption(
            name="final_max_results",
            value_type=ConfigValueType.INTEGER,
            default_value=10,
            description="Maximum final results after second stage",
            min_value=1,
            max_value=100,
        ),
        ConfigOption(
            name="keyword_boost",
            value_type=ConfigValueType.FLOAT,
            default_value=0.2,
            description="Boost factor for keyword matches",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="first_stage_k",
            value_type=ConfigValueType.INTEGER,
            default_value=10,
            description="Number of results to retrieve in first stage",
            min_value=1,
            max_value=100,
        ),
        ConfigOption(
            name="first_stage_threshold_factor",
            value_type=ConfigValueType.FLOAT,
            default_value=0.7,
            description="Threshold adjustment factor for first stage",
            min_value=0.0,
            max_value=1.0,
        ),
        # Nested configurations for two-stage strategy
        ConfigOption(
            name="first_stage_config",
            value_type=ConfigValueType.DICT,
            default_value={},
            description="Configuration for first stage retrieval strategy",
        ),
        ConfigOption(
            name="second_stage_config",
            value_type=ConfigValueType.DICT,
            default_value={},
            description="Configuration for second stage retrieval strategy",
        ),
    ],
)

# Query Analyzer Configuration
QUERY_ANALYZER_CONFIG = ComponentConfig(
    component_type="query_analyzer",
    description="Configuration for query analysis components",
    options=[
        ConfigOption(
            name="min_keyword_length",
            value_type=ConfigValueType.INTEGER,
            default_value=3,
            description="Minimum length for extracted keywords",
            min_value=1,
            max_value=10,
        ),
        ConfigOption(
            name="max_keywords",
            value_type=ConfigValueType.INTEGER,
            default_value=10,
            description="Maximum number of keywords to extract",
            min_value=1,
            max_value=50,
        ),
        ConfigOption(
            name="stopwords",
            value_type=ConfigValueType.LIST,
            default_value=[],
            description="Additional stopwords to exclude from keywords",
        ),
        ConfigOption(
            name="personal_patterns",
            value_type=ConfigValueType.LIST,
            default_value=[],
            description="Additional patterns for personal queries",
        ),
        ConfigOption(
            name="factual_patterns",
            value_type=ConfigValueType.LIST,
            default_value=[],
            description="Additional patterns for factual queries",
        ),
        ConfigOption(
            name="personal_keywords",
            value_type=ConfigValueType.LIST,
            default_value=["my", "me", "i", "mine", "favorite"],
            description="Keywords indicating personal queries",
        ),
        ConfigOption(
            name="factual_keywords",
            value_type=ConfigValueType.LIST,
            default_value=["what", "when", "where", "who", "why", "how"],
            description="Keywords indicating factual queries",
        ),
        ConfigOption(
            name="conceptual_keywords",
            value_type=ConfigValueType.LIST,
            default_value=["concept", "explain", "theory", "idea", "principle"],
            description="Keywords indicating conceptual queries",
        ),
    ],
)

# Query Adapter Configuration
QUERY_ADAPTER_CONFIG = ComponentConfig(
    component_type="query_adapter",
    description="Configuration for query adaptation components",
    options=[
        ConfigOption(
            name="apply_keyword_boost",
            value_type=ConfigValueType.BOOLEAN,
            default_value=True,
            description="Whether to apply keyword boosting to retrieval",
        ),
        ConfigOption(
            name="scale_params_by_length",
            value_type=ConfigValueType.BOOLEAN,
            default_value=True,
            description="Whether to adjust parameters based on query length",
        ),
        ConfigOption(
            name="length_threshold",
            value_type=ConfigValueType.INTEGER,
            default_value=50,
            description="Character threshold for considering a query 'long'",
            min_value=10,
            max_value=1000,
        ),
        ConfigOption(
            name="type_configs",
            value_type=ConfigValueType.DICT,
            default_value={},
            description="Type-specific configuration overrides",
        ),
        # Thresholds for different query types
        ConfigOption(
            name="personal_threshold",
            value_type=ConfigValueType.FLOAT,
            default_value=0.6,
            description="Similarity threshold for personal queries",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="factual_threshold",
            value_type=ConfigValueType.FLOAT,
            default_value=0.7,
            description="Similarity threshold for factual queries",
            min_value=0.0,
            max_value=1.0,
        ),
        ConfigOption(
            name="conceptual_threshold",
            value_type=ConfigValueType.FLOAT,
            default_value=0.5,
            description="Similarity threshold for conceptual queries",
            min_value=0.0,
            max_value=1.0,
        ),
    ],
)

# Pipeline Configuration
PIPELINE_CONFIG = ComponentConfig(
    component_type="pipeline",
    description="Configuration for pipeline components",
    options=[
        ConfigOption(
            name="pipeline_stages",
            value_type=ConfigValueType.LIST,
            default_value=[],
            description="List of component IDs to include in the pipeline",
            required=True,
        ),
        ConfigOption(
            name="pipeline_name",
            value_type=ConfigValueType.STRING,
            default_value="default_pipeline",
            description="Name of the pipeline",
        ),
        ConfigOption(
            name="stage_configs",
            value_type=ConfigValueType.DICT,
            default_value={},
            description="Configuration for individual pipeline stages",
        ),
    ],
)

# All component configurations
COMPONENT_CONFIGS = {
    "memory_store": MEMORY_STORE_CONFIG,
    "vector_store": VECTOR_STORE_CONFIG,
    "retrieval_strategy": RETRIEVAL_STRATEGY_CONFIG,
    "query_analyzer": QUERY_ANALYZER_CONFIG,
    "query_adapter": QUERY_ADAPTER_CONFIG,
    "pipeline": PIPELINE_CONFIG,
}


def get_component_config(component_type: str) -> Optional[ComponentConfig]:
    """Get configuration options for a component type."""
    return COMPONENT_CONFIGS.get(component_type)


def get_default_config(component_type: str) -> Dict[str, Any]:
    """Get default configuration for a component type."""
    config = get_component_config(component_type)
    if not config:
        return {}

    return {option.name: option.default_value for option in config.options}
