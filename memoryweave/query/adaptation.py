"""Query parameter adaptation for MemoryWeave.

This module provides implementations for adapting retrieval parameters
based on query type and characteristics.
"""

from dataclasses import dataclass
from typing import Any, Dict

from memoryweave.interfaces.query import IQueryAdapter
from memoryweave.interfaces.retrieval import Query, QueryType, RetrievalParameters


@dataclass
class QueryTypeConfig:
    """Configuration for a specific query type."""

    similarity_threshold: float
    max_results: int
    recency_bias: float
    activation_boost: float
    keyword_weight: float
    min_results: int


class QueryTypeAdapter(IQueryAdapter):
    """Adapter that adjusts retrieval parameters based on query type."""

    def __init__(self):
        """Initialize the query type adapter."""
        # Default parameters for different query types
        self._type_configs = {
            QueryType.PERSONAL: QueryTypeConfig(
                similarity_threshold=0.65,
                max_results=8,
                recency_bias=0.4,
                activation_boost=0.3,
                keyword_weight=0.2,
                min_results=2,
            ),
            QueryType.FACTUAL: QueryTypeConfig(
                similarity_threshold=0.75,
                max_results=5,
                recency_bias=0.1,
                activation_boost=0.1,
                keyword_weight=0.3,
                min_results=1,
            ),
            QueryType.TEMPORAL: QueryTypeConfig(
                similarity_threshold=0.6,
                max_results=10,
                recency_bias=0.5,
                activation_boost=0.2,
                keyword_weight=0.2,
                min_results=3,
            ),
            QueryType.CONCEPTUAL: QueryTypeConfig(
                similarity_threshold=0.7,
                max_results=7,
                recency_bias=0.2,
                activation_boost=0.15,
                keyword_weight=0.3,
                min_results=2,
            ),
            QueryType.HISTORICAL: QueryTypeConfig(
                similarity_threshold=0.65,
                max_results=10,
                recency_bias=0.3,
                activation_boost=0.2,
                keyword_weight=0.25,
                min_results=2,
            ),
            QueryType.UNKNOWN: QueryTypeConfig(
                similarity_threshold=0.7,
                max_results=7,
                recency_bias=0.3,
                activation_boost=0.2,
                keyword_weight=0.25,
                min_results=2,
            ),
        }

        # Default configuration
        self._config = {
            "apply_keyword_boost": True,
            "scale_params_by_length": True,
            "length_threshold": 50,  # Character length to consider a long query
        }

    def adapt_parameters(self, query: Query) -> RetrievalParameters:
        """Adapt retrieval parameters based on query type."""
        # Get config for the query type
        query_type = query.query_type
        type_config = self._type_configs.get(query_type, self._type_configs[QueryType.UNKNOWN])

        # Create parameters from type config
        params = RetrievalParameters(
            similarity_threshold=type_config.similarity_threshold,
            max_results=type_config.max_results,
            recency_bias=type_config.recency_bias,
            activation_boost=type_config.activation_boost,
            keyword_weight=type_config.keyword_weight,
            min_results=type_config.min_results,
        )

        # Add extracted keywords if keyword boost is enabled
        if self._config["apply_keyword_boost"] and query.extracted_keywords:
            params["keywords"] = query.extracted_keywords

        # Add extracted entities
        if query.extracted_entities:
            params["entities"] = query.extracted_entities

        # Adjust parameters based on query length if enabled
        if self._config["scale_params_by_length"]:
            params = self._adjust_for_query_length(params, query.text)

        return params

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the query adapter."""
        if "apply_keyword_boost" in config:
            self._config["apply_keyword_boost"] = config["apply_keyword_boost"]

        if "scale_params_by_length" in config:
            self._config["scale_params_by_length"] = config["scale_params_by_length"]

        if "length_threshold" in config:
            self._config["length_threshold"] = config["length_threshold"]

        # Configure type-specific parameters
        if "type_configs" in config:
            for type_name, type_config in config["type_configs"].items():
                query_type = getattr(QueryType, type_name.upper(), None)
                if query_type and query_type in self._type_configs:
                    # Update specific fields
                    for field, value in type_config.items():
                        if hasattr(self._type_configs[query_type], field):
                            setattr(self._type_configs[query_type], field, value)

    def _adjust_for_query_length(
        self, params: RetrievalParameters, query_text: str
    ) -> RetrievalParameters:
        """Adjust parameters based on query length."""
        # For longer queries, we may want to:
        # - Lower the similarity threshold (more lenient matching)
        # - Increase max results (more comprehensive retrieval)
        # - Increase min results (ensure sufficient context)

        if len(query_text) > self._config["length_threshold"]:
            # Make a copy to avoid modifying the original
            adjusted = dict(params)

            # Scale threshold down for longer queries (but not below 0.5)
            length_factor = min(1.0, self._config["length_threshold"] / len(query_text))
            adjusted["similarity_threshold"] = max(
                0.5, params["similarity_threshold"] * (0.8 + 0.2 * length_factor)
            )

            # Increase max results for longer queries
            adjusted["max_results"] = min(
                20,  # Cap at 20 to avoid excessive results
                int(params["max_results"] * (1.0 + (1.0 - length_factor))),
            )

            # Increase min results for longer queries
            adjusted["min_results"] = min(
                5,  # Cap at 5 to avoid forcing too many results
                int(params["min_results"] * (1.0 + (1.0 - length_factor))),
            )

            return adjusted

        return params
