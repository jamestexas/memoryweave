"""
Query type adaptation component for MemoryWeave.

This module provides components for adapting retrieval parameters based on
query type analysis.
"""

import logging
from typing import Any

from pydantic import Field

from memoryweave.components.base import RetrievalComponent
from memoryweave.interfaces.retrieval import QueryType

logger = logging.getLogger(__name__)


class QueryTypeAdapter(RetrievalComponent):
    """
    Adapts retrieval parameters based on query type analysis.

    This component takes query type information from the QueryAnalyzer
    and adapts retrieval parameters accordingly, passing them to
    the retrieval strategies.
    """

    adaptation_strength: float = Field(1.0, description="How strongly to adapt (0.0-1.0)")
    use_recommendations: bool = Field(True, description="Whether to use recommendations")
    default_confidence_threshold: float = Field(0.3, description="Default confidence threshold")
    default_adaptive_k_factor: float = Field(0.3, description="Default adaptive k factor")
    default_first_stage_k: int = Field(20, description="Default first stage k")
    default_first_stage_threshold_factor: float = Field(
        0.7,
        description="Default first stage threshold factor",
    )
    default_keyword_boost_weight: float = Field(
        0.5,
        description="Default keyword boost weight",
    )
    config_name: str | None = None

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.adaptation_strength = config.get("adaptation_strength", self.adaptation_strength)
        self.use_recommendations = config.get("use_recommendations", self.use_recommendations)
        self.default_confidence_threshold = config.get(
            "confidence_threshold",
            self.default_confidence_threshold,
        )
        self.default_adaptive_k_factor = config.get(
            "adaptive_k_factor",
            self.default_adaptive_k_factor,
        )
        self.default_first_stage_k = config.get(
            "first_stage_k",
            self.default_first_stage_k,
        )
        self.default_first_stage_threshold_factor = config.get(
            "first_stage_threshold_factor",
            self.default_first_stage_threshold_factor,
        )
        self.default_keyword_boost_weight = config.get("keyword_boost_weight", 0.5)

    def process(self, input_data: Any) -> Any:
        """
        Process the input data as a pipeline stage.

        This method provides backward compatibility with the IQueryAdapter interface.

        Args:
            input_data: Input data to process

        Returns:
            Processed data
        """
        # If input is a Query object, adapt it
        if hasattr(input_data, "query_type"):
            # Adapt parameters
            params = self.adapt_parameters(input_data)

            # Return the query with adapted parameters
            result = dict(input_data)
            result["parameters"] = params
            return result

        # Otherwise pass through
        return input_data

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query to adapt retrieval parameters.

        Args:
            query: The query string
            context: Context from the pipeline, including query_types and
                    retrieval_param_recommendations if available

        Returns:
            Updated context with adapted parameters
        """
        import logging
        import traceback

        logger = logging.getLogger(__name__)

        logger.debug(
            f"[QueryTypeAdapter.process_query] Called on adapter id={id(self)} with query='{query[:50]}...'"
        )
        logger.debug("Call stack:\n" + "".join(traceback.format_stack(limit=5)))

        # Check if query type adaptation is explicitly enabled in context
        enable_query_type_adaptation = context.get("enable_query_type_adaptation", False)

        # Get config name from context, memory manager, or default
        if "config_name" in context:
            config_name = context["config_name"]
        elif "memory_manager" in context and hasattr(context["memory_manager"], "config_name"):
            config_name = context["memory_manager"].config_name
        else:
            config_name = "unknown"

        # Log context for debugging
        logger.debug(
            f"[QueryTypeAdapter.process_query] enable_query_type_adaptation={enable_query_type_adaptation}, "
            f"config_name={config_name}, self.adaptation_strength={self.adaptation_strength}"
        )

        # If adaptation is disabled, return default params
        if not enable_query_type_adaptation:
            logger.debug(
                f"[QueryTypeAdapter.process_query] Skipping - query type adaptation not enabled for config '{config_name}'"
            )
            default_params = dict(
                adapter_retrieval_params=dict(
                    confidence_threshold=self.default_confidence_threshold,
                    adaptive_k_factor=self.default_adaptive_k_factor,
                    first_stage_k=self.default_first_stage_k,
                    first_stage_threshold_factor=self.default_first_stage_threshold_factor,
                    keyword_boost_weight=self.default_keyword_boost_weight,
                    adapted_by_query_type=False,
                )
            )
            logger.debug(
                f"[QueryTypeAdapter.process_query] Returning default params: {default_params}"
            )
            return default_params

        # If adaptation_strength=0, also skip
        if self.adaptation_strength <= 0:
            logger.warning(
                f"[QueryTypeAdapter.process_query] adaptation_strength={self.adaptation_strength}, so adaptation disabled"
            )
            return {}

        # Attempt to get a primary_query_type
        primary_type = context.get("primary_query_type")
        if not primary_type:
            logger.warning(
                "[QueryTypeAdapter.process_query] No primary_query_type in context, skipping adaptation"
            )
            return {}

        # Check for param recommendations
        param_recommendations = context.get("retrieval_param_recommendations", {})

        logger.info(
            f"[QueryTypeAdapter.process_query] primary_type={primary_type}, existing recommendations={param_recommendations}"
        )

        # Base parameters
        adapted_params = {
            "confidence_threshold": self.default_confidence_threshold,
            "adaptive_k_factor": self.default_adaptive_k_factor,
            "first_stage_k": self.default_first_stage_k,
            "first_stage_threshold_factor": self.default_first_stage_threshold_factor,
            "keyword_boost_weight": self.default_keyword_boost_weight,
            "expand_keywords": False,
        }

        logger.info(f"[QueryTypeAdapter.process_query] Base adapted_params={adapted_params}")

        # If we have param recommendations, we can merge them
        if self.use_recommendations and param_recommendations:
            for key, recommended_value in param_recommendations.items():
                if key in adapted_params:
                    current_value = adapted_params[key]
                    if isinstance(recommended_value, (int, float)) and isinstance(
                        current_value, (int, float)
                    ):
                        # linear interpolation
                        adapted_params[key] = current_value + self.adaptation_strength * (
                            recommended_value - current_value
                        )
                    else:
                        if self.adaptation_strength > 0.5:
                            adapted_params[key] = recommended_value
        else:
            # Or do your manual adaptation
            self._manually_adapt_params(adapted_params, primary_type)

        # Mark that we adapted
        adapted_params["adapted_by_query_type"] = True
        logger.info(f"[QueryTypeAdapter.process_query] Final adapted_params={adapted_params}")

        return {"adapted_retrieval_params": adapted_params}

    def _manually_adapt_params(self, params: dict[str, Any], query_type: str) -> None:
        """
        Manually adapt parameters based on query type when recommendations aren't available.

        Args:
            params: Parameters to adapt (modified in place)
            query_type: The query type to adapt for
        """
        import logging

        logger = logging.getLogger(__name__)
        config_name = getattr(self, "config_name", "unknown")
        logger.info(
            f"QueryTypeAdapter._manually_adapt_params: Adapting for query_type={query_type}, config={config_name}"
        )

        # Store original values for logging
        orig_params = params.copy()

        # Unified parameter adaptation based on query type
        if query_type == "personal":
            # Personal queries need higher precision - focus on quality of results
            # The stronger the adaptation_strength, the more we adjust for precision
            adjustment_factor = 0.2 * self.adaptation_strength

            params["confidence_threshold"] = min(
                0.9, params["confidence_threshold"] + adjustment_factor
            )
            params["adaptive_k_factor"] = min(0.9, params["adaptive_k_factor"] + adjustment_factor)
            params["first_stage_threshold_factor"] = min(
                1.0, params["first_stage_threshold_factor"] + (0.15 * self.adaptation_strength)
            )
            params["top_k"] = max(
                1, params.get("top_k", 5) - int(1 * self.adaptation_strength)
            )  # Reduce results for precision

        elif query_type == "factual":
            # Factual queries need better recall - prioritize finding all relevant information
            # The stronger the adaptation_strength, the more we adjust for recall
            adjustment_factor = 0.15 * self.adaptation_strength

            params["confidence_threshold"] = max(
                0.1, params["confidence_threshold"] - adjustment_factor
            )
            params["adaptive_k_factor"] = max(0.1, params["adaptive_k_factor"] - adjustment_factor)
            params["first_stage_k"] = params["first_stage_k"] + int(12 * self.adaptation_strength)
            params["first_stage_threshold_factor"] = max(
                0.5, params["first_stage_threshold_factor"] - (0.1 * self.adaptation_strength)
            )
            params["expand_keywords"] = self.adaptation_strength > 0.3
            params["top_k"] = min(
                20, params.get("top_k", 5) + int(2 * self.adaptation_strength)
            )  # More results for recall

        # Log the parameter changes
        changes = {
            k: (orig_params.get(k), params[k])
            for k in params
            if k in orig_params and params[k] != orig_params[k]
        }
        logger.info(f"QueryTypeAdapter._manually_adapt_params: Changes for {query_type}: {changes}")

    def adapt_parameters(self, query_obj: dict[str, Any]) -> dict[str, Any]:
        """
        Adapt retrieval parameters based on query characteristics.

        Args:
            query_obj: Query object with type, keywords, etc.

        Returns:
            Dictionary of adapted parameters
        """
        query_type = query_obj.get("query_type")

        # Default parameters
        params = {
            "confidence_threshold": 0.1,
            "max_results": 10,
            "use_keyword_boost": True,
        }

        # Adapt based on query type
        if query_type == QueryType.FACTUAL:
            params["confidence_threshold"] = 0.15  # Higher precision
            params["max_results"] = 5  # Fewer, more focused results
        elif query_type == QueryType.PERSONAL:
            params["confidence_threshold"] = 0.05  # Higher recall
            params["max_results"] = 10  # More results to ensure personal info is found
        elif query_type == QueryType.TEMPORAL:
            params["confidence_threshold"] = 0.1  # Balanced
            params["max_results"] = 7  # Medium number of results

        # Adapt based on keyword presence
        keywords = query_obj.get("extracted_keywords", [])
        if keywords:
            params["use_keyword_boost"] = True
            # Adjust threshold based on keyword count
            if len(keywords) > 3:
                params["confidence_threshold"] *= 0.9  # Lower threshold with more keywords

        return params
