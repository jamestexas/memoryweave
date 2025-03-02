"""
Query type adaptation component for MemoryWeave.

This module provides components for adapting retrieval parameters based on
query type analysis.
"""

from typing import Any

from memoryweave.components.base import RetrievalComponent


class QueryTypeAdapter(RetrievalComponent):
    """
    Adapts retrieval parameters based on query type analysis.

    This component takes query type information from the QueryAnalyzer
    and adapts retrieval parameters accordingly, passing them to
    the retrieval strategies.
    """

    def __init__(self):
        self.adaptation_strength = 1.0  # How strongly to adapt (0.0-1.0)
        self.use_recommendations = True
        self.config_name = None  # Will be set by benchmark

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.adaptation_strength = config.get("adaptation_strength", 1.0)
        self.use_recommendations = config.get("use_recommendations", True)
        self.default_confidence_threshold = config.get("confidence_threshold", 0.3)
        self.default_adaptive_k_factor = config.get("adaptive_k_factor", 0.3)
        self.default_first_stage_k = config.get("first_stage_k", 20)
        self.default_first_stage_threshold_factor = config.get("first_stage_threshold_factor", 0.7)
        self.default_keyword_boost_weight = config.get("keyword_boost_weight", 0.5)

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
        logger = logging.getLogger(__name__)
        
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
        logger.info(f"QueryTypeAdapter: Context: enable_query_type_adaptation={enable_query_type_adaptation}, config_name={config_name}")
        
        # Skip processing if query type adaptation is not enabled for this configuration
        if not enable_query_type_adaptation:
            logger.info(f"QueryTypeAdapter: Skipping - query type adaptation not enabled for config {config_name}")
            # Return default params instead of empty dict to ensure consistent behavior
            default_params = {
                "adapted_retrieval_params": {
                    "confidence_threshold": self.default_confidence_threshold,
                    "adaptive_k_factor": self.default_adaptive_k_factor,
                    "first_stage_k": self.default_first_stage_k,
                    "first_stage_threshold_factor": self.default_first_stage_threshold_factor,
                    "keyword_boost_weight": self.default_keyword_boost_weight,
                    "adapted_by_query_type": False
                }
            }
            logger.info(f"QueryTypeAdapter: Returning default params: {default_params}")
            return default_params
        
        # Log initial state
        logger.info(f"QueryTypeAdapter.process_query: adaptation_strength={self.adaptation_strength} for config {config_name}")
        
        # Don't process if adaptation strength is zero
        if self.adaptation_strength <= 0:
            logger.warning(f"QueryTypeAdapter: adaptation_strength={self.adaptation_strength}, adaptation disabled")
            return {}

        # Get query type information
        primary_type = context.get("primary_query_type")
        if not primary_type:
            logger.warning("QueryTypeAdapter: No primary_query_type in context, skipping adaptation")
            return {}

        # Get parameter recommendations if available
        param_recommendations = context.get("retrieval_param_recommendations", {})
        
        logger.info(f"QueryTypeAdapter.process_query: primary_type={primary_type}, existing recommendations={param_recommendations}")

        # Base parameters to adapt
        adapted_params = {
            "confidence_threshold": self.default_confidence_threshold,
            "adaptive_k_factor": self.default_adaptive_k_factor,
            "first_stage_k": self.default_first_stage_k,
            "first_stage_threshold_factor": self.default_first_stage_threshold_factor,
            "keyword_boost_weight": self.default_keyword_boost_weight,
            "expand_keywords": False,
        }
        
        logger.info(f"QueryTypeAdapter.process_query: Base adapted_params={adapted_params}")

        # Use recommendations if available and enabled
        if self.use_recommendations and param_recommendations:
            # Interpolate between default and recommended parameters based on adaptation strength
            for key, recommended_value in param_recommendations.items():
                if key in adapted_params:
                    current_value = adapted_params[key]
                    # For numerical values, linearly interpolate
                    if isinstance(recommended_value, (int, float)) and isinstance(
                        current_value, (int, float)
                    ):
                        adapted_params[key] = current_value + self.adaptation_strength * (
                            recommended_value - current_value
                        )
                    # For booleans or other types, use recommended value if adaptation strength > 0.5
                    else:
                        if self.adaptation_strength > 0.5:
                            adapted_params[key] = recommended_value
        else:
            # Manually adapt based on query type if no recommendations
            self._manually_adapt_params(adapted_params, primary_type)

        # Store the adapted parameters for use by retrieval strategies
        logger.info(f"QueryTypeAdapter.process_query: Final adapted_params={adapted_params}")
        
        # Add a flag to indicate that parameters were adapted by this component
        adapted_params["adapted_by_query_type"] = True
        
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
        logger.info(f"QueryTypeAdapter._manually_adapt_params: Adapting for query_type={query_type}, config={config_name}")
        
        # Store original values for logging
        orig_params = params.copy()
        
        # Unified parameter adaptation based on query type
        if query_type == "personal":
            # Personal queries need higher precision - focus on quality of results
            # The stronger the adaptation_strength, the more we adjust for precision
            adjustment_factor = 0.2 * self.adaptation_strength
            
            params["confidence_threshold"] = min(0.9, params["confidence_threshold"] + adjustment_factor)
            params["adaptive_k_factor"] = min(0.9, params["adaptive_k_factor"] + adjustment_factor)
            params["first_stage_threshold_factor"] = min(1.0, params["first_stage_threshold_factor"] + (0.15 * self.adaptation_strength))
            params["top_k"] = max(1, params.get("top_k", 5) - int(1 * self.adaptation_strength))  # Reduce results for precision
            
        elif query_type == "factual":
            # Factual queries need better recall - prioritize finding all relevant information
            # The stronger the adaptation_strength, the more we adjust for recall
            adjustment_factor = 0.15 * self.adaptation_strength
            
            params["confidence_threshold"] = max(0.1, params["confidence_threshold"] - adjustment_factor)
            params["adaptive_k_factor"] = max(0.1, params["adaptive_k_factor"] - adjustment_factor)
            params["first_stage_k"] = params["first_stage_k"] + int(12 * self.adaptation_strength)
            params["first_stage_threshold_factor"] = max(0.5, params["first_stage_threshold_factor"] - (0.1 * self.adaptation_strength))
            params["expand_keywords"] = self.adaptation_strength > 0.3
            params["top_k"] = min(20, params.get("top_k", 5) + int(2 * self.adaptation_strength))  # More results for recall
            
        # Log the parameter changes
        changes = {k: (orig_params.get(k), params[k]) for k in params if k in orig_params and params[k] != orig_params[k]}
        logger.info(f"QueryTypeAdapter._manually_adapt_params: Changes for {query_type}: {changes}")
