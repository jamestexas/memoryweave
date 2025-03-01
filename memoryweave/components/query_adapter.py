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
        # Don't process if query type adaptation is disabled
        if self.adaptation_strength <= 0:
            return {}

        # Get query type information
        primary_type = context.get("primary_query_type")
        if not primary_type:
            return {}

        # Get parameter recommendations if available
        param_recommendations = context.get("retrieval_param_recommendations", {})

        # Base parameters to adapt
        adapted_params = {
            "confidence_threshold": self.default_confidence_threshold,
            "adaptive_k_factor": self.default_adaptive_k_factor,
            "first_stage_k": self.default_first_stage_k,
            "first_stage_threshold_factor": self.default_first_stage_threshold_factor,
            "keyword_boost_weight": self.default_keyword_boost_weight,
            "expand_keywords": False,
        }

        # Use recommendations if available and enabled
        if self.use_recommendations and param_recommendations:
            # Interpolate between default and recommended parameters based on adaptation strength
            for key, recommended_value in param_recommendations.items():
                if key in adapted_params:
                    current_value = adapted_params[key]
                    # For numerical values, linearly interpolate
                    if isinstance(recommended_value, (int, float)) and isinstance(current_value, (int, float)):
                        adapted_params[key] = current_value + self.adaptation_strength * (recommended_value - current_value)
                    # For booleans or other types, use recommended value if adaptation strength > 0.5
                    else:
                        if self.adaptation_strength > 0.5:
                            adapted_params[key] = recommended_value
        else:
            # Manually adapt based on query type if no recommendations
            self._manually_adapt_params(adapted_params, primary_type)

        # Store the adapted parameters for use by retrieval strategies
        return {"adapted_retrieval_params": adapted_params}

    def _manually_adapt_params(self, params: dict[str, Any], query_type: str) -> None:
        """
        Manually adapt parameters based on query type when recommendations aren't available.
        
        Args:
            params: Parameters to adapt (modified in place)
            query_type: The query type to adapt for
        """
        if query_type == "personal":
            # Personal queries need higher precision
            params["confidence_threshold"] = params["confidence_threshold"] + 0.1 * self.adaptation_strength
            params["adaptive_k_factor"] = params["adaptive_k_factor"] + 0.1 * self.adaptation_strength
            params["first_stage_threshold_factor"] = min(1.0, params["first_stage_threshold_factor"] + 0.1 * self.adaptation_strength)
        elif query_type == "factual":
            # Factual queries need better recall
            params["confidence_threshold"] = max(0.0, params["confidence_threshold"] - 0.1 * self.adaptation_strength)
            params["adaptive_k_factor"] = max(0.1, params["adaptive_k_factor"] - 0.15 * self.adaptation_strength)
            params["first_stage_k"] = params["first_stage_k"] + int(10 * self.adaptation_strength)
            params["first_stage_threshold_factor"] = max(0.5, params["first_stage_threshold_factor"] - 0.1 * self.adaptation_strength)
            params["expand_keywords"] = self.adaptation_strength > 0.5
