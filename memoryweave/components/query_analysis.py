# memoryweave/components/query_analysis.py
from typing import Any

from memoryweave.components.base import RetrievalComponent
from memoryweave.utils.nlp_extraction import NLPExtractor


class QueryAnalyzer(RetrievalComponent):
    """
    Analyzes queries to determine type and extract important information.
    """

    def __init__(self, nlp_model_name: str = "en_core_web_sm"):
        self.nlp_extractor = NLPExtractor(model_name=nlp_model_name)

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        pass

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Process a query to identify type and extract keywords."""
        # Identify query type
        query_types = self.nlp_extractor.identify_query_type(query)
        primary_type = max(query_types.items(), key=lambda x: x[1])[0]

        # Extract important keywords
        keywords = self.nlp_extractor.extract_important_keywords(query)

        # Recommend retrieval parameters based on query type
        param_recommendations = self._get_parameter_recommendations(primary_type, query_types)

        return {
            "query_types": query_types,
            "primary_query_type": primary_type,
            "important_keywords": keywords,
            "retrieval_param_recommendations": param_recommendations,
        }
        
    def _get_parameter_recommendations(self, primary_type: str, query_types: dict[str, float]) -> dict[str, Any]:
        """
        Get recommended retrieval parameters based on query type.
        
        Args:
            primary_type: The primary query type
            query_types: Dictionary of all query types and their scores
            
        Returns:
            Dictionary of recommended parameter values
        """
        # Default parameters (balanced)
        recommendations = {
            "confidence_threshold": 0.3,
            "adaptive_k_factor": 0.3,
            "first_stage_k": 20,
            "first_stage_threshold_factor": 0.7,
            "keyword_boost_weight": 0.5,
            "expand_keywords": False,
        }
        
        # Adapt based on query type
        if primary_type == "personal":
            # Personal queries need higher precision
            recommendations.update({
                "confidence_threshold": 0.4,  # Higher threshold for better precision
                "adaptive_k_factor": 0.4,     # More conservative K selection
                "first_stage_k": 15,          # Smaller candidate set
                "first_stage_threshold_factor": 0.8,  # Less aggressive first stage
                "keyword_boost_weight": 0.6,   # Higher keyword boost
                "expand_keywords": False,       # Don't expand keywords for personal queries
            })
        elif primary_type == "factual":
            # Factual queries need better recall
            recommendations.update({
                "confidence_threshold": 0.2,    # Lower threshold for better recall
                "adaptive_k_factor": 0.15,      # Less conservative K selection
                "first_stage_k": 30,            # Larger candidate set
                "first_stage_threshold_factor": 0.6,  # More aggressive first stage
                "keyword_boost_weight": 0.4,     # Lower keyword boost
                "expand_keywords": True,         # Expand keywords for factual queries
            })
        elif primary_type == "opinion":
            # Opinion queries need balanced approach with emphasis on recency
            recommendations.update({
                "confidence_threshold": 0.25,    # Moderate threshold
                "adaptive_k_factor": 0.25,       # Moderate K selection
                "recency_weight": 0.4,           # Higher recency weight
                "relevance_weight": 0.6,         # Lower relevance weight
                "expand_keywords": False,        # Don't expand keywords
            })
        elif primary_type == "instruction":
            # Instruction queries often need precise, factual responses
            recommendations.update({
                "confidence_threshold": 0.35,    # Higher threshold
                "adaptive_k_factor": 0.3,        # Standard K selection
                "keyword_boost_weight": 0.6,     # Higher keyword boost
                "expand_keywords": True,         # Expand keywords for better recall
            })
            
        # Adjust further based on certainty of classification
        primary_score = query_types.get(primary_type, 0.0)
        if primary_score > 0.8:
            # High confidence in classification - use more extreme parameters
            if primary_type == "personal":
                recommendations["confidence_threshold"] = 0.45  # Even higher threshold
            elif primary_type == "factual":
                recommendations["confidence_threshold"] = 0.15  # Even lower threshold
        elif primary_score < 0.4:
            # Low confidence in classification - use more balanced parameters
            recommendations["confidence_threshold"] = 0.3  # Default threshold
            recommendations["adaptive_k_factor"] = 0.3    # Default adaptive K
            recommendations["expand_keywords"] = False    # Don't expand keywords
            
        return recommendations
