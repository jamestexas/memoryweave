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

        return {
            "query_types": query_types,
            "primary_query_type": primary_type,
            "important_keywords": keywords,
        }
