# memoryweave/components/post_processors.py
from typing import Any

from memoryweave.components.base import PostProcessor


class KeywordBoostProcessor(PostProcessor):
    """
    Boosts relevance scores of results containing important keywords.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.keyword_boost_weight = config.get("keyword_boost_weight", 0.5)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results by boosting for keyword matches."""
        # Get important keywords from query analysis
        keywords = context.get("important_keywords", set())
        if not keywords:
            return results

        # Apply keyword boost
        for result in results:
            content = str(result.get("content", "")).lower()

            # Count keyword matches
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content)

            # Apply boost proportional to matches and weight
            if keyword_matches > 0:
                boost = min(self.keyword_boost_weight * keyword_matches / len(keywords), 0.5)

                # Apply boost to relevance score
                current_score = result.get("relevance_score", 0)
                new_score = min(current_score + boost * (1.0 - current_score), 1.0)
                result["relevance_score"] = new_score
                result["keyword_boost_applied"] = True

        return results


class SemanticCoherenceProcessor(PostProcessor):
    """
    Adjusts relevance scores based on semantic coherence with query.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.coherence_threshold = config.get("coherence_threshold", 0.2)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results checking semantic coherence."""
        # Get query type from context
        query_type = context.get("primary_query_type", "factual")

        # Penalize incoherent results based on query type
        for result in results:
            # Check for type mismatch
            result_type = result.get("type", "unknown")

            if query_type == "personal" and result_type == "factual":
                # Penalize factual results for personal queries
                result["relevance_score"] = max(0, result.get("relevance_score", 0) - 0.2)

            elif query_type == "factual" and result_type == "personal":
                # Slightly penalize personal results for factual queries
                result["relevance_score"] = max(0, result.get("relevance_score", 0) - 0.1)

        return results


class AdaptiveKProcessor(PostProcessor):
    """
    Adjusts the number of results based on query characteristics.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.adaptive_k_factor = config.get("adaptive_k_factor", 0.3)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results by adjusting the number based on scores."""
        if not results:
            return results

        # Get original top_k
        original_k = context.get("top_k", 5)

        # Check result quality
        avg_score = sum(r.get("relevance_score", 0) for r in results) / len(results)

        # Adjust number of results based on quality
        if avg_score > 0.7:
            # High quality results - keep fewer
            adaptive_k = max(1, int(original_k * (1.0 - self.adaptive_k_factor)))
            return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)[:adaptive_k]
        elif avg_score < 0.3:
            # Low quality results - keep more for diversity
            return results
        else:
            # Medium quality - sort and return original amount
            return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)[:original_k]
