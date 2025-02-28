# memoryweave/components/post_processors.py
from typing import Any

import numpy as np

from memoryweave.components.base import PostProcessor


class KeywordBoostProcessor(PostProcessor):
    """
    Boosts retrieval scores based on keyword matches.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.keyword_boost_weight = config.get("keyword_boost_weight", 0.5)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results by applying keyword boosting."""
        if not results:
            return results

        # Get keywords from context
        important_keywords = context.get("important_keywords", set())
        if not important_keywords:
            return results

        # Apply keyword boosting
        boosted_results = []
        for result in results:
            boost = self._calculate_keyword_boost(result, important_keywords)

            # Create a copy of the result with boosted score
            boosted_result = dict(result)
            boosted_result["original_score"] = result["relevance_score"]
            boosted_result["keyword_boost"] = boost
            boosted_result["relevance_score"] = result["relevance_score"] * boost

            boosted_results.append(boosted_result)

        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return boosted_results

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query by applying post-processing to results.

        Args:
            query: The query string
            context: Context dictionary containing results, etc.

        Returns:
            Updated context with processed results
        """
        results = context.get("results", [])

        # Process results
        processed_results = self.process_results(results, query, context)

        # Update context with processed results
        return {"results": processed_results}

    def _calculate_keyword_boost(
        self, memory_metadata: dict[str, Any], important_keywords: set[str]
    ) -> float:
        """Calculate a boost factor based on keyword matching."""
        if not important_keywords:
            return 1.0

        # Combine relevant text fields from memory
        memory_text = ""

        # Check different text fields that might exist in the metadata
        for field in ["text", "content", "description", "name"]:
            if field in memory_metadata:
                memory_text += " " + str(memory_metadata[field]).lower()

        # For interaction type, check response field too
        if memory_metadata.get("type") == "interaction" and "response" in memory_metadata:
            memory_text += " " + str(memory_metadata["response"]).lower()

        # Count matching keywords
        matches = sum(1 for keyword in important_keywords if keyword in memory_text)

        # Calculate boost factor (more matches = higher boost)
        if matches > 0:
            # Exponential boost for multiple keyword matches
            boost = 1.0 + min(2.5, 0.7 * matches)  # More aggressive boosting
            return boost

        return 1.0


class SemanticCoherenceProcessor(PostProcessor):
    """
    Filters memories to ensure semantic coherence among results.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.coherence_threshold = config.get("coherence_threshold", 0.2)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results by ensuring semantic coherence."""
        if len(results) <= 1:
            return results

        # Get query embedding from context
        query_embedding = context.get("query_embedding")
        if query_embedding is None:
            return results

        # Get embeddings for results
        result_embeddings = []
        for result in results:
            memory_id = result.get("memory_id")
            if isinstance(memory_id, int):
                memory = context.get("memory")
                if memory and memory_id < len(memory.memory_embeddings):
                    result_embeddings.append(memory.memory_embeddings[memory_id])

        if not result_embeddings:
            return results

        # Calculate pairwise similarities
        result_embeddings = np.array(result_embeddings)
        pairwise_similarities = np.dot(result_embeddings, result_embeddings.T)

        # Calculate average similarity for each result
        avg_similarities = (pairwise_similarities.sum(axis=1) - 1) / (len(results) - 1)

        # Filter results based on coherence threshold
        coherent_indices = np.where(avg_similarities >= self.coherence_threshold)[0]

        if len(coherent_indices) == 0:
            # Keep the highest scoring result if nothing passes threshold
            return [results[0]]

        return [results[i] for i in coherent_indices]

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query by applying post-processing to results.

        Args:
            query: The query string
            context: Context dictionary containing results, etc.

        Returns:
            Updated context with processed results
        """
        results = context.get("results", [])

        # Process results
        processed_results = self.process_results(results, query, context)

        # Update context with processed results
        return {"results": processed_results}


class AdaptiveKProcessor(PostProcessor):
    """
    Adaptively selects number of results based on score distribution.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.adaptive_k_factor = config.get("adaptive_k_factor", 0.3)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results by adaptively selecting k."""
        if len(results) <= 1:
            return results

        # Extract scores
        scores = np.array([r["relevance_score"] for r in results])
        diffs = np.diff(scores)

        # Find significant drops
        significance_threshold = self.adaptive_k_factor * scores[0]
        significant_drops = np.where((-diffs) > significance_threshold)[0]

        if len(significant_drops) > 0:
            # Use the first significant drop as the cut point
            cut_idx = significant_drops[0] + 1
            return results[:cut_idx]

        return results

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query by applying post-processing to results.

        Args:
            query: The query string
            context: Context dictionary containing results, etc.

        Returns:
            Updated context with processed results
        """
        results = context.get("results", [])

        # Process results
        processed_results = self.process_results(results, query, context)

        # Update context with processed results
        return {"results": processed_results}
