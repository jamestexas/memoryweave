# memoryweave/components/dynamic_threshold_adjuster.py
"""
DynamicThresholdAdjuster

A PostProcessor that dynamically adjusts confidence thresholds based on
query characteristics and retrieval metrics to optimize memory retrieval performance.
"""

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from memoryweave.components.base import PostProcessor
from memoryweave.interfaces.retrieval import QueryType


class DynamicThresholdConfig(BaseModel):
    window_size: int = Field(5, description="Number of recent metrics to keep")
    adjustment_step: float = Field(0.05, description="Threshold adjustment step")
    min_threshold: float = Field(0.0, description="Minimum allowed threshold")
    max_threshold: float = Field(0.9, description="Maximum allowed threshold")
    min_result_count: int = Field(5, description="Minimum number of results to ensure")
    confidence_threshold: float = Field(0.0, description="Initial confidence threshold")
    enable_query_type_adaptation: bool = Field(
        True, description="Enable query type specific threshold adjustments"
    )
    enable_distribution_based_adjustment: bool = Field(
        True, description="Enable adjustment based on score distribution"
    )
    enable_feedback_learning: bool = Field(
        False, description="Enable learning from feedback (if available)"
    )
    target_result_count: int = Field(7, description="Target number of results to aim for")
    target_min_relevance: float = Field(0.3, description="Target minimum relevance score")
    smoothing_factor: float = Field(0.3, description="Smoothing factor for threshold updates (0-1)")


class DynamicThresholdAdjuster(PostProcessor):
    """
    Dynamically adjusts confidence thresholds based on query characteristics and
    retrieval metrics to optimize memory retrieval performance.

    Features:
    - Maintains a rolling window of retrieval metrics
    - Adjusts thresholds based on query type and characteristics
    - Analyzes relevance score distribution to optimize precision/recall tradeoff
    - Provides adaptive threshold adjustment for different retrieval strategies
    - Learns from explicit or implicit feedback (when available)
    """

    # Declare the expected configuration model
    config_model: ClassVar[DynamicThresholdConfig] = DynamicThresholdConfig

    def __init__(self) -> None:
        self.min_threshold = 0.1
        self.max_threshold = 0.8
        self.learning_rate = 0.05
        self.query_thresholds = {}  # Maps query types to optimal thresholds
        self.query_stats = {}  # Maps query types to retrieval stats

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the dynamic threshold adjuster using the given config."""
        parsed = self.config_model.parse_obj(config)
        self.window_size = parsed.window_size
        self.adjustment_step = parsed.adjustment_step
        self.min_threshold = parsed.min_threshold
        self.max_threshold = parsed.max_threshold
        self.min_result_count = parsed.min_result_count
        self.current_threshold = parsed.confidence_threshold
        self.enable_query_type_adaptation = parsed.enable_query_type_adaptation
        self.enable_distribution_based_adjustment = parsed.enable_distribution_based_adjustment
        self.enable_feedback_learning = parsed.enable_feedback_learning
        self.target_result_count = parsed.target_result_count
        self.target_min_relevance = parsed.target_min_relevance
        self.smoothing_factor = parsed.smoothing_factor

        # Initialize metrics tracking
        self.recent_metrics: list[dict[str, Any]] = []

        # Initialize query type specific thresholds
        self.query_type_thresholds = {
            "personal": 0.4,  # Personal questions need higher precision
            "factual": 0.25,  # Factual questions need higher recall
            "temporal": 0.3,  # Questions about time need medium threshold
            "conceptual": 0.35,  # Conceptual questions need higher precision
            "default": self.current_threshold,
        }

        # Initialize performance metrics
        self.performance_metrics = {
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_result_count": 0.0,
            "avg_relevance": 0.0,
            "threshold_history": [self.current_threshold],
        }

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Update retrieval metrics and adjust the confidence threshold dynamically.

        This method:
        1. Updates metrics for this retrieval
        2. Analyzes relevance score distribution
        3. Adjusts thresholds based on query type and past performance
        4. Ensures minimum result count
        5. Updates context with new thresholds

        Args:
            results: list of retrieval result dicts (each with a 'relevance_score')
            query: The query string
            context: The context dictionary (which we update with the new threshold)

        Returns:
            Updated list of results.
        """
        # Extract query type if available
        query_type = context.get("primary_query_type", "default")

        # Update metrics for this retrieval
        result_count = len(results)
        relevance_scores = [r.get("relevance_score", 0) for r in results] if results else []
        avg_score = np.mean(relevance_scores) if relevance_scores else 0.0

        # Calculate score distribution metrics if available
        score_distribution = self._analyze_score_distribution(relevance_scores)

        # Calculate retrieval metrics
        retrieval_metrics = {
            "result_count": result_count,
            "avg_score": avg_score,
            "query_type": query_type,
            "score_distribution": score_distribution,
            "query_length": len(query.split()),
            "has_entities": bool(context.get("entities", [])),
        }

        # Add feedback if available
        if "feedback" in context:
            retrieval_metrics["feedback"] = context["feedback"]

        # Add metrics to recent window
        self.recent_metrics.append(retrieval_metrics)
        if len(self.recent_metrics) > self.window_size:
            self.recent_metrics.pop(0)

        # Calculate and update thresholds
        self._update_thresholds(context)

        # Ensure minimum result guarantee
        results = self._ensure_minimum_results(results, query, context)

        # Update context with current threshold
        context["dynamic_confidence_threshold"] = self.current_threshold

        # Add query type specific thresholds to context
        context["query_type_thresholds"] = self.query_type_thresholds.copy()

        return results

    def _analyze_score_distribution(self, scores: list[float]) -> dict[str, float]:
        """Analyze the distribution of relevance scores."""
        if not scores:
            return {"variance": 0.0, "skew": 0.0, "gap_ratio": 0.0, "top_score": 0.0}

        # Calculate basic statistics
        sorted_scores = sorted(scores, reverse=True)
        variance = np.var(scores) if len(scores) > 1 else 0.0
        top_score = sorted_scores[0] if scores else 0.0

        # Calculate skew as a measure of distribution asymmetry
        try:
            from scipy import stats

            skew = float(stats.skew(scores)) if len(scores) > 2 else 0.0
        except (ImportError, Exception):
            # Simpler approximation if scipy not available
            if len(scores) > 2:
                median = np.median(scores)
                mean = np.mean(scores)
                std = np.std(scores)
                skew = 3 * (mean - median) / std if std > 0 else 0.0
            else:
                skew = 0.0

        # Calculate the gap ratio (ratio of largest gap to average gap)
        if len(sorted_scores) > 1:
            gaps = [sorted_scores[i] - sorted_scores[i + 1] for i in range(len(sorted_scores) - 1)]
            max_gap = max(gaps) if gaps else 0.0
            avg_gap = np.mean(gaps) if gaps else 0.0
            gap_ratio = max_gap / avg_gap if avg_gap > 0 else 0.0
        else:
            gap_ratio = 0.0

        return {
            "variance": float(variance),
            "skew": float(skew),
            "gap_ratio": float(gap_ratio),
            "top_score": float(top_score),
        }

    def _update_thresholds(self, context: dict[str, Any]) -> None:
        """Update thresholds based on recent metrics and context."""
        if len(self.recent_metrics) < 2:
            return

        # Get query type
        query_type = context.get("primary_query_type", "default")

        # Calculate average metrics
        avg_result_count = np.mean([m["result_count"] for m in self.recent_metrics])
        avg_score = np.mean([m["avg_score"] for m in self.recent_metrics if m["avg_score"] > 0])

        # Update general threshold based on retrieval performance
        threshold_adjustment = 0.0

        # Adjust for result count (target is self.target_result_count)
        if avg_result_count < self.target_result_count * 0.7:  # Too few results
            threshold_adjustment -= self.adjustment_step
        elif avg_result_count > self.target_result_count * 1.5:  # Too many results
            threshold_adjustment += self.adjustment_step * 0.5  # Less aggressive adjustment upward

        # Adjust for relevance quality
        if avg_score < self.target_min_relevance and avg_result_count > 0:
            threshold_adjustment += self.adjustment_step * 0.7
        elif (
            avg_score > self.target_min_relevance * 1.5
            and avg_result_count < self.target_result_count
        ):
            threshold_adjustment -= self.adjustment_step * 0.3

        # Apply distribution-based adjustment if enabled
        if self.enable_distribution_based_adjustment:
            # Get the most recent distribution
            latest_distribution = self.recent_metrics[-1].get("score_distribution", {})

            # High variance and gap ratio indicate clear relevant/irrelevant separation
            # - Can increase threshold to improve precision
            variance = latest_distribution.get("variance", 0.0)
            gap_ratio = latest_distribution.get("gap_ratio", 0.0)

            if gap_ratio > 2.0:  # Large gap between relevant and irrelevant results
                threshold_adjustment += self.adjustment_step * 0.4
            elif variance < 0.01 and avg_result_count > 3:  # Very uniform scores
                threshold_adjustment -= (
                    self.adjustment_step * 0.3
                )  # Lower threshold to get more diverse results

        # Apply feedback-based learning if enabled and feedback available
        if self.enable_feedback_learning:
            feedback_adjustments = []
            for metric in self.recent_metrics:
                if "feedback" in metric:
                    feedback = metric["feedback"]
                    if feedback.get("too_few_results", False):
                        feedback_adjustments.append(-self.adjustment_step)
                    if feedback.get("irrelevant_results", False):
                        feedback_adjustments.append(self.adjustment_step)

            if feedback_adjustments:
                avg_feedback_adjustment = np.mean(feedback_adjustments)
                threshold_adjustment += avg_feedback_adjustment

        # Update the general threshold with smoothing
        new_threshold = self.current_threshold + threshold_adjustment
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))

        # Apply smoothing to avoid rapid oscillation
        self.current_threshold = (
            1 - self.smoothing_factor
        ) * self.current_threshold + self.smoothing_factor * new_threshold

        # Update query type specific threshold if enabled
        if self.enable_query_type_adaptation and query_type in self.query_type_thresholds:
            # Calculate query-type specific metrics
            type_metrics = [m for m in self.recent_metrics if m.get("query_type") == query_type]
            if type_metrics:
                type_avg_count = np.mean([m["result_count"] for m in type_metrics])
                type_avg_score = np.mean(
                    [m["avg_score"] for m in type_metrics if m["avg_score"] > 0]
                )

                # Calculate appropriate threshold for this query type
                if type_avg_count < self.target_result_count * 0.7:
                    self.query_type_thresholds[query_type] = max(
                        self.min_threshold,
                        self.query_type_thresholds[query_type] - self.adjustment_step * 0.7,
                    )
                elif (
                    type_avg_count > self.target_result_count * 1.5
                    and type_avg_score < self.target_min_relevance
                ):
                    self.query_type_thresholds[query_type] = min(
                        self.max_threshold,
                        self.query_type_thresholds[query_type] + self.adjustment_step * 0.7,
                    )

        # Keep track of threshold history
        self.performance_metrics["threshold_history"].append(self.current_threshold)
        if len(self.performance_metrics["threshold_history"]) > self.window_size * 2:
            self.performance_metrics["threshold_history"].pop(0)

    def _ensure_minimum_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Ensure at least min_result_count results are returned."""
        # If we already have enough results, return as is
        if len(results) >= self.min_result_count:
            return results

        # If we have some results but need more, append more specific "not enough information" entries
        if results:
            missing_count = self.min_result_count - len(results)
            keywords = context.get("important_keywords", [])
            keywords_str = ", ".join(keywords) if keywords else query

            for _i in range(missing_count):
                results.append(
                    {
                        "memory_id": -1,  # Use negative ID to indicate synthetic result
                        "relevance_score": 0.1,
                        "content": f"Limited information available about: {keywords_str}",
                        "type": "synthetic",
                        "synthetic_reason": "minimum_result_guarantee",
                    }
                )
        else:
            # If no results at all, create a more specific "no information" entry
            results.append(
                {
                    "memory_id": -1,
                    "relevance_score": 0.1,
                    "content": f"No information found about: {query}",
                    "type": "synthetic",
                    "synthetic_reason": "no_results",
                }
            )

            # Add additional context to help with follow-up queries
            results.append(
                {
                    "memory_id": -2,
                    "relevance_score": 0.1,
                    "content": "You may want to provide more specific details or try a different phrasing.",
                    "type": "synthetic",
                    "synthetic_reason": "suggestion",
                }
            )

            # If query analysis is available, add some additional context
            query_type = context.get("primary_query_type", "")
            if query_type:
                if query_type == "personal":
                    results.append(
                        {
                            "memory_id": -3,
                            "relevance_score": 0.1,
                            "content": "I don't have personal information about that yet.",
                            "type": "synthetic",
                            "synthetic_reason": "query_type_response",
                        }
                    )
                elif query_type == "factual":
                    results.append(
                        {
                            "memory_id": -3,
                            "relevance_score": 0.1,
                            "content": "I don't have factual information about that in my memory.",
                            "type": "synthetic",
                            "synthetic_reason": "query_type_response",
                        }
                    )

        return results

    def get_adjusted_threshold(self, query_type, base_threshold: float = 0.1) -> float:
        """
        Get an adjusted confidence threshold based on query type and historical performance.

        Args:
            query_type: Type of the query (e.g., factual, personal)
            base_threshold: Base threshold to adjust from

        Returns:
            Adjusted confidence threshold
        """
        # Use query-specific threshold if available
        if hasattr(self, "query_thresholds") and query_type in self.query_thresholds:
            return self.query_thresholds[query_type]

        # If we don't have a specific threshold for this query type yet,
        # use a sensible default based on query type
        if query_type == QueryType.FACTUAL:
            # Higher threshold for factual queries - we want precision
            return min(base_threshold * 1.2, self.max_threshold)
        elif query_type == QueryType.PERSONAL:
            # Lower threshold for personal queries - we want recall
            return max(base_threshold * 0.8, self.min_threshold)
        elif query_type == QueryType.TEMPORAL:
            # Medium threshold for temporal queries
            return base_threshold

        # For unknown query types, use the base threshold
        return base_threshold

    def update_threshold(self, query_type, result_count: int, had_good_results: bool) -> None:
        """
        Update threshold based on retrieval results.

        Args:
            query_type: Type of the query
            result_count: Number of results retrieved
            had_good_results: Whether the results were satisfactory
        """
        # Initialize stats for this query type if not already present
        if query_type not in self.query_stats:
            self.query_stats[query_type] = {
                "count": 0,
                "good_results": 0,
                "total_results": 0,
            }

        # Update stats
        self.query_stats[query_type]["count"] += 1
        self.query_stats[query_type]["good_results"] += 1 if had_good_results else 0
        self.query_stats[query_type]["total_results"] += result_count

        # Get current threshold or use default
        current_threshold = self.query_thresholds.get(query_type, 0.1)

        # Adjust threshold based on results
        if had_good_results and result_count > 0:
            if result_count > 10:
                # Too many results, increase threshold
                new_threshold = current_threshold + self.learning_rate
            else:
                # Good number of results, small adjustment
                new_threshold = current_threshold + (self.learning_rate * 0.1)
        else:
            # Poor results, decrease threshold
            new_threshold = current_threshold - self.learning_rate

        # Ensure threshold is within bounds
        new_threshold = max(min(new_threshold, self.max_threshold), self.min_threshold)

        # Update threshold
        self.query_thresholds[query_type] = new_threshold
